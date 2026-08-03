# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from time import perf_counter, time
from typing import Any, List
from uuid import uuid4

from fastapi import Request, Response
from pydantic import ConfigDict, ValidationError

from nemo_gym.base_resources_server import (
    AggregateMetrics,
    AggregateMetricsRequest,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
)
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig,
    Body,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    accumulate_response_usage,
)
from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body
from nemo_gym.rollout_observability import (
    AgentEpisode,
    AgentInvocation,
    AgentObservationBundle,
    ModelCallRef,
    TrajectoryTurn,
)
from nemo_gym.server_utils import get_response_json, raise_for_status
from nemo_gym.tool_observability import ToolObservationRecorder, classify_http_status


class SimpleAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_steps: int = None


class SimpleAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class SimpleAgentVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")


class SimpleAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    ng_agent_observations: AgentObservationBundle | None = None


class SimpleAgent(SimpleResponsesAPIAgent):
    config: SimpleAgentConfig

    async def _create_episode(
        self,
        body: NeMoGymResponseCreateParamsNonStreaming,
        *,
        model_url_path: str,
        resources_server_cookies: Any = None,
        task_id: str = "unscoped",
        rollout_id: str = "unscoped",
    ) -> tuple[AgentEpisode, Any, Any]:
        started = perf_counter()
        invocation_id = str(uuid4())
        recorder: ToolObservationRecorder[Any] = ToolObservationRecorder(invocation_id)
        model_calls: list[ModelCallRef] = []
        turns: list[TrajectoryTurn] = []
        body = body.model_copy(deep=True)

        if isinstance(body.input, str):
            body.input = [NeMoGymEasyInputMessage(role="user", content=body.input)]

        new_outputs = []
        usage = None
        step = 0
        model_server_cookies = None

        while True:
            step += 1
            new_body = body.model_copy(update={"input": body.input + new_outputs})
            turn_timestamp = time()

            model_response = await self.server_client.post(
                server_name=self.config.model_server.name,
                url_path=model_url_path,
                json=new_body,
                cookies=model_server_cookies,
            )
            # We raise for status here since we expect model calls to always work.
            await raise_for_status(model_response)
            model_response_json = await get_response_json(model_response)
            model_server_cookies = model_response.cookies
            try:
                model_response = NeMoGymResponse.model_validate(model_response_json)
            except ValidationError as e:
                raise RuntimeError(
                    f"Received an invalid response from model server: {json.dumps(model_response_json)}"
                ) from e

            model_call_ref = ModelCallRef(model_ref=self.config.model_server, response_id=model_response.id)
            model_calls.append(model_call_ref)
            output = model_response.output
            new_outputs.extend(output)
            reasoning = [item.model_dump(mode="json") for item in output if item.type == "reasoning"] or None
            turn = TrajectoryTurn(
                invocation_id=invocation_id,
                task_id=task_id,
                rollout_id=rollout_id,
                turn_no=step,
                timestamp=turn_timestamp,
                question=new_body.input,
                answer=output,
                reasoning_content=reasoning,
                step_count=len(recorder.records),
                model_calls=[model_call_ref],
            )
            turns.append(turn)

            usage = accumulate_response_usage(usage, model_response.usage)
            model_response.usage = None

            if model_response.incomplete_details:
                break

            all_fn_calls: List[NeMoGymResponseFunctionToolCall] = [o for o in output if o.type == "function_call"]
            all_output_messages: List[NeMoGymResponseOutputMessage] = [
                o for o in output if o.type == "message" and o.role == "assistant"
            ]
            if not all_fn_calls and all_output_messages:
                break

            for output_function_call in all_fn_calls:
                try:
                    parsed_arguments = json.loads(output_function_call.arguments)
                except (json.JSONDecodeError, TypeError) as e:
                    # Model produced malformed tool-call arguments. Surface the
                    # error back as a tool response so the rollout can continue
                    # (or terminate with a low reward) instead of crashing the
                    # whole batch on json.loads.
                    argument_error_type = type(e).__name__
                    # Use repr(e) so the exception type name is always
                    # included even when str(e) would be empty.
                    tool_output = json.dumps({"error": f"Invalid tool call arguments: {e!r}"})

                    async def malformed_arguments() -> None:
                        return None

                    _, observation = await recorder.run(
                        malformed_arguments,
                        tool_call_id=output_function_call.call_id,
                        tool_name=output_function_call.name,
                        classify_result=lambda _: ("failed", argument_error_type),
                    )
                    observation.output = tool_output
                    tool_response = NeMoGymFunctionCallOutput(
                        type="function_call_output",
                        call_id=output_function_call.call_id,
                        output=tool_output,
                    )
                    new_outputs.append(tool_response)
                    continue

                # We don't raise for status here since resource-server errors are valid tool outputs.
                async def execute_tool() -> tuple[Any, str]:
                    api_response = await self.server_client.post(
                        server_name=self.config.resources_server.name,
                        url_path=f"/{output_function_call.name}",
                        json=parsed_arguments,
                        cookies=resources_server_cookies,
                    )
                    return api_response, (await api_response.content.read()).decode()

                (api_response, tool_output), observation = await recorder.run(
                    execute_tool,
                    tool_call_id=output_function_call.call_id,
                    tool_name=output_function_call.name,
                    classify_result=lambda result: classify_http_status(result[0].status),
                )
                resources_server_cookies = api_response.cookies
                observation.output = tool_output

                tool_response = NeMoGymFunctionCallOutput(
                    type="function_call_output",
                    call_id=output_function_call.call_id,
                    output=tool_output,
                )
                new_outputs.append(tool_response)

            # Check if max steps is not None and if we have exhausted it.
            turn.step_count = len(recorder.records)
            if self.config.max_steps and step >= self.config.max_steps:
                break

        model_response.output = new_outputs
        model_response.usage = usage
        invocation = AgentInvocation(
            invocation_id=invocation_id,
            status="completed",
            duration_ms=(perf_counter() - started) * 1000,
            model_calls=model_calls,
            conversation=[*body.input, *new_outputs],
        )
        observations = AgentObservationBundle(
            source="simple_agent",
            records=[invocation, *turns, *recorder.records],
        )
        return (
            AgentEpisode(response=model_response, observations=observations),
            model_server_cookies,
            resources_server_cookies,
        )

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        episode, model_server_cookies, resources_server_cookies = await self._create_episode(
            body,
            model_url_path=self.url_path_for_request("/v1/responses", request),
            resources_server_cookies=request.cookies,
        )
        # Propogate any extra cookies necessary for downstream verification
        for k, v in (*resources_server_cookies.items(), *model_server_cookies.items()):
            response.set_cookie(k, v)
        return episode.response

    async def run(self, request: Request, body: SimpleAgentRunRequest) -> SimpleAgentVerifyResponse:
        cookies = request.cookies

        seed_session_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(seed_session_response)
        cookies = seed_session_response.cookies

        extra = body.model_extra or {}
        task_id = next(
            (
                str(extra[key])
                for key in ("task_id", "problem_id", "instance_id", "_ng_task_index")
                if extra.get(key) is not None
            ),
            "unknown",
        )
        episode, model_server_cookies, cookies = await self._create_episode(
            body.responses_create_params,
            model_url_path=self.url_path_for_run("/v1/responses", body),
            resources_server_cookies=cookies,
            task_id=task_id,
            rollout_id=maybe_rollout_id_from_run_body(body) or "unknown",
        )
        if model_server_cookies:
            cookies.update(model_server_cookies)

        verify_request = SimpleAgentVerifyRequest.model_validate(
            body.model_dump() | {"response": episode.response.model_dump(mode="json")}
        )

        verify_response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/verify",
            json=verify_request.model_dump(),
            cookies=cookies,
        )
        await raise_for_status(verify_response)
        result = await get_response_json(verify_response)
        resolved = result.get("resolved")
        turn_records = [record for record in episode.observations.records if isinstance(record, TrajectoryTurn)]
        if isinstance(resolved, bool) and turn_records:
            turn_records[-1].resolved = resolved
        result["ng_agent_observations"] = episode.observations.model_dump(mode="json")
        return SimpleAgentVerifyResponse.model_validate(result)

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        """Proxy aggregate_metrics to the resources server."""
        response = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/aggregate_metrics",
            json=body,
        )
        await raise_for_status(response)
        return AggregateMetrics.model_validate(await get_response_json(response))


if __name__ == "__main__":
    SimpleAgent.run_webserver()
