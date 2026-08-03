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
from unittest.mock import AsyncMock, MagicMock, call

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseReasoningItem,
    NeMoGymSummary,
)
from nemo_gym.rollout_observability import AgentObservationBundle, ToolCallObservation, TrajectoryTurn
from nemo_gym.server_utils import ServerClient
from responses_api_agents.simple_agent.app import (
    ModelServerRef,
    ResourcesServerRef,
    SimpleAgent,
    SimpleAgentConfig,
)


class TestApp:
    def test_sanity(self) -> None:
        config = SimpleAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="",
            ),
            model_server=ModelServerRef(
                type="responses_api_models",
                name="",
            ),
        )
        SimpleAgent(config=config, server_client=MagicMock(spec=ServerClient))

    async def test_responses(self, monkeypatch: MonkeyPatch) -> None:
        config = SimpleAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(
                type="responses_api_models",
                name="my server name",
            ),
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="",
            ),
        )
        server = SimpleAgent(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        client = TestClient(app)

        mock_response_data = {
            "id": "resp_688babb004988199b26c5250ba69c1e80abdf302bcd600d3",
            "created_at": 1753983920.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "id": "msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                    "content": [
                        {
                            "annotations": [],
                            "text": "Hello! How can I help you today?",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }

        dotjson_mock = AsyncMock()
        dotjson_mock.read.return_value = json.dumps(mock_response_data)
        dotjson_mock.cookies = MagicMock()
        server.server_client.post.return_value = dotjson_mock

        # No model provided should use the one from the config
        res_no_model = client.post("/v1/responses", json={"input": [{"role": "user", "content": "hello"}]})
        assert res_no_model.status_code == 200
        server.server_client.post.assert_called_with(
            server_name="my server name",
            url_path="/v1/responses",
            json=NeMoGymResponseCreateParamsNonStreaming(
                input=[NeMoGymEasyInputMessage(content="hello", role="user", type="message")]
            ),
            cookies=None,
        )

        actual_responses_dict = res_no_model.json()
        expected_responses_dict = {
            "id": "resp_688babb004988199b26c5250ba69c1e80abdf302bcd600d3",
            "created_at": 1753983920.0,
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": None,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "id": "msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                    "content": [
                        {
                            "annotations": [],
                            "text": "Hello! How can I help you today?",
                            "type": "output_text",
                            "logprobs": None,
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": True,
            "temperature": None,
            "tool_choice": "auto",
            "tools": [],
            "top_p": None,
            "background": None,
            "max_output_tokens": None,
            "max_tool_calls": None,
            "previous_response_id": None,
            "prompt": None,
            "reasoning": None,
            "service_tier": None,
            "status": None,
            "text": None,
            "top_logprobs": None,
            "truncation": None,
            "usage": None,
            "user": None,
            "conversation": None,
            "prompt_cache_key": None,
            "safety_identifier": None,
        }
        assert expected_responses_dict == actual_responses_dict

    def test_run_emits_standard_turns_and_tool_observation(self) -> None:
        config = SimpleAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="simple",
            model_server=ModelServerRef(type="responses_api_models", name="model"),
            resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
        )
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = {"observability_enabled": True}
        server = SimpleAgent(config=config, server_client=server_client)
        client = TestClient(server.setup_webserver())

        model_payloads = iter(
            [
                {
                    "id": "resp-tool",
                    "created_at": 1.0,
                    "model": "model",
                    "object": "response",
                    "output": [
                        {
                            "id": "reasoning-1",
                            "summary": [{"text": "look up the answer", "type": "summary_text"}],
                            "status": "completed",
                            "type": "reasoning",
                        },
                        {
                            "id": "fc-1",
                            "call_id": "call-1",
                            "name": "lookup",
                            "arguments": '{"q":"x"}',
                            "type": "function_call",
                            "status": "completed",
                        },
                    ],
                    "parallel_tool_calls": True,
                    "tool_choice": "auto",
                    "tools": [],
                },
                {
                    "id": "resp-final",
                    "created_at": 2.0,
                    "model": "model",
                    "object": "response",
                    "output": [
                        {
                            "id": "msg-1",
                            "content": [{"annotations": [], "text": "done", "type": "output_text"}],
                            "role": "assistant",
                            "status": "completed",
                            "type": "message",
                        }
                    ],
                    "parallel_tool_calls": True,
                    "tool_choice": "auto",
                    "tools": [],
                },
            ]
        )
        verify_cookies = None

        def mock_response(payload, *, cookies, status=200, content=None):
            response = MagicMock()
            response.ok = status < 400
            response.status = status
            response.cookies = cookies
            response.read = AsyncMock(return_value=json.dumps(payload))
            response.content.read = AsyncMock(return_value=(content or "").encode())
            return response

        async def post(*, server_name, url_path, **kwargs):
            nonlocal verify_cookies
            if url_path == "/seed_session":
                return mock_response({}, cookies={"seed": "1"})
            if url_path.endswith("/v1/responses"):
                payload = next(model_payloads)
                cookie = "m1" if payload["id"] == "resp-tool" else "m2"
                return mock_response(payload, cookies={"model": cookie})
            if url_path == "/lookup":
                return mock_response({}, cookies={"tool": "2"}, status=422, content="bad input")
            assert url_path == "/verify"
            verify_cookies = kwargs["cookies"]
            verify_request = kwargs["json"]
            return mock_response(
                verify_request | {"reward": 0.0, "resolved": False},
                cookies={},
            )

        server_client.post = AsyncMock(side_effect=post)
        response = client.post(
            "/run",
            json={
                "responses_create_params": {"input": [{"role": "user", "content": "question"}]},
                "instance_id": 0,
                "_ng_task_index": 4,
                "_ng_rollout_index": 1,
            },
        )

        assert response.status_code == 200
        bundle = AgentObservationBundle.model_validate(response.json()["ng_agent_observations"])
        turns = [record for record in bundle.records if isinstance(record, TrajectoryTurn)]
        tools = [record for record in bundle.records if isinstance(record, ToolCallObservation)]
        assert [(turn.task_id, turn.rollout_id) for turn in turns] == [("0", "4-1"), ("0", "4-1")]
        assert [turn.turn_no for turn in turns] == [1, 2]
        assert all(turn.timestamp > 0 for turn in turns)
        assert [turn.model_calls[0].response_id for turn in turns] == ["resp-tool", "resp-final"]
        assert turns[0].model_dump(mode="json")["question"] == [
            {"role": "user", "content": "question", "type": "message"}
        ]
        assert [item["type"] for item in turns[0].model_dump(mode="json")["answer"]] == [
            "reasoning",
            "function_call",
        ]
        assert turns[0].reasoning_content is not None
        assert turns[0].reasoning_content[0]["summary"][0]["text"] == "look up the answer"
        assert [turn.step_count for turn in turns] == [1, 1]
        assert turns[-1].resolved is False
        assert len(tools) == 1
        assert (tools[0].output, tools[0].status, tools[0].error_type) == ("bad input", "failed", "http_422")
        assert (
            tools[0].started_at is not None and tools[0].completed_at is not None and tools[0].duration_ms is not None
        )
        assert verify_cookies == {"tool": "2", "model": "m2"}

    async def test_responses_continues_on_malformed_tool_call_arguments(self, monkeypatch: MonkeyPatch) -> None:
        """Malformed JSON in a tool-call's arguments must not crash the rollout.

        The agent should surface the parse error back to the model as a
        function_call_output and let the loop continue (ultimately terminating
        on a normal assistant message).
        """
        config = SimpleAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(
                type="responses_api_models",
                name="my server name",
            ),
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="my resources server",
            ),
        )
        server = SimpleAgent(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        client = TestClient(app)

        mock_response_bad_tool_call = {
            "id": "resp_bad_tool_call",
            "created_at": 1753983920.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "my_tool",
                    # Not valid JSON.
                    "arguments": "{not json",
                    "type": "function_call",
                    "status": "completed",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }

        mock_response_chat_data = {
            "id": "resp_final",
            "created_at": 1753983921.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "id": "msg_final",
                    "content": [
                        {
                            "annotations": [],
                            "text": "Sorry, I'll stop calling that tool.",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }

        dotjson_mock = AsyncMock()
        dotjson_mock.read.side_effect = [
            json.dumps(mock_response_bad_tool_call),
            json.dumps(mock_response_chat_data),
        ]
        dotjson_mock.cookies = MagicMock()
        server.server_client.post.return_value = dotjson_mock

        res = client.post("/v1/responses", json={"input": [{"role": "user", "content": "hello"}]})
        assert res.status_code == 200

        # The resources server must not be called for a malformed tool call —
        # only the two model calls should hit server_client.post.
        post_call_kwargs = [c.kwargs for c in server.server_client.post.call_args_list]
        server_names_called = [kw["server_name"] for kw in post_call_kwargs]
        assert server_names_called == ["my server name", "my server name"]

        # The second model call's input must include the original function_call
        # plus a function_call_output describing the parse error.
        second_call_input = post_call_kwargs[1]["json"].input
        assert any(
            isinstance(item, NeMoGymResponseFunctionToolCall) and item.call_id == "call_1"
            for item in second_call_input
        )
        error_outputs = [
            item
            for item in second_call_input
            if isinstance(item, NeMoGymFunctionCallOutput) and item.call_id == "call_1"
        ]
        assert len(error_outputs) == 1
        error_payload = json.loads(error_outputs[0].output)
        assert "error" in error_payload
        assert "Invalid tool call arguments" in error_payload["error"]
        # The exception type must be visible to the model — repr(e) on a
        # JSONDecodeError starts with the class name.
        assert "JSONDecodeError" in error_payload["error"]

    async def test_responses_continues_on_reasoning_only(self, monkeypatch: MonkeyPatch) -> None:
        config = SimpleAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(
                type="responses_api_models",
                name="my server name",
            ),
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="",
            ),
        )
        server = SimpleAgent(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        client = TestClient(app)

        mock_response_reasoning_data = {
            "id": "resp_688babb004988199b26c5250ba69c1e80abdf302bcd600d3",
            "created_at": 1753983920.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "id": "msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                    "summary": [
                        {
                            "text": "I'm thinking how to respond",
                            "type": "summary_text",
                        }
                    ],
                    "status": "completed",
                    "type": "reasoning",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }

        mock_response_chat_data = {
            "id": "resp_688babb004988199b26c5250ba69c1e80abdf302bcd600d3",
            "created_at": 1753983920.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "id": "msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                    "content": [
                        {
                            "annotations": [],
                            "text": "Hello! How can I help you today?",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }

        dotjson_mock = AsyncMock()
        dotjson_mock.read.side_effect = [json.dumps(mock_response_reasoning_data), json.dumps(mock_response_chat_data)]
        dotjson_mock.cookies = MagicMock()
        server.server_client.post.return_value = dotjson_mock

        # No model provided should use the one from the config
        res_no_model = client.post("/v1/responses", json={"input": [{"role": "user", "content": "hello"}]})
        assert res_no_model.status_code == 200

        expected_calls = [
            call(
                server_name="my server name",
                url_path="/v1/responses",
                json=NeMoGymResponseCreateParamsNonStreaming(
                    input=[NeMoGymEasyInputMessage(content="hello", role="user", type="message")]
                ),
                cookies=None,
            ),
            call().ok.__bool__(),
            call().read(),
            call(
                server_name="my server name",
                url_path="/v1/responses",
                json=NeMoGymResponseCreateParamsNonStreaming(
                    input=[
                        NeMoGymEasyInputMessage(content="hello", role="user", type="message"),
                        NeMoGymResponseReasoningItem(
                            id="msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                            summary=[NeMoGymSummary(text="I'm thinking how to respond", type="summary_text")],
                            type="reasoning",
                            encrypted_content=None,
                            status="completed",
                        ),
                    ]
                ),
                cookies=dotjson_mock.cookies,
            ),
            call().ok.__bool__(),
            call().read(),
            call().cookies.items(),
            call().cookies.items().__iter__(),
            call().cookies.items().__len__(),
        ]
        server.server_client.post.assert_has_calls(expected_calls)

        actual_responses_dict = res_no_model.json()
        expected_responses_dict = {
            "id": "resp_688babb004988199b26c5250ba69c1e80abdf302bcd600d3",
            "created_at": 1753983920.0,
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": None,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "id": "msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                    "encrypted_content": None,
                    "summary": [
                        {
                            "text": "I'm thinking how to respond",
                            "type": "summary_text",
                        }
                    ],
                    "type": "reasoning",
                },
                {
                    "id": "msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                    "content": [
                        {
                            "annotations": [],
                            "text": "Hello! How can I help you today?",
                            "type": "output_text",
                            "logprobs": None,
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                },
            ],
            "parallel_tool_calls": True,
            "temperature": None,
            "tool_choice": "auto",
            "tools": [],
            "top_p": None,
            "background": None,
            "max_output_tokens": None,
            "max_tool_calls": None,
            "previous_response_id": None,
            "prompt": None,
            "reasoning": None,
            "service_tier": None,
            "status": None,
            "text": None,
            "top_logprobs": None,
            "truncation": None,
            "usage": None,
            "user": None,
            "conversation": None,
            "prompt_cache_key": None,
            "safety_identifier": None,
        }
        assert expected_responses_dict == actual_responses_dict

    async def test_usage_sanity(self, monkeypatch: MonkeyPatch) -> None:
        config = SimpleAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(
                type="responses_api_models",
                name="my server name",
            ),
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="",
            ),
            max_steps=3,
        )
        server = SimpleAgent(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        client = TestClient(app)

        mock_response_data = {
            "id": "resp_688babb004988199b26c5250ba69c1e80abdf302bcd600d3",
            "created_at": 1753983920.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "id": "msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                    "summary": [
                        {
                            "text": "Hello! How can I help you today?",
                            "type": "summary_text",
                        }
                    ],
                    "status": "completed",
                    "type": "reasoning",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }

        response_1 = mock_response_data | {
            "usage": {
                "input_tokens": 1,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 2,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 3,
            },
        }
        response_2 = mock_response_data | {"usage": None}
        response_3 = mock_response_data | {
            "usage": {
                "input_tokens": 100,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 200,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 300,
            },
        }

        dotjson_mock = AsyncMock()
        dotjson_mock.read.side_effect = [json.dumps(response_1), json.dumps(response_2), json.dumps(response_3)]
        dotjson_mock.cookies = MagicMock()
        server.server_client.post.return_value = dotjson_mock

        # No model provided should use the one from the config
        res_no_model = client.post("/v1/responses", json={"input": [{"role": "user", "content": "hello"}]})
        assert res_no_model.status_code == 200

        actual_responses_dict = res_no_model.json()
        actual_usage_dict = actual_responses_dict["usage"]
        expected_usage_dict = {
            "input_tokens": 101,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 202,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 303,
        }
        assert expected_usage_dict == actual_usage_dict

    async def test_incomplete_details(self, monkeypatch: MonkeyPatch) -> None:
        await self._test_incomplete_details_helper(monkeypatch, {"reason": "max_output_tokens"})
        await self._test_incomplete_details_helper(monkeypatch, {"reason": "content_filter"})

    async def _test_incomplete_details_helper(self, monkeypatch: MonkeyPatch, incomplete_details) -> None:
        config = SimpleAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(
                type="responses_api_models",
                name="my server name",
            ),
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="",
            ),
        )
        server = SimpleAgent(config=config, server_client=MagicMock(spec=ServerClient))
        app = server.setup_webserver()
        client = TestClient(app)

        mock_response_reasoning_data = {
            "id": "resp_688babb004988199b26c5250ba69c1e80abdf302bcd600d3",
            "created_at": 1753983920.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "id": "msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                    "summary": [
                        {
                            "text": "I'm thinking how to respond",
                            "type": "summary_text",
                        }
                    ],
                    "status": "completed",
                    "type": "reasoning",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "incomplete_details": incomplete_details,
        }

        mock_response_chat_data = {
            "id": "resp_688babb004988199b26c5250ba69c1e80abdf302bcd600d3",
            "created_at": 1753983920.0,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "id": "msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                    "content": [
                        {
                            "annotations": [],
                            "text": "Hello! How can I help you today?",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }

        dotjson_mock = AsyncMock()
        dotjson_mock.read.side_effect = [json.dumps(mock_response_reasoning_data), json.dumps(mock_response_chat_data)]
        dotjson_mock.cookies = MagicMock()
        server.server_client.post.return_value = dotjson_mock

        # No model provided should use the one from the config
        res_no_model = client.post("/v1/responses", json={"input": [{"role": "user", "content": "hello"}]})
        assert res_no_model.status_code == 200

        expected_calls = [
            call(
                server_name="my server name",
                url_path="/v1/responses",
                json=NeMoGymResponseCreateParamsNonStreaming(
                    input=[NeMoGymEasyInputMessage(content="hello", role="user", type="message")]
                ),
                cookies=None,
            ),
            call().ok.__bool__(),
            call().read(),
            call().cookies.items(),
            call().cookies.items().__iter__(),
            call().cookies.items().__len__(),
        ]
        server.server_client.post.assert_has_calls(expected_calls)

        actual_responses_dict = res_no_model.json()
        expected_responses_dict = {
            "id": "resp_688babb004988199b26c5250ba69c1e80abdf302bcd600d3",
            "created_at": 1753983920.0,
            "error": None,
            "incomplete_details": incomplete_details,
            "instructions": None,
            "metadata": None,
            "model": "dummy_model",
            "object": "response",
            "output": [
                {
                    "id": "msg_688babb17a7881998cc7a42d53c8e5790abdf302bcd600d3",
                    "encrypted_content": None,
                    "summary": [
                        {
                            "text": "I'm thinking how to respond",
                            "type": "summary_text",
                        }
                    ],
                    "type": "reasoning",
                },
            ],
            "parallel_tool_calls": True,
            "temperature": None,
            "tool_choice": "auto",
            "tools": [],
            "top_p": None,
            "background": None,
            "max_output_tokens": None,
            "max_tool_calls": None,
            "previous_response_id": None,
            "prompt": None,
            "reasoning": None,
            "service_tier": None,
            "status": None,
            "text": None,
            "top_logprobs": None,
            "truncation": None,
            "usage": None,
            "user": None,
            "conversation": None,
            "prompt_cache_key": None,
            "safety_identifier": None,
        }
        assert expected_responses_dict == actual_responses_dict
