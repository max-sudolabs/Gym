# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared timing wrapper for agent-owned asynchronous tool execution."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Generic, Literal, Optional, TypeVar

from nemo_gym.rollout_observability import ToolCallObservation


T = TypeVar("T")
ToolStatus = Literal["completed", "failed", "timeout", "cancelled", "incomplete", "unknown"]
ResultClassifier = Callable[[T], tuple[ToolStatus, Optional[str]]]


def classify_http_status(status_code: int) -> tuple[ToolStatus, Optional[str]]:
    """Classify a resource-server HTTP result without raising away its body."""

    if 200 <= status_code < 400:
        return "completed", None
    return "failed", f"http_{status_code}"


class ToolObservationRecorder(Generic[T]):
    """Record independent timing and status for asynchronous tool calls."""

    def __init__(self, invocation_id: str) -> None:
        self.invocation_id = invocation_id
        self.records: list[ToolCallObservation] = []

    async def run(
        self,
        operation: Callable[[], Awaitable[T]],
        *,
        tool_call_id: str,
        tool_name: Optional[str] = None,
        classify_result: Optional[ResultClassifier[T]] = None,
    ) -> tuple[T, ToolCallObservation]:
        started_at = time.time()
        started_monotonic = time.perf_counter()
        status: ToolStatus = "completed"
        error_type: Optional[str] = None
        try:
            result = await operation()
            if classify_result is not None:
                status, error_type = classify_result(result)
        except asyncio.CancelledError:
            status = "cancelled"
            error_type = "cancelled"
            raise
        except TimeoutError:
            status = "timeout"
            error_type = "timeout"
            raise
        except Exception as exc:
            status = "failed"
            error_type = type(exc).__name__
            raise
        finally:
            completed_at = max(started_at, time.time())
            duration_ms = (time.perf_counter() - started_monotonic) * 1000
            observation = ToolCallObservation(
                invocation_id=self.invocation_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                timing_source="executor",
                status=status,
                error_type=error_type,
            )
            self.records.append(observation)
        return result, observation
