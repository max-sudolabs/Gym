# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from nemo_gym.tool_observability import ToolObservationRecorder, classify_http_status


async def test_records_success_and_http_failure() -> None:
    recorder: ToolObservationRecorder[int] = ToolObservationRecorder("root")

    async def ok() -> int:
        return 200

    async def rejected() -> int:
        return 422

    ok_result, ok_record = await recorder.run(ok, tool_call_id="call-ok", classify_result=classify_http_status)
    rejected_result, rejected_record = await recorder.run(
        rejected, tool_call_id="call-bad", classify_result=classify_http_status
    )
    assert (ok_result, rejected_result) == (200, 422)
    assert (ok_record.tool_call_id, rejected_record.tool_call_id) == ("call-ok", "call-bad")

    assert [record.status for record in recorder.records] == ["completed", "failed"]
    assert recorder.records[1].error_type == "http_422"
    assert all(record.duration_ms is not None and record.duration_ms >= 0 for record in recorder.records)


async def test_records_exception_before_reraising() -> None:
    recorder: ToolObservationRecorder[None] = ToolObservationRecorder("root")

    async def fail() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await recorder.run(fail, tool_call_id="call-1", tool_name="broken")

    [record] = recorder.records
    assert record.status == "failed"
    assert record.error_type == "RuntimeError"


async def test_wall_clock_rollback_does_not_mask_tool_result(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamps = iter((10.0, 9.0))
    monkeypatch.setattr("nemo_gym.tool_observability.time.time", lambda: next(timestamps))
    recorder: ToolObservationRecorder[str] = ToolObservationRecorder("root")

    result, record = await recorder.run(lambda: asyncio.sleep(0, result="ok"), tool_call_id="call-1")

    assert result == "ok"
    assert (record.started_at, record.completed_at) == (10.0, 10.0)


async def test_concurrent_calls_keep_independent_intervals_and_completion_order() -> None:
    recorder: ToolObservationRecorder[str] = ToolObservationRecorder("root")

    async def execute(value: str, delay: float) -> str:
        await asyncio.sleep(delay)
        return value

    slow_result, fast_result = await asyncio.gather(
        recorder.run(lambda: execute("slow", 0.02), tool_call_id="slow"),
        recorder.run(lambda: execute("fast", 0), tool_call_id="fast"),
    )

    assert (slow_result[0], fast_result[0]) == ("slow", "fast")
    assert slow_result[1].tool_call_id == "slow"
    assert fast_result[1].tool_call_id == "fast"
    slow_result[1].output = slow_result[0]
    fast_result[1].output = fast_result[0]
    by_id = {record.tool_call_id: record for record in recorder.records}
    assert {call_id: record.output for call_id, record in by_id.items()} == {"slow": "slow", "fast": "fast"}
    assert by_id["slow"].started_at <= by_id["fast"].completed_at
    assert by_id["fast"].started_at <= by_id["slow"].completed_at
    assert [record.tool_call_id for record in recorder.records] == ["fast", "slow"]
