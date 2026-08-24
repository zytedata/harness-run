"""Shared offline fakes for the Claude Agent SDK surface the harness touches.

``make_sdk_client(script)`` builds a stand-in for ``claude_agent_sdk.ClaudeSDKClient``
(monkeypatch it onto the ``claude_agent_sdk`` module — the harness imports it lazily at
call time). The script is a list of items played back by ``receive_messages()``:

* an SDK message            → yielded to the harness
* an ``Exception`` instance → raised out of the stream
* the string ``"hang"``     → blocks forever (lets the harness's timeout paths fire)
* a callable                → invoked with the client (e.g. to poke ``options.stderr``)
"""

from __future__ import annotations

import asyncio

from claude_agent_sdk import (
    ResultMessage,
    SystemMessage,
    TaskNotificationMessage,
    TaskStartedMessage,
    ToolResultBlock,
    UserMessage,
)


def result_msg(subtype="success", is_error=False, num_turns=1, cost=0.01, result="done"):
    return ResultMessage(
        subtype=subtype, duration_ms=10, duration_api_ms=8, is_error=is_error,
        num_turns=num_turns, session_id="claude-sid", total_cost_usd=cost, result=result,
        usage={"input_tokens": 1},
    )


def init_msg():
    # The CLI's init payload as it really arrives: the session id it picked and the
    # resolved MCP servers ride along with the model (see run_harness_conformance).
    return SystemMessage(
        subtype="init",
        data={"model": "m", "session_id": "claude-sid", "mcp_servers": []},
    )


def task_started_msg(task_id, description="bg work"):
    return TaskStartedMessage(
        subtype="task_started", data={}, task_id=task_id, description=description,
        uuid="u", session_id="claude-sid",
    )


def task_done_msg(task_id, status="completed"):
    return TaskNotificationMessage(
        subtype="task_notification", data={}, task_id=task_id, status=status,
        output_file="", summary=f"task {task_id} {status}", uuid="u",
        session_id="claude-sid",
    )


def tool_result_msg(tool_use_id="tu1", content="ok"):
    """A tool result flowing back to the model — the point where the CLI assembles the
    next model call (and injects any queued task notifications into it)."""
    return UserMessage(content=[ToolResultBlock(tool_use_id=tool_use_id, content=content)])


def make_sdk_client(script):
    """Build a fake ``ClaudeSDKClient`` class playing back ``script``; returns the class.

    The class records its instances on ``.instances`` so tests can assert on the prompt
    sent, the options received, and that ``disconnect()`` ran.
    """

    class FakeSDKClient:
        instances: list = []

        def __init__(self, options=None):
            self.options = options
            self.prompt = None
            self.disconnected = False
            type(self).instances.append(self)

        async def connect(self):
            pass

        async def query(self, prompt, session_id="default"):
            self.prompt = prompt

        async def receive_messages(self):
            for item in script:
                if isinstance(item, Exception):
                    raise item
                if callable(item):
                    item(self)
                    continue
                if item == "hang":
                    await asyncio.Event().wait()
                yield item

        async def disconnect(self):
            self.disconnected = True

    return FakeSDKClient
