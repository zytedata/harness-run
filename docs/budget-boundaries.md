# Budget admission versus a hard spending ceiling

For the toolkit OpenRouter proxy, `max_budget_usd=0` permits no upstream requests,
even before the first cost is recorded. An exhausted allowance returns HTTP 402 before
contacting OpenRouter. Negative and non-finite limits are rejected; `None` means no
proxy budget gate.

This does **not** make a positive budget a hard monetary ceiling. Accounting is based
on completed responses: in-flight/concurrent requests can exceed the remaining balance,
and a response without known cost is not a reserved maximum charge. Other providers,
agent subagents, tool/service calls, runtime jobs and direct cloud credentials have
separate costs and limits. Cancellation is not instantaneous and observing a stream
ending is not confirmation that billable remote execution has stopped.

Use provider-side prepaid/hard limits where available, explicit quota/concurrency
admission, scoped capabilities and a budget authority covering all enabled services
when promising a platform-wide ceiling. The zero-budget regression is intentionally
narrow and uses only a loopback listener plus a fake upstream; no paid overspend test
is necessary. It does not remedy the wider platform cost-control design by itself.
