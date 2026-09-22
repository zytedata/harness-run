# Structured output

Set `output_schema` to a **pydantic model** (or a JSON-schema `dict`) and `result.structured_output` holds
the parsed, validated object. The harness sends the schema through the CLI's native structured-output
option. Tell the agent what content to emit in the prompt; the schema describes its shape. OpenRouter may
leave schema enforcement to the selected model, so those turns also receive the schema in their prompt.
The library validates the SDK's structured value when available and can parse JSON from the final text (a
```` ```json ```` block, a bare object, or the whole message) as a fallback.

```python
import json
import pydantic
from harness_run import AgentSpec, local

class PriceCheck(pydantic.BaseModel):
    in_stock: bool
    price_gbp: float

spec = AgentSpec(name="price-check", model="claude-sonnet-4-6", harness="claude-code", output_schema=PriceCheck)
prompt = (
    "Check the product at <url> and report the result as a JSON object matching this schema:\n"
    f"{json.dumps(PriceCheck.model_json_schema())}"
)
result = await local.deploy(spec).start_session().run(prompt)
print(result.structured_output)    # PriceCheck(in_stock=True, price_gbp=51.77), or None if unparseable
```

Parsing is best-effort: if the final message has no JSON matching the schema, `structured_output` is `None`
and you still have `result.text`. A runnable version is in [`examples/minimal`](../examples/minimal).
