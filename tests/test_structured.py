"""Best-effort structured-output parsing."""

from __future__ import annotations

from harness_run.structured import extract_json, parse_structured_output


def test_extract_from_fenced_block():
    text = "Here is the result:\n```json\n{\"a\": 1, \"b\": [2, 3]}\n```\nDone."
    assert extract_json(text) == {"a": 1, "b": [2, 3]}


def test_extract_bare_object_amid_prose():
    assert extract_json('prefix {"x": true} suffix') == {"x": True}


def test_extract_returns_none_for_no_json():
    assert extract_json("no json here") is None
    assert extract_json("") is None


def test_parse_with_jsonschema_dict_passes_valid_and_rejects_invalid():
    schema = {"type": "object", "required": ["n"], "properties": {"n": {"type": "integer"}}}
    assert parse_structured_output('{"n": 5}', schema) == {"n": 5}
    # invalid against schema → None (jsonschema is installed in dev)
    assert parse_structured_output('{"n": "not-int"}', schema) is None


def test_parse_with_pydantic_model():
    import pydantic

    class Book(pydantic.BaseModel):
        title: str
        pages: int

    out = parse_structured_output('```json\n{"title": "Dune", "pages": 412}\n```', Book)
    assert isinstance(out, Book) and out.title == "Dune" and out.pages == 412
    # malformed → None, never raises
    assert parse_structured_output('{"title": "x"}', Book) is None


def test_parse_no_schema_returns_parsed_json():
    # The runtime only calls this when output_schema is set; if a None schema reaches the
    # helper it returns the parsed JSON as-is (documented behavior), and None for no text.
    assert parse_structured_output('{"k": 1}', None) == {"k": 1}
    assert parse_structured_output(None, None) is None
