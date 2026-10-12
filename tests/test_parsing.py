from toolcall.data import is_noisy
from toolcall.metrics import calls_equal, categorize
from toolcall.parsing import parse_lenient, parse_strict

CALL = '<tool_call>\n{"name": "circle_area", "arguments": {"radius": 6}}\n</tool_call>'
EXPECTED = [{"name": "circle_area", "arguments": {"radius": 6}}]


def test_strict_accepts_well_formed_calls():
    assert parse_strict(CALL + "\n" + CALL) == EXPECTED * 2


def test_strict_rejects_the_old_finetunes_broken_opening_tag():
    broken = CALL.replace("<tool_call>", "</tool_call>", 1)
    assert parse_strict(broken) is None
    assert parse_lenient(broken) == EXPECTED
    assert categorize(broken, None, EXPECTED) == "format"


def test_no_tool_call_is_an_empty_list_not_false():
    assert parse_strict("I cannot help with that.") == []
    assert categorize("I cannot help with that.", [], EXPECTED) == "no_call"


def test_order_and_int_float_do_not_matter():
    a = [{"name": "f", "arguments": {"x": 1.0}}, {"name": "g", "arguments": {}}]
    b = [{"name": "g", "arguments": {}}, {"name": "f", "arguments": {"x": 1}}]
    assert calls_equal(a, b)
    assert not calls_equal(a, a[:1])


def test_string_expression_for_a_float_is_noisy():
    tools = [{"name": "circle_area", "parameters": {"radius": {"type": "float"}}}]
    assert is_noisy({"tools": tools, "answers": [{"name": "circle_area", "arguments": {"radius": "7 / 2"}}]})
    assert not is_noisy({"tools": tools, "answers": [{"name": "circle_area", "arguments": {"radius": 3}}]})
