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


from toolcall.semantic import calls_equivalent, resolve

CIRCLE = [{"name": "circle_area", "parameters": {"radius": {"type": "float"}}}]
CHI = [{"name": "chi", "parameters": {"table": {"type": "list"},
                                      "significance_level": {"type": "float, optional", "default": 0.05}}}]


def test_arithmetic_strings_are_evaluated_without_eval():
    assert resolve("7 / 2") == 3.5 and resolve("10 ** 2") == 100 and resolve("10^2") == 100
    assert abs(resolve("math.sqrt(2)") - 2 ** 0.5) < 1e-12
    assert resolve("__import__('os').system('x')") == "__import__('os').system('x')"
    assert resolve("[1, [2, 3]]") == [1, [2, 3]]


def test_equivalent_values_and_defaults_match():
    pred = [{"name": "circle_area", "arguments": {"radius": 3.5}}]
    assert calls_equivalent(pred, [{"name": "circle_area", "arguments": {"radius": "7 / 2"}}], CIRCLE)
    with_default = [{"name": "chi", "arguments": {"table": [[1, 2]], "significance_level": 0.05}}]
    omitted = [{"name": "chi", "arguments": {"table": [[1, 2]]}}]
    assert calls_equivalent(with_default, omitted, CHI)
    other = [{"name": "chi", "arguments": {"table": [[1, 2]], "significance_level": 0.01}}]
    assert not calls_equivalent(other, omitted, CHI)


def test_different_values_stay_different():
    hours = [{"name": "m", "arguments": {"intervals": [[12, 13]]}}]
    hhmm = [{"name": "m", "arguments": {"intervals": [[1200, 1300]]}}]
    assert not calls_equivalent(hours, hhmm, [{"name": "m", "parameters": {}}])
    assert not calls_equivalent(hours, hours * 2, [{"name": "m", "parameters": {}}])


def test_unparseable_literals_stay_strings():
    assert resolve("{[1, 2]}") == "{[1, 2]}"
    assert resolve("(" * 500) == "(" * 500
