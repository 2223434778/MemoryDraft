from pathlib import Path

import pytest

from memorydraft.evaluation.asd_extension_contracts import evaluate,json_payloads,load_plan,question_sha

ROOT=Path(__file__).resolve().parents[1]


def test_question_hash_excludes_reference_answer():
    record=dict(id='alpaca:1',benchmark='alpaca',instruction='Return three items',input='',output='reference')
    first=question_sha(record);record['output']='changed private evaluation reference'
    assert question_sha(record)==first
    record['instruction']='Return four items'
    assert question_sha(record)!=first


@pytest.mark.parametrize('check,good,bad',[
    ('json_groups','```json\n{"people":["A"],"org":[],"locations":[]}\n```','{"people":"A","org":[],"locations":[]}'),
    ('json_integer_ratings','[5,1,3]','[true,1,3]'),
    ('single_line_json','{"people":["A"]}','```json\n{"people":["A"]}\n```'),
    ('gendered_pronouns','They found the purse while hearing the question.',"They asked; she's unsure."),
    ('section_order','**Header:** Intro\n**Body:** Content\n**Footer:** Links','Footer: links\nBody: content\nHeader: intro'),
])
def test_structural_properties_have_positive_and_negative_controls(check,good,bad):
    assert evaluate(dict(check=check),good)['verdict']=='pass'
    assert evaluate(dict(check=check),bad)['verdict']=='fail'


def test_years_must_be_inside_the_json_payload():
    rule=dict(check='json_years',years=[2019,2022,2018])
    assert evaluate(rule,'[{"date":"2019-11-18"},{"date":2022},{"date":"Feb 2018"}]')['verdict']=='pass'
    assert evaluate(rule,'2019 2022 2018\n[5,1,3]')['verdict']=='fail'


def test_word_count_convention_and_partial_scope_are_explicit():
    rule=dict(check='word_limit',maximum=3)
    assert evaluate(rule,"Don't re-count 25")['verdict']=='pass'
    assert evaluate(rule,'one two three four')['verdict']=='fail'
    assert evaluate(dict(check='manual',criterion='Interpret the final answer'),'any text')['verdict']=='needs_review'
    with pytest.raises(ValueError):evaluate(dict(check='invented'),'anything')


def test_redaction_checks_literal_fields_without_substring_false_positives():
    rule=dict(check='redaction_literals',patterns={'state':r'\bCA\b','name':r'\bJohn\b'})
    assert evaluate(rule,'[REDACTED] contains no original location')['verdict']=='pass'
    assert evaluate(rule,'State CA, name [REDACTED]')['verdict']=='fail'


def test_invalid_json_is_not_executed_or_silently_accepted():
    assert json_payloads('{bad}')==[]
    assert evaluate(dict(check='json_integer_ratings'),'[5,1,3,4]')['verdict']=='fail'
    assert evaluate(dict(check='json_integer_ratings'),'{"ratings":[5,1,3]}')['verdict']=='fail'
    assert evaluate(dict(check='json_integer_ratings'),'{"ratings":[5,1,3]')['verdict']=='fail'


def test_json_section_labels_are_a_valid_alternative_to_markdown():
    assert evaluate(dict(check='section_order'),'{"Header":"intro","Body":"text","Footer":"links"}')['verdict']=='pass'


def test_frozen_property_plan_binds_only_selected_question_fields():
    plan=load_plan(ROOT)
    assert len(plan['rules'])==34
    assert sum(r['check']=='manual' for r in plan['rules'])==25
    assert len(plan['source_flags'])==6
