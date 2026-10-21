from copy import deepcopy

import pytest

from scripts.run_asd_suite import prefix_agreement
from scripts.write_asd_suite_report import aggregate, paired


def test_bootstrap_uses_question_ratios_and_is_reproducible():
    one=paired([.8,1.2,.9,1.1],"fixed")
    assert one==paired([.8,1.2,.9,1.1],"fixed")
    assert one["n"]==4 and one["slower"]==2
    assert one["ci95"][0]<1<one["ci95"][1]
    assert paired([],"fixed") is None
    with pytest.raises(ValueError):
        paired([0],"fixed")


def test_repeats_are_merged_before_uncertainty_and_ar_calls_are_not_zero():
    rows=[dict(request_id="q",benchmark="gsm8k",method="AR",stage="natural",latency_seconds=s,
               token_ids=[1]*10,finish_reason="eos",quality=dict(correct=False),asd=None,rounds=[])
          for s in (1.,2.,3.)]
    item=aggregate(rows)["q","AR","natural"]
    assert item["seconds"]==2 and item["target_forwards"]==10 and not item["quality"]["correct"]


def test_length_cropped_last_block_is_not_used_to_hide_earlier_divergence():
    full=dict(token_ids=list(range(30)))
    prefix=dict(token_ids=list(range(10)),rounds=[dict(generated_before=1,proposed=15),dict(generated_before=8,proposed=2)])
    assert prefix_agreement(full,prefix)["agrees"]
    prefix["token_ids"][9]=99
    assert prefix_agreement(full,prefix)["agrees"]
    prefix["token_ids"][3]=99
    assert not prefix_agreement(full,prefix)["agrees"]
