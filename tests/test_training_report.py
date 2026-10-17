import copy

import pytest

from scripts.write_training_report import validate_training_rows


def test_training_report_requires_complete_repeats_and_original_teacher_labels():
    labels = [dict(request_id="a", token_ids=[1, 2], target_prompt_sha256="teacher")]
    data = [dict(request_id="a", repeat=repeat, method=method, token_ids=[1, 2],
                 target_prompt_sha256="teacher", correct=True)
            for repeat in range(3) for method in ("AR", "TRAIN_D", "TRAIN_H")]
    validate_training_rows(data, labels)
    with pytest.raises(ValueError, match="Incomplete"):
        validate_training_rows(data[:-1], labels)
    altered = copy.deepcopy(data)
    altered[-1]["token_ids"] = [9, 2]
    with pytest.raises(ValueError, match="frozen"):
        validate_training_rows(altered, labels)
