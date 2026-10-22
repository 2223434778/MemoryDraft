import json

from scripts import prepare_asd_adaptation_followup as module


def test_reexport_to_another_directory_does_not_exclude_its_own_frozen_selection(tmp_path, monkeypatch):
    root = tmp_path/"repo"
    suite = root/"data/research/asd-suite"
    suite.mkdir(parents=True)
    source_rows = [dict(question=f"Fixture question {i}?", answer=f"#### {i}") for i in range(80)]
    observed = [dict(source_rows[i], id=f"gsm8k:{i}", benchmark="gsm8k", split="development") for i in range(8)]
    (suite/"gsm8k.json").write_text(json.dumps(observed), encoding="utf-8")
    license_dir = root/"data/research/asd-adaptation"
    license_dir.mkdir()
    (license_dir/"GSM8K_LICENSE.txt").write_text("Test fixture license", encoding="utf-8")
    source = tmp_path/"source.jsonl"
    source.write_text("\n".join(json.dumps(r) for r in source_rows)+"\n", encoding="utf-8")
    monkeypatch.setattr(module,"ROOT",root)
    monkeypatch.setattr(module,"SOURCE_SHA",module.digest(source.read_bytes()))
    monkeypatch.setattr(module,"previous_rows",lambda *args: [])
    monkeypatch.setattr(module,"aligned_controls",lambda *args: [dict(split="train" if i < 64 else "calibration",matched_content_equal=False) for i in range(80)])
    frozen = root/f"data/research/{module.STUDY}"
    module.prepare(source,frozen)
    exported = tmp_path/"exported"
    module.prepare(source,exported)
    assert (exported/"records.json").read_bytes() == (frozen/"records.json").read_bytes()
    a = json.loads((exported/"manifest.json").read_text())
    b = json.loads((frozen/"manifest.json").read_text())
    assert a["exclusion_files"] == b["exclusion_files"]
