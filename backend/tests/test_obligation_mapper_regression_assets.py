import json
from pathlib import Path


BASE = Path(__file__).parent / "regression" / "obligation_mapper"


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def test_golden_assets_exist_and_have_minimum_cases():
    cases_path = BASE / "golden_cases.jsonl"
    expected_path = BASE / "golden_expected.jsonl"

    assert cases_path.exists()
    assert expected_path.exists()

    cases = _read_jsonl(cases_path)
    expected = _read_jsonl(expected_path)

    assert 20 <= len(cases) <= 50
    assert len(cases) == len(expected)

    case_ids = {row["case_id"] for row in cases}
    expected_ids = {row["case_id"] for row in expected}
    assert case_ids == expected_ids


def test_golden_expected_required_fields():
    expected_path = BASE / "golden_expected.jsonl"
    expected = _read_jsonl(expected_path)

    required = {"applicable_regulators", "risk_types", "obligations", "evidence_chunks"}
    for row in expected:
        assert required.issubset(row.keys())
        assert isinstance(row["applicable_regulators"], list)
        assert isinstance(row["risk_types"], list)
        assert isinstance(row["obligations"], list)
        assert isinstance(row["evidence_chunks"], list)

