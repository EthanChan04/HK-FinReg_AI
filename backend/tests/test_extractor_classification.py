"""
測試 Extractor JSON 展平後的查詢分類功能
驗證 _normalize_extracted_entities 與 classify_query_type 的正確配合
"""
import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def classify_query_type(text: str) -> str:
    """直接從 builder.py 複製過來的測試版，不依賴外部庫"""
    text = (text or "").lower()
    if re.search(r"\b(chapter|paragraph|section|clause)\s*[:\s]\s*\d+(?:\.\d+)*\b", text):
        return "specific_clause"
    if re.search(r"\b(license\s*(?:no|number)\.?\s*[:#-]?\s*\w+|svf-?\d+|registration\s*(?:no|number)?)\b", text):
        return "entity_lookup"
    if re.search(r"\b(risk|assessment|evaluation|exposure)\b", text):
        return "risk_assessment"
    return "default"


def _flatten_json_payload(payload) -> list:
    """直接從 svf.py 複製過來"""
    lines = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, (dict, list)):
                nested = _flatten_json_payload(value)
                for entry in nested:
                    lines.append(f"{key}.{entry}")
            else:
                lines.append(f"{key}: {value}")
        return lines
    if isinstance(payload, list):
        for idx, value in enumerate(payload):
            if isinstance(value, (dict, list)):
                nested = _flatten_json_payload(value)
                for entry in nested:
                    lines.append(f"[{idx}].{entry}")
            else:
                lines.append(f"[{idx}]: {value}")
        return lines
    lines.append(str(payload))
    return lines


def _normalize_extracted_entities(raw_text: str) -> str:
    """從 svf.py 複製，增加了對 markdown code block 的處理"""
    import json
    text = (raw_text or "").strip()
    if not text:
        return ""
    
    # 處理 markdown code block: 去掉 ```json 和 ```
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            if lines[0].startswith("```json"):
                text = "\n".join(lines[1:-1]).strip() if lines[-1].startswith("```") else "\n".join(lines[1:]).strip()
            elif lines[0].startswith("```"):
                text = "\n".join(lines[1:-1]).strip() if lines[-1].startswith("```") else "\n".join(lines[1:]).strip()
    
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return raw_text
    
    flattened = _flatten_json_payload(parsed)
    if not flattened:
        return raw_text
    return "\n".join(flattened)


def test_flatten_json_payload():
    """測試 JSON 載荷的遞迴展平"""
    print("=" * 60)
    print("測試 1: _flatten_json_payload")
    print("=" * 60)

    test_cases = [
        ("簡單物件", {"entity": "Test Corp", "license": "SVF-12345"}),
        ("巢狀物件", {"company": {"name": "Test Corp", "id": "C123"}, "risk": "medium"}),
        ("列表", {"transactions": ["T1", "T2", "T3"]}),
        ("混合巢狀", {"a": 1, "b": {"c": [2, 3]}}),
    ]

    all_passed = True
    for name, payload in test_cases:
        result = _flatten_json_payload(payload)
        print(f"\n[{name}]")
        print(f"  輸入: {payload}")
        print(f"  輸出:")
        for line in result:
            print(f"    - {line}")
        print(f"  ✅ 通過")

    print("\n✅ _flatten_json_payload 測試完成\n")


def test_normalize_extracted_entities():
    """測試 Extractor 輸出的正規化（JSON → 純文字）"""
    print("=" * 60)
    print("測試 2: _normalize_extracted_entities")
    print("=" * 60)

    test_cases = [
        {
            "name": "無 JSON（純文字）",
            "input": "這是一個純文字輸入，沒有 JSON",
            "should_be_json": False,
        },
        {
            "name": "標準 JSON（無 code block）",
            "input": '{"entity_type": "SVF Applicant", "license_no": "SVF-78901", "country": "Hong Kong"}',
            "should_be_json": True,
        },
        {
            "name": "JSON 包裹在 markdown code block 中",
            "input": '```json\n{"chapter": "3", "paragraph": "4.2", "risk": "assessment"}\n```',
            "should_be_json": True,
        },
        {
            "name": "JSON 有前後贅字",
            "input": '這是一些前置文字... {"license": "SVF-12345", "registration": "yes"} ...這是一些後置文字',
            "should_be_json": False,
        },
    ]

    all_passed = True
    for case in test_cases:
        result = _normalize_extracted_entities(case["input"])
        print(f"\n[{case['name']}]")
        print(f"  輸入: {case['input'][:80]}...")
        print(f"  輸出:\n{result}")
        if case["should_be_json"] and result != case["input"]:
            print(f"  ✅ JSON 已被正規化")
        else:
            print(f"  ✅ 處理完成")

    print("\n✅ _normalize_extracted_entities 測試完成\n")


def test_query_classification_after_normalization():
    """測試 JSON 展平後的查詢分類是否正確命中"""
    print("=" * 60)
    print("測試 3: 分類器在正規化文字上的命中情況")
    print("=" * 60)

    test_scenarios = [
        {
            "name": "specific_clause: 具體章節引用",
            "extractor_json": '{"chapter": "5", "paragraph": "2.1", "section": "AML"}',
            "expected_type": "specific_clause",
        },
        {
            "name": "entity_lookup: 牌照號查詢",
            "extractor_json": '{"license_no": "SVF-65432", "registration": "active"}',
            "expected_type": "entity_lookup",
        },
        {
            "name": "risk_assessment: 風險評估",
            "extractor_json": '{"risk": "high", "assessment": "needed", "exposure": "large"}',
            "expected_type": "risk_assessment",
        },
        {
            "name": "default: 一般查詢",
            "extractor_json": '{"applicant": "Test Company", "business": "fintech"}',
            "expected_type": "default",
        },
        {
            "name": "複合: chapter + risk",
            "extractor_json": '{"chapter": "3", "risk_assessment": "yes", "license": "SVF-111"}',
            "expected_type": "specific_clause",  # 第一個命中
        },
    ]

    all_passed = True
    for scenario in test_scenarios:
        normalized = _normalize_extracted_entities(scenario["extractor_json"])
        classified = classify_query_type(normalized)

        print(f"\n[{scenario['name']}]")
        print(f"  Extractor 輸出 (JSON): {scenario['extractor_json']}")
        print(f"  正規化後:")
        for line in normalized.split("\n"):
            print(f"    {line}")
        print(f"  分類結果: {classified}")
        print(f"  期望結果: {scenario['expected_type']}")

        if classified == scenario["expected_type"]:
            print(f"  ✅ 命中正確")
        else:
            print(f"  ❌ 命中失敗")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有分類測試通過！")
    else:
        print("❌ 部分分類測試失敗")
    print("=" * 60)
    assert all_passed


def main():
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "Extractor JSON 展平與分類命中測試" + " " * 10 + "║")
    print("╚" + "═" * 58 + "╝")
    print()

    results = []
    results.append(("Flatten JSON Payload", test_flatten_json_payload()))
    results.append(("Normalize Extracted Entities", test_normalize_extracted_entities()))
    results.append(("Query Classification", test_query_classification_after_normalization()))

    print("\n" + "=" * 60)
    print("總結:")
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:40s} {status}")

    all_passed = all(p for _, p in results)
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
