"""Repeated public-runtime checks for the three competition demonstration cases."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
from starlette.testclient import TestClient

from dy3_polaris.l5.unified_app import UnifiedApp


CASES = (
    (
        "yellow-blue-emission",
        "Dy³⁺为什么具有黄蓝双发射？",
        (("4F9/2",), ("6H15/2",), ("6H13/2",)),
        (),
    ),
    (
        "cct-health-boundary",
        "3000K低色温是否一定没有蓝光风险？",
        (("光谱", "蓝光"), ("色温", "CCT")),
        ("3000K一定安全", "低色温一定没有蓝光风险"),
    ),
    (
        "material-comparison-boundary",
        "没有数据时，一种Dy³⁺发光材料是否一定优于另一种材料？",
        (("指标", "评价"), ("数据", "证据", "条件")),
        ("一定优于", "必然优于"),
    ),
)


def _public_data(response) -> dict[str, object]:
    payload = response.json()
    assert payload["code"] == 0
    return dict(payload["data"])


@pytest.mark.parametrize(
    ("case_name", "query", "required_groups", "forbidden_claims"),
    CASES,
)
def test_competition_case_is_stable_across_three_rounds(
    tmp_path,
    case_name: str,
    query: str,
    required_groups: tuple[tuple[str, ...], ...],
    forbidden_claims: tuple[str, ...],
) -> None:
    builder = UnifiedApp.create_full_app_builder(
        data_dir=str(tmp_path / case_name),
    )
    client = TestClient(builder.create_app())
    signatures: list[tuple[str, str, str, bool]] = []

    for round_index in range(3):
        data = _public_data(client.post(
            "/api/query",
            json={
                "query": query,
                "learner_id": f"competition-{case_name}",
            },
        ))
        answer = str(data.get("answer") or "")
        review = data.get("review") or {}
        release = data.get("quality_release") or {}
        evidence = data.get("evidence") or []

        assert isinstance(review, Mapping)
        assert isinstance(release, Mapping)
        assert not any(claim in answer for claim in forbidden_claims)

        if answer:
            outcome = "published"
            assert review.get("status") == "completed"
            assert review.get("verdict") in {"approved", "pass"}
            assert release.get("eligible") is True
            assert release.get("status") in {"FULL_RELEASE", "LIMITED_RELEASE"}
            assert evidence
            assert all(
                any(term.casefold() in answer.casefold() for term in group)
                for group in required_groups
            )
        else:
            outcome = "safe_withheld"
            assert data.get("knowledge_unavailable") is True
            assert release.get("eligible") is False
            assert release.get("status") not in {"FULL_RELEASE", "LIMITED_RELEASE"}
            assert review.get("verdict") not in {"approved", "pass"}

        signatures.append((
            outcome,
            str(review.get("verdict")),
            str(release.get("status")),
            bool(release.get("eligible")),
        ))

    assert len(set(signatures)) == 1
