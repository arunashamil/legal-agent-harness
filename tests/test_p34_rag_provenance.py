"""P34: RAG и provenance — ответ без проверяемого источника. Сеть не используется.

Фикстуры и описание ожидаемого поведения: tests/fixtures/p34/README.md.
Тесты с xfail(strict=True) фиксируют найденные пробелы: когда harness начнёт
проверять provenance, они пройдут и упадут как XPASS — тогда снимите маркер.
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from legal_agent_harness import (  # noqa: E402
    ReplayAdapter,
    load_cases,
    load_responses,
    run_case,
)

FIXTURES = os.path.join(ROOT, "tests", "fixtures", "p34")


def _fixture(name: str) -> str:
    return os.path.join(FIXTURES, name + ".jsonl")


def _run(response: str):
    (case,) = load_cases(_fixture("case"))
    adapter = ReplayAdapter(load_responses(_fixture(response)))
    return run_case(case, adapter, max_rounds=1)


def test_case_accepted():
    (case,) = load_cases(_fixture("case"))
    assert case["expected"]["norm_id"] == "гк-рф/ст.1406.1"
    assert case["source"]["coordinate"] == "ст. 3 п. 124"


def test_answer_with_provenance_approved_and_logged():
    result = _run("response_with_provenance")
    assert result.approved is True
    assert result.needs_expert_approval is True
    (answer,) = [e for e in result.events if e["event"] == "agent_answer"]
    fragments = answer["payload"]["response"]["provenance"]["fragments"]
    assert [f["chunk_id"] for f in fragments] == [338, 337]


def test_similar_norm_from_top1_not_approved():
    # Верхний результат поиска по формуле компенсации — ст. 1515, а не ст. 1406.1.
    assert _run("response_similar_norm").approved is False


@pytest.mark.xfail(
    strict=True,
    reason="P34: ответ без retrieved-фрагмента и координаты получает approved=True, "
    "provenance не требуется",
)
def test_answer_without_provenance_not_approved():
    assert _run("response_no_provenance").approved is False


@pytest.mark.xfail(
    strict=True,
    reason="P34: provenance не проверяется — фрагмент без «1406.1» и «35-ФЗ» "
    "(список литературы) получает approved=True",
)
def test_false_provenance_not_approved():
    assert _run("response_false_provenance").approved is False
