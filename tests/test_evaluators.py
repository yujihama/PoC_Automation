from poc_automation.evaluators import EvaluatorSuite
from poc_automation.models import Case, Citation, ExpectedOutput, NormalizedResult, RationaleItem, Split


def test_evaluator_suite_scores_expected_output():
    case = Case(
        case_id="c1",
        split=Split.TRAIN,
        procedure_csv_path="x.csv",
        evidence_bundle_path="evidence",
        expected_output=ExpectedOutput(
            judgement="適合",
            required_claim_keywords=["住所", "一致"],
            citations=[Citation(evidence_id="doc1", page=1)],
        ),
    )
    output = NormalizedResult(
        judgement="適合",
        rationale_items=[RationaleItem(claim="住所が一致している", citations=[Citation(evidence_id="doc1", page=1)])],
    )
    results = EvaluatorSuite().evaluate_case(case=case, output=output)
    scores = {result.evaluator_name: result.score for result in results}
    assert scores["judgement_match"] == 1.0
    assert scores["rationale_support"] == 1.0
    assert scores["citation_quality"] == 1.0
    assert scores["total_score"] and scores["total_score"] > 0.8
