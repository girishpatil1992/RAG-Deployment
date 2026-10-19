import unittest
from types import SimpleNamespace
from unittest.mock import patch

from evaluation import (
    GroundedQAPair,
    GroundedQABatch,
    METRIC_NAMES,
    evaluate_cases,
    generate_grounded_qa_pairs,
)


class FakeGraph:
    def invoke(self, state):
        self.state = state
        return {
            "answer": "The recorded value is 42.",
            "context": [
                {"source": "notes.txt", "score": 0.95, "text": "The recorded value is 42."}
            ],
        }


class FakeMetric:
    next_score = 0.0

    def __init__(self, **kwargs):
        self.score = None
        self.reason = "mock metric reason"

    def measure(self, test_case):
        self.score = FakeMetric.next_score
        FakeMetric.next_score += 0.01


class EvaluationTests(unittest.TestCase):
    def test_generates_qa_pairs_mapped_to_document_excerpts(self):
        batch = GroundedQABatch(
            pairs=[
                GroundedQAPair(
                    question="What is the recorded value?",
                    expected_answer="42",
                    source_chunk_id=0,
                )
            ]
        )
        fake_model = SimpleNamespace(invoke=lambda messages: batch)
        structured_model = SimpleNamespace(with_structured_output=lambda schema: fake_model)

        with patch("evaluation.ChatOpenAI", return_value=structured_model):
            cases = generate_grounded_qa_pairs(
                [{"filename": "notes.txt", "text": "The recorded value is 42."}],
                question_count=1,
            )

        self.assertEqual(cases[0]["expected_answer"], "42")
        self.assertEqual(cases[0]["source"], "notes.txt")
        self.assertIn("The recorded value is 42.", cases[0]["source_context"])

    def test_evaluates_all_ten_metrics_and_scopes_retrieval(self):
        FakeMetric.next_score = 0.0
        graph = FakeGraph()
        case = {
            "question": "What is the recorded value?",
            "expected_answer": "42",
            "source": "notes.txt",
            "source_context": "The recorded value is 42.",
        }
        metric_patches = [
            patch(f"evaluation.{name}", side_effect=FakeMetric)
            for name in (
                "AnswerRelevancyMetric",
                "FaithfulnessMetric",
                "ContextualRelevancyMetric",
                "ContextualPrecisionMetric",
                "ContextualRecallMetric",
                "HallucinationMetric",
                "GEval",
                "BiasMetric",
                "ToxicityMetric",
            )
        ]
        for metric_patch in metric_patches:
            metric_patch.start()
        try:
            rows = evaluate_cases(
                [case],
                graph,
                document_hashes=["document-hash"],
                collection_name="rag_evaluation",
            )
        finally:
            for metric_patch in metric_patches:
                metric_patch.stop()

        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]["metric_scores"]), set(METRIC_NAMES))
        self.assertEqual(rows[0]["ai_answer"], "The recorded value is 42.")
        self.assertEqual(graph.state["document_hashes"], ["document-hash"])
        self.assertEqual(graph.state["collection_name"], "rag_evaluation")
        self.assertEqual(len(rows[0]["metric_reasons"]), 10)


if __name__ == "__main__":
    unittest.main()
