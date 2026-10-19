import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, TypedDict

from deepeval.metrics import (
    AnswerRelevancyMetric,
    BiasMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
    ToxicityMetric,
)
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig
from deepeval.test_case import LLMTestCase, SingleTurnParams
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from rag_workflow import answer_question, build_rag_graph
from storage import SQLiteStorage
from vector_store import QdrantVectorStore


EVALUATION_COLLECTION = "rag_evaluation"


class GroundedQAPair(BaseModel):
    question: str = Field(description="A specific question answerable from one source excerpt.")
    expected_answer: str = Field(
        description="A concise, factual answer supported by that source excerpt."
    )
    source_chunk_id: int = Field(
        description="The exact source ID containing evidence for the answer."
    )


class GroundedQABatch(BaseModel):
    pairs: list[GroundedQAPair]


METRIC_NAMES = (
    "Answer relevancy",
    "Faithfulness",
    "Contextual relevancy",
    "Contextual precision",
    "Contextual recall",
    "Hallucination risk",
    "Answer correctness",
    "Answer completeness",
    "Bias risk",
    "Toxicity risk",
)


def load_cases(dataset_path: str | Path) -> list[dict[str, str]]:
    with Path(dataset_path).open(encoding="utf-8") as dataset_file:
        cases = json.load(dataset_file)
    if not isinstance(cases, list) or not cases:
        raise ValueError("The evaluation dataset must be a non-empty JSON array.")
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or not all(
            isinstance(case.get(field), str) and case[field].strip()
            for field in ("question", "expected_answer")
        ):
            raise ValueError(
                f"Evaluation case {index + 1} must contain non-empty "
                "'question' and 'expected_answer' strings."
            )
    return cases


def generate_grounded_qa_pairs(
    documents: list[dict[str, str]],
    question_count: int = 5,
    model_name: str | None = None,
) -> list[dict[str, str]]:
    if not 1 <= question_count <= 10:
        raise ValueError("Question count must be between 1 and 10.")
    if not documents:
        raise ValueError("Upload at least one document for evaluation.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    chunks: list[dict[str, str]] = []
    for document in documents:
        filename = document.get("filename", "").strip()
        text = document.get("text", "")
        if not filename or not text.strip():
            raise ValueError("Each evaluation document needs a filename and readable text.")
        chunks.extend(
            {"filename": filename, "text": chunk}
            for chunk in splitter.split_text(text)
            if chunk.strip()
        )
    if not chunks:
        raise ValueError("No readable text was found in the evaluation documents.")

    selected_count = min(question_count, len(chunks))
    selected_indices = [
        0
        if selected_count == 1
        else round(index * (len(chunks) - 1) / (selected_count - 1))
        for index in range(selected_count)
    ]
    selected_chunks = [chunks[index] for index in selected_indices]
    source_contexts = "\n\n".join(
        f"[Source ID: {index} | File: {chunk['filename']}]\n{chunk['text']}"
        for index, chunk in enumerate(selected_chunks)
    )
    model = ChatOpenAI(
        model=model_name or os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        temperature=0,
    ).with_structured_output(GroundedQABatch)
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "Create exactly the requested number of question-and-answer pairs using "
                    "only the supplied source excerpts. Every question must have a specific "
                    "answer explicitly supported by one excerpt. Keep answers concise, do not "
                    "use outside knowledge, and set source_chunk_id to the matching Source ID."
                )
            ),
            HumanMessage(
                content=(
                    f"Create exactly {question_count} distinct QA pairs.\n\n"
                    f"Source excerpts:\n{source_contexts}"
                )
            ),
        ]
    )
    batch = (
        response
        if isinstance(response, GroundedQABatch)
        else GroundedQABatch.model_validate(response)
    )
    if len(batch.pairs) != question_count:
        raise ValueError(
            f"Expected {question_count} generated QA pairs, received {len(batch.pairs)}."
        )

    cases = []
    for index, pair in enumerate(batch.pairs, start=1):
        if not pair.question.strip() or not pair.expected_answer.strip():
            raise ValueError(f"Generated QA pair {index} has an empty question or answer.")
        if not 0 <= pair.source_chunk_id < len(selected_chunks):
            raise ValueError(
                f"Generated QA pair {index} references invalid source "
                f"{pair.source_chunk_id}."
            )
        source = selected_chunks[pair.source_chunk_id]
        cases.append(
            {
                "question": pair.question.strip(),
                "expected_answer": pair.expected_answer.strip(),
                "source": source["filename"],
                "source_context": source["text"],
            }
        )
    return cases


def generate_pdf_evaluation_cases(
    pdf_path: str | Path,
    question_count: int = 2,
) -> list[dict[str, str]]:
    if not 1 <= question_count <= 10:
        raise ValueError("Question count must be between 1 and 10.")

    document_path = Path(pdf_path)
    if not document_path.is_file():
        raise FileNotFoundError(f"Evaluation PDF does not exist: {document_path}")

    goldens = Synthesizer().generate_goldens_from_docs(
        document_paths=[str(document_path)],
        include_expected_output=True,
        max_goldens_per_context=question_count,
        context_construction_config=ContextConstructionConfig(
            max_contexts_per_document=1
        ),
    )
    if len(goldens) != question_count:
        raise ValueError(
            f"Expected {question_count} synthesized QA pairs, received {len(goldens)}."
        )

    cases = []
    for index, golden in enumerate(goldens, start=1):
        context = golden.context or []
        if not golden.input or not golden.expected_output or not context:
            raise ValueError(
                f"Synthesized QA pair {index} is missing its question, expected "
                "answer, or source context."
            )
        cases.append(
            {
                "question": golden.input,
                "expected_answer": golden.expected_output,
                "source": document_path.name,
                "source_context": "\n\n".join(context),
            }
        )
    return cases


def build_metrics() -> list[Any]:
    return [
        AnswerRelevancyMetric(threshold=0.7, async_mode=False),
        FaithfulnessMetric(threshold=0.7, async_mode=False),
        ContextualRelevancyMetric(threshold=0.7, async_mode=False),
        ContextualPrecisionMetric(threshold=0.7, async_mode=False),
        ContextualRecallMetric(threshold=0.7, async_mode=False),
        HallucinationMetric(threshold=0.3, async_mode=False),
        GEval(
            name="Answer correctness",
            criteria=(
                "Determine whether the actual answer is factually correct and semantically "
                "equivalent to the expected answer, using the question and retrieved context."
            ),
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
                SingleTurnParams.RETRIEVAL_CONTEXT,
            ],
            threshold=0.7,
            async_mode=False,
        ),
        GEval(
            name="Answer completeness",
            criteria=(
                "Determine whether the actual answer includes the important information "
                "required by the expected answer without omitting key facts."
            ),
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
                SingleTurnParams.RETRIEVAL_CONTEXT,
            ],
            threshold=0.7,
            async_mode=False,
        ),
        BiasMetric(threshold=0.3, async_mode=False),
        ToxicityMetric(threshold=0.3, async_mode=False),
    ]


def evaluate_cases(
    cases: list[dict[str, str]],
    graph: Any,
    document_hashes: list[str] | None = None,
    collection_name: str = EVALUATION_COLLECTION,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    if not cases:
        raise ValueError("At least one QA pair is required for evaluation.")
    metrics = build_metrics()
    total = len(cases) * len(metrics)
    completed = 0
    rows = []
    for case in cases:
        result = answer_question(
            graph,
            case["question"],
            document_hashes=document_hashes,
            collection_name=collection_name,
        )
        test_case = LLMTestCase(
            input=case["question"],
            actual_output=result["answer"],
            expected_output=case["expected_answer"],
            context=[case["source_context"]] if case.get("source_context") else None,
            retrieval_context=result["retrieval_context"],
        )
        scores: dict[str, float] = {}
        reasons: dict[str, str] = {}
        for metric, metric_name in zip(metrics, METRIC_NAMES, strict=True):
            metric.measure(test_case)
            if metric.score is None:
                raise ValueError(f"DeepEval did not return a score for {metric_name}.")
            scores[metric_name] = float(metric.score)
            if metric.reason:
                reasons[metric_name] = str(metric.reason)
            completed += 1
            if progress_callback:
                progress_callback(completed, total)
        rows.append(
            {
                "question": case["question"],
                "grounded_answer": case["expected_answer"],
                "ai_answer": result["answer"],
                "source": case.get("source", ""),
                "source_context": case.get("source_context", ""),
                "retrieved_context": result["retrieval_context"],
                "metric_scores": scores,
                "metric_reasons": reasons,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the document Q&A app with DeepEval."
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--dataset",
        help="JSON array containing question and expected_answer fields.",
    )
    inputs.add_argument(
        "--pdf",
        help="PDF to synthesize evaluation questions from with DeepEval.",
    )
    parser.add_argument(
        "--question-count",
        type=int,
        default=2,
        help="Number of QA pairs to synthesize from --pdf (default: 2).",
    )
    args = parser.parse_args()
    if not 1 <= args.question_count <= 10:
        parser.error("--question-count must be between 1 and 10.")
    if args.dataset and args.question_count != 2:
        parser.error("--question-count can only be used with --pdf.")

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in .env before running evaluation.")

    storage = SQLiteStorage(os.getenv("SQLITE_DATABASE_PATH", "data/rag_app.sqlite3"))
    vector_store = QdrantVectorStore(
        storage=storage,
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        collection_name=os.getenv("QDRANT_COLLECTION", "rag_documents"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
    )
    document_hashes: list[str] | None = None
    collection_name = vector_store.collection_name
    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.is_file():
            raise SystemExit(f"Evaluation PDF does not exist: {pdf_path}")
        pdf_content = pdf_path.read_bytes()
        document_hashes = [hashlib.sha256(pdf_content).hexdigest()]
        vector_store.ingest_document(
            pdf_path.name,
            pdf_content,
            collection_name=EVALUATION_COLLECTION,
        )
        cases = generate_pdf_evaluation_cases(pdf_path, args.question_count)
        collection_name = EVALUATION_COLLECTION
    else:
        if args.dataset is None:
            parser.error("Either --dataset or --pdf must be provided.")
        cases = load_cases(args.dataset)

    rows = evaluate_cases(
        cases,
        build_rag_graph(vector_store),
        document_hashes=document_hashes,
        collection_name=collection_name,
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
