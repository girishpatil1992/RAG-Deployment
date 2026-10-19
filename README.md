# Document Q&A with RAG

A small Streamlit document Q&A app using LangGraph for retrieval and answer generation, Qdrant for vector search, and SQLite for conversation history and a persistent chunk/embedding cache.

## Setup

1. From this folder, create and activate a virtual environment, then install dependencies:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Update `OPENAI_API_KEY` in `.env`. The other settings can remain at their defaults. `.env` is ignored by Git; `.env.example` documents the available settings.
3. Start the app:

   ```powershell
   streamlit run app.py
   ```

4. Upload PDF, TXT, or Markdown files in the sidebar, then ask questions in the conversation.

## Storage and caching

- Conversations, messages, extracted chunks, and embeddings are stored in `data/rag_app.sqlite3`.
- Qdrant runs in local persistent mode under `data/qdrant` by default. To use Qdrant Cloud or another Qdrant server, set `QDRANT_URL` and `QDRANT_API_KEY`.
- Re-indexing the same document with the same chunk settings reuses its cached chunks and embeddings. Cached embeddings are keyed by model and text, so repeated chunks and question vectors are reused as well.
- To rebuild the local index and cache, stop Streamlit and remove the `data` folder.

## DeepEval

The Streamlit app includes an **Evaluate the application** section. Upload PDF, TXT, or Markdown documents there, choose 1–10 grounded QA pairs, and select **Generate QA pairs and evaluate**. Evaluation documents are indexed in a separate Qdrant collection, so they do not alter normal chat retrieval. The UI displays each question, document-grounded answer, AI answer, source, and ten DeepEval scores; expanding a row shows the source/retrieved context and metric reasons.

The ten tracked parameters are:

1. Answer relevancy
2. Faithfulness
3. Contextual relevancy
4. Contextual precision
5. Contextual recall
6. Hallucination risk
7. Answer correctness
8. Answer completeness
9. Bias risk
10. Toxicity risk

Relevancy, faithfulness, precision, recall, correctness, and completeness are quality scores (higher is better). Hallucination, bias, and toxicity are risk scores (lower is better).

The GitHub Actions evaluation check runs the evaluation unit tests on pushes and pull requests to `main`. These tests mock model and metric calls, so they do not require an API key or make paid evaluation requests.

The standalone evaluator is also available. Create a UTF-8 JSON file containing a non-empty array of question/expected-answer pairs:

```json
[
  {
    "question": "What is the document's main topic?",
    "expected_answer": "A concise answer supported by the uploaded document."
  }
]
```

Run evaluation from this folder:

```powershell
python evaluation.py --dataset evaluation_cases.json
```

The evaluator requires `OPENAI_API_KEY`. Each question is answered with the RAG workflow and scored against its expected answer and retrieved context using all ten metrics. A full evaluation runs ten metric checks per QA pair and may use additional model calls and incur API costs.
