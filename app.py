import os
import uuid
import hashlib

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from evaluation import (
    EVALUATION_COLLECTION,
    METRIC_NAMES,
    evaluate_cases,
    generate_grounded_qa_pairs,
)
from rag_workflow import answer_question, build_rag_graph
from storage import SQLiteStorage
from vector_store import QdrantVectorStore


load_dotenv()
st.set_page_config(page_title="Document Q&A", page_icon="📚", layout="wide")


@st.cache_resource
def get_storage() -> SQLiteStorage:
    return SQLiteStorage(os.getenv("SQLITE_DATABASE_PATH", "data/rag_app.sqlite3"))


@st.cache_resource
def get_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore(
        storage=get_storage(),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        collection_name=os.getenv("QDRANT_COLLECTION", "rag_documents"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
    )


@st.cache_resource
def get_graph():
    return build_rag_graph(get_vector_store())


storage = get_storage()
st.title("📚 Document Q&A")
st.caption("Upload documents and ask questions grounded in their content.")

threads = storage.list_threads()
thread_ids = {thread["thread_id"] for thread in threads}
if st.session_state.get("thread_id") not in thread_ids:
    st.session_state.thread_id = threads[0]["thread_id"] if threads else None

with st.sidebar:
    st.header("Conversations")
    if st.button("＋ New conversation", use_container_width=True):
        thread_id = str(uuid.uuid4())
        storage.create_thread(thread_id)
        st.session_state.thread_id = thread_id
        st.rerun()

    for thread in threads:
        label = thread["title"]
        if st.button(
            label,
            key=f"thread_{thread['thread_id']}",
            use_container_width=True,
            type="primary" if thread["thread_id"] == st.session_state.thread_id else "secondary",
        ):
            st.session_state.thread_id = thread["thread_id"]
            st.rerun()

    st.divider()
    st.header("Add documents")
    uploaded_files = st.file_uploader(
        "Choose PDF, TXT, or Markdown files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if st.button("Index documents", disabled=not uploaded_files, use_container_width=True):
        with st.spinner("Extracting text and indexing documents..."):
            vector_store = get_vector_store()
            for uploaded_file in uploaded_files:
                try:
                    count = vector_store.ingest_document(
                        uploaded_file.name, uploaded_file.getvalue()
                    )
                    st.success(f"Indexed {count} chunks from {uploaded_file.name}.")
                except Exception as error:
                    st.error(f"Could not index {uploaded_file.name}: {error}")

if st.session_state.thread_id is None:
    thread_id = str(uuid.uuid4())
    storage.create_thread(thread_id)
    st.session_state.thread_id = thread_id
    st.rerun()

thread_id = st.session_state.thread_id
for message in storage.get_messages(thread_id):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message["sources"]:
            with st.expander("Retrieved sources"):
                for source in message["sources"]:
                    st.caption(f"{source['source']} · relevance {source['score']:.3f}")
                    st.write(source["text"])

question = st.chat_input("Ask a question about your documents")
if question:
    storage.add_message(thread_id, "user", question)
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching documents and preparing an answer..."):
                history = storage.get_messages(thread_id)[:-1]
                result = answer_question(get_graph(), question, history=history)
            st.markdown(result["answer"])
            storage.add_message(
                thread_id, "assistant", result["answer"], sources=result["sources"]
            )
            if result["sources"]:
                with st.expander("Retrieved sources"):
                    for source in result["sources"]:
                        st.caption(f"{source['source']} · relevance {source['score']:.3f}")
                        st.write(source["text"])
        except Exception as error:
            st.error(f"Unable to answer this question: {error}")


st.divider()
st.header("Evaluate the application")
st.write(
    "Upload evaluation documents to generate grounded QA pairs, run ten DeepEval "
    "metrics, and review each model answer and score."
)
evaluation_files = st.file_uploader(
    "Choose evaluation PDF, TXT, or Markdown documents",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
    key="evaluation_documents",
)
question_count = st.slider(
    "Grounded QA pairs to evaluate",
    min_value=1,
    max_value=10,
    value=5,
    key="evaluation_question_count",
)
if st.button(
    "Generate QA pairs and evaluate",
    disabled=not evaluation_files,
    type="primary",
    key="run_deepeval",
):
    st.session_state["evaluation_results"] = None
    st.session_state["evaluation_qa_pairs"] = []
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Set OPENAI_API_KEY in .env before running evaluation.")
    else:
        progress = None
        try:
            vector_store = get_vector_store()
            documents = []
            document_hashes = []
            with st.spinner("Reading evaluation documents and preparing the isolated index..."):
                for uploaded_file in evaluation_files:
                    content = uploaded_file.getvalue()
                    document_hash = hashlib.sha256(content).hexdigest()
                    text = vector_store.extract_text(uploaded_file.name, content)
                    vector_store.ingest_document(
                        uploaded_file.name,
                        content,
                        collection_name=EVALUATION_COLLECTION,
                    )
                    documents.append({"filename": uploaded_file.name, "text": text})
                    document_hashes.append(document_hash)

            with st.spinner("Generating document-grounded QA pairs..."):
                qa_pairs = generate_grounded_qa_pairs(documents, question_count)
            st.session_state["evaluation_qa_pairs"] = qa_pairs
            progress = st.progress(0, text="Running DeepEval metrics...")

            def update_evaluation_progress(completed: int, total: int) -> None:
                progress.progress(
                    completed / total,
                    text=f"Running DeepEval metrics: {completed} / {total}",
                )

            results = evaluate_cases(
                qa_pairs,
                build_rag_graph(vector_store),
                document_hashes=document_hashes,
                collection_name=EVALUATION_COLLECTION,
                progress_callback=update_evaluation_progress,
            )
            st.session_state["evaluation_results"] = results
            progress.empty()
        except Exception as error:
            if progress is not None:
                progress.empty()
            st.error(f"Evaluation failed: {error}")
            st.exception(error)

evaluation_results = st.session_state.get("evaluation_results")
if evaluation_results:
    st.subheader("Evaluation results")
    st.caption(
        "Quality scores are higher-is-better. Hallucination, bias, and toxicity are "
        "risk scores where lower is better."
    )
    table_rows = []
    for result in evaluation_results:
        table_row = {
            "Question": result["question"],
            "Grounded answer": result["grounded_answer"],
            "AI answer": result["ai_answer"],
            "Source": result["source"],
        }
        table_row.update(result["metric_scores"])
        table_rows.append(table_row)
    st.dataframe(
        pd.DataFrame(table_rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            metric_name: st.column_config.NumberColumn(
                metric_name, format="%.2f", min_value=0.0, max_value=1.0
            )
            for metric_name in METRIC_NAMES
        },
    )
    for index, result in enumerate(evaluation_results, start=1):
        with st.expander(f"QA pair {index}: {result['question']}"):
            st.markdown(f"**Grounded answer:** {result['grounded_answer']}")
            st.markdown(f"**AI answer:** {result['ai_answer']}")
            st.markdown(f"**Grounding source:** {result['source']}")
            st.markdown(f"**Source excerpt:** {result['source_context']}")
            if result["retrieved_context"]:
                st.markdown("**Retrieved context:**")
                for context in result["retrieved_context"]:
                    st.write(context)
            if result["metric_reasons"]:
                st.markdown("**DeepEval metric reasons:**")
                for metric_name, reason in result["metric_reasons"].items():
                    score = result["metric_scores"][metric_name]
                    st.write(f"- **{metric_name} ({score:.2f}):** {reason}")
