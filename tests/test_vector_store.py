import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from storage import SQLiteStorage
from vector_store import QdrantVectorStore


class FakeEmbeddings:
    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


class FakeLargeEmbeddings:
    def embed_documents(self, texts):
        return [[1.0] + [0.0] * 1535 for _ in texts]

    def embed_query(self, text):
        return [1.0] + [0.0] * 1535


class QdrantEvaluationIsolationTests(unittest.TestCase):
    def test_evaluation_filter_only_retrieves_uploaded_evaluation_documents(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.dict(
                os.environ,
                {"QDRANT_PATH": str(root / "qdrant"), "QDRANT_URL": ""},
            ):
                store = QdrantVectorStore(
                    SQLiteStorage(root / "app.sqlite3"),
                    collection_name="chat_documents",
                    chunk_size=100,
                    chunk_overlap=0,
                )
                store.embeddings.embeddings = FakeEmbeddings()
                chat_content = b"Chat collection document."
                evaluation_content = b"Evaluation collection document."
                store.ingest_document("chat.txt", chat_content)
                store.ingest_document(
                    "evaluation.txt",
                    evaluation_content,
                    collection_name="evaluation_documents",
                )

                evaluation_hits = store.search(
                    "document",
                    collection_name="evaluation_documents",
                    document_hashes=[
                        hashlib.sha256(evaluation_content).hexdigest()
                    ],
                )
                chat_hits = store.search("document")

                self.assertEqual({hit["source"] for hit in evaluation_hits}, {"evaluation.txt"})
                self.assertEqual({hit["source"] for hit in chat_hits}, {"chat.txt"})
                store.client.close()

    def test_stale_three_dimensional_collection_uses_compatible_collection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.dict(
                os.environ,
                {"QDRANT_PATH": str(root / "qdrant"), "QDRANT_URL": ""},
            ):
                store = QdrantVectorStore(
                    SQLiteStorage(root / "app.sqlite3"),
                    collection_name="chat_documents",
                    chunk_size=100,
                    chunk_overlap=0,
                )
                store.embeddings.embeddings = FakeEmbeddings()
                store.ingest_document(
                    "old-evaluation.txt",
                    b"Previously indexed with a three dimensional test vector.",
                    collection_name="evaluation_documents",
                )

                store.embeddings.embeddings = FakeLargeEmbeddings()
                new_content = b"Index this evaluation document using the current model."
                store.ingest_document(
                    "current-evaluation.txt",
                    new_content,
                    collection_name="evaluation_documents",
                )

                original = store.client.get_collection("evaluation_documents")
                compatible = store.client.get_collection("evaluation_documents_dim_1536")
                hits = store.search(
                    "current model",
                    collection_name="evaluation_documents",
                    document_hashes=[hashlib.sha256(new_content).hexdigest()],
                )

                self.assertEqual(original.config.params.vectors.size, 3)
                self.assertEqual(compatible.config.params.vectors.size, 1536)
                self.assertEqual(
                    {hit["source"] for hit in hits}, {"current-evaluation.txt"}
                )
                store.client.close()


if __name__ == "__main__":
    unittest.main()
