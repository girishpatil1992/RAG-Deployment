import hashlib
import io
import os
import uuid
from pathlib import Path
from typing import Any

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_client import QdrantClient, models

from storage import SQLiteStorage


class CachedEmbeddings:
    def __init__(self, model: str, storage: SQLiteStorage) -> None:
        self.model = model
        self.storage = storage
        self.embeddings = OpenAIEmbeddings(model=model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        cached = self.storage.get_cached_embeddings(self.model, texts)
        missing = list(dict.fromkeys(text for text in texts if text not in cached))
        if missing:
            vectors = self.embeddings.embed_documents(missing)
            self.storage.cache_embeddings(self.model, dict(zip(missing, vectors, strict=True)))
            cached.update(dict(zip(missing, vectors, strict=True)))
        return [cached[text] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class QdrantVectorStore:
    def __init__(
        self,
        storage: SQLiteStorage,
        embedding_model: str = "text-embedding-3-small",
        collection_name: str = "rag_documents",
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ) -> None:
        self.storage = storage
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = CachedEmbeddings(embedding_model, storage)

        qdrant_url = os.getenv("QDRANT_URL", "").strip()
        qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip() or None
        self._requires_payload_index = bool(qdrant_url)
        if qdrant_url:
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            qdrant_path = Path(os.getenv("QDRANT_PATH", "data/qdrant"))
            qdrant_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(qdrant_path))

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )

    @staticmethod
    def extract_text(filename: str, content: bytes) -> str:
        suffix = Path(filename).suffix.lower()
        if suffix == ".pdf":
            pages = PdfReader(io.BytesIO(content)).pages
            text = "\n".join(page.extract_text() or "" for page in pages)
        elif suffix in {".txt", ".md"}:
            text = content.decode("utf-8-sig")
        else:
            raise ValueError("Unsupported file type. Upload a PDF, TXT, or Markdown file.")
        if not text.strip():
            raise ValueError(f"No readable text was found in {filename}.")
        return text

    def _resolve_collection(
        self,
        collection_name: str,
        vector_size: int,
        create_if_missing: bool,
    ) -> str | None:
        candidate_names = [collection_name]
        compatible_name = f"{collection_name[: 255 - len(f'_dim_{vector_size}')]}_dim_{vector_size}"
        if compatible_name != collection_name:
            candidate_names.append(compatible_name)

        for candidate in candidate_names:
            if self.client.collection_exists(candidate):
                collection = self.client.get_collection(candidate)
                configured_vectors = collection.config.params.vectors
                if isinstance(configured_vectors, models.VectorParams):
                    if configured_vectors.size == vector_size:
                        return candidate
                    continue
                raise ValueError(
                    f"Qdrant collection '{candidate}' uses named vectors. "
                    "This app requires a collection with a single unnamed vector."
                )
            if candidate == collection_name and create_if_missing:
                self.client.create_collection(
                    collection_name=candidate,
                    vectors_config=models.VectorParams(
                        size=vector_size, distance=models.Distance.COSINE
                    ),
                )
                return candidate
            if candidate == compatible_name and create_if_missing:
                self.client.create_collection(
                    collection_name=candidate,
                    vectors_config=models.VectorParams(
                        size=vector_size, distance=models.Distance.COSINE
                    ),
                )
                return candidate
        return None

    def ingest_document(
        self,
        filename: str,
        content: bytes,
        collection_name: str | None = None,
    ) -> int:
        text = self.extract_text(filename, content)
        document_hash = hashlib.sha256(content).hexdigest()
        target_collection = collection_name or self.collection_name
        cache_key = hashlib.sha256(
            f"{document_hash}:{self.chunk_size}:{self.chunk_overlap}".encode("utf-8")
        ).hexdigest()
        cached_chunks = self.storage.get_cached_chunks(cache_key)
        if cached_chunks is None:
            cached_chunks = [
                chunk.page_content
                for chunk in self.splitter.create_documents([text])
                if chunk.page_content.strip()
            ]
            if not cached_chunks:
                raise ValueError(f"No indexable text was found in {filename}.")
            self.storage.cache_chunks(cache_key, cached_chunks)

        vectors = self.embeddings.embed_documents(cached_chunks)
        target_collection = self._resolve_collection(
            target_collection, len(vectors[0]), create_if_missing=True
        )
        if target_collection is None:
            raise RuntimeError("Unable to select a Qdrant collection for document indexing.")
        if self._requires_payload_index and not self.client.get_collection(
            target_collection
        ).payload_schema.get("document_hash"):
            self.client.create_payload_index(
                collection_name=target_collection,
                field_name="document_hash",
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )

        points = []
        for index, (chunk, vector) in enumerate(zip(cached_chunks, vectors, strict=True)):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_hash}:{index}"))
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text": chunk,
                        "source": filename,
                        "document_hash": document_hash,
                        "chunk_index": index,
                    },
                )
            )
        self.client.upsert(collection_name=target_collection, points=points, wait=True)
        return len(points)

    def search(
        self,
        question: str,
        limit: int = 4,
        document_hashes: list[str] | None = None,
        collection_name: str | None = None,
    ) -> list[dict[str, Any]]:
        vector = self.embeddings.embed_query(question)
        target_collection = self._resolve_collection(
            collection_name or self.collection_name,
            len(vector),
            create_if_missing=False,
        )
        if target_collection is None:
            return []
        response = self.client.query_points(
            collection_name=target_collection,
            query=vector,
            limit=limit,
            with_payload=True,
            query_filter=(
                models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_hash",
                            match=models.MatchAny(any=document_hashes),
                        )
                    ]
                )
                if document_hashes
                else None
            ),
        )
        results = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                {
                    "text": str(payload.get("text", "")),
                    "source": str(payload.get("source", "Unknown source")),
                    "score": float(point.score or 0.0),
                }
            )
        return results
