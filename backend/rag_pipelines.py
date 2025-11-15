from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import uuid
import os
from dotenv import load_dotenv
load_dotenv()
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

GROQ_MODEL = "llama-3.1-8b-instant"

# 🔹 Single shared embedder to save RAM
GLOBAL_EMBEDDER = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


@dataclass
class RagPipelineConfig:
    id: str
    name: str
    chunk_size: int
    overlap: int


class RagPipeline:
    def __init__(self, config: RagPipelineConfig):
        self.config = config
        # all pipelines share the same embedder instance
        self.embedder = GLOBAL_EMBEDDER
        self.index = None  # FAISS index
        self.chunks: List[str] = []
        self.chunk_ids: List[str] = []

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        self.client = Groq(api_key=api_key)

    def _chunk_text(self, text: str) -> List[str]:
        """Simple character-based chunking with overlap."""
        size = self.config.chunk_size
        overlap = self.config.overlap
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = start + size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
            if start < 0:
                start = 0
        return chunks

    def index_documents(self, documents: List[str]) -> None:
        """Chunk documents and build FAISS index for this pipeline."""
        all_chunks: List[str] = []
        for doc in documents:
            all_chunks.extend(self._chunk_text(doc))

        if not all_chunks:
            return

        self.chunks = all_chunks
        self.chunk_ids = [str(uuid.uuid4()) for _ in all_chunks]

        embeddings = self.embedder.encode(
            all_chunks,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        self.index = index

    def _retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve top_k chunks for a query."""
        if self.index is None or not self.chunks:
            return []
        q_vec = self.embedder.encode(
            [query], convert_to_numpy=True, show_progress_bar=False
        ).astype("float32")
        faiss.normalize_L2(q_vec)
        scores, idxs = self.index.search(q_vec, top_k)
        idxs = idxs[0]
        scores = scores[0]
        results: List[Tuple[str, float]] = []
        for i, s in zip(idxs, scores):
            if i == -1:
                continue
            results.append((self.chunks[i], float(s)))
        return results

    def _call_llm(self, prompt: str) -> str:
        """Call Groq LLM to generate an answer based on context."""
        completion = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that answers strictly "
                        "based on the provided context."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        return completion.choices[0].message.content

    def answer(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Run retrieval + generation and return answer + metadata."""
        retrieved = self._retrieve(question, top_k=top_k)
        context = "\n\n".join([c for c, _ in retrieved])

        prompt = (
            "Use ONLY the context below to answer the question.\n"
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        answer = self._call_llm(prompt)

        ctx_tokens = len(context.split())
        ans_tokens = len(answer.split())
        approx_tokens = ctx_tokens + ans_tokens

        return {
            "pipeline_id": self.config.id,
            "pipeline_name": self.config.name,
            "answer": answer,
            "retrieved_chunks": [c for c, _ in retrieved],
            "approx_tokens": approx_tokens,
        }


def build_pipelines() -> List[RagPipeline]:
    """Create 4 pipelines with different chunking configs (same embedder)."""
    configs = [
        RagPipelineConfig(
            id="A",
            name="MiniLM, chunk 256, overlap 32",
            chunk_size=256,
            overlap=32,
        ),
        RagPipelineConfig(
            id="B",
            name="MiniLM, chunk 512, overlap 64",
            chunk_size=512,
            overlap=64,
        ),
        RagPipelineConfig(
            id="C",
            name="MiniLM, chunk 768, overlap 96",
            chunk_size=768,
            overlap=96,
        ),
        RagPipelineConfig(
            id="D",
            name="MiniLM, chunk 1024, overlap 128",
            chunk_size=1024,
            overlap=128,
        ),
    ]

    return [RagPipeline(cfg) for cfg in configs]