"""
rag.py — FAISS-backed RAG module with sentence-transformer embeddings.

Provides DocumentLoader (ingest) and RAGRetriever (query-time) classes.
Seed documents are embedded on first run; subsequent runs reload
the persisted FAISS index for sub-millisecond retrieval.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from config import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — these are heavy; only loaded when the module is first used.
# ---------------------------------------------------------------------------

def _load_sentence_transformer():
    from sentence_transformers import SentenceTransformer  # type: ignore
    return SentenceTransformer(config.rag.embedding_model)


def _load_faiss():
    import faiss  # type: ignore
    return faiss


# ---------------------------------------------------------------------------
# Built-in seed documents
# These are embedded into the FAISS index if no external docs are found.
# Replace with your own domain-specific documents in the docs/ directory.
# ---------------------------------------------------------------------------

TELECOM_SEED_DOCS: List[str] = [
    # ---------------------------------------------------------------------------
    # General assistant identity and behaviour
    # ---------------------------------------------------------------------------
    "This is a friendly AI voice assistant that supports Telugu and Kannada languages. "
    "The assistant should always respond in the language the user is speaking. "
    "The assistant is helpful, warm, and conversational — like a knowledgeable friend on a phone call.",

    "The assistant must keep responses short and natural — 1 to 3 sentences maximum. "
    "This is a voice call, so never use bullet points, markdown, lists, or formatting symbols. "
    "Speak in plain, natural conversational language.",

    "If the user asks something the assistant does not know, it should say so honestly "
    "and offer to help with something else. Never make up information.",

    "The assistant should always be empathetic. If the user sounds confused or frustrated, "
    "acknowledge their feeling first before providing information. "
    "In Telugu: 'క్షమించండి, నేను మీకు సహాయం చేయడానికి ఇక్కడ ఉన్నాను.' "
    "In Kannada: 'ಕ್ಷಮಿಸಿ, ನಾನು ಇಲ್ಲಿ ಸಹಾಯ ಮಾಡಲು ಇದ್ದೇನೆ.'",

    # ---------------------------------------------------------------------------
    # Telugu language knowledge
    # ---------------------------------------------------------------------------
    "Telugu greeting: నమస్కారం! నేను మీ AI అసిస్టెంట్‌ను. మీకు ఎలా సహాయం చేయగలను?",

    "Telugu goodbye: మీరు మాట్లాడినందుకు ధన్యవాదాలు. మీ రోజు శుభంగా గడవాలి.",

    "Telugu clarification: నేను మీరు చెప్పింది పూర్తిగా అర్థం చేసుకోలేదు. "
    "దయచేసి మళ్ళీ చెప్పగలరా?",

    "Telugu acknowledgement phrases: అర్థమైంది. సరే. చాలా ధన్యవాదాలు. "
    "మీరు చెప్పింది నాకు అర్థమైంది.",

    # ---------------------------------------------------------------------------
    # Kannada language knowledge
    # ---------------------------------------------------------------------------
    "Kannada greeting: ನಮಸ್ಕಾರ! ನಾನು ನಿಮ್ಮ AI ಅಸಿಸ್ಟೆಂಟ್. ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಬಲ್ಲೆ?",

    "Kannada goodbye: ಮಾತಾಡಿದ್ದಕ್ಕೆ ಧನ್ಯವಾದಗಳು. ನಿಮ್ಮ ದಿನ ಚೆನ್ನಾಗಿ ಕಳೆಯಲಿ.",

    "Kannada clarification: ನೀವು ಹೇಳಿದ್ದು ನನಗೆ ಸರಿಯಾಗಿ ಅರ್ಥವಾಗಲಿಲ್ಲ. "
    "ದಯವಿಟ್ಟು ಮತ್ತೊಮ್ಮೆ ಹೇಳಬಹುದೇ?",

    "Kannada acknowledgement phrases: ಅರ್ಥವಾಯಿತು. ಸರಿ. ತುಂಬಾ ಧನ್ಯವಾದಗಳು. "
    "ನೀವು ಹೇಳಿದ್ದು ನನಗೆ ಅರ್ಥವಾಯಿತು.",

    # ---------------------------------------------------------------------------
    # Conversation best practices
    # ---------------------------------------------------------------------------
    "When a user asks a question, answer it directly and clearly. "
    "Do not repeat the question back to the user. "
    "Do not add unnecessary filler phrases.",

    "If the user says goodbye or indicates they are done, always respond warmly and end gracefully. "
    "Do not ask if there is anything else after the user has said goodbye.",

    "The assistant supports both Telugu and Kannada equally well. "
    "It should detect which language the user is speaking and respond in that same language.",
]


# ---------------------------------------------------------------------------
# DocumentLoader
# ---------------------------------------------------------------------------

class DocumentLoader:
    """
    Loads, chunks, and indexes text documents for RAG retrieval.

    Usage:
        loader = DocumentLoader()
        loader.load_from_directory("./docs")   # optional external docs
        loader.build_index()
    """

    def __init__(self) -> None:
        self._raw_chunks: List[str] = []

    def load_seed_documents(self) -> None:
        """Load the built-in telecom knowledge base."""
        self._raw_chunks.extend(TELECOM_SEED_DOCS)
        logger.info("Loaded %d seed documents", len(TELECOM_SEED_DOCS))

    def load_from_directory(self, directory: str) -> None:
        """
        Load all .txt files from the given directory, chunk them, and add
        to the internal document pool.

        Args:
            directory: Path to directory containing .txt knowledge-base files.
        """
        doc_dir = Path(directory)
        if not doc_dir.exists():
            logger.warning("Docs directory %s does not exist — skipping", directory)
            return

        txt_files = list(doc_dir.glob("*.txt"))
        if not txt_files:
            logger.warning("No .txt files found in %s", directory)
            return

        for fpath in txt_files:
            try:
                text = fpath.read_text(encoding="utf-8")
                chunks = self.chunk_text(text)
                self._raw_chunks.extend(chunks)
                logger.info("Loaded %d chunks from %s", len(chunks), fpath.name)
            except Exception as exc:
                logger.error("Failed to load %s: %s", fpath, exc)

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping fixed-size word chunks.

        Args:
            text: Raw document text.

        Returns:
            List of string chunks.
        """
        words = text.split()
        chunk_size = config.rag.chunk_size
        overlap = config.rag.chunk_overlap
        chunks: List[str] = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            if chunk.strip():
                chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    def get_chunks(self) -> List[str]:
        """Return the accumulated raw text chunks."""
        return list(self._raw_chunks)

    def build_index(self) -> "FAISSIndex":
        """
        Embed all loaded chunks and build a FAISS index.

        Returns:
            FAISSIndex instance ready for retrieval.
        """
        if not self._raw_chunks:
            raise ValueError("No documents loaded. Call load_seed_documents() first.")
        return FAISSIndex.build(self._raw_chunks)


# ---------------------------------------------------------------------------
# FAISSIndex
# ---------------------------------------------------------------------------

class FAISSIndex:
    """
    Wraps a FAISS flat L2 index together with the text chunks it indexes.

    Supports persistence to disk so the index is not rebuilt on every start.
    """

    def __init__(
        self,
        index,          # faiss.Index
        chunks: List[str],
        model,          # SentenceTransformer
    ) -> None:
        self._index = index
        self._chunks = chunks
        self._model = model

    @classmethod
    def build(cls, chunks: List[str]) -> "FAISSIndex":
        """
        Embed chunks and create a new FAISS IndexFlatIP (inner-product / cosine).

        Args:
            chunks: List of text strings to embed and index.

        Returns:
            Populated FAISSIndex.
        """
        faiss = _load_faiss()
        model = _load_sentence_transformer()

        logger.info("Embedding %d chunks (this may take a moment)…", len(chunks))
        embeddings = model.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)   # cosine similarity via normalized vectors
        index.add(embeddings)
        logger.info("FAISS index built: %d vectors, dim=%d", index.ntotal, dim)
        return cls(index, chunks, model)

    @classmethod
    def load(cls, path: str) -> "FAISSIndex":
        """
        Load a persisted index from disk.

        Args:
            path: Directory where index.faiss and chunks.pkl were saved.

        Returns:
            Loaded FAISSIndex.

        Raises:
            FileNotFoundError: If the index files do not exist.
        """
        faiss = _load_faiss()
        model = _load_sentence_transformer()

        index_file = os.path.join(path, "index.faiss")
        chunks_file = os.path.join(path, "chunks.pkl")

        if not os.path.exists(index_file) or not os.path.exists(chunks_file):
            raise FileNotFoundError(f"No saved index found at {path}")

        index = faiss.read_index(index_file)
        with open(chunks_file, "rb") as f:
            chunks = pickle.load(f)

        logger.info("FAISS index loaded from %s (%d vectors)", path, index.ntotal)
        return cls(index, chunks, model)

    def save(self, path: str) -> None:
        """
        Persist the index to disk.

        Args:
            path: Directory to write index.faiss and chunks.pkl.
        """
        faiss = _load_faiss()
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self._index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "chunks.pkl"), "wb") as f:
            pickle.dump(self._chunks, f)
        logger.info("FAISS index saved to %s", path)

    def search(
        self, query: str, top_k: int = None, threshold: float = None
    ) -> List[Tuple[str, float]]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query: Natural language query string.
            top_k: Number of results to return. Defaults to config.rag.top_k.
            threshold: Minimum cosine similarity. Defaults to config.rag.similarity_threshold.

        Returns:
            List of (chunk_text, score) tuples sorted by descending score.
        """
        top_k = top_k or config.rag.top_k
        threshold = threshold if threshold is not None else config.rag.similarity_threshold

        q_emb = self._model.encode(
            [query], show_progress_bar=False, normalize_embeddings=True
        )
        q_emb = np.array(q_emb, dtype=np.float32)

        scores, indices = self._index.search(q_emb, top_k)

        results: List[Tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if float(score) < threshold:
                continue
            results.append((self._chunks[idx], float(score)))

        logger.debug(
            "RAG search returned %d results for query: %s",
            len(results),
            query[:60],
        )
        return results


# ---------------------------------------------------------------------------
# RAGRetriever — high-level interface used by the LLM client
# ---------------------------------------------------------------------------

class RAGRetriever:
    """
    High-level retriever that manages FAISSIndex lifecycle.

    On construction it will:
    1. Try to load a persisted index from disk.
    2. If none exists, build one from seed + external docs and persist it.

    Usage:
        retriever = RAGRetriever()
        await retriever.initialize()
        context_chunks = retriever.retrieve("cheap internet bundle")
    """

    def __init__(self) -> None:
        self._index: Optional[FAISSIndex] = None

    def initialize(self) -> None:
        """
        Build or reload the FAISS index.  Call once at startup (synchronous,
        acceptable at boot time before the server accepts connections).
        """
        index_path = config.rag.index_path
        try:
            self._index = FAISSIndex.load(index_path)
            logger.info("RAG index loaded from cache at %s", index_path)
        except FileNotFoundError:
            logger.info("No cached RAG index found — building from documents…")
            loader = DocumentLoader()
            loader.load_seed_documents()
            loader.load_from_directory(config.rag.docs_directory)
            self._index = loader.build_index()
            self._index.save(index_path)

    def retrieve(self, query: str) -> List[str]:
        """
        Return the top-k most relevant text chunks for the given query.

        Args:
            query: The user's current utterance or LLM prompt fragment.

        Returns:
            List of relevant text strings (may be empty if index not built
            or no results exceed the similarity threshold).
        """
        if self._index is None:
            logger.warning("RAGRetriever.retrieve called before initialize()")
            return []

        results = self._index.search(query)
        return [chunk for chunk, _score in results]

    def format_context(self, query: str) -> str:
        """
        Build the RAG context block to inject into the LLM prompt.

        Args:
            query: The user's utterance.

        Returns:
            Formatted context string, or empty string if no relevant docs.
        """
        chunks = self.retrieve(query)
        if not chunks:
            return ""
        joined = "\n---\n".join(chunks)
        return f"Relevant knowledge:\n{joined}"
