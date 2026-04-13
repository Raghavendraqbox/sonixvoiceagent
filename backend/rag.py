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
    # Etisalat — IVR System 888 — Identity & Assistant Rules
    # ---------------------------------------------------------------------------
    "This is the Etisalat IVR voice assistant. The assistant name is Etisalat. "
    "The IVR system number is 888. The default language is Telugu. "
    "The assistant must never call itself by any other name.",

    "Etisalat IVR rules: Always follow the IVR flow in order. Do not skip steps. "
    "Do not invent menu options. Only use the provided menu items. "
    "The assistant name must always be Etisalat. Default language is Telugu unless the user specifies otherwise.",

    # ---------------------------------------------------------------------------
    # IVR Flow — Step by Step
    # ---------------------------------------------------------------------------
    "Etisalat IVR Step 1 — Preferred Language: The user selects their preferred language. "
    "The default language is Telugu. If the user does not specify, continue in Telugu.",

    "Etisalat IVR Step 2 — Promotional Announcement: Play current Etisalat promotions and announcements "
    "to the user before reaching the main menu.",

    "Etisalat IVR Step 3 — Greeting Script: Play the standard Etisalat greeting message. "
    "In Telugu: \u0c0e\u0c1f\u0c3f\u0c38\u0c32\u0c3e\u0c1f\u0c4d\u200c\u0c15\u0c41 \u0c38\u0c4d\u0c35\u0c3e\u0c17\u0c24\u0c02. In Kannada: \u0c8e\u0c9f\u0cbf\u0cb8\u0cb2\u0cbe\u0c9f\u0ccd\u200c\u0c97\u0cc6 \u0cb8\u0ccd\u0cb5\u0cbe\u0c97\u0ca4.",

    "Etisalat IVR Step 4 — Main Menu: The user must choose one of 9 options. "
    "Option 1: My Best Offers. "
    "Option 2: Data Bundles. "
    "Option 3: Voice Bundles. "
    "Option 4: Mixed and Other Bundles. "
    "Option 5: Services. "
    "Option 6: Package and Migration. "
    "Option 7: Balance Inquiry. "
    "Option 8: Further Assistance. "
    "Option 9: DRM Bundles Deactivation.",

    # ---------------------------------------------------------------------------
    # Dialogue Scripts — Telugu (తెలుగు)
    # ---------------------------------------------------------------------------
    "Etisalat Telugu greeting (Step 1): ఎటిసలాట్‌కు స్వాగతం. దయచేసి మీ భాషను ఎంచుకోండి. డిఫాల్ట్ భాష తెలుగు.",
    "Etisalat Telugu promotion (Step 2): ఎటిసలాట్ ఈ నెలలో మీకు ప్రత్యేక ఆఫర్లు అందిస్తోంది.",
    "Etisalat Telugu main menu (Step 4): దయచేసి కింది ఎంపికలలో ఒకదాన్ని ఎంచుకోండి: "
    "ఒకటి నా ఉత్తమ ఆఫర్లకు, రెండు ఇంటర్నెట్ ప్యాకేజీలకు, మూడు కాల్ ప్యాకేజీలకు, "
    "నాలుగు మిక్స్డ్ ప్యాకేజీలకు, అయిదు సేవలకు, ఆరు ప్యాకేజీ మరియు మైగ్రేషన్‌కు, "
    "ఏడు బ్యాలెన్స్ విచారణకు, ఎనిమిది మరింత సహాయానికి, తొమ్మిది DRM నిష్క్రియం చేయడానికి.",

    "Etisalat Telugu option 1 — నా ఉత్తమ ఆఫర్లు (My Best Offers): వినియోగదారు వాడుక ఆధారంగా వ్యక్తిగతీకరించిన ఉత్తమ ఆఫర్లు.",
    "Etisalat Telugu option 2 — ఇంటర్నెట్ ప్యాకేజీలు (Data Bundles): రోజువారీ, వారపు మరియు నెలవారీ డేటా ప్యాకేజీలు.",
    "Etisalat Telugu option 3 — కాల్ ప్యాకేజీలు (Voice Bundles): దేశీయ మరియు అంతర్జాతీయ కాల్ ప్యాకేజీలు.",
    "Etisalat Telugu option 4 — మిక్స్డ్ ప్యాకేజీలు (Mixed Bundles): డేటా, కాల్ మరియు SMS కలిపిన ప్యాకేజీలు.",
    "Etisalat Telugu option 5 — సేవలు (Services): రింగ్‌టోన్, వార్తలు, గేమ్స్, సంగీతం వంటి సేవలు.",
    "Etisalat Telugu option 6 — ప్యాకేజీ మరియు మైగ్రేషన్ (Package & Migration): ప్యాకేజీ మార్పు లేదా కొత్త ప్లాన్‌కు మారడం.",
    "Etisalat Telugu option 7 — బ్యాలెన్స్ విచారణ (Balance Inquiry): ఖాతా బ్యాలెన్స్, గడువు తేదీ మరియు వినియోగ చరిత్ర.",
    "Etisalat Telugu option 8 — మరింత సహాయం (Further Assistance): లైవ్ ఆపరేటర్ లేదా అదనపు మద్దతు.",
    "Etisalat Telugu option 9 — DRM నిష్క్రియం (DRM Deactivation): చురుకైన DRM బండిల్స్ రద్దు.",

    # ---------------------------------------------------------------------------
    # Dialogue Scripts — Kannada (ಕನ್ನಡ)
    # ---------------------------------------------------------------------------
    "Etisalat Kannada greeting (Step 1): ಎಟಿಸಲಾಟ್‌ಗೆ ಸ್ವಾಗತ. ದಯವಿಟ್ಟು ನಿಮ್ಮ ಭಾಷೆಯನ್ನು ಆಯ್ಕೆ ಮಾಡಿ. ಡೀಫಾಲ್ಟ್ ಭಾಷೆ ಕನ್ನಡ.",
    "Etisalat Kannada promotion (Step 2): ಎಟಿಸಲಾಟ್ ಈ ತಿಂಗಳು ನಿಮಗೆ ವಿಶೇಷ ಕೊಡುಗೆಗಳನ್ನು ನೀಡುತ್ತಿದೆ.",
    "Etisalat Kannada main menu (Step 4): ದಯವಿಟ್ಟು ಕೆಳಗಿನ ಆಯ್ಕೆಗಳಲ್ಲಿ ಒಂದನ್ನು ಆಯ್ಕೆ ಮಾಡಿ: "
    "ಒಂದು ನನ್ನ ಉತ್ತಮ ಕೊಡುಗೆಗಳಿಗೆ, ಎರಡು ಇಂಟರ್ನೆಟ್ ಪ್ಯಾಕೇಜ್‌ಗಳಿಗೆ, ಮೂರು ಕರೆ ಪ್ಯಾಕೇಜ್‌ಗಳಿಗೆ, "
    "ನಾಲ್ಕು ಮಿಕ್ಸ್ಡ್ ಪ್ಯಾಕೇಜ್‌ಗಳಿಗೆ, ಐದು ಸೇವೆಗಳಿಗೆ, ಆರು ಪ್ಯಾಕೇಜ್ ಮತ್ತು ಮೈಗ್ರೇಷನ್‌ಗೆ, "
    "ಏಳು ಬ್ಯಾಲೆನ್ಸ್ ವಿಚಾರಣೆಗೆ, ಎಂಟು ಹೆಚ್ಚಿನ ಸಹಾಯಕ್ಕೆ, ಒಂಬತ್ತು DRM ನಿಷ್ಕ್ರಿಯಗೊಳಿಸಲು.",

    "Etisalat Kannada option 1 — ನನ್ನ ಉತ್ತಮ ಕೊಡುಗೆಗಳು (My Best Offers): ಬಳಕೆದಾರರ ಬಳಕೆ ಮಾದರಿ ಆಧಾರದ ಮೇಲೆ ವ್ಯಕ್ತಿಗತ ಕೊಡುಗೆಗಳು.",
    "Etisalat Kannada option 2 — ಇಂಟರ್ನೆಟ್ ಪ್ಯಾಕೇಜ್‌ಗಳು (Data Bundles): ದಿನಸರಿ, ವಾರಸರಿ ಮತ್ತು ತಿಂಗಳ ಡೇಟಾ ಪ್ಯಾಕೇಜ್‌ಗಳು.",
    "Etisalat Kannada option 3 — ಕರೆ ಪ್ಯಾಕೇಜ್‌ಗಳು (Voice Bundles): ದೇಶೀಯ ಮತ್ತು ಅಂತರರಾಷ್ಟ್ರೀಯ ಕರೆ ಪ್ಯಾಕೇಜ್‌ಗಳು.",
    "Etisalat Kannada option 4 — ಮಿಕ್ಸ್ಡ್ ಪ್ಯಾಕೇಜ್‌ಗಳು (Mixed Bundles): ಡೇಟಾ, ಕರೆ ಮತ್ತು SMS ಸೇರಿದ ಪ್ಯಾಕೇಜ್‌ಗಳು.",
    "Etisalat Kannada option 5 — ಸೇವೆಗಳು (Services): ರಿಂಗ್‌ಟೋನ್, ಸುದ್ದಿ, ಆಟಗಳು, ಸಂಗೀತ ಸೇರಿದ ಸೇವೆಗಳು.",
    "Etisalat Kannada option 6 — ಪ್ಯಾಕೇಜ್ ಮತ್ತು ಮೈಗ್ರೇಷನ್ (Package & Migration): ಪ್ಯಾಕೇಜ್ ಬದಲಿಸುವುದು ಅಥವಾ ಹೊಸ ಯೋಜನೆಗೆ ಬದಲಾಯಿಸುವುದು.",
    "Etisalat Kannada option 7 — ಬ್ಯಾಲೆನ್ಸ್ ವಿಚಾರಣೆ (Balance Inquiry): ಖಾತೆ ಬ್ಯಾಲೆನ್ಸ್, ಮುಕ್ತಾಯ ದಿನಾಂಕ ಮತ್ತು ಬಳಕೆ ಇತಿಹಾಸ.",
    "Etisalat Kannada option 8 — ಹೆಚ್ಚಿನ ಸಹಾಯ (Further Assistance): ನೇರ ಆಪರೇಟರ್ ಅಥವಾ ಹೆಚ್ಚುವರಿ ಬೆಂಬಲ.",
    "Etisalat Kannada option 9 — DRM ನಿಷ್ಕ್ರಿಯ (DRM Deactivation): ಸಕ್ರಿಯ DRM ಬಂಡಲ್‌ಗಳ ರದ್ದು.",

    # ---------------------------------------------------------------------------
    # Etisalat — General Telecom Knowledge
    # ---------------------------------------------------------------------------
    "Etisalat customer service is available by calling 888. "
    "The IVR system provides self-service options 24 hours a day, 7 days a week.",

    "To check Etisalat balance in Telugu: మీ బ్యాలెన్స్‌ను *888# డయల్ చేసి తెలుసుకోండి. "
    "To check balance in Kannada: ನಿಮ್ಮ ಬ್ಯಾಲೆನ್ಸ್ *888# ಡಯಲ್ ಮಾಡಿ ತಿಳಿಯಿರಿ.",

    "Etisalat offers prepaid and postpaid plans. "
    "Migration between plans is available through IVR option 6 — Package & Migration.",

    "Etisalat DRM (Digital Rights Management) bundles are premium content subscriptions. "
    "To deactivate any active DRM bundle, select option 9 from the main menu.",

    "Etisalat data bundles range from hourly to monthly. "
    "Select option 2 from the main menu to view and activate data bundles.",

    "Etisalat voice bundles provide discounted local and international call minutes. "
    "Select option 3 from the main menu to view and activate voice bundles.",

    "Etisalat mixed bundles combine data, voice minutes, and SMS in a single package. "
    "Select option 4 from the main menu to view mixed and other bundles.",

    "Etisalat My Best Offers (option 1) shows personalized promotions tailored to the customer's usage history. "
    "These change regularly and are unique to each subscriber.",

    "Etisalat Services (option 5) includes value-added services such as music streaming, "
    "games, news alerts, caller tunes, and other digital content.",

    "Etisalat Further Assistance (option 8) connects the caller to a live customer service agent "
    "or provides advanced troubleshooting support.",

    # Frustration / emotional handling
    "Etisalat emotional handling: If a caller expresses frustration, confusion, or asks why they must "
    "follow the IVR, the assistant should respond empathetically first before re-presenting options. "
    "In Telugu: 'క్షమించండి, చింతించకండి. నేను మీకు సహాయం చేయడానికి ఇక్కడ ఉన్నాను.' "
    "In Kannada: 'ಕ್ಷಮಿಸಿ, ಚಿಂತಿಸಬೇಡಿ. ನಾನು ಇಲ್ಲಿ ಸಹಾಯ ಮಾಡಲು ಇದ್ದೇನೆ.'",

    # Flow continuation rule
    "Etisalat IVR flow rule: After the caller confirms a menu selection, the assistant must "
    "present the sub-options for that selection. It must NOT return to the main menu unless "
    "the caller explicitly requests it. The conversation continues within the chosen option.",
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
