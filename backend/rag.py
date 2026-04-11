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
    # Etisalat Afghanistan — IVR System 888 — Identity & Assistant Rules
    # ---------------------------------------------------------------------------
    "This is the Etisalat Afghanistan IVR voice assistant. The assistant name is Etisalat. "
    "The IVR system number is 888. The default language is Dari. "
    "The assistant must never call itself by any other name.",

    "Etisalat IVR rules: Always follow the IVR flow in order. Do not skip steps. "
    "Do not invent menu options. Only use the provided menu items. "
    "The assistant name must always be Etisalat. Default language is Dari unless the user specifies otherwise.",

    # ---------------------------------------------------------------------------
    # IVR Flow — Step by Step
    # ---------------------------------------------------------------------------
    "Etisalat IVR Step 1 — Preferred Language: The user selects their preferred language. "
    "The default language is Dari. If the user does not specify, continue in Dari.",

    "Etisalat IVR Step 2 — Promotional Announcement: Play current Etisalat promotions and announcements "
    "to the user before reaching the main menu.",

    "Etisalat IVR Step 3 — Greeting Script: Play the standard Etisalat greeting message. "
    "In Dari: خوش آمدید به اتصالات. In Pashto: اتصالات ته ښه راغلاست.",

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
    # Dialogue Scripts — Dari (دری)
    # ---------------------------------------------------------------------------
    "Etisalat Dari greeting (Step 1): خوش آمدید به اتصالات. لطفاً زبان مورد نظر خود را انتخاب کنید. زبان پیش‌فرض دری است.",

    "Etisalat Dari promotion (Step 2): اینجا آخرین پیشنهادات و تبلیغات اتصالات برای شما است.",

    "Etisalat Dari main menu (Step 4): لطفاً یکی از گزینه‌های زیر را انتخاب کنید: "
    "یک برای بهترین آفرهای من، دو برای بسته‌های انترنت، سه برای بسته‌های مکالمه، "
    "چهار برای بسته‌های مختلط، پنج برای خدمات، شش برای پکیج و مهاجرت، "
    "هفت برای پرسش موجودی، هشت برای کمک بیشتر، نه برای غیرفعال کردن بسته‌های DRM.",

    # Dari — menu option details
    "Etisalat گزینه ۱ — بهترین آفرهای من (My Best Offers): بهترین پیشنهادات شخصی‌سازی‌شده برای مشترک بر اساس الگوی استفاده.",
    "Etisalat گزینه ۲ — بسته‌های انترنت (Data Bundles): بسته‌های داده روزانه، هفتگی و ماهانه اتصالات.",
    "Etisalat گزینه ۳ — بسته‌های مکالمه (Voice Bundles): بسته‌های مکالمه محلی و بین‌المللی.",
    "Etisalat گزینه ۴ — بسته‌های مختلط (Mixed & Other Bundles): بسته‌هایی که شامل داده، مکالمه و پیامک هستند.",
    "Etisalat گزینه ۵ — خدمات (Services): خدمات ارزش افزوده اتصالات شامل موسیقی، بازی و بیشتر.",
    "Etisalat گزینه ۶ — پکیج و مهاجرت (Package & Migration): تغییر پکیج یا مهاجرت به پلن جدید.",
    "Etisalat گزینه ۷ — پرسش موجودی (Balance Inquiry): بررسی موجودی حساب، تاریخ انقضا و استفاده.",
    "Etisalat گزینه ۸ — کمک بیشتر (Further Assistance): اتصال به اپراتور زنده یا پشتیبانی بیشتر.",
    "Etisalat گزینه ۹ — غیرفعال کردن بسته‌های DRM (DRM Bundles Deactivation): لغو اشتراک بسته‌های DRM فعال.",

    # ---------------------------------------------------------------------------
    # Dialogue Scripts — Pashto (پښتو)
    # ---------------------------------------------------------------------------
    "Etisalat Pashto greeting (Step 1): اتصالات ته ښه راغلاست. مهرباني وکړئ خپله ژبه وټاکئ. د ډیفالټ ژبه دري ده.",

    "Etisalat Pashto promotion (Step 2): دلته د اتصالات وروستي وړاندیزونه او اعلانونه دي.",

    "Etisalat Pashto main menu (Step 4): مهرباني وکړئ لاندې انتخابونو څخه یو غوره کړئ: "
    "یو د زما غوره وړاندیزونو لپاره، دوه د انټرنیټ بنډلونو لپاره، درې د غږ بنډلونو لپاره، "
    "څلور د مخلوط بنډلونو لپاره، پنځه د خدماتو لپاره، شپږ د پکیج او مهاجرت لپاره، "
    "اوه د بیلانس پوښتنې لپاره، اته د نورې مرستې لپاره، نهه د DRM بنډلونو د غیر فعالولو لپاره.",

    # Pashto — menu option details
    "Etisalat انتخاب ۱ — زما غوره وړاندیزونه (My Best Offers): د مشترک د کارونې نمونې پر بنسټ شخصي شوي غوره وړاندیزونه.",
    "Etisalat انتخاب ۲ — د انټرنیټ بنډلونه (Data Bundles): د اتصالات ورځني، اونیزي او میاشتني ډیټا بنډلونه.",
    "Etisalat انتخاب ۳ — د غږ بنډلونه (Voice Bundles): محلي او نړیوال د غږ بنډلونه.",
    "Etisalat انتخاب ۴ — مخلوط بنډلونه (Mixed & Other Bundles): هغه بنډلونه چې ډیټا، غږ او SMS لري.",
    "Etisalat انتخاب ۵ — خدمات (Services): د اتصالات ارزش اضافه خدمات لکه موسیقي، لوبې او نور.",
    "Etisalat انتخاب ۶ — پکیج او مهاجرت (Package & Migration): د پکیج بدلول یا نوي پلان ته مهاجرت.",
    "Etisalat انتخاب ۷ — د بیلانس پوښتنه (Balance Inquiry): د حساب بیلانس، د پای نیټه او د کارونې کتنه.",
    "Etisalat انتخاب ۸ — نوره مرسته (Further Assistance): ژوندي اپریټر یا اضافي ملاتړ ته اتصال.",
    "Etisalat انتخاب ۹ — د DRM بنډلونو غیر فعالول (DRM Bundles Deactivation): د فعالو DRM بنډلونو لغو.",

    # ---------------------------------------------------------------------------
    # Etisalat — General Telecom Knowledge (Afghanistan)
    # ---------------------------------------------------------------------------
    "Etisalat Afghanistan customer service is available by calling 888. "
    "The IVR system provides self-service options 24 hours a day, 7 days a week.",

    "To check Etisalat balance in Dari: موجودی خود را با گرفتن *888# بررسی کنید. "
    "To check balance in Pashto: خپل بیلانس د *888# ډایل کولو سره وګورئ.",

    "Etisalat Afghanistan offers prepaid and postpaid plans. "
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

    # ---------------------------------------------------------------------------
    # Etisalat — Sub-menu details per option (Dari + Pashto)
    # ---------------------------------------------------------------------------

    # Option 1 — My Best Offers
    "Etisalat My Best Offers sub-options (Dari): "
    "الف) بسته انترنت ۱ گیگ رایگان  ب) ۵۰ دقیقه مکالمه رایگان  ج) مشاهده همه آفرها. "
    "Offers are personalised per subscriber and change monthly.",

    # Option 2 — Data Bundles
    "Etisalat Data Bundle sub-options (Dari): "
    "یک برای بسته روزانه، دو برای بسته هفتگی، سه برای بسته ماهانه. "
    "Daily bundle: 1GB for 24 hours. Weekly bundle: 5GB for 7 days. Monthly bundle: 20GB for 30 days.",

    "Etisalat د انټرنیټ بنډل فرعي انتخابونه (Pashto): "
    "یو ورځني بنډل، دوه اونیز بنډل، درې میاشتني بنډل. "
    "ورځنی: ۱ ګیګ ۲۴ ساعته. اونیز: ۵ ګیګ ۷ ورځې. میاشتنی: ۲۰ ګیګ ۳۰ ورځې.",

    # Option 3 — Voice Bundles
    "Etisalat Voice Bundle sub-options (Dari): "
    "یک برای مکالمه داخلی، دو برای مکالمه بین‌المللی. "
    "Local voice bundle: 100 minutes for 7 days. International: rates vary by country.",

    # Option 4 — Mixed Bundles
    "Etisalat Mixed Bundle sub-options (Dari): "
    "یک برای بسته کوچک (۵۰۰MB + ۳۰ دقیقه)، دو برای متوسط (۲GB + ۱۰۰ دقیقه)، "
    "سه برای بزرگ (۵GB + ۳۰۰ دقیقه + ۵۰ پیامک).",

    # Option 5 — Services
    "Etisalat Services sub-options (Dari): "
    "یک آهنگ پشت خط، دو اخبار روزانه، سه بازی آنلاین، چهار موسیقی. "
    "All services are subscription-based and can be deactivated via option 9.",

    # Option 6 — Package & Migration
    "Etisalat Package and Migration sub-options (Dari): "
    "یک برای تغییر پکیج فعلی، دو برای مهاجرت به پلن جدید، سه برای مشاهده پکیج فعلی. "
    "Migration is free and takes effect immediately.",

    # Option 7 — Balance Inquiry
    "Etisalat Balance Inquiry sub-options (Dari): "
    "یک برای مشاهده موجودی، دو برای تاریخ انقضا، سه برای تاریخچه مصرف. "
    "Balance can also be checked by dialling *888#.",

    # Option 8 — Further Assistance
    "Etisalat Further Assistance sub-options (Dari): "
    "یک برای اتصال به اپراتور زنده، دو برای پشتیبانی تخنیکی، سه برای ثبت شکایت. "
    "Live agents are available 24/7.",

    # Option 9 — DRM Deactivation
    "Etisalat DRM Deactivation sub-options (Dari): "
    "یک برای مشاهده بسته‌های DRM فعال، دو برای غیرفعال کردن همه بسته‌های DRM، "
    "سه برای غیرفعال کردن یک بسته خاص. "
    "DRM deactivation takes effect within 24 hours.",

    # Frustration / emotional handling
    "Etisalat emotional handling: If a caller expresses frustration, confusion, or asks why they must "
    "follow the IVR, the assistant should respond empathetically first before re-presenting options. "
    "In Dari: 'بخشش می‌خواهم، نگران نباشید. من اینجا هستم که کمکتان کنم.' "
    "In Pashto: 'بخښنه وغواړئ، اندیښنه مه کوئ. زه دلته یم چې مرسته وکړم.'",

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
