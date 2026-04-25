"""
llm.py — Open-source LLM client for the Telugu & Kannada Voice AI Agent.

Runs entirely locally via Ollama — no external API keys required.

Recommended models for Telugu/Kannada (pull before starting):
  ollama pull qwen2.5:7b         ← good multilingual quality, ~6GB VRAM
  ollama pull qwen2.5:14b        ← better quality, ~10GB VRAM
  ollama pull qwen2.5:32b        ← best quality, ~20GB VRAM
  ollama pull qwen2.5:72b        ← best overall (80GB GPU)
  ollama pull aya-expanse         ← Cohere multilingual, Indian languages supported

Set OLLAMA_MODEL in .env to choose, default is qwen2.5:7b.

Sentence fragments are yielded as they arrive so TTS can begin speaking
after the first complete sentence rather than waiting for the full response.
"""

import asyncio
import json as _json
import logging
import re
from typing import AsyncIterator, Optional

import httpx

from config import config, get_language_config
from memory import ConversationMemory
from rag import RAGRetriever

logger = logging.getLogger(__name__)

# Sentence boundaries: sentence-ending punctuation only.
# Do not split on commas/pipes, otherwise long utterances fragment unnaturally
# and generate many tiny TTS requests.
_SENTENCE_BOUNDARY = re.compile(r"([.!?؟۔])\s*(?=\S|$)")

# Word-count dispatch: yield after this many words even without punctuation.
# Value comes from config so deployments can tune latency without code changes.
_DEFAULT_WORD_DISPATCH_THRESHOLD = 6


class VoiceLLMClient:
    """
    Async streaming LLM client using local Ollama.

    Each call to `stream_response` yields sentence fragments as soon as they
    are complete, so the TTS pipeline can start speaking immediately.

    Supports Telugu and Kannada via language-specific system prompts.
    """

    def __init__(
        self,
        retriever: Optional[RAGRetriever] = None,
        language: str = "telugu",
    ) -> None:
        self._retriever = retriever
        self._language  = language

        lang_cfg = get_language_config(language)
        self._system_prompt: str = lang_cfg["system_prompt"]
        self._neutral_stubs: list = lang_cfg["neutral_stubs"]
        self._language_display: str = lang_cfg["display_name"]

        self._http = httpx.AsyncClient(
            base_url=config.ollama.base_url,
            # 72b models need up to 3-4 min for the very first inference while
            # layers are paged into VRAM.  Subsequent requests are much faster.
            timeout=httpx.Timeout(300.0, connect=15.0),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def stream_response(
        self,
        user_query: str,
        memory: ConversationMemory,
        session_id: str = "unknown",
    ) -> AsyncIterator[str]:
        """
        Stream sentence fragments for the given user query via Ollama.

        Args:
            user_query:  Latest user utterance (Dari, Pashto, or English).
            memory:      Session conversation history.
            session_id:  For structured log context.

        Yields:
            Sentence fragments suitable for direct TTS synthesis.
        """
        prompt = self._build_prompt(user_query, memory)
        async for fragment in self._stream_ollama(prompt, session_id):
            yield fragment

    async def close(self) -> None:
        """Cleanly close the HTTP client."""
        await self._http.aclose()

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, user_query: str, memory: ConversationMemory) -> str:
        """
        Assemble the full prompt: system persona + RAG context +
        conversation history + current user turn.
        """
        parts: list[str] = [self._system_prompt]

        # RAG context (may be empty if no relevant docs found)
        if self._retriever is not None:
            rag_context = self._retriever.format_context(user_query)
            if rag_context:
                parts.append(rag_context)

        # Full conversation history — every turn since the call started
        history = memory.format_history()
        if history:
            parts.append(
                "FULL CONVERSATION HISTORY (complete record from the start of this call — "
                "read this carefully, maintain full context, do not repeat information "
                "already exchanged, and do not ask for anything the user has already provided):\n"
                + history
            )

        # Current turn — "Bot:" suffix primes the model to continue
        parts.append(f"User: {user_query}\nBot:")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Ollama streaming
    # ------------------------------------------------------------------

    async def _stream_ollama(
        self, prompt: str, session_id: str
    ) -> AsyncIterator[str]:
        """
        Stream tokens from local Ollama and yield complete sentence fragments.

        Uses the /api/generate endpoint with stream=True.
        On connection failure, falls back to a neutral acknowledgement in the
        session language.
        """
        payload = {
            "model": config.ollama.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": config.ollama.temperature,
                "top_p": config.ollama.top_p,
                "num_predict": config.ollama.max_tokens,
                "num_ctx": config.ollama.num_ctx,
            },
        }

        buffer = ""
        full_response = ""

        try:
            async with self._http.stream(
                "POST", "/api/generate", json=payload
            ) as response:
                response.raise_for_status()

                async for raw_line in response.aiter_lines():
                    if not raw_line:
                        continue
                    try:
                        data = _json.loads(raw_line)
                    except _json.JSONDecodeError:
                        continue

                    token = data.get("response", "")
                    done  = data.get("done", False)

                    buffer       += token
                    full_response += token

                    fragment, buffer = self._split_fragment(buffer)
                    if fragment:
                        logger.debug(
                            "LLM fragment [%s]: %s",
                            self._language_display,
                            fragment[:50],
                            extra={"session_id": session_id},
                        )
                        yield fragment

                    if done:
                        break

            # Flush any remaining text that didn't end with punctuation
            if buffer.strip():
                yield buffer.strip()

            logger.info(
                "Ollama response complete (%d chars): %s",
                len(full_response), full_response[:80],
                extra={"session_id": session_id},
            )

        except (httpx.ConnectError, httpx.HTTPStatusError) as exc:
            logger.error(
                "Ollama not reachable at %s: %s\n"
                "Make sure Ollama is running: ollama serve\n"
                "Model pulled: ollama pull %s",
                config.ollama.base_url, exc, config.ollama.model,
                extra={"session_id": session_id},
            )
            async for fragment in self._neutral_stub(session_id):
                yield fragment

        except Exception as exc:
            logger.error(
                "Ollama unexpected error: %s", exc,
                extra={"session_id": session_id},
            )
            async for fragment in self._neutral_stub(session_id):
                yield fragment

    # ------------------------------------------------------------------
    # Last-resort stub
    # ------------------------------------------------------------------

    async def _neutral_stub(self, session_id: str) -> AsyncIterator[str]:
        """Neutral acknowledgement in the session language when Ollama is unreachable."""
        import random
        logger.warning(
            "Using neutral stub — is Ollama running? Run: ollama serve",
            extra={"session_id": session_id},
        )
        await asyncio.sleep(0.1)
        yield random.choice(self._neutral_stubs)

    # ------------------------------------------------------------------
    # Fragment splitter
    # ------------------------------------------------------------------

    @staticmethod
    def _split_fragment(buffer: str) -> "tuple[str, str]":
        """
        Extract the next TTS-ready fragment from the buffer.

        Priority:
          1. Sentence boundary (.  !  ?  ,  | and Arabic equivalents) — cleanest split.
          2. Word-count trigger — if no punctuation arrives after
             _WORD_DISPATCH_THRESHOLD words, dispatch what we have so TTS
             can start speaking immediately instead of waiting for the full
             sentence.  This is critical for Telugu/Kannada where the LLM
             often generates a full sentence (~50-70 tokens) with no early
             comma, causing 4-5 s of silence before the first audio chunk.

        Returns (fragment, remaining_buffer).
        fragment is empty string if neither condition is met yet.
        """
        # ── Priority 1: sentence boundary ──────────────────────────────────
        match = None
        for m in _SENTENCE_BOUNDARY.finditer(buffer):
            match = m  # take the last boundary

        if match is not None:
            split_pos = match.end()
            return buffer[:split_pos].strip(), buffer[split_pos:]

        # ── Priority 2: word-count trigger ─────────────────────────────────
        words = buffer.split()
        threshold = max(
            3,
            int(getattr(config.ollama, "word_dispatch_threshold", _DEFAULT_WORD_DISPATCH_THRESHOLD)),
        )
        if len(words) >= threshold:
            # Split at the last space so we never break a word mid-character.
            last_space = buffer.rfind(" ")
            if last_space > 0:
                return buffer[:last_space].strip(), buffer[last_space + 1:]

        return "", buffer


# ---------------------------------------------------------------------------
# Backwards-compat alias
# ---------------------------------------------------------------------------
TeluguLLMClient = VoiceLLMClient


# ---------------------------------------------------------------------------
# Gemini LLM client — cloud, best natural Telugu/Kannada quality
# ---------------------------------------------------------------------------

class GeminiLLMClient:
    """
    Async streaming LLM client using Google Gemini cloud API (google-genai SDK).

    Same public interface as VoiceLLMClient — drop-in replacement.
    Requires GEMINI_API_KEY environment variable.
    Recommended model: gemini-2.0-flash (fast streaming, excellent Telugu).
    """

    def __init__(
        self,
        retriever: Optional[RAGRetriever] = None,
        language: str = "telugu",
    ) -> None:
        self._retriever = retriever
        self._language  = language

        lang_cfg = get_language_config(language)
        self._system_prompt: str = lang_cfg["system_prompt"]
        self._neutral_stubs: list = lang_cfg["neutral_stubs"]
        self._language_display: str = lang_cfg["display_name"]

        try:
            from google import genai
            from google.genai import types as genai_types
            self._client = genai.Client(api_key=config.gemini.api_key)
            self._genai_types = genai_types
            logger.info(
                "GeminiLLMClient ready: model=%s language=%s",
                config.gemini.model,
                self._language_display,
            )
        except ImportError:
            raise RuntimeError(
                "google-genai not installed. Run: pip install google-genai"
            )

    async def stream_response(
        self,
        user_query: str,
        memory: ConversationMemory,
        session_id: str = "unknown",
    ) -> AsyncIterator[str]:
        """Stream sentence fragments from Gemini for the given user query."""
        contents = self._build_contents(user_query, memory)
        gen_config = self._genai_types.GenerateContentConfig(
            system_instruction=self._system_prompt,
            temperature=config.gemini.temperature,
            max_output_tokens=config.gemini.max_tokens,
        )
        buffer = ""
        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=config.gemini.model,
                contents=contents,
                config=gen_config,
            )
            async for chunk in stream:
                token = chunk.text or ""
                if not token:
                    continue

                buffer += token
                fragment, buffer = VoiceLLMClient._split_fragment(buffer)
                if fragment:
                    logger.debug(
                        "Gemini fragment [%s]: %s",
                        self._language_display,
                        fragment[:50],
                        extra={"session_id": session_id},
                    )
                    yield fragment

            if buffer.strip():
                yield buffer.strip()

        except Exception as exc:
            logger.error(
                "Gemini error: %s", exc,
                extra={"session_id": session_id},
            )
            import random
            yield random.choice(self._neutral_stubs)

    async def close(self) -> None:
        pass

    def _build_contents(self, user_query: str, memory: ConversationMemory) -> list:
        """Convert conversation history + query into Gemini contents format."""
        contents = []
        for role, text in memory.get_turns():
            gemini_role = "user" if role == "User" else "model"
            contents.append({"role": gemini_role, "parts": [{"text": text}]})

        user_text = user_query
        if self._retriever is not None:
            rag_ctx = self._retriever.format_context(user_query)
            if rag_ctx:
                user_text = f"{rag_ctx}\n\nUser: {user_query}"

        contents.append({"role": "user", "parts": [{"text": user_text}]})
        return contents


# ---------------------------------------------------------------------------
# Factory — returns the right client for the requested backend
# ---------------------------------------------------------------------------

def create_llm_client(
    backend: str = "ollama",
    retriever: Optional[RAGRetriever] = None,
    language: str = "telugu",
) -> "VoiceLLMClient | GeminiLLMClient":
    """
    Return an LLM client for the requested backend.

    Args:
        backend:   "ollama" (local) or "gemini" (cloud).
        retriever: Optional RAG retriever (passed through to client).
        language:  "telugu" or "kannada".
    """
    backend = backend.lower().strip()
    if backend == "gemini":
        if not config.gemini.api_key:
            logger.warning(
                "GEMINI_API_KEY not set — falling back to Ollama. "
                "Get a free key at https://aistudio.google.com"
            )
            return VoiceLLMClient(retriever=retriever, language=language)
        return GeminiLLMClient(retriever=retriever, language=language)
    return VoiceLLMClient(retriever=retriever, language=language)
