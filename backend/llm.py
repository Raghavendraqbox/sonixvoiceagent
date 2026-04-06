"""
llm.py — Open-source LLM client for the Telugu Voice AI Agent.

Runs entirely locally via Ollama — no external API keys required.

Recommended models for Telugu (pull before starting):
  ollama pull qwen2.5:32b        ← best Telugu quality, needs ~20GB VRAM
  ollama pull qwen2.5:14b        ← good Telugu, needs ~10GB VRAM
  ollama pull qwen2.5:7b         ← lighter option, needs ~6GB VRAM
  ollama pull aya:35b            ← Cohere's multilingual model
  ollama pull gemma3:27b         ← Google Gemma 3, good Telugu

Set OLLAMA_MODEL in .env to choose, default is qwen2.5:32b.

Sentence fragments are yielded as they arrive so TTS can begin speaking
after the first complete sentence rather than waiting for the full response.
"""

import asyncio
import json as _json
import logging
import re
from typing import AsyncIterator, Optional

import httpx

from config import config
from memory import ConversationMemory
from rag import RAGRetriever

logger = logging.getLogger(__name__)

# Sentence boundaries: Telugu danda (।) + Western punctuation
_SENTENCE_BOUNDARY = re.compile(r"([.!?,।|])\s*(?=\S|$)")


class TeluguLLMClient:
    """
    Async streaming LLM client using local Ollama.

    Each call to `stream_response` yields sentence fragments as soon as they
    are complete, so the TTS pipeline can start speaking immediately.

    No external API keys required — all inference runs on the local GPU.
    """

    def __init__(self, retriever: Optional[RAGRetriever] = None) -> None:
        self._retriever = retriever
        self._http = httpx.AsyncClient(
            base_url=config.ollama.base_url,
            timeout=httpx.Timeout(60.0, connect=10.0),
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
            user_query:  Latest user utterance (Telugu or English).
            memory:      Session conversation history.
            session_id:  For structured log context.

        Yields:
            Sentence fragments suitable for direct TTS synthesis.
        """
        prompt = self._build_prompt(user_query, memory)
        logger.debug(
            "LLM prompt built (%d chars): %s",
            len(prompt), user_query[:60],
            extra={"session_id": session_id},
        )
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
        parts: list[str] = [config.system_prompt]

        # RAG context (may be empty if no relevant docs found)
        if self._retriever is not None:
            rag_context = self._retriever.format_context(user_query)
            if rag_context:
                parts.append(rag_context)

        # Conversation history
        history = memory.format_history()
        if history:
            parts.append(f"Conversation so far:\n{history}")

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
        On connection failure, falls back to a neutral Telugu acknowledgement.
        """
        payload = {
            "model": config.ollama.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": config.ollama.temperature,
                "top_p": config.ollama.top_p,
                "num_predict": config.ollama.max_tokens,
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
                    done = data.get("done", False)

                    buffer += token
                    full_response += token

                    fragment, buffer = self._split_fragment(buffer)
                    if fragment:
                        logger.debug(
                            "LLM fragment: %s", fragment[:50],
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
        """
        Neutral Telugu acknowledgement when Ollama is unreachable.
        Includes a hint to the operator in logs.
        """
        import random
        stubs = [
            "క్షమించండి, నేను అర్థం చేసుకున్నాను. కొంచెం వేచి ఉండండి.",
            "అర్థమైంది. నేను మళ్ళీ ప్రయత్నిస్తాను.",
            "Sorry, please give me a moment while I check on that.",
        ]
        logger.warning(
            "Using neutral stub — is Ollama running? Run: ollama serve",
            extra={"session_id": session_id},
        )
        await asyncio.sleep(0.1)
        yield random.choice(stubs)

    # ------------------------------------------------------------------
    # Fragment splitter
    # ------------------------------------------------------------------

    @staticmethod
    def _split_fragment(buffer: str) -> tuple[str, str]:
        """
        Extract the longest complete sentence fragment from the buffer.

        Telugu uses । (danda, U+0964) as sentence terminator; Western
        punctuation (.  !  ?  ,) is also detected.

        Returns (fragment, remaining_buffer).
        fragment is empty string if no sentence boundary has been reached yet.
        """
        match = None
        for m in _SENTENCE_BOUNDARY.finditer(buffer):
            match = m  # take the last boundary

        if match is None:
            return "", buffer

        split_pos = match.end()
        fragment = buffer[:split_pos].strip()
        remaining = buffer[split_pos:]
        return fragment, remaining
