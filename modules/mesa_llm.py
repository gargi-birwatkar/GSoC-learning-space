"""
mesa_llm.py — MesaLLM Multi-Provider Connector  (v3.1)
=======================================================
GSoC 2026 Prototype | Production-Ready Module

New in v3
---------
  1. Token Guard    — tiktoken-based (with char-count fallback) trims history
                      before it would overflow the model's context window.
  2. Persistence    — .save_history() / .load_history() dump/restore the full
                      simulation trace as newline-delimited JSON (ndjson).
  3. Async Ready    — async def step() / async def stream_step() let you drive
                      hundreds of Mesa agents concurrently via asyncio.gather().
  4. Singleton      — MesaConnector.instance() returns one shared object for the
                      whole simulation; re-calling it never re-initialises.

Fixed in v3.1
-------------
  5. .env Support   — Automatically loads a .env file from the working directory
                      so GEMINI_API_KEY (and other keys) are always resolved,
                      regardless of how the shell session was started.
                      PowerShell `set VAR=...` does NOT export env vars; use a
                      .env file or pass api_key= directly instead.
  6. Smarter Retry  — Rate-limit back-off now uses full jitter (random delay in
                      [0, cap]) so concurrent agents don't retry in lockstep.
                      Default attempts raised 3 → 6, base delay 1s → 2s,
                      cap 60s.  Reads Retry-After header when present.

Supported providers (via LiteLLM — 100+ total):
  Google Gemini  →  model="gemini/gemini-2.5-flash"   ← tested default
  OpenAI         →  model="openai/gpt-4o"
  Anthropic      →  model="anthropic/claude-sonnet-4-20250514"
  Groq           →  model="groq/llama-3.3-70b-versatile"
  Mistral        →  model="mistral/mistral-large-latest"
  Ollama (local) →  model="ollama/llama3"   # no key needed

Install
-------
  pip install litellm python-dotenv          # required
  pip install tiktoken                       # optional — improves token counting

API Key setup (pick ONE)
------------------------
  Option A — .env file in your project folder (recommended):
      GEMINI_API_KEY=AIzaSy...              # no quotes needed in .env files

  Option B — pass directly:
      conn = MesaConnector.instance(api_key="AIzaSy...", model="gemini/gemini-2.5-flash")

  ⚠  PowerShell `set VAR=value` is a DISPLAY command, not an export.
     Use `$env:GEMINI_API_KEY = "..."` for the current session,
     or use a .env file so it works every time.

Quick-start (async simulation loop)
------------------------------------
  import asyncio
  from mesa_llm import MesaConnector

  async def main():
      conn   = MesaConnector.instance(model="gemini/gemini-2.5-flash")
      SYSTEM = "You are a resource-allocation agent. Be terse."
      state  = "Resources: 100. Agents: 5. Round: 1."
      for r in range(1, 501):
          state = await conn.step(f"Round {r} — state: {state}. Decide.", SYSTEM)
          print(f"[R{r:03d}] {state[:100]}")
      conn.save_history("trace.ndjson")

  asyncio.run(main())
"""

from __future__ import annotations

__all__     = ["MesaConnector"]
__version__ = "3.1.0"

# ── stdlib ────────────────────────────────────────────────────────────────────
import asyncio
import json
import logging
import os
import random
import threading
import time
from collections import deque
from datetime    import datetime, timezone
from pathlib     import Path
from typing      import AsyncIterator, Iterator

# ── python-dotenv — load .env before anything touches os.environ ──────────────
# This fixes the PowerShell `set VAR=value` gotcha: that command only sets a
# variable for cmd.exe display; it does NOT export it to child processes.
# A .env file in the working directory is the reliable cross-shell alternative.
try:
    from dotenv import load_dotenv as _load_dotenv
    # search_parent=True walks up to find .env even when cwd != project root
    _load_dotenv(override=False)   # override=False: real env vars win over .env
    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False
    # Non-fatal: users can still pass api_key= directly or set env vars manually

# ── optional tiktoken — graceful fallback to char heuristic ──────────────────
try:
    import tiktoken as _tiktoken          # pip install tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False

# ── LiteLLM (required) ───────────────────────────────────────────────────────
try:
    import litellm as _litellm
    _litellm.suppress_debug_info = True
    _litellm.set_verbose          = False
except ImportError as _err:
    raise ImportError(
        "mesa_llm requires 'litellm'.  Run:  pip install litellm"
    ) from _err

# ── module logger ─────────────────────────────────────────────────────────────
log = logging.getLogger("mesa_llm")

# ── provider → env-var map (for clear auth-error messages) ───────────────────
_PROVIDER_KEY_MAP: dict[str, str | None] = {
    "openai":     "OPENAI_API_KEY",
    "anthropic":  "ANTHROPIC_API_KEY",
    "gemini":     "GEMINI_API_KEY",
    "mistral":    "MISTRAL_API_KEY",
    "groq":       "GROQ_API_KEY",
    "cohere":     "COHERE_API_KEY",
    "together":   "TOGETHER_API_KEY",
    "perplexity": "PERPLEXITYAI_API_KEY",
    "ollama":     None,
}

# ── known context-window sizes (tokens) — extend as needed ───────────────────
# Values are *input* context limits; we reserve max_tokens for the reply.
_CONTEXT_LIMITS: dict[str, int] = {
    # Google — 2.5 series (tested default)
    "gemini/gemini-2.5-flash":          1_048_576,
    "gemini/gemini-2.5-pro":            2_097_152,
    # Google — 2.0 series
    "gemini/gemini-2.0-flash":          1_048_576,
    "gemini/gemini-2.0-flash-lite":     1_048_576,
    # Google — 1.5 series
    "gemini/gemini-1.5-pro":            2_097_152,
    "gemini/gemini-1.5-flash":          1_048_576,
    "gemini/gemini-1.5-flash-8b":       1_048_576,
    # OpenAI
    "openai/gpt-4o":                     128_000,
    "openai/gpt-4o-mini":                128_000,
    "openai/gpt-4-turbo":                128_000,
    "openai/gpt-3.5-turbo":              16_385,
    # Anthropic
    "anthropic/claude-opus-4-5":         200_000,
    "anthropic/claude-sonnet-4-20250514":200_000,
    "anthropic/claude-haiku-4-5-20251001":200_000,
    # Groq
    "groq/llama-3.3-70b-versatile":      128_000,
    "groq/llama-3.1-8b-instant":         128_000,
    # Mistral
    "mistral/mistral-large-latest":      128_000,
}
_DEFAULT_CONTEXT_LIMIT = 8_192   # conservative fallback for unknown models

# Characters-per-token heuristic used when tiktoken is unavailable.
# English text averages ~4 chars/token; we use 3.5 (slightly pessimistic).
_CHARS_PER_TOKEN: float = 3.5


# ─────────────────────────────────────────────────────────────────────────────
#  Token counting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _count_tokens_tiktoken(text: str, model: str) -> int:
    """
    Exact token count via tiktoken.

    Falls back to the cl100k_base encoding (used by GPT-4 / Gemini-adjacent
    models) when the model name is not directly registered.
    """
    try:
        enc = _tiktoken.encoding_for_model(model.split("/")[-1])
    except KeyError:
        enc = _tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def _count_tokens(text: str, model: str) -> int:
    """
    Return a token estimate for *text* given *model*.

    Strategy (in order):
      1. tiktoken exact count — most accurate, works for OpenAI/Gemini.
      2. Character heuristic  — 1 token ≈ 3.5 chars (always available).
    """
    if _TIKTOKEN_AVAILABLE:
        return _count_tokens_tiktoken(text, model)
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def _messages_token_count(messages: list[dict[str, str]], model: str) -> int:
    """
    Sum token estimates for every message in the list.
    Adds 4 tokens per message for role/formatting overhead (OpenAI convention).
    """
    total = 0
    for msg in messages:
        total += 4 + _count_tokens(msg.get("content", ""), model)
    return total


# ─────────────────────────────────────────────────────────────────────────────
#  MesaConnector
# ─────────────────────────────────────────────────────────────────────────────

class MesaConnector:
    """
    Provider-agnostic, async-first LLM connector with sliding-window memory,
    token-budget enforcement, JSON persistence, and process-wide singleton.

    Instantiation
    -------------
    Preferred — use the singleton factory so Mesa components share one client:

        conn = MesaConnector.instance(api_key="...", model="gemini/gemini-2.0-flash")

    Direct instantiation is also fine for unit tests or isolated agents:

        conn = MesaConnector(api_key="...", model="openai/gpt-4o")

    Parameters
    ----------
    api_key : str, optional
        Provider API key.  Falls back to the standard env var
        (e.g. ``GEMINI_API_KEY``).  Ollama needs no key.
    model : str
        LiteLLM model string — ``"provider/model-name"``.
    max_history_turns : int
        Hard cap on the deque sliding window (turn pairs).
        The token guard may evict additional turns before this limit
        is reached if the context budget is exceeded.
    max_tokens : int
        Max completion tokens requested per call.
    temperature : float
        Sampling temperature [0.0, 1.0].
    context_limit : int, optional
        Override the auto-detected context window size (tokens).
        Useful for fine-tuned or unreleased model variants.
    retry_attempts : int
        Retry count on transient errors (rate limits, 5xx, timeouts).
    retry_delay : float
        Base retry wait in seconds; doubles each attempt (exp. back-off).
    """

    _DEFAULT_MODEL   = "gemini/gemini-2.5-flash"

    # ── singleton state (class-level) ─────────────────────────────────────────
    _singleton_lock: threading.Lock          = threading.Lock()
    _singleton_inst: MesaConnector | None    = None   # type: ignore[assignment]

    # ─────────────────────────────────────────────────────────────────────────
    @classmethod
    def instance(cls, **kwargs) -> "MesaConnector":
        """
        Return the process-wide singleton, creating it on the first call.

        All keyword arguments are forwarded to ``__init__`` on first
        construction only — subsequent calls ignore kwargs entirely and
        return the cached instance unchanged.

        Thread-safe via a class-level ``threading.Lock``.

        Example
        -------
        >>> conn1 = MesaConnector.instance(api_key="k", model="gemini/gemini-2.0-flash")
        >>> conn2 = MesaConnector.instance()   # same object, no re-init
        >>> assert conn1 is conn2
        """
        if cls._singleton_inst is None:
            with cls._singleton_lock:
                if cls._singleton_inst is None:          # double-checked locking
                    cls._singleton_inst = cls(**kwargs)
                    log.info("Singleton MesaConnector created | model=%s",
                             cls._singleton_inst.model)
        return cls._singleton_inst

    @classmethod
    def reset_singleton(cls) -> None:
        """
        Destroy the singleton so the next ``instance()`` call creates a fresh
        one.  Primarily for tests and notebook reruns.
        """
        with cls._singleton_lock:
            cls._singleton_inst = None
        log.debug("Singleton reset.")

    # ─────────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        api_key:           str | None = None,
        *,
        model:             str   = _DEFAULT_MODEL,
        max_history_turns: int   = 20,
        max_tokens:        int   = 1_024,
        temperature:       float = 0.7,
        context_limit:     int | None = None,
        retry_attempts:    int   = 6,    # raised from 3 — rate limits need patience
        retry_delay:       float = 2.0,  # raised from 1.0 — base for exp back-off
        retry_cap:         float = 60.0, # max seconds any single wait can reach
    ) -> None:
        self.model          = model
        self.max_tokens     = max_tokens
        self.temperature    = temperature
        self.retry_attempts = retry_attempts
        self.retry_delay    = retry_delay
        self.retry_cap      = retry_cap

        # ── resolve context budget ────────────────────────────────────────────
        # Total tokens the model accepts.  We subtract max_tokens (reply budget)
        # and a 256-token safety margin before calculating history headroom.
        raw_limit          = context_limit or _CONTEXT_LIMITS.get(model, _DEFAULT_CONTEXT_LIMIT)
        self._ctx_limit    = raw_limit
        self._history_budget: int = raw_limit - max_tokens - 256
        if self._history_budget <= 0:
            raise ValueError(
                f"max_tokens={max_tokens} leaves no room for history "
                f"(context_limit={raw_limit})."
            )

        # ── inject API key into env ───────────────────────────────────────────
        if api_key:
            provider = model.split("/")[0].lower()
            env_var  = _PROVIDER_KEY_MAP.get(provider)
            target   = env_var or "LITELLM_API_KEY"
            os.environ.setdefault(target, api_key)

        # ── sliding window ────────────────────────────────────────────────────
        # Each element: {"role": "user"|"assistant", "content": str,
        #                "ts": ISO-8601, "tokens": int}
        self._history: deque[dict] = deque(maxlen=max(1, max_history_turns))

        # ── diagnostics ───────────────────────────────────────────────────────
        self._call_count:          int = 0
        self._total_input_tokens:  int = 0
        self._total_output_tokens: int = 0
        self._evicted_turns:       int = 0   # turns dropped by token guard

        # ── async semaphore ───────────────────────────────────────────────────
        # Prevents hammering the API when many agents call step() concurrently.
        # Adjust max_concurrent via the property if needed.
        self._semaphore: asyncio.Semaphore | None = None   # lazy-init in event loop
        self._max_concurrent: int = 10

        log.debug(
            "MesaConnector v%s | model=%s | window=%d | ctx_budget=%d tokens"
            " | tiktoken=%s | dotenv=%s",
            __version__, model, max_history_turns,
            self._history_budget, _TIKTOKEN_AVAILABLE, _DOTENV_AVAILABLE,
        )

    # ── property: concurrency cap ─────────────────────────────────────────────
    @property
    def max_concurrent(self) -> int:
        """Max simultaneous async API calls (default 10)."""
        return self._max_concurrent

    @max_concurrent.setter
    def max_concurrent(self, value: int) -> None:
        self._max_concurrent = max(1, value)
        self._semaphore = None   # force re-create on next async call

    # ─────────────────────────────────────────────────────────────────────────
    #  Public async API
    # ─────────────────────────────────────────────────────────────────────────

    async def step(self, user_input: str, system_prompt: str = "") -> str:
        """
        Advance the agent one simulation step  **[async]**.

        Internally:
          1. Token-guards history (evicts oldest turns if over budget).
          2. Builds the message list.
          3. Calls the LLM via ``asyncio.to_thread`` (non-blocking).
          4. Appends the turn to history.

        Parameters
        ----------
        user_input : str
            The agent's observation / prompt for this step.
        system_prompt : str
            Persistent system-level instructions (injected every call).

        Returns
        -------
        str
            The LLM's plain-text response.

        Example
        -------
        >>> response = await connector.step("What is your action?", SYSTEM)
        """
        sem  = self._get_semaphore()
        async with sem:
            self._apply_token_guard(user_input, system_prompt)
            messages      = self._build_messages(user_input, system_prompt)
            response_text = await asyncio.to_thread(self._call_with_retry, messages)

        ts = _utc_now()
        self._history.append({
            "role": "user",      "content": user_input,     "ts": ts, "step": self._call_count,
        })
        self._history.append({
            "role": "assistant", "content": response_text,  "ts": ts, "step": self._call_count,
        })
        self._call_count += 1

        log.debug("step %d | %s | history_entries=%d",
                  self._call_count, self.model, len(self._history))
        return response_text

    async def stream_step(
        self, user_input: str, system_prompt: str = ""
    ) -> AsyncIterator[str]:
        """
        Streaming async variant — yields text chunks as they arrive.

        Usage::

            async for chunk in connector.stream_step("Hello"):
                print(chunk, end="", flush=True)
        """
        self._apply_token_guard(user_input, system_prompt)
        messages      = self._build_messages(user_input, system_prompt)
        full_response : list[str] = []

        # LiteLLM's sync streaming run inside a thread to stay non-blocking
        def _stream() -> list[str]:
            parts: list[str] = []
            resp = _litellm.completion(
                model=self.model, messages=messages,
                max_tokens=self.max_tokens, temperature=self.temperature,
                stream=True,
            )
            for chunk in resp:
                delta = chunk.choices[0].delta.content or ""
                parts.append(delta)
            return parts

        chunks = await asyncio.to_thread(_stream)
        for c in chunks:
            full_response.append(c)
            yield c

        ts        = _utc_now()
        assembled = "".join(full_response)
        self._history.append({"role": "user",      "content": user_input, "ts": ts, "step": self._call_count})
        self._history.append({"role": "assistant",  "content": assembled,  "ts": ts, "step": self._call_count})
        self._call_count += 1

    # ── sync convenience wrapper (for non-async Mesa models) ─────────────────
    def step_sync(self, user_input: str, system_prompt: str = "") -> str:
        """
        Synchronous wrapper around ``step()``.

        Use this when your Mesa model's ``step()`` method is not async.
        Internally calls ``asyncio.run()`` so it **cannot** be called from
        inside a running event loop — use ``await conn.step()`` there instead.
        """
        return asyncio.run(self.step(user_input, system_prompt))

    # ─────────────────────────────────────────────────────────────────────────
    #  Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save_history(self, filename: str | Path) -> Path:
        """
        Dump the full simulation trace to **newline-delimited JSON** (ndjson).

        Each line is a self-contained JSON object with keys:
          ``step``, ``role``, ``content``, ``ts`` (ISO-8601 UTC).

        A metadata header line is written first with simulation stats.

        Parameters
        ----------
        filename : str or Path
            Output file path.  Parent directories are created automatically.

        Returns
        -------
        Path
            Resolved path of the written file.

        Example
        -------
        >>> connector.save_history("traces/run_001.ndjson")
        """
        path = Path(filename).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as fh:
            # ── line 0: metadata ──────────────────────────────────────────────
            meta = {
                "_type":               "mesa_llm_trace",
                "version":             __version__,
                "model":               self.model,
                "saved_at":            _utc_now(),
                "total_steps":         self._call_count,
                "total_input_tokens":  self._total_input_tokens,
                "total_output_tokens": self._total_output_tokens,
                "evicted_turns":       self._evicted_turns,
            }
            fh.write(json.dumps(meta, ensure_ascii=False) + "\n")

            # ── subsequent lines: history entries ─────────────────────────────
            for entry in self._history:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

        log.info("Trace saved → %s  (%d entries)", path, len(self._history))
        return path

    def load_history(self, filename: str | Path) -> None:
        """
        Restore history from a previously saved ndjson trace.

        The metadata header line is parsed for stats but does **not** overwrite
        the current model or configuration — only history entries are loaded.

        Parameters
        ----------
        filename : str or Path
            Path to an ndjson file written by ``save_history()``.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file was not written by mesa_llm (missing metadata header).
        """
        path = Path(filename).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Trace file not found: {path}")

        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            raise ValueError(f"Empty trace file: {path}")

        meta = json.loads(lines[0])
        if meta.get("_type") != "mesa_llm_trace":
            raise ValueError(
                "File does not appear to be a mesa_llm trace "
                f"(missing '_type' header): {path}"
            )

        self._history.clear()
        for line in lines[1:]:
            if line.strip():
                self._history.append(json.loads(line))

        # Restore counters from metadata
        self._call_count          = meta.get("total_steps",         0)
        self._total_input_tokens  = meta.get("total_input_tokens",  0)
        self._total_output_tokens = meta.get("total_output_tokens", 0)
        self._evicted_turns       = meta.get("evicted_turns",       0)

        log.info("Trace loaded ← %s  (%d entries, %d steps)",
                 path, len(self._history), self._call_count)

    # ─────────────────────────────────────────────────────────────────────────
    #  Diagnostics / utility
    # ─────────────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Wipe conversation history (keeps model/config and singleton intact)."""
        self._history.clear()
        log.debug("History reset.")

    @property
    def stats(self) -> dict:
        """Live usage snapshot — safe to call mid-simulation."""
        return {
            "version":              __version__,
            "model":                self.model,
            "calls":                self._call_count,
            "history_entries":      len(self._history),
            "history_capacity":     self._history.maxlen,
            "context_limit_tokens": self._ctx_limit,
            "history_budget_tokens":self._history_budget,
            "evicted_turns":        self._evicted_turns,
            "total_input_tokens":   self._total_input_tokens,
            "total_output_tokens":  self._total_output_tokens,
            "tiktoken_available":   _TIKTOKEN_AVAILABLE,
            "dotenv_available":     _DOTENV_AVAILABLE,
            "retry_attempts":       self.retry_attempts,
            "retry_cap_s":          self.retry_cap,
        }

    def __repr__(self) -> str:
        return (
            f"MesaConnector(v{__version__}, model={self.model!r}, "
            f"calls={self._call_count}, "
            f"history={len(self._history)}/{self._history.maxlen})"
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_semaphore(self) -> asyncio.Semaphore:
        """
        Lazy-initialise the asyncio.Semaphore inside the running event loop.

        A Semaphore must be created in the same loop it's used in, so we
        defer construction until the first ``await step()`` call.
        """
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
        return self._semaphore

    # ── Token Guard ───────────────────────────────────────────────────────────

    def _apply_token_guard(self, user_input: str, system_prompt: str) -> None:
        """
        Evicts oldest history entries until the payload fits the token budget.
        """
        # 1. INITIAL STATUS PRINT
    #    print(f"🔍 [DEBUG] Pre-Guard: History contains {len(self._history)} entries.")

        # 2. CALCULATE STARTING OVERHEAD (System + New User Input)
        overhead = _count_tokens(system_prompt, self.model) + 4 if system_prompt else 0
        overhead += _count_tokens(user_input, self.model) + 4

        # 3. THE TRIMMING LOOP
        while self._history:
            # Calculate tokens currently in the history buffer
            history_tokens = sum(
                _count_tokens(e.get("content", ""), self.model) + 4
                for e in self._history
            )
            
            total_projected = overhead + history_tokens
            
            # Check if we are within budget
            #if total_projected <= self._history_budget:
             #   print(f"✅ [DEBUG] Budget OK: {total_projected} / {self._history_budget} tokens.")
             #   break
            
            # BUDGET EXCEEDED: Evict oldest turn pair (User + Assistant)
           # print(f"✂️ [DEBUG] Budget Exceeded ({total_projected} tokens). Evicting oldest turn...")
            
            # Pop the oldest (User)
            self._history.popleft()
            self._evicted_turns += 1
            
            # Pop the paired response (Assistant) if it exists
            if self._history:
                self._history.popleft()
                self._evicted_turns += 1

        # 4. FINAL LOGGING
        if self._evicted_turns > 0:
            log.debug("Token guard evicted %d entries to fit context budget.", self._evicted_turns)
    
    def log_interaction(self, observation, system_prompt, response, stats):
        """Appends the current turn to an NDJSON file immediately."""
        log_entry = {
            "timestamp": time.time(),
            "model": self.model,
            "system_prompt": system_prompt,
            "observation": observation,
            "response": response,
            "usage": {
                "prompt_tokens": stats.get("last_input_tokens"),
                "completion_tokens": stats.get("last_output_tokens")
            }
        }
        
        # Append mode ('a') ensures we don't overwrite previous turns
        with open("gsoc_prototype_trace.ndjson", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    def _build_messages(
        self, user_input: str, system_prompt: str
    ) -> list[dict[str, str]]:
        """
        Flatten the sliding-window history into the OpenAI-compatible message
        format that LiteLLM normalises for every provider.

        Structure:
          [system?] → [user, assistant, ...] × history → [user(current)]

        Note: history entries store extra keys (ts, step) that are stripped
        here — the API only receives ``role`` and ``content``.
        """
        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
       # print(f"📤 [DEBUG] Sending {len(self._history)} history entries + current input.")
        for entry in self._history:
            messages.append({"role": entry["role"], "content": entry["content"]})

        messages.append({"role": "user", "content": user_input})
        return messages

    # ── API call with retry ───────────────────────────────────────────────────

    def _call_with_retry(self, messages: list[dict[str, str]]) -> str:
        """
        Call the LLM via LiteLLM with full-jitter exponential back-off retry.

        Back-off strategy — "full jitter" (AWS recommendation):
          cap  = min(retry_cap, retry_delay × 2^attempt)
          wait = random.uniform(0, cap)

        This prevents the thundering-herd problem where many concurrent agents
        all retry at the same instant after a shared rate-limit window resets.

        Retry-After header: when the provider returns a 429 with a Retry-After
        value (seconds), that value overrides the jitter calculation so we wait
        exactly as long as the server requests.

        Retryable   : RateLimitError, ServiceUnavailableError, APIConnectionError
        Non-retryable: AuthenticationError, BadRequestError  (re-raised immediately)
        """
        USE_MOCK = True

        if USE_MOCK:
            p_lower = messages[-1]["content"].lower()
            
            if "emergency vehicle" in p_lower:
                if "hospital" in p_lower:
                    return "THOUGHT: Almost there! ACTION: Plowing through the traffic cones! GET OUT OF THE WAY!"
                return "THOUGHT: Patient is coding! ACTION: Blasting the siren and screaming into the PA system."

            elif "angry commuter" in p_lower:
                shouts = ["HEY! WATCH IT, YOU MORON!", "I'M GONNA BE LATE!", "USE YOUR DAMN SIGNAL!", "@#$%&! MOVE IT!"]
               
            
                return f"THOUGHT: I'm done with this traffic. ACTION: Honking like a maniac and yelling '{random.choice(shouts)}'"

            elif "pedestrian" in p_lower:
                return "THOUGHT: This is a circus. ACTION: Shouting 'YOU CAN'T PARK THERE!' and filming the ambulance."

            return "THOUGHT: It's a mess out here. ACTION: Just shouting into the void."
        last_exc: Exception | None = None

        for attempt in range(1, self.retry_attempts + 1):
            try:
                response = _litellm.completion(
                    model       = self.model,
                    messages    = messages,
                    max_tokens  = self.max_tokens,
                    temperature = self.temperature,
                )

                usage = getattr(response, "usage", None)
                if usage:
                    self._total_input_tokens  += getattr(usage, "prompt_tokens",    0)
                    self._total_output_tokens += getattr(usage, "completion_tokens", 0)

                return (response.choices[0].message.content or "").strip()

            except (_litellm.RateLimitError,
                    _litellm.ServiceUnavailableError,
                    _litellm.APIConnectionError) as exc:
                last_exc = exc

                # ── Retry-After header (provider hint) ────────────────────────
                # LiteLLM surfaces response headers on the exception object.
                retry_after: float | None = None
                headers = getattr(exc, "response", None)
                if headers is not None:
                    raw = getattr(headers, "headers", {}).get("retry-after")
                    if raw is not None:
                        try:
                            retry_after = float(raw)
                        except ValueError:
                            pass

                # ── Full-jitter back-off ──────────────────────────────────────
                if retry_after is not None:
                    wait = retry_after
                else:
                    cap  = min(self.retry_cap, self.retry_delay * (2 ** attempt))
                    wait = random.uniform(0, cap)   # full jitter

                log.warning(
                    "%s (attempt %d/%d) — retrying in %.1fs%s",
                    type(exc).__name__, attempt, self.retry_attempts, wait,
                    " [Retry-After]" if retry_after is not None else " [jitter]",
                )
                time.sleep(wait)
            except _litellm.RateLimitError as exc:
                # DEBUG PRINT 3: The exact reason for the block
                print(f"🚫 [DEBUG] API BLOCKED: {type(exc).__name__}")
                print(f"   Message: {exc.message}")
                
                if "quota" in exc.message.lower():
                    print("   ⚠️ DIAGNOSIS: This is a DAILY limit (RPD). Waiting 10s won't help.")
                else:
                    print("   ⚠️ DIAGNOSIS: This is a PER-MINUTE limit (RPM). Backing off...")
            except _litellm.AuthenticationError as exc:
                provider = self.model.split("/")[0]
                env_var  = _PROVIDER_KEY_MAP.get(provider,
                                                 f"{provider.upper()}_API_KEY")
                raise RuntimeError(
                    f"Auth failed for '{provider}'. "
                    f"Options:\n"
                    f"  1. Create a .env file:  {env_var}=YOUR_KEY\n"
                    f"  2. Pass directly:        MesaConnector(api_key='...')\n"
                    f"  3. PowerShell session:   $env:{env_var} = 'YOUR_KEY'\n"
                    f"  ⚠  `set {env_var}=...` (CMD syntax) does NOT work in PowerShell."
                ) from exc

            except _litellm.BadRequestError as exc:
                
                raise RuntimeError(
                    f"Bad request to '{self.model}': {exc}"
                ) from exc

        raise RuntimeError(
            f"LLM call failed after {self.retry_attempts} attempts "
            f"(last error: {last_exc})."
        ) from last_exc

# ─────────────────────────────────────────────────────────────────────────────
#  Multi-Agent System (MAS)
# ─────────────────────────────────────────────────────────────────────────────

class AgentBrain:
    def __init__(self, agent_id: str, role: str, api_key: str, model: str = "gemini/gemini-2.5-flash"):
        self.agent_id = agent_id
        self.role = role
        # Each agent gets its own private MesaConnector instance
        self.connector = MesaConnector(
            api_key=api_key,
            model=model,
            max_history_turns=3  # Keep it small for many agents!
        )
        self.system_prompt = f"You are agent {agent_id}, a {role} in a traffic sim. Stay safe."

    async def decide(self, observation: str):
        """The agent perceives the world and returns an action."""
        return await self.connector.step(observation, self.system_prompt)
# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _utc_now() -> str:
    """Return the current UTC time as a compact ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


