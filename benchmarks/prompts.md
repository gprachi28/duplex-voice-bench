# Prompt Versions

Dated changelog of `SYSTEM_PROMPT` (`agent/worker.py`) — exact text per
version, which model(s) it was used/tested against, and why it changed.
`SYSTEM_PROMPT_VERSION` travels into every turn's metrics record
(`TurnMetrics.prompt_version` -> JSONL -> `eval_latency.py` groups by
`(combination_id, prompt_version)`), so this file is what turns that
version string back into readable text and intent. Git already tracks the
line-level diff in `worker.py`; this file makes the history readable
without digging through blame, and ties each version explicitly to the
model and the finding that motivated it.

Entry template:

```markdown
## <version> -- YYYY-MM-DD
- model(s): <LLM this was written/tested for>
- prompt text:
  > <verbatim SYSTEM_PROMPT text>
- reason: <what motivated this version -- link a benchmarks/experiments.md entry if applicable>
```

---

## v1-concise-en -- 2026-07-22

- **model(s)**: `llama3.2:3b` (Ollama, local)
- **prompt text**:
  > You are a helpful voice assistant. Your replies are spoken aloud, so
  > keep them short and conversational -- one to three sentences unless
  > the user explicitly asks for more detail, a list, or a recipe/steps.
  > Always reply in English, even if a question references another
  > language or asks about earlier turns in the conversation.
- **reason**: First version — replaces having no system prompt at all.
  See [experiments.md#2026-07-22](experiments.md) — fixes chatty replies
  and a live Hindi-language-drift/garbled-TTS bug found during the first
  benchmark session.
