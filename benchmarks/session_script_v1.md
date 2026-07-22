# Session Script v1

A fixed talk-track for benchmark sessions. Read each line into the mic in
order; the fixed wording is what makes runs comparable across combinations
later (design.md: "same prompt, different backend, audibly different
result"). Versioned (`v1`) so a future script revision doesn't silently
break comparability with earlier runs.

## Before you start

1. Terminal 1: `uvicorn server.main:app --host 127.0.0.1 --port 8000`
2. Terminal 2: `python -m agent.worker dev`
3. Open http://127.0.0.1:8000, click **Connect**, allow mic.
4. Note the room name (for the logbook entry).

Unless a section says otherwise: **wait for the agent to fully finish
speaking before you speak again.** Talk at a normal pace, normal volume.

---

## A — Warm-up (not counted in the baseline)

Just confirms the pipeline is alive. Don't log this turn.

1. "Hello, can you hear me?"

---

## B — Baseline turns (the real sample — ~16 turns)

Natural pace, full replies, no interrupting. Mix of short factual asks,
one multi-turn reference, and one longer request.

1. "What's a good name for a coffee shop?"
2. "What's the capital of Australia?"
3. "Can you explain what a jitter buffer does, in one sentence?"
4. "What was the first thing I asked you today?" *(tests conversation history)*
5. "Give me a three-step recipe for scrambled eggs."
6. "What's 17 times 6?"
7. "Recommend a book if I liked science fiction."
8. "Why is the sky blue?"
9. "What did I just ask you about, two questions ago?" *(tests history again)*
10. "Tell me a short fact about octopuses."
11. "What's a good name for a pet turtle?"
12. "Can you summarize what a voice activity detector does?"
13. "What's the weather like in general in November in Delhi?"
14. "Give me two options for a weekend day trip near a big city."
15. "What's the difference between weather and climate?"
16. "Thanks, that's helpful."

---

## C — Turn-detection stress (Smart Turn v3)

Tests the gate, not raw latency — expect these to look different from
section B in the logbook, not worse.

1. Speak with a genuine ~2s pause mid-sentence, then finish:
   **"I think I want to..."** *(pause ~2 seconds)* **"...order a pizza."**
   It should wait through the pause and transcribe one sentence, not two.
2. A short, clearly complete utterance:
   **"No, that's all."**
   It should fire immediately, not wait for more.

---

## D — Barge-in test (clearly separate — don't mix into baseline stats)

1. Ask something that gets a longer reply: "Explain how WebRTC handles
   packet loss."
2. While the agent is still mid-reply, immediately talk over it:
   **"Wait, stop."** Confirm playback pauses/interrupts within about a
   frame.
3. Ask a normal follow-up afterward to confirm the session recovered:
   "Okay, what's your favorite color?"

---

## After the session

```bash
cp /tmp/voice-agent-metrics.jsonl \
  benchmarks/results/runs/$(date +%Y%m%d_%H%M%S)_local-lv3-ollama3b-kokoro.jsonl

python -m benchmarks.eval_latency benchmarks/results/runs
python -m benchmarks.plot
```

Then add a `logbook.md` entry. Note the room name, turn count, and flag
that sections C and D are stress/interrupt turns, not baseline latency
samples — `eval_latency.py` currently reports all turns together, so call
that out by hand in the "reading" line until the analyzer can filter by
`interrupted`/turn-type itself.
