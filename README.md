# Local LLM Real-World Task Marathon

A transparent, reproducible benchmark of local LLMs on real daily-work tasks — not
synthetic benchmarks. All models ran **one at a time, standalone**, on the same Apple
Silicon host (M5 Max / 128 GB), with the chat workload moved off the machine so no
run was competed for GPU.

**Why this exists:** coding benchmarks (HumanEval, MMLU) don't tell you which model
writes a correct invoice, summarizes a spec accurately, or drafts a customer email.
This suite measures exactly those.

**Housekeeping note (v2):** an earlier harness bug (models loaded at their 262K-token
default context, which silently disabled the "reasoning off" knob) produced invalid
empty answers and inflated timings for a subset of models. The affected rows were
re-run under the corrected protocol and are flagged with a **v2** marker in the
tables. Everything below reflects the corrected numbers.

---

## The 4 Tasks

Each task is graded against **known-correct facts** — a model passes only if the
answer contains them. Full prompts were identical across all models and modes.

| # | Task | What it tests | Graded on |
|---|------|---------------|-----------|
| 1 | **Grounded Q&A** — answer 3 questions from a fleet-scheduling document | Hallucination control; instruction following ("only from the document") | Falling back through the exact node order; the per-node model+port; the all-busy fallback rule |
| 2 | **Invoice generation** — draft a commercial invoice under house style | Format compliance + arithmetic | Math: 42×4,500 = 189,000; −10% → 170,100; +13% → **192,213**; "Bank Deposit" wording; number format |
| 3 | **Technical summary** — 5 bullets from a dense ML-pipeline description | Fact retention under compression | Resolution 512×768; 43 GB model; fp8-unsupported-on-MPS note; 150-word prompt limit |
| 4 | **Client email** — reply to a delivery + damage complaint | Tone, structure, completeness | Both issues acknowledged; courier confirmation before escalation; photo inspection; 5-working-day replacement promise |

**Harness parameters (identical for every run):** `temperature 0.2`, `max_tokens`
varies per model (see "Token budget findings" below), `stream=true`, one-shot
(no few-shot examples, no tools). Thinking-off knob: oMLX
`chat_template_kwargs.enable_thinking=false`; LM Studio `reasoning_effort="none"`
(top-level). Models loaded at **32K context** (`-c 32768`) — the guardrail-safe
config where reasoning-off is honored.

---

## Results (4 tasks, pass/fail per task, answer tok/s)

Legend: ✅ = passed (4/4) · 🟡 = partial (3/4) · ❌ = failed (<3/4)
"v2" = corrected re-run (initial harness bug). Models are grouped by engine.

### Tier 1 — 4/4 with the best speed/quality balance

| Model | Engine | Thinking | Score | Speed |
|---|---|---|---|---|
| SuperQwen AgentWorld 35B A3B (abliterated) | LM Studio | ON | ✅ 4/4 | 78.6 tok/s |
| SuperQwen AgentWorld 35B A3B (abliterated) | LM Studio | OFF | ✅ 4/4 | 78.6 tok/s |
| Laguna S-2.1 | LM Studio | OFF | ✅ 4/4 | 64.0 tok/s |
| Laguna S-2.1 | LM Studio | ON | ✅ 4/4 | 56.6 tok/s |
| GLM-4.7-Flash | LM Studio | ON | ✅ 4/4 | 50.5 tok/s |
| Magistral Small 2509 | LM Studio | ON/OFF | ✅ 4/4 | 33.7-35.2 tok/s |
| Ornith 1.5 35B A3B | LM Studio | ON | ✅ 4/4 | 24.4 tok/s |
| Bonsai 27B (v2) | LM Studio | OFF | ✅ 4/4 | — |

### Tier 2 — correct, mid speed (8-20 tok/s)

| Model | Engine | Thinking | Speed |
|---|---|---|---|
| GLM-4.6V-Flash (v2) | LM Studio | OFF | 8.7 tok/s — BUT on 8,000 |
| LFM2 24B (oMLX) | oMLX | OFF | 8.2 tok/s |
| Ornith 1.5-35B (oMLX) | oMLX | OFF | 7.6 tok/s |

### Tier 3 — correct but slow (oMLX lane, thinking-off)

Qwen3.8-27B AWQ (5.7) / OBLITERATED (4.9) / grug-27b (5.7) / Ternary-Bonsai (6.6) /
Seed-OSS (4.3) / Fable-Fusion (0.9) / Muse-Glimmer (0.6) / LFM2 (1.1) / Devstral-24B /
and the ON-lane of every thinking model (2.0-0.1 tok/s — see matrix). All 4/4 —
fine for batch, too slow for interactive chat.

### Notable failures & behavioral quirks

| Model | Finding |
|---|---|
| qwen2.5:0.5b (Ollama) | ❌ 2/4 — too small for these tasks |
| MiniCPM-V (Ollama, vision-tagged) | 🟡 3/4 at best |
| Gemma 4 26B A4B QAT (v2, thinking OFF) | 🟡 2/4 — *degrades sharply* when reasoning is disabled (genuine reasoning-dependent model) |
| OLMo 3 32B Think (v2, thinking OFF) | 🟡 2/4 — reasoning-dependent by design |
| DRK Coding v1 (v2, thinking OFF) | 🟡 3/4 — fails the summary task regardless of mode |
| GLM-4.7-Flash | ⚠️ passes but **over-generates** (31K tokens for a 50-token task) — its "speed" is inflated by verbosity |

### Token-budget findings (the "failed then passed" story)

Several models initially failed because the harness's 6,000-token cap was consumed
entirely by reasoning before any answer token appeared. Re-tested with larger budgets,
**every one of them passed** — proof the failures were budget, not capability:

| Model | Task | Failed at | **Passed at** |
|---|---|---|---|
| OLMo 3 32B Think (OFF) | summary | 6,000 | **16,000** |
| GLM-4.6V-Flash (ON) | summary | 6,000 | **8,000** |
| DRK Coding v1 (ON) | invoice | 6,000 | **8,000** |
| DRK Coding v1 (OFF) | invoice/summary | 6,000 | 6,000 |
| Qwen3.8-27B Heretic (ON+OFF) | invoice | 6,000 (harness) | 6,000 (v2) |
| Qwen3.8-27B Obliterated (OFF) | invoice | 6,000 (harness) | 6,000 (v2) |
| Qwen3.8-27B (OFF) | all | 6,000 (harness) | 6,000 (v2) |

**Lesson for practitioners:** a hard `max_tokens` cap silently turns reasoning models
into "empty answer" models. Always verify the **reasoning-off knob is actually honored**
(compare thinking-token counts), and give thinking models ≥ 16K output budget.

---

## Harness gotchas (for reproducibility)

- oMLX: unload between models via `POST /admin/api/models/{id}/unload` — never kill
  the server between runs (connection-refused storms on next load).
- oMLX streams keepalive chunks without `choices` during long generations — parsers
  must tolerate them.
- LM Studio `lms ls --json` lists models once per device/variant — filter
  `deviceIdentifier == null` for local-only sweeps.
- 100GB-class models (REAP37-class) need `iogpu.wired_limit_mb` raised (~120G) plus
  aggressive memory guard; without it inference crawls or OOMs.
- Apple Silicon: models load at their *default* context (e.g. 262K) unless you pass
  `-c 32768` — and the default context changes reasoning-off behavior.

## Full data

- `leaderboard.json` — machine-readable per-task scoring (all 88 runs, v1+v2 reconciled)
- `progress.jsonl` (+ `progress_v2.jsonl`) — raw per-task records incl. full answers
- Interactive comparison: `comparison.html` (filter by engine & thinking mode)

---
*Methodology questions welcome — open an issue. All runs were standalone; token
counts and timings are from the same streaming client used for every model.*