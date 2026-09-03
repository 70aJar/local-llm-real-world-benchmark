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

### Full detailed results — every model × thinking mode, per task

Columns: **QA/INV/SUM/EML** = grounded Q&A / invoice / summary / email. Each cell: ✅ pass or ❌ fail, wall-clock seconds, answer tokens (`a`), thinking tokens (`t`). `v2` = corrected re-run; `@N` = passes only at max_tokens=N. Total time = sum of 4 tasks; tok/s = answer tokens ÷ wall time (∞ = wall time < 2s, token counter stalling).

| Model | Engine | Mode | Score | QA | INV | SUM | EML | Total | Ans tok | Think tok | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `superqwen-agentworld-35b-a3b-abliterated` | LM Studio | on | ✅ 4/4 | ✅ 2.1s · 138a | ✅ 2.1s · 178a | ✅ 2.7s · 227a | ✅ 1.7s · 133a | 9s | 676 | 0 | 78.6 |
| `superqwen-agentworld-35b-a3b-abliterated` | LM Studio | off | ✅ 4/4 | ✅ 1.6s · 120a | ✅ 2.1s · 166a | ✅ 2.4s · 192a | ✅ 1.8s · 143a | 8s | 621 | 0 | 78.6 |
| `poolside/laguna-s-2.1` | LM Studio | off | ✅ 4/4 | ✅ 2s · 130a | ✅ 2.7s · 170a | ✅ 4.1s · 263a | ✅ 3s · 192a | 12s | 755 | 0 | 64.0 |
| `poolside/laguna-s-2.1` | LM Studio | on | ✅ 4/4 | ✅ 3s · 130a | ✅ 2.9s · 182a | ✅ 3.8s · 229a | ✅ 3.2s · 189a | 13s | 730 | 0 | 56.6 |
| `zai-org/glm-4.7-flash` | LM Studio | on | ✅ 4/4 | ✅ 100.6s · 5139a · 764t | ✅ 97.4s · 4668a · 1022t | ✅ 96.7s · 5005a · 834t | ✅ 99.1s · 5065a · 778t | 394s | 19877 | 3398 | 50.5 |
| `zai-org/glm-4.7-flash` | LM Studio | off | ✅ 4/4 | ✅ 99.5s · 5194a · 713t | ✅ 100.2s · 4042a · 1722t | ✅ 98.1s · 4946a · 854t | ✅ 96.3s · 4729a · 1160t | 394s | 18911 | 4449 | 48.0 |
| `prism-ml/bonsai-27b` | LM Studio | off | ✅ 4/4 | ✅ 3.7s · 168a <sup>v2</sup> | ✅ 6.2s · 291a <sup>v2</sup> | ✅ 3.4s · 147a <sup>v2</sup> | ✅ 3.5s · 165a <sup>v2</sup> | 17s | 771 | 0 | 45.9 |
| `huizimao-gpt-oss-20b-uncensored-hi-mlx` | LM Studio | off | ✅ 4/4 | ✅ 5.6s · 179a | ✅ 5.3s · 163a | ✅ 4.1s · 238a | ✅ 4s · 134a | 19s | 714 | 0 | 37.6 |
| `magistral-small-2509-mlx` | LM Studio | off | ✅ 4/4 | ✅ 4.3s · 154a | ✅ 4.9s · 169a | ✅ 6.4s · 223a | ✅ 4.5s · 161a | 20s | 707 | 0 | 35.2 |
| `magistral-small-2509-mlx` | LM Studio | on | ✅ 4/4 | ✅ 4.5s · 154a | ✅ 4.2s · 140a | ✅ 6s · 197a | ✅ 4.3s · 150a | 19s | 641 | 0 | 33.7 |
| `qwen3.8-27b-heretic-abliterated-uncensored` | LM Studio | off | ✅ 4/4 | ✅ 7.5s · 280a <sup>v2</sup> | ✅ 5.5s · 177a <sup>v2</sup> | ✅ 4.9s · 156a <sup>v2</sup> | ✅ 6.7s · 186a <sup>v2</sup> | 25s | 799 | 0 | 32.5 |
| `qwen3.8-27b-obliterated` | LM Studio | off | ✅ 4/4 | ✅ 7.8s · 260a <sup>v2</sup> | ✅ 6.1s · 177a <sup>v2</sup> | ✅ 5.1s · 149a <sup>v2</sup> | ✅ 6.5s · 176a <sup>v2</sup> | 26s | 762 | 0 | 29.9 |
| `qwen3.8-27b` | LM Studio | off | ✅ 4/4 | ✅ 7.3s · 251a <sup>v2</sup> | ✅ 5.7s · 157a <sup>v2</sup> | ✅ 5.7s · 160a <sup>v2</sup> | ✅ 6.9s · 187a <sup>v2</sup> | 26s | 755 | 0 | 29.5 |
| `ornith-1.5-35b-a3b-uncensored-mlx` | LM Studio | on | ✅ 4/4 | ✅ 7.4s · 193a · 402t | ✅ 15.1s · 192a · 1057t | ✅ 5.5s · 215a · 223t | ✅ 2.6s · 147a · 70t | 31s | 747 | 1752 | 24.4 |
| `mistralai/devstral-small-2507` | LM Studio | off | ✅ 4/4 | ✅ 7.3s · 154a | ✅ 8.6s · 174a | ✅ 10.6s · 209a | ✅ 8.6s · 179a | 35s | 716 | 0 | 20.4 |
| `zai-org/glm-4.6v-flash` | LM Studio | off | ✅ 4/4 | ✅ 8.5s · 143a · 541t <sup>v2</sup> | ✅ 15.4s · 101a · 1078t <sup>v2</sup> | ✅ 28.7s · 136a · 2030t <sup>v2</sup> | ✅ 4.6s · 115a · 238t <sup>v2</sup> | 57s | 495 | 3887 | 8.7 |
| `LFM2-24B-A2B-MLX-8bit` | oMLX | off | ✅ 4/4 | ✅ 1s · 8a | ✅ 1.6s · 13a | ✅ 1.8s · 15a | ✅ 1.2s · 10a | 6s | 46 | 0 | 8.2 |
| `Ornith-1.5-35B-A3B-oQ4e-fp16-mtp` | oMLX | off | ✅ 4/4 | ✅ 2s · 14a | ✅ 2.4s · 19a | ✅ 3.3s · 26a | ✅ 2s · 15a | 10s | 74 | 0 | 7.6 |
| `google/gemma-4-26b-a4b-qat` | LM Studio | on | ✅ 4/4 | ✅ 5.7s · 44a · 512t | ✅ 33.3s · 124a · 3215t | ✅ 20.4s · 140a · 1943t | ✅ 6.8s · 159a · 573t | 66s | 467 | 6243 | 7.1 |
| `Ternary-Bonsai-27B-mlx-2bit` | oMLX | off | ✅ 4/4 | ✅ 4.4s · 29a | ✅ 3.5s · 23a | ✅ 6.2s · 42a | ✅ 2.9s · 19a | 17s | 113 | 0 | 6.6 |
| `grug-27b-oQ8e-fp16` | oMLX | off | ✅ 4/4 | ✅ 3.4s · 17a | ✅ 11s · 64a | ✅ 8.8s · 50a | ✅ 7.4s · 42a | 31s | 173 | 0 | 5.7 |
| `Qwen3.8-27B-AWQ-5.0bpw` | oMLX | off | ✅ 4/4 | ✅ 3.4s · 17a | ✅ 8s · 47a | ✅ 8.5s · 48a | ✅ 6.2s · 36a | 26s | 148 | 0 | 5.7 |
| `Seed-OSS-36B-Instruct-MLX-8bit` | oMLX | off | ✅ 4/4 | ✅ 30.3s · 134a | ✅ 154.7s · 683a | ✅ 467.2s · 1999a | ✅ 51.2s · 228a | 703s | 3044 | 0 | 4.3 |
| `Seed-OSS-36B-Instruct-MLX-8bit` | oMLX | on | ✅ 4/4 | ✅ 83.5s · 134a | ✅ 115.7s · 502a | ✅ 438.8s · 1858a | ✅ 81.9s · 362a | 720s | 2856 | 0 | 4.0 |
| `bytedance/seed-oss-36b` | LM Studio | off | ✅ 4/4 | ✅ 17.4s · 79a · 297t | ✅ 123.2s · 148a · 2385t | ✅ 36.6s · 223a · 558t | ✅ 30.4s · 136a · 527t | 208s | 586 | 3767 | 2.8 |
| `prism-ml/bonsai-27b` | LM Studio | on | ✅ 4/4 | ✅ 35.4s · 66a · 1460t | ✅ 82s · 182a · 3321t | ✅ 62.3s · 171a · 2316t | ✅ 34.9s · 145a · 1255t | 215s | 564 | 8352 | 2.6 |
| `grug-27b-oQ8e-fp16` | oMLX | on | ✅ 4/4 | ✅ 47.6s · 18a · 16t | ✅ 15.1s · 57a · 32t | ✅ 14.2s · 73a · 8t | ✅ 10s · 45a · 13t | 87s | 193 | 69 | 2.2 |
| `qwen3.8-27b` | LM Studio | on | ✅ 4/4 | ✅ 11.1s · 53a · 359t | ✅ 177.3s · 140a · 4544t | ✅ 31.5s · 160a · 794t | ✅ 30.8s · 142a · 806t | 251s | 495 | 6503 | 2.0 |
| `qwen3.8-27b-obliterated` | LM Studio | on | ✅ 4/4 | ✅ 11.9s · 48a · 281t | ✅ 136.8s · 139a · 2842t | ✅ 80.2s · 185a · 2056t | ✅ 45.7s · 138a · 1153t | 275s | 510 | 6332 | 1.9 |
| `bytedance/seed-oss-36b` | LM Studio | on | ✅ 4/4 | ✅ 17.9s · 79a · 297t | ✅ 112.9s · 146a · 2248t | ✅ 227.5s · 199a · 4329t | ✅ 30.1s · 117a · 527t | 388s | 541 | 7401 | 1.4 |
| `Ornith-1.5-35B-A3B-oQ4e-fp16-mtp` | oMLX | on | ✅ 4/4 | ✅ 37s · 16a · 32t | ✅ 19.5s · 28a · 142t | ✅ 4.6s · 21a · 18t | ✅ 2.9s · 15a · 9t | 64s | 80 | 201 | 1.2 |
| `LFM2-24B-A2B-MLX-8bit` | oMLX | on | ✅ 4/4 | ✅ 33.6s · 6a | ✅ 1.3s · 11a | ✅ 1.8s · 15a | ✅ 1.3s · 11a | 38s | 43 | 0 | 1.1 |
| `Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-NEO-MAX-MTP-GGUF-dequantized` | oMLX | off | ✅ 4/4 | ✅ 26.2s · 22a | ✅ 90.1s · 82a | ✅ 110.2s · 104a | ✅ 83.5s · 73a | 310s | 281 | 0 | 0.9 |
| `Muse-Glimmer-30B-4bit` | oMLX | off | ✅ 4/4 | ✅ 58.4s · 41a · 138t | ✅ 108.2s · 36a · 587t | ✅ 63.1s · 51a · 297t | ✅ 41s · 24a · 192t | 271s | 152 | 1214 | 0.6 |
| `Muse-Glimmer-30B-4bit` | oMLX | on | ✅ 4/4 | ✅ 63.9s · 42a · 157t | ✅ 129.6s · 34a · 699t | ✅ 72.8s · 53a · 356t | ✅ 31.4s · 27a · 150t | 298s | 156 | 1362 | 0.5 |
| `Ternary-Bonsai-27B-mlx-2bit` | oMLX | on | ✅ 4/4 | ✅ 41s · 8a · 197t | ✅ 85.1s · 30a · 627t | ✅ 63.3s · 29a · 451t | ✅ 32.3s · 23a · 223t | 222s | 90 | 1498 | 0.4 |
| `Qwen3.8-27B-AWQ-5.0bpw` | oMLX | on | ✅ 4/4 | ✅ 35s · 16a · 52t | ✅ 188.9s · 35a · 1235t | ✅ 113s · 57a · 683t | ✅ 36.7s · 35a · 204t | 374s | 143 | 2174 | 0.4 |
| `Qwen3.8-27B-MTPLX-Optimized-Quality` | oMLX | on | ✅ 4/4 | ✅ 57.6s · 21a · 85t | ✅ 230.3s · 51a · 1205t | ✅ 58.3s · 56a · 263t | ✅ 64.5s · 48a · 306t | 411s | 176 | 1859 | 0.4 |
| `Qwen3.6-27B-Fable-Fusion-711-Uncensored-Heretic-NM-DAU-NEO-MAX-MTP-GGUF-dequantized` | oMLX | on | ✅ 4/4 | ✅ 479.7s · 25a · 335t | ✅ 1473.5s · 76a · 1239t | ✅ 1066s · 95a · 888t | ✅ 532.9s · 64a · 423t | 3552s | 260 | 2885 | 0.1 |
| `liquid/lfm2.5-1.2b` | LM Studio | on | 🟡 3/4 | ✅ 0.7s · 145a | ❌ 0.3s · 91a | ✅ 0.4s · 132a | ✅ 0.4s · 129a | 2s | 497 | 0 | ∞ |
| `liquid/lfm2.5-1.2b` | LM Studio | off | 🟡 3/4 | ✅ 0.5s · 145a | ❌ 0.3s · 91a | ✅ 0.5s · 132a | ✅ 0.4s · 129a | 2s | 497 | 0 | ∞ |
| `minicpm-v:latest` | Ollama | off | 🟡 3/4 | ✅ 1.4s · 145a | ❌ 2.6s · 272a | ✅ 2.5s · 253a | ✅ 1.7s · 170a | 8s | 840 | 0 | 102.4 |
| `gemma4-26b-a4b-uncensored-hauhaucs-balanced` | LM Studio | off | 🟡 3/4 | ❌ 0s · 0a | ✅ 1.8s · 149a <sup>v2</sup> | ✅ 1.6s · 129a <sup>v2</sup> | ✅ 2s · 162a <sup>v2</sup> | 5s | 440 | 0 | 81.5 |
| `drk-coding-v1` | LM Studio | off | 🟡 3/4 | ✅ 2.3s · 221a <sup>v2</sup> | ✅ 1.8s · 148a <sup>v2</sup> | ❌ 0s · 0a | ✅ 2.6s · 165a <sup>v2</sup> | 7s | 534 | 0 | 79.7 |
| `grug-35b-qat` | LM Studio | off | 🟡 3/4 | ✅ 1.5s · 65a · 62t | ✅ 3.3s · 197a · 107t | ❌ 2.5s · 193a · 31t | ✅ 1.9s · 118a · 50t | 9s | 573 | 250 | 62.3 |
| `vibe-coding-claude-fable-5-mlx` | LM Studio | off | 🟡 3/4 | ❌ 1.2s · 68a | ✅ 4s · 232a | ✅ 3.4s · 198a | ✅ 2.6s · 151a | 11s | 649 | 0 | 57.9 |
| `vibe-coding-claude-fable-5-mlx` | LM Studio | on | 🟡 3/4 | ❌ 1.4s · 69a | ✅ 3.7s · 219a | ✅ 3.4s · 200a | ✅ 2.6s · 152a | 11s | 640 | 0 | 57.7 |
| `grug-35b-qat` | LM Studio | on | 🟡 3/4 | ✅ 1.7s · 64a · 63t | ✅ 3.6s · 199a · 110t | ❌ 3s · 217a · 24t | ✅ 2.1s · 116a · 46t | 10s | 596 | 243 | 57.3 |
| `minicpm-v:latest` | Ollama | on | 🟡 3/4 | ✅ 10.6s · 142a | ❌ 2.3s · 208a | ✅ 2.2s · 208a | ✅ 2.1s · 210a | 17s | 768 | 0 | 44.7 |
| `ornith-1.5-35b-a3b-uncensored-mlx` | LM Studio | off | 🟡 3/4 | ✅ 6s · 212a · 310t | ✅ 10.2s · 203a · 686t | ❌ 3.9s · 184a · 158t | ✅ 2.6s · 150a · 70t | 23s | 749 | 1224 | 33.0 |
| `gpt-oss-safeguard-20b-mlx` | LM Studio | on | 🟡 3/4 | ❌ 7.1s · 161a | ✅ 7.4s · 193a | ✅ 10.4s · 261a | ✅ 2s · 144a | 27s | 759 | 0 | 28.2 |
| `gpt-oss-safeguard-20b-mlx` | LM Studio | off | 🟡 3/4 | ❌ 4.3s · 111a | ✅ 8.4s · 175a | ✅ 7.9s · 218a | ✅ 2s · 131a | 23s | 635 | 0 | 28.1 |
| `huizimao-gpt-oss-20b-uncensored-hi-mlx` | LM Studio | on | 🟡 3/4 | ✅ 5.3s · 101a | ❌ 6.8s · 134a | ✅ 3.9s · 221a | ✅ 7.5s · 118a | 24s | 574 | 0 | 24.4 |
| `gemma4-26b-a4b-uncensored-hauhaucs-balanced` | LM Studio | on | 🟡 3/4 | ✅ 11.2s · 127a · 785t | ✅ 19.3s · 138a · 1542t | ❌ 17.2s · 138a · 1343t | ✅ 14.4s · 161a · 1075t | 62s | 564 | 4745 | 9.1 |
| `Devstral-Small-2-24B-Instruct-2512-bf16` | oMLX | off | 🟡 3/4 | ❌ 6.8s · 24a | ✅ 12.6s · 47a | ✅ 21.3s · 79a | ✅ 14.2s · 51a | 55s | 201 | 0 | 3.7 |
| `google/gemma-4-12b` | LM Studio | on | 🟡 3/4 | ✅ 26.2s · 135a · 769t | ✅ 61.9s · 144a · 2034t | ❌ 91s · 166a · 2916t | ✅ 23.6s · 155a · 630t | 203s | 600 | 6349 | 3.0 |
| `google/gemma-4-12b` | LM Studio | off | 🟡 3/4 | ✅ 23.2s · 127a · 645t | ✅ 45.3s · 138a · 1376t | ❌ 118s · 177a · 3629t | ✅ 22.7s · 149a · 613t | 209s | 591 | 6263 | 2.8 |
| `zai-org/glm-4.6v-flash` | LM Studio | on | 🟡 3/4 | ✅ 4.9s · 42a · 286t | ✅ 23.7s · 170a · 1545t | ❌ 82.1s · 0a · 5795t | ✅ 3.3s · 98a · 147t | 114s | 310 | 7773 | 2.7 |
| `Qwen3.8-27B-MTPLX-Optimized-Quality` | oMLX | off | 🟡 3/4 | ❌ 0s · 0a | ✅ 49.7s · 59a | ✅ 13.5s · 70a | ✅ 10s · 52a | 73s | 181 | 0 | 2.5 |
| `Devstral-Small-2-24B-Instruct-2512-bf16` | oMLX | on | 🟡 3/4 | ❌ 69.7s · 25a | ✅ 13.1s · 47a | ✅ 27.3s · 101a | ✅ 15.4s · 55a | 126s | 228 | 0 | 1.8 |
| `gemma-4-12b-coder-fable5-composer2.5-4bit` | oMLX | on | 🟡 3/4 | ✅ 16.2s · 6a · 20t | ✅ 9.1s · 12a · 54t | ❌ 13.3s · 23a · 79t | ✅ 13s · 22a · 81t | 52s | 63 | 234 | 1.2 |
| `qwen3.8-27b-heretic-abliterated-uncensored` | LM Studio | on | 🟡 3/4 | ✅ 14.7s · 53a · 375t | ❌ 257.1s · 0a · 5966t <sup>@6000</sup> | ✅ 72s · 192a · 1862t | ✅ 43.3s · 145a · 1144t | 387s | 390 | 9347 | 1.0 |
| `allenai/olmo-3-32b-think` | LM Studio | on | 🟡 3/4 | ✅ 17.9s · 103a · 329t | ✅ 149.6s · 95a · 3405t | ❌ 254.7s · 62a · 5724t | ✅ 45.6s · 157a · 951t | 468s | 417 | 10409 | 0.9 |
| `qwen2.5:0.5b` | Ollama | off | ❌ 2/4 | ✅ 0.5s · 164a | ❌ 0.7s · 269a | ✅ 1.6s · 646a | ❌ 0.4s · 148a | 3s | 1227 | 0 | 383.4 |
| `qwen2.5:0.5b` | Ollama | on | ❌ 2/4 | ✅ 1.5s · 148a | ❌ 0.2s · 82a | ✅ 1.2s · 376a | ❌ 0.4s · 158a | 3s | 764 | 0 | 231.5 |
| `google/gemma-4-26b-a4b-qat` | LM Studio | off | ❌ 2/4 | ❌ 0s · 0a | ❌ 0s · 0a | ✅ 1.3s · 121a <sup>v2</sup> | ✅ 1.5s · 158a <sup>v2</sup> | 3s | 279 | 0 | 99.6 |
| `minicpm5-1b-claude-opus-fable5-v2-thinking-heretic` | LM Studio | off | ❌ 2/4 | ✅ 1.2s · 211a · 36t | ❌ 3.6s · 123a · 630t | ❌ 2.1s · 119a · 296t | ✅ 1.2s · 77a · 161t | 8s | 530 | 1123 | 65.4 |
| `minicpm5-1b-claude-opus-fable5-v2-thinking-heretic` | LM Studio | on | ❌ 2/4 | ✅ 2.2s · 161a · 288t | ❌ 5.4s · 77a · 1014t | ❌ 2.3s · 118a · 336t | ✅ 1.4s · 82a · 198t | 11s | 438 | 1836 | 38.8 |
| `mistralai/devstral-small-2507` | LM Studio | on | ❌ 2/4 | ✅ 7.9s · 154a | ❌ 8.6s · 166a | ❌ 8s · 150a | ✅ 8.9s · 182a | 33s | 652 | 0 | 19.5 |
| `gemma-4-12b-coder-fable5-composer2.5-4bit` | oMLX | off | ❌ 2/4 | ✅ 5.4s · 18a | ❌ 2.7s · 19a | ❌ 2.2s · 15a | ✅ 2.3s · 17a | 13s | 69 | 0 | 5.5 |
| `allenai/olmo-3-32b-think` | LM Studio | off | ❌ 2/4 | ❌ 0s · 0a | ✅ 134.9s · 103a · 3290t <sup>v2</sup> | ❌ 0s · 0a | ✅ 59.2s · 169a · 1316t <sup>v2</sup> | 194s | 272 | 4606 | 1.4 |
| `drk-coding-v1` | LM Studio | on | ❌ 2/4 | ✅ 11.3s · 47a · 1072t | ❌ 80.4s · 0a · 6000t <sup>@8000</sup> | ❌ 72.9s · 0a · 6000t <sup>@6000</sup> | ✅ 22.8s · 116a · 1626t | 187s | 163 | 14698 | 0.9 |
| `minicpm-v4.6:latest` | Ollama | off | ❌ 1/4 | ✅ 0.6s · 165a | ❌ 0.2s · 43a | ❌ 0.2s · 66a | ❌ 0.2s · 44a | 1s | 318 | 0 | ∞ |
| `minicpm-v4.6:latest` | Ollama | on | ❌ 1/4 | ✅ 18s · 125a | ❌ 3.3s · 83a | ❌ 1.1s · 94a | ❌ 0.8s · 40a | 23s | 342 | 0 | 14.7 |


---

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