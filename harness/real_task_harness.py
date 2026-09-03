#!/usr/bin/env python3
"""Local LLM real-world task harness.

Runs the 4-task suite (grounded QA / invoice / technical summary / client email)
against any OpenAI-compatible local endpoint, one model at a time, thinking on and off.

Supports the two engines used in the benchmark:
  - LM Studio  : `--engine lmstudio` -> http://127.0.0.1:1234/v1/chat/completions
                 thinking-off knob = top-level "reasoning_effort": "none"
                 (only honored if the model is LOADED with a sane context length,
                  e.g. `lms load <model> -c 32768`)
  - oMLX       : `--engine omlx` -> http://127.0.0.1:8002/v1/chat/completions
                 (Bearer token from the env var in --api-key-env)
                 thinking-off knob = chat_template_kwargs.enable_thinking=false

Pick --engine EXPLICITLY. Do NOT sniff it from the URL — a URL ending in
`/v1/chat/completions` does not end in `:1234`, and sending the wrong knob is
silently ignored by the other engine (a lesson learned the hard way).

Usage:
  python3 real_task_harness.py --url http://127.0.0.1:1234/v1/chat/completions \
      --model qwen3.8-27b --thinking off --max-tokens 16000
  OMLX_API_KEY=... python3 real_task_harness.py --url http://127.0.0.1:8002/v1/chat/completions ...

Notes / lessons baked in:
  - max_tokens matters: a hard cap < 8-16K silently turns thinking models into
    "empty answer" models (reasoning consumes the whole budget). Use 16000+.
  - Verify your thinking-off knob is actually honored: inspect the thinking-token
    column — it should be ~0 when off. If a model still burns thinking at default
    context, re-load it with -c 32768.
  - Streaming parsers must tolerate chunks without `choices` (keepalives).

Tasks, grading rules and results: see README.md in this repo.
"""
import argparse, json, os, sys, time, urllib.request

TASKS = [
    {
        "name": "1_grounded_qa",
        "system": ("You are a fleet operations assistant. Answer ONLY from the "
                   "provided document. If the answer is not in the document, say "
                   "'not stated'."),
        "user": (
            "DOCUMENT:\n"
            "Gang Scheduling pattern: requests go to a central router. If the "
            "primary node is at or above its concurrency limit (3), requests fall "
            "back in this exact order: Node-B (10.0.0.2:9134, model-m), then "
            "Node-C (10.0.0.3:9134, model-v), then Node-D (10.0.0.4:9134, model-g). "
            "If every fallback node is busy, the request falls through to the "
            "primary node with a warning.\n\n"
            "QUESTIONS:\n"
            "1. What is the exact fallback order when the primary node is at or "
            "above its concurrency limit?\n"
            "2. Which model does Node-C serve, and on which port?\n"
            "3. What happens if every fallback node is busy?"
        ),
        # passes only if the three facts from the document survived verbatim
        "check": lambda a: ("node-b" in a.lower() and "node-c" in a.lower()
                            and "node-d" in a.lower() and "9134" in a),
    },
    {
        "name": "2_invoice",
        "system": "You are the bookkeeping assistant for a small home-decor brand.",
        "user": (
            "Draft an invoice for this job. Follow these house rules EXACTLY:\n"
            "- Invoice number format: INV-2026-XXXX (this is number 147)\n"
            "- Header must be logo-only, no tagline\n"
            "- Body in small type\n"
            "- Payment section wording must say 'Bank Deposit'\n"
            "- Footer contains only the business address and email\n"
            "Job details: 42 units of ceramic wall panels, MRP Rs 4,500 each; "
            "trade discount 10%; HST 13% on the discounted subtotal.\n"
            "Output the invoice as plain text."
        ),
        # math: 42*4500=189000; -10% -> 170100; +13% -> 192,213 and payment wording
        "check": lambda a: ("192,213" in a or "192213" in a)
                            and "deposit" in a.lower(),
    },
    {
        "name": "3_summary",
        "system": "You are a technical editor. Summarize accurately and concisely.",
        "user": (
            "Summarize the following in exactly 5 bullet points, under 100 words "
            "total:\n\n"
            "Video Generation Pipeline. A desktop app with a Python backend calls "
            "a distilled text-to-video pipeline. Models: text-to-video distilled "
            "bf16 43GB, official distilled 43GB, 12B text encoder at 4-bit (24GB), "
            "spatial upscaler x2 (349MB). Pipeline: prompt -> video model + text "
            "encoder conditioning -> spatial upscaler x2 -> 512x768 MP4, generated "
            "at 256x384 first then upscaled. Performance on Apple Silicon MPS for "
            "193 frames / 8s: cold run about 3 minutes, warm runs about 2m45s. "
            "Known issues: fp8 does not work on MPS; generation is about 3x slower "
            "than CUDA fp8; keep prompts under 150 words."
        ),
        # facts that must survive compression
        "check": lambda a: ("512x768" in a or "512" in a) and "43" in a
                            and ("fp8" in a or "mps" in a),
    },
    {
        "name": "4_email",
        "system": "You are the owner of a home-decor brand. Write professional, warm emails.",
        "user": (
            "A customer emailed: 'I ordered 6 wall panels last week (order #8812). "
            "The tracking shows delivered since Tuesday but I never got anything at "
            "my office. Also one of the pieces I picked has a chip on the corner - "
            "can you look at both?'\n"
            "Draft the reply. Requirements: acknowledge both issues separately, "
            "ask for a delivery confirmation or receiver signature from the courier "
            "before escalating, offer to inspect the chipped piece via photo today "
            "and promise replacement within 5 working days if confirmed, close "
            "warmly and professionally. Under 130 words."
        ),
        "check": lambda a: ("replacement" in a or "replace" in a
                            or "5 working" in a or "5 business" in a)
                            and ("photo" in a or "picture" in a or "image" in a),
    },
]


def stream_completion(url, model, system, user, thinking, max_tokens,
                      api_key=None, timeout=600, engine="lmstudio"):
    """Stream one completion. Returns answer text + token counters.

    Tolerates chunks without `choices` (e.g. keepalives during long generations).
    """
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": True,
    }
    if thinking == "off":
        if engine == "lmstudio":
            body["reasoning_effort"] = "none"   # LM Studio knob
        else:
            body["chat_template_kwargs"] = {"enable_thinking": False}  # oMLX knob

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers=headers)
    t0 = time.time()
    think_chunks = content_chunks = 0
    out = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                j = json.loads(data)
            except Exception:
                continue
            if "choices" not in j or not j["choices"]:
                continue  # keepalive / non-choice chunk
            delta = j["choices"][0].get("delta", {})
            rc = delta.get("reasoning_content") or ""
            c = delta.get("content") or ""
            if rc:
                think_chunks += 1
            if c:
                content_chunks += 1
                out.append(c)
    return {
        "answer": "".join(out),
        "thinking_tokens": think_chunks,
        "answer_tokens": content_chunks,
        "elapsed": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="OpenAI-compatible endpoint")
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine", choices=["lmstudio", "omlx"], default="lmstudio",
                    help="which reasoning-off knob to use (do NOT sniff from URL)")
    ap.add_argument("--thinking", choices=["on", "off"], default="off")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--api-key-env", default="OMLX_API_KEY",
                    help="env var holding the Bearer token (never pass keys inline)")
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env)

    results = []
    for t in TASKS:
        print("=" * 72, flush=True)
        print("TASK:", t["name"], flush=True)
        r = stream_completion(args.url, args.model, t["system"], t["user"],
                              args.thinking, args.max_tokens, api_key,
                              engine=args.engine)
        passed = t["check"](r["answer"])
        results.append({"task": t["name"], "passed": passed, **r})
        print(f"  passed={passed} elapsed={r['elapsed']}s "
              f"think={r['thinking_tokens']} answer={r['answer_tokens']}", flush=True)
        print("  answer head:", r["answer"][:180].replace("\n", " "), flush=True)

    n_pass = sum(1 for r in results if r["passed"])
    print("=" * 72, flush=True)
    print(f"SCORE: {n_pass}/4", flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()