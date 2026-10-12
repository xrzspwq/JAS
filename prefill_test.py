#!/usr/bin/env python3
"""
prefill_test.py - measures how fast a model READS a brand-new prompt (no cache hits).
Use it to compare llama-server settings such as -ub 512 vs -ub 1024.

    python prefill_test.py                    # model "main", ~2000-token prompts, 3 runs
    python prefill_test.py --model coder --tokens 4000 --runs 2
    python prefill_test.py --url http://127.0.0.1:2004/v1

Every prompt starts with a random id, so the server cannot reuse its cache.
"""
import argparse
import os
import random
import time
import uuid

from openai import OpenAI

WORDS = ("river stone lamp window garden engine copper violet signal harbor pocket thunder ribbon marble "
         "orbit canvas hollow timber anchor mirror velvet crater paper bridge ladder pepper quartz meadow "
         "falcon button tunnel saddle wander ember glacier compass lantern whisper pillow dagger ocean "
         "village shadow engineer carpet rocket silver forest basket circuit morning summer winter autumn "
         "kitchen planet jungle market bottle camera helmet island castle pencil spider ticket yellow").split()


def make_prompt(tokens: int) -> str:
    n_words = int(tokens / 1.3)
    body = " ".join(random.choice(WORDS) for _ in range(n_words))
    return f"[id {uuid.uuid4()}] Summarize the following text in one word.\n\n{body}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="main")
    ap.add_argument("--tokens", type=int, default=2000, help="approximate prompt size")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--url", default=os.environ.get("JAS_URL", "http://127.0.0.1:8080/v1"))
    args = ap.parse_args()

    c = OpenAI(base_url=args.url, api_key="none")
    print(f"Loading '{args.model}' with a tiny request first (not counted)...")
    c.chat.completions.create(model=args.model, max_tokens=1, messages=[{"role": "user", "content": "hi"}])

    speeds = []
    for i in range(1, args.runs + 1):
        t0 = time.time()
        r = c.chat.completions.create(model=args.model, max_tokens=1,
                                      messages=[{"role": "user", "content": make_prompt(args.tokens)}])
        wall = time.time() - t0
        t = (getattr(r, "model_extra", None) or {}).get("timings") or {}
        n_new = t.get("prompt_n") or (r.usage.prompt_tokens if r.usage else 0)
        cached = t.get("cache_n", 0)
        speed = t.get("prompt_per_second") or (n_new / wall if wall else 0)
        speeds.append(speed)
        print(f"run {i}: {n_new} new tokens ({cached} cached) in {wall:.1f}s  ->  {speed:.0f} tok/s")
    print(f"\naverage prompt-reading speed: {sum(speeds) / len(speeds):.0f} tok/s")


if __name__ == "__main__":
    main()