# Quick speed test of the models behind llama-swap.
#   python test.py                 -> fast, main, coder, thinker
#   python test.py main            -> only the models you name
# pip install openai
import sys
import time

from openai import OpenAI

c = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
GREY, RST = "\033[90m", "\033[0m"

for model in sys.argv[1:] or ["fast", "main", "coder", "thinker"]:
    print(f"\n=== {model} ===")
    t0 = time.time()
    first = None
    n = 0
    try:
        stream = c.chat.completions.create(
            model=model, stream=True, max_tokens=2000,
            messages=[{"role": "user", "content": "Explain how a hash table works in about 150 words."}],
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            d = chunk.choices[0].delta
            thinking = getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None)   # thinking models
            if d.content or thinking:
                if first is None:
                    first = time.time()
                    print(f"[load + first token: {first - t0:.1f}s]")
                n += 1
                if thinking:
                    print(f"{GREY}{thinking}{RST}", end="", flush=True)
                if d.content:
                    print(d.content, end="", flush=True)
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}\nIs llama-swap running on port 8080, and is '{model}' in its config?")
        continue
    if first is None:
        print("No tokens received. Check the llama-swap window for a load error or crash.")
    else:
        print(f"\n[{n / max(time.time() - first, 1e-6):.1f} tokens/s]")
