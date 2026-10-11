# from openai import OpenAI   # pip install openai
# c = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
# r = c.chat.completions.create(model="fast", messages=[{"role": "user", "content": "Say hi as Jas."}])
# print(r.choices[0].message.content)

import time
from openai import OpenAI

c = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")

for model in ["fast", "main", "coder", "thinker"]:   # add "coder", "thinker" later
# for model in ["thinker"]:   # add "coder", "thinker" later
    print(f"\n=== {model} ===")
    t0 = time.time()
    first = None
    n = 0
    stream = c.chat.completions.create(
        model=model, stream=True, max_tokens=2000,
        messages=[{"role": "user", "content": "Explain how a hash table works in about 150 words."}],
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if first is None:
                first = time.time()
                print(f"[load + first token: {first - t0:.1f}s]")
            n += 1
            print(chunk.choices[0].delta.content, end="", flush=True)
    # print(first)
    print(f"\n[{n / (time.time() - first):.1f} tokens/s]")