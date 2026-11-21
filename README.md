# Jas - Just Artificial Stupidity

A local, private AI agent for Windows: it chats, researches the web, writes code, remembers things and can operate
your PC and graphical apps. Everything runs on your machine with [llama.cpp](https://github.com/ggml-org/llama.cpp);
every action is shown to you and needs your OK.

## Setup (once)

```powershell
python -m venv env
.\env\Scripts\activate
pip install -r requirements.txt

python jas_install_llama.py            # downloads the llama.cpp build for YOUR graphics card (CUDA/Vulkan/...), checks it works
python jas_download_models.py          # downloads the models and writes models\llama-swap.config.yaml
```

You also need **llama-swap** (a single `llama-swap.exe` from https://github.com/mostlygeek/llama-swap/releases):
put it on your PATH or next to these files.

## Run

Double-click `start_jas.bat` (or run `.\start_jas.ps1`). It opens llama-swap, Jas and the monitor in one Windows
Terminal window. Options: `-Layout tabs|windows`, `-NoMonitor`, `-Port 2004`, `-Stop` (stops llama-swap/llama-server),
`-DryRun`. Manual start: `llama-swap --config models\llama-swap.config.yaml --listen 127.0.0.1:8080`, then
`python jas.py`, and `python jas_monitor.py` in a third terminal.

## Using Jas

* **Approval:** whenever Jas wants to act it shows ONE list of everything it wants to run; answer `a` (all), `n` (none,
  or just Enter), `1,3`, `except 2`, or `d2` for the details of #2. `/approve actions` lets read-only actions run without
  asking; `/yolo on` turns all prompts off (the terminal Jas runs in stays off limits either way).
* **Commands:** `/model main|coder|thinker|fast`, `/think on|off`, `/approve all|actions`, `/yolo on|off`, `/ctx`,
  `/notes [query]`, `/remember <text>`, `/forget <id>`, `/learn`, `/reset`, `/exit`.
* **Memory:** notes live in `%USERPROFILE%\.jas\memory.db` (unencrypted); long chats are summarised automatically.
* **Context size** comes from `-c` in your llama-swap config; `JAS_CTX` overrides it.
* **Web:** by default traffic goes straight from your PC (your real IP is visible). Set `JAS_PROXY` (e.g.
  `socks5h://127.0.0.1:9150` for Tor) to route it elsewhere, or `SEARXNG_URL` for your own search engine. Jas prints
  its web route at startup. See the top of `jas_web.py` for all settings.
* **Graphical apps:** Jas reads an app's controls and clicks/types by number (`jas_ui.py`). Push the mouse into a
  screen corner to abort. `JAS_NO_UI=1` turns this off.

## Files

| File | What it does |
|---|---|
| `jas.py` | the agent |
| `jas_web.py`, `jas_ui.py`, `jas_memory.py` | web access, graphical-app control, long-term notes |
| `jas_monitor.py` | live dashboard (CPU, RAM, GPU, VRAM, speed, context) |
| `jas_install_llama.py`, `jas_download_models.py` | install llama.cpp, download the models |
| `start_jas.ps1` / `.bat` | start everything |
| `test.py`, `prefill_test.py` | model speed tests |

## If something is slow

* Make sure the K and V cache types in your llama-swap config **match** (`--cache-type-k q8_0 --cache-type-v q8_0`).
  A mismatched pair (e.g. q8_0 with q4_0) silently falls back to CPU attention and runs about 10x slower.
* Keep VRAM under about 3.8 GB on a 4 GB card (raise `--n-cpu-moe` if it overflows) and RAM below ~90%.
* Run `python prefill_test.py` to measure how fast prompts are read; `/reset` starts a fresh conversation.
