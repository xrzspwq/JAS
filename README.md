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
Terminal window, and prepares the `.jas` data folder next to them (copying in an older `%USERPROFILE%\.jas` once).
Options: `-Layout panes|tabs|windows`, `-Config <path>`, `-Port 2004`, `-NoMonitor`, `-Stop`
(stops llama-swap/llama-server), `-DryRun` (prints exactly what it would run). Manual start: `llama-swap --config models\llama-swap.config.yaml --listen 127.0.0.1:8080`, then
`python jas.py`, and `python jas_monitor.py` in a third terminal.

## Using Jas

* **Approval:** whenever Jas wants to act it shows ONE list of everything it wants to run; answer `a` (all), `n` (none,
  or just Enter), `1,3`, `except 2`, or `d2` for the details of #2. `/approve actions` lets read-only actions run without
  asking; `/yolo on` turns all prompts off (the terminal Jas runs in stays off limits either way).
* **Commands:** `/model main|coder|thinker|fast`, `/think on|off`, `/approve all|actions`, `/yolo on|off`, `/ctx`,
  `/notes [query]`, `/remember <text>`, `/forget <id>`, `/learn`, `/reset`, `/exit`.
* **Memory:** notes live in `\.jas\memory.db` **inside the Jas folder** (unencrypted, like before); long chats are
  summarised automatically. `status.json` and `backups\` live there too. `JAS_HOME` moves the whole folder; on first
  use anything from an older `%USERPROFILE%\.jas` is copied in.
* **Context size** comes from `-c` in your llama-swap config; `JAS_CTX` overrides it.
* **Web:** by default traffic goes straight from your PC (your real IP is visible). Set `JAS_PROXY` (e.g.
  `socks5h://127.0.0.1:9150` for Tor) to route it elsewhere, or `SEARXNG_URL` for your own search engine. Jas prints
  its web route at startup. See the top of `jas_web.py` for all settings.
* **Grounding check:** every bullet/numbered item in Jas's answer must appear in a tool output from that turn, and
  a single landing-page read is explicitly labelled as *not explored* - a list Jas was not given is rejected and
  re-asked, and after two failures it is marked "unconfirmed" rather than shipped as fact.
* **Never invents app content:** if `ui_inspect` returns only window chrome (title bar, menus), Jas is told plainly
  that nothing was read, and if it still describes the content the answer is rejected and re-asked. Documents are
  read with TextPattern, so a custom-drawn app's text is available when the app exposes it.
* **Graphical apps:** Jas reads an app's controls and clicks/types by number (`jas_ui.py`); `open_app` launches an app
  (optionally with a deep link, e.g. `spotify:search:lofi` or `spotify:playlist:<id>`). When an app cannot be found — or
  launching it puts no window on screen — Jas asks, then falls back to the **Windows Start menu search**: Win key,
  type the app's name, Enter (`start_menu_search`). It prefers the **Start menu's own app list** (`shell:AppsFolder`) and launches the best match by
  its AppUserModelId — no keystrokes, so it also works when the Win key is ignored (elevated foreground window). It
  checks both of those catalogues - the app list and the Start menu's `.lnk` shortcuts - so desktop programs the
  app list misses are still found. It only falls back to pressing Win and typing when both fail (and never while a
  terminal has the keyboard, or an elevated window would swallow the keys), and the search **verifies** the outcome — a new window, a web result opened instead, or nothing at
  all — and reports it honestly instead of treating "pressed Enter" as success. It refuses to type unless the Start
  menu is really open, and never types its own name. It never touches the desktop,
  taskbar or a terminal window, and a tool call that keeps failing stops the turn instead of looping.
  Push the mouse into a screen corner to abort. `JAS_NO_UI=1` turns this off.

## Files

| File | What it does |
|---|---|
| `jas.py` | the agent |
| `jas_web.py`, `jas_ui.py`, `jas_memory.py` | web access, graphical-app control, long-term notes |
| `jas_monitor.py` | live dashboard (CPU, RAM, GPU, VRAM, speed, context) |
| `jas_install_llama.py`, `jas_download_models.py` | install llama.cpp, download the models |
| `start_jas.ps1` / `.bat` | start everything |
| `test.py`, `prefill_test.py` | model speed tests |
| `jas_paths.py` | where Jas keeps its files: `<jas folder>\.jas` (memory, status, backups) |
| `jas_selfcheck.py` | diagnostic: prints every window with its process/class while you open Start |
| `start_jas.bat`, `start_jas.ps1` | launcher (llama-swap + Jas + monitor) |
| `test_*.py` | regression tests for the UI control, the agent and the paths - `python test_ui_fixes.py` etc. |

## If something is slow

* Make sure the K and V cache types in your llama-swap config **match** (`--cache-type-k q8_0 --cache-type-v q8_0`).
  A mismatched pair (e.g. q8_0 with q4_0) silently falls back to CPU attention and runs about 10x slower.
* Keep VRAM under about 3.8 GB on a 4 GB card (raise `--n-cpu-moe` if it overflows) and RAM below ~90%.
* Run `python prefill_test.py` to measure how fast prompts are read; `/reset` starts a fresh conversation.
