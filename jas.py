#!/usr/bin/env python3
"""
jas.py - Jas v0: a local Jarvis-style terminal agent for Windows.

Setup:
    pip install openai ddgs trafilatura
    llama-swap --config <your config> --listen 127.0.0.1:8080     (keep it running)
    python jas.py

Commands inside the chat:
    /model main|coder|thinker|fast   switch the active brain
    /think on|off                    show/hide the thinking model's reasoning
    /yolo on|off                     skip the approval prompt for shell commands (default: off = ask every time)
    /reset                           forget the conversation (the cached system prompt stays valid)
    /exit                            quit
Override the server with the JAS_URL environment variable.
"""
import atexit
import datetime
import difflib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from openai import OpenAI

BASE_URL = os.environ.get("JAS_URL", "http://127.0.0.1:8080/v1")
client = OpenAI(base_url=BASE_URL, api_key="none")

HOME = Path.home()
BACKUP_DIR = HOME / ".jas" / "backups"
MAX_STEPS = 16            # tool-call rounds per user message
MAX_TOOL_CHARS = 4000     # cap on a single tool result
CONTEXT_BUDGET = 24000    # rough char budget before old tool output is shrunk

STATE = {"model": "main", "yolo": False, "think": False}

STATUS_FILE = HOME / ".jas" / "status.json"
DEFAULT_EXPECT = {"fast": 120, "main": 250, "coder": 700, "thinker": 900}   # typical reply lengths (tokens)


class Status:
    """Publishes live agent state to ~/.jas/status.json so jas_monitor.py can display it."""

    def __init__(self):
        self.d = {"pid": os.getpid(), "state": "idle", "model": STATE["model"], "step": 0,
                  "max_steps": MAX_STEPS, "tokens": 0, "tok_s": 0.0, "max_tokens": 4096,
                  "expect_tokens": 250, "prompt_est": None, "task_start": None, "reply_start": None, "first_token": None,
                  "tool": "", "last_task_s": None, "last_reply": None, "updated": time.time()}
        self.hist = {}      # model -> recent reply lengths, used to estimate time left
        self._last = 0.0

    def set(self, force=True, **kw):
        self.d.update(kw)
        now = time.time()
        self.d["updated"] = now
        if not force and now - self._last < 0.2:
            return
        self._last = now
        try:
            STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = STATUS_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.d), encoding="utf-8")
            os.replace(tmp, STATUS_FILE)
        except OSError:
            pass            # monitor is optional; never break the agent over it

    def expect(self, model):
        h = self.hist.get(model)
        return int(sum(h) / len(h)) if h else DEFAULT_EXPECT.get(model, 300)

    def begin_reply(self, model, max_tokens=4096, prompt_est=None):
        self.set(state="waiting", model=model, tokens=0, tok_s=0.0, reply_start=time.time(),
                 first_token=None, max_tokens=max_tokens, expect_tokens=self.expect(model),
                 prompt_est=prompt_est)

    def tick(self, n, first, state):
        dt = time.time() - first
        self.set(force=False, state=state, tokens=n, first_token=first,
                 tok_s=round((n - 1) / dt, 2) if dt > 0.3 else 0.0)

    def end_reply(self, model, n, pstats=None):
        if n <= 0:
            return self.set(tokens=n)
        self.hist[model] = (self.hist.get(model, []) + [n])[-8:]
        now = time.time()
        first, start = self.d.get("first_token"), self.d.get("reply_start")
        last = {"model": model, "tokens": n,
                "tok_s": round((n - 1) / (now - first), 2) if first and n > 1 and now - first > 0.05 else 0.0,
                "duration_s": round(now - start, 1) if start else None,
                "first_token_s": round(first - start, 1) if first and start else None}
        last.update(pstats or {})
        self.set(tokens=n, last_reply=last)


STATUS = Status()


def stream_stats(chunk) -> dict:
    """Prompt size and prefill speed from llama-server's final streamed chunk (when present)."""
    out = {}
    u = getattr(chunk, "usage", None)
    if u is not None and getattr(u, "prompt_tokens", None):
        out["prompt_tokens"] = u.prompt_tokens
    t = (getattr(chunk, "model_extra", None) or {}).get("timings") or {}
    if t.get("prompt_n") is not None:
        out["prompt_new"] = t.get("prompt_n", 0)
        out["prompt_cached"] = t.get("cache_n", 0)
        out["prompt_tok_s"] = round(t.get("prompt_per_second") or 0, 1)
        out.setdefault("prompt_tokens", out["prompt_new"] + out["prompt_cached"])
    return out

# ---------- terminal colours ----------
if os.name == "nt":
    os.system("")  # enable ANSI escapes in the Windows console
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
CYAN, GREY, YEL, RED, GRN, RST = "\033[96m", "\033[90m", "\033[93m", "\033[91m", "\033[92m", "\033[0m"


def system_prompt() -> str:
    now = datetime.datetime.now().strftime("%A %d %B %Y")
    return f"""You are Jas, a personal AI assistant running locally on the user's Windows PC.
Personality: calm, capable, quietly witty, in the spirit of a butler-grade AI. Be concise; no emojis.
Environment: Windows, PowerShell, home folder {HOME}, today is {now} (run Get-Date for the exact time).

How you work:
- Act, don't just describe. When the user asks for something you can do with a tool, call the tool.
- Look before you change: read or list files before editing them. Prefer edit_file for small changes.
- For coding tasks, you may call ask_specialist("coder", ...) to draft code, then save it with write_file and test it with run_shell.
- For hard analysis or research synthesis, call ask_specialist("thinker", ...) with the gathered facts.
- For research: web_search, then fetch_url the best 2-3 results, then answer and cite the URLs.
- Never invent tool results, file contents or URLs. If a tool fails, read the error and try another approach.
- Keep answers short unless the user asks for depth."""


# ---------- safety: confirmation + backups ----------
DESTRUCTIVE = re.compile(
    r"(?<![\w-])(rm|del|erase|rmdir|rd|remove-item|ri|format|diskpart|shutdown|restart-computer|stop-computer|"
    r"taskkill|stop-process|kill|move-item|mv|clear-content|set-executionpolicy|netsh|takeown|icacls|"
    r"reg\s+(delete|add)|sc\s+(delete|stop)|schtasks|"
    r"pip\s+uninstall|npm\s+uninstall|git\s+(reset\s+--hard|clean|push\s+(-f|--force)))(?![\w-])",
    re.I,
)


def confirm(prompt: str) -> bool:
    if STATE["yolo"]:
        return True
    ans = input(f"{YEL}! {prompt}  Allow? [y/N] {RST}").strip().lower()
    return ans in ("y", "yes", "o", "oui")


def backup(path: Path) -> str:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    dest = BACKUP_DIR / f"{datetime.datetime.now():%Y%m%d-%H%M%S-%f}_{path.name}"
    dest.write_bytes(path.read_bytes())
    return str(dest)


def expand(p: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(p)))


# ---------- tools ----------
def approve_command(command: str, timeout: int) -> bool:
    """Show the exact command and ask for approval. Used for EVERY shell command (unless /yolo on)."""
    if STATE["yolo"]:
        return True
    print(f"\n{YEL}Jas wants to run this command:{RST}")
    for line in command.splitlines() or [""]:
        print(f"{CYAN}    {line}{RST}")
    print(f"{GREY}    (PowerShell, timeout {timeout}s, folder: {os.getcwd()}){RST}")
    risky = DESTRUCTIVE.search(command)
    if risky:
        print(f"{RED}    WARNING: contains a potentially destructive operation: '{risky.group(0)}'{RST}")
    STATUS.set(state="approval")
    ans = input(f"{YEL}Run it? [y/N] {RST}").strip().lower()
    STATUS.set(state="tool")
    return ans in ("y", "yes", "o", "oui")


def tool_run_shell(command: str, timeout: int = 60):
    if not approve_command(command, int(timeout)):
        return "DENIED by the user. Do not retry the same command; ask what they would prefer."
    wrapped = "[Console]::OutputEncoding=[Text.Encoding]::UTF8; " + command
    try:
        p = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", wrapped],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=int(timeout), cwd=os.getcwd(),
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: command timed out after {timeout}s"
    out = p.stdout or ""
    if p.stderr:
        out += "\n[stderr]\n" + p.stderr
    return f"exit code {p.returncode}\n{out}".strip()


def tool_read_file(path: str, max_chars: int = 8000):
    p = expand(path)
    if not p.is_file():
        return f"ERROR: not a file: {p}"
    text = p.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + f"\n...[truncated, {len(text)} chars total]"
    return text


def approve_change(p: Path, old: str, new: str, verb: str) -> bool:
    """Show a diff of what will change in an EXISTING file and ask for approval (unless /yolo on)."""
    if STATE["yolo"]:
        return True
    print(f"\n{YEL}Jas wants to {verb} an existing file:{RST}")
    print(f"{CYAN}    {p}{RST}  {GREY}({len(old.splitlines())} lines -> {len(new.splitlines())} lines){RST}")
    diff = list(difflib.unified_diff(old.splitlines(), new.splitlines(), "current", "proposed", lineterm="", n=2))
    if not diff:
        print(f"{GREY}    (no textual changes){RST}")
    for line in diff[:40]:
        color = GRN if line.startswith("+") and not line.startswith("+++") else \
            RED if line.startswith("-") and not line.startswith("---") else GREY
        print(f"{color}    {line[:160]}{RST}")
    if len(diff) > 40:
        print(f"{GREY}    ... ({len(diff) - 40} more diff lines){RST}")
    print(f"{GREY}    (the current version is backed up automatically if you approve){RST}")
    STATUS.set(state="approval")
    ok = input(f"{YEL}Apply it? [y/N] {RST}").strip().lower() in ("y", "yes", "o", "oui")
    STATUS.set(state="tool")
    return ok


def tool_write_file(path: str, content: str):
    p = expand(path)
    note = ""
    if p.exists():
        if not p.is_file():
            return f"ERROR: {p} is a folder, not a file."
        old = p.read_text(encoding="utf-8", errors="replace")
        if not approve_change(p, old, content, "OVERWRITE"):
            return "DENIED by the user. The file was not changed. Do not retry; ask what they would prefer."
        note = f" (previous version backed up to {backup(p)})"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} chars to {p}{note}"


def tool_edit_file(path: str, old: str, new: str):
    p = expand(path)
    if not p.is_file():
        return f"ERROR: not a file: {p}"
    text = p.read_text(encoding="utf-8", errors="replace")
    count = text.count(old)
    if count == 0:
        return "ERROR: 'old' text not found. Read the file again and copy it exactly."
    if count > 1:
        return f"ERROR: 'old' text appears {count} times; include more surrounding context so it is unique."
    updated = text.replace(old, new)
    if not approve_change(p, text, updated, "EDIT"):
        return "DENIED by the user. The file was not changed. Do not retry; ask what they would prefer."
    b = backup(p)
    p.write_text(updated, encoding="utf-8")
    return f"Edited {p} (backup: {b})"


def tool_list_dir(path: str = "."):
    p = expand(path)
    if not p.is_dir():
        return f"ERROR: not a directory: {p}"
    rows = []
    for e in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))[:200]:
        rows.append(("[dir]  " if e.is_dir() else "[file] ") + e.name)
    return "\n".join(rows) or "(empty)"


def tool_open(target: str):
    """Open an app, file, folder or URL with its default handler."""
    if hasattr(os, "startfile"):
        os.startfile(os.path.expandvars(target))  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["xdg-open", target])
    return f"Opened {target}"


def tool_web_search(query: str, max_results: int = 20):
    try:
        from ddgs import DDGS
    except ImportError:
        return "ERROR: run  pip install ddgs"
    results = DDGS().text(query, max_results=int(max_results))
    if not results:
        return "No results."
    return "\n\n".join(f"{r.get('title')}\n{r.get('href')}\n{r.get('body')}" for r in results)


def tool_fetch_url(url: str, max_chars: int = 6000):
    try:
        import trafilatura
    except ImportError:
        return "ERROR: run  pip install trafilatura"
    html = trafilatura.fetch_url(url)
    if not html:
        return "ERROR: could not download the page."
    text = trafilatura.extract(html, include_links=False) or ""
    if not text:
        return "ERROR: no readable text found."
    return text[:max_chars] + ("\n...[truncated]" if len(text) > max_chars else "")


SPECIALISTS = {
    "coder": "You are an expert software engineer. Return correct, complete, runnable code with brief notes.",
    "thinker": "You are a rigorous analyst. Reason carefully, then give a clear, well-structured answer.",
}


def tool_ask_specialist(specialist: str, task: str):
    if specialist not in SPECIALISTS:
        return f"ERROR: specialist must be one of {list(SPECIALISTS)}"
    print(f"{GREY}  [consulting {specialist}... model swap may take a few seconds]{RST}")
    STATUS.begin_reply(specialist, prompt_est=int((len(task) + len(SPECIALISTS[specialist])) / 3.5))
    stream = client.chat.completions.create(
        model=specialist, max_tokens=4096, stream=True, stream_options={"include_usage": True},
        messages=[{"role": "system", "content": SPECIALISTS[specialist]},
                  {"role": "user", "content": task}],
    )
    text, n, first, pst = "", 0, None, {}
    for chunk in stream:
        pst.update(stream_stats(chunk))
        if not chunk.choices:
            continue
        d = chunk.choices[0].delta
        reasoning = getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None)
        if d.content:
            text += d.content
        if reasoning or d.content:
            first = first or time.time()
            n += 1
            STATUS.tick(n, first, "thinking" if reasoning and not d.content else "generating")
    STATUS.end_reply(specialist, n, pst)
    STATUS.set(state="tool")
    return text or "(empty reply)"


def _fn(name, desc, props, required):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required}}}


S = {"type": "string"}
TOOLS = [
    _fn("run_shell", "Run a PowerShell command on the user's PC and return its output. Use for launching programs, "
        "git, pip, system info, running scripts and tests.", {"command": S, "timeout": {"type": "integer"}}, ["command"]),
    _fn("read_file", "Read a text file.", {"path": S, "max_chars": {"type": "integer"}}, ["path"]),
    _fn("write_file", "Create or overwrite a text file (an existing file is backed up first).",
        {"path": S, "content": S}, ["path", "content"]),
    _fn("edit_file", "Replace one exact, unique piece of text in a file.",
        {"path": S, "old": S, "new": S}, ["path", "old", "new"]),
    _fn("list_dir", "List the contents of a folder.", {"path": S}, []),
    _fn("open", "Open an application, file, folder or URL with its default handler (e.g. 'notepad', "
        "'https://example.com', 'C:\\\\Users').", {"target": S}, ["target"]),
    _fn("web_search", "Search the web. Returns titles, URLs and snippets.", {"query": S, "max_results": {"type": "integer"}}, ["query"]),
    _fn("fetch_url", "Download a web page and return its readable text.", {"url": S, "max_chars": {"type": "integer"}}, ["url"]),
    _fn("ask_specialist", "Delegate to a stronger specialist model: 'coder' for writing code, 'thinker' for deep reasoning "
        "or synthesis. Include all needed context in the task.", {"specialist": {"type": "string", "enum": ["coder", "thinker"]}, "task": S},
        ["specialist", "task"]),
]
IMPL = {
    "run_shell": tool_run_shell, "read_file": tool_read_file, "write_file": tool_write_file,
    "edit_file": tool_edit_file, "list_dir": tool_list_dir, "open": tool_open,
    "web_search": tool_web_search, "fetch_url": tool_fetch_url, "ask_specialist": tool_ask_specialist,
}


def dispatch(call: dict) -> str:
    fn = IMPL.get(call["name"])
    if not fn:
        return f"ERROR: unknown tool '{call['name']}'"
    try:
        args = json.loads(call["args"] or "{}")
    except json.JSONDecodeError as e:
        return f"ERROR: arguments were not valid JSON ({e}). Call the tool again with valid JSON."
    try:
        return str(fn(**args))[:MAX_TOOL_CHARS]
    except TypeError as e:
        return f"ERROR: bad arguments: {e}"
    except Exception as e:  # tool failures go back to the model, not up to the user
        return f"ERROR: {type(e).__name__}: {e}"


# ---------- model I/O ----------
def stream_completion(model: str, messages: list):
    """Stream one completion; print text live; return (content, tool_calls)."""
    est = int((sum(len(json.dumps(m)) for m in messages) + len(json.dumps(TOOLS))) / 3.5)
    STATUS.begin_reply(model, prompt_est=est)
    stream = client.chat.completions.create(
        model=model, messages=messages, tools=TOOLS, stream=True, max_tokens=4096,
        stream_options={"include_usage": True})
    content, calls = "", {}
    n, first, pst = 0, None, {}
    for chunk in stream:
        pst.update(stream_stats(chunk))
        if not chunk.choices:
            continue
        d = chunk.choices[0].delta
        reasoning = getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None)
        if reasoning and STATE["think"]:
            print(f"{GREY}{reasoning}{RST}", end="", flush=True)
        if d.content:
            print(d.content, end="", flush=True)
            content += d.content
        for tc in d.tool_calls or []:
            slot = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
            if tc.id:
                slot["id"] = tc.id
            if tc.function:
                if tc.function.name:
                    slot["name"] = tc.function.name
                if tc.function.arguments:
                    slot["args"] += tc.function.arguments
        if reasoning or d.content or d.tool_calls:
            first = first or time.time()
            n += 1
            STATUS.tick(n, first, "thinking" if reasoning and not d.content else "generating")
    STATUS.end_reply(model, n, pst)
    out = [calls[i] for i in sorted(calls)]
    for k, c in enumerate(out):
        c["id"] = c["id"] or f"call_{int(time.time() * 1000)}_{k}"
    return content, out


def compact(messages: list):
    """Keep the prompt inside CONTEXT_BUDGET without touching it every turn.

    Editing an old message invalidates the server's prompt cache (everything after the edit must be re-read),
    so when we are over budget we shrink a LOT at once (down to ~50%) and then leave the history alone for
    several turns, instead of trimming a little on every turn."""
    def size() -> int:
        return sum(len(m.get("content") or "") + sum(len(tc["function"]["arguments"]) for tc in m.get("tool_calls") or [])
                   for m in messages)

    total = size()
    if total <= CONTEXT_BUDGET:
        return
    target = int(CONTEXT_BUDGET * 0.5)
    for m in messages[1:-6]:                         # 1) shrink old tool outputs (newest messages stay intact)
        if total <= target:
            break
        if m["role"] == "tool" and len(m["content"]) > 300:
            total -= len(m["content"]) - 300
            m["content"] = m["content"][:300] + "\n...[old output trimmed]"
    while total > target and len(messages) > 8:      # 2) still too big: drop the oldest whole turns
        i = next((k for k in range(1, len(messages)) if messages[k]["role"] == "user"), None)
        j = next((k for k in range(i + 1, len(messages)) if messages[k]["role"] == "user"), None) if i else None
        if not j:
            break
        total -= sum(len(m.get("content") or "") for m in messages[i:j])
        del messages[i:j]


def warm_up(model: str, system_msg: dict):
    """Load the model and pre-read the system prompt + tool definitions, so the first real request is fast."""
    print(f"{GREY}Warming up '{model}' (loading the model, caching the prompt; Ctrl+C to skip)...{RST}", end="", flush=True)
    t0 = time.time()
    STATUS.begin_reply(model)
    try:
        client.chat.completions.create(
            model=model, max_tokens=1, tools=TOOLS,
            messages=[system_msg, {"role": "user", "content": "hi"}])
        print(f"{GREY} ready in {time.time() - t0:.0f}s{RST}")
    except KeyboardInterrupt:
        print(f"{GREY} skipped{RST}")
    except Exception as e:
        print(f"\n{RED}Warm-up failed: {type(e).__name__}: {e}{RST}")
    finally:
        STATUS.set(state="idle", model=model, tokens=0)


def run_turn(messages: list, user_text: str):
    messages.append({"role": "user", "content": user_text})
    for step in range(MAX_STEPS):
        STATUS.set(step=step + 1)
        compact(messages)
        print(f"{CYAN}jas>{RST} ", end="", flush=True)
        content, calls = stream_completion(STATE["model"], messages)
        print()
        if not calls:
            messages.append({"role": "assistant", "content": content})
            return
        messages.append({
            "role": "assistant", "content": content or None,
            "tool_calls": [{"id": c["id"], "type": "function",
                            "function": {"name": c["name"], "arguments": c["args"] or "{}"}} for c in calls],
        })
        for c in calls:
            STATUS.set(state="tool", tool=c["name"])
            short = (c["args"] or "")[:140].replace("\n", " ")
            print(f"{GREY}  > {c['name']}({short}){RST}")
            result = dispatch(c)
            print(f"{GREY}    {result[:200].replace(chr(10), ' ')}{RST}")
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
    print(f"{YEL}(step limit reached; tell me to continue if needed){RST}")


def main():
    print(f"{CYAN}Jas online.{RST} {GREY}server {BASE_URL} | model {STATE['model']} | /model /think /yolo /reset /exit{RST}")
    messages = [{"role": "system", "content": system_prompt()}]
    STATUS.set(state="idle")
    atexit.register(lambda: STATUS.set(state="offline"))
    if os.environ.get("JAS_NO_WARMUP") != "1":
        warm_up(STATE["model"], messages[0])
    while True:
        try:
            text = input(f"\n{GRN}you>{RST} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.startswith("/"):
            cmd, *rest = text.split()
            arg = rest[0].lower() if rest else ""
            if cmd == "/exit":
                break
            elif cmd == "/reset":
                messages = [{"role": "system", "content": system_prompt()}]
                print("Memory cleared.")
            elif cmd == "/model" and arg in ("main", "coder", "thinker", "fast"):
                STATE["model"] = arg
                STATUS.set(model=arg)
                print(f"Active model: {arg}")
                if os.environ.get("JAS_NO_WARMUP") != "1":
                    warm_up(arg, messages[0])
            elif cmd == "/think" and arg in ("on", "off"):
                STATE["think"] = arg == "on"
                print(f"Show reasoning: {arg}")
            elif cmd == "/yolo" and arg in ("on", "off"):
                STATE["yolo"] = arg == "on"
                print(f"Confirmations {'DISABLED' if arg == 'on' else 'enabled'}.")
            else:
                print("Unknown command.")
            continue
        STATUS.set(state="waiting", task_start=time.time(), step=0, tool="", tokens=0, tok_s=0.0)
        try:
            run_turn(messages, text)
        except KeyboardInterrupt:
            print(f"\n{YEL}(interrupted){RST}")
        except Exception as e:
            print(f"\n{RED}Error talking to the server: {type(e).__name__}: {e}{RST}")
            print(f"{GREY}Is llama-swap running at {BASE_URL}?{RST}")
        finally:
            STATUS.set(state="idle", model=STATE["model"], tool="", task_start=None,
                       last_task_s=round(time.time() - (STATUS.d["task_start"] or time.time()), 1))


if __name__ == "__main__":
    main()