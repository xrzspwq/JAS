#!/usr/bin/env python3
"""
jas_monitor.py - live dashboard to run NEXT TO jas.py.

Shows: CPU, RAM, GPU, VRAM, which model Jas is using, generation speed (tokens/s),
an estimate of the time left for the current reply, and how full the context window is.

    pip install psutil rich
    python jas_monitor.py                 # refreshes ~2x per second, Ctrl+C to quit
    python jas_monitor.py --interval 1    # slower refresh
    python jas_monitor.py --url http://127.0.0.1:2004   # llama-swap address (optional)

Data sources:
  * ~/.jas/status.json   written by jas.py (model, tokens, speed, step, state)
  * psutil               CPU / RAM / llama-server processes
  * nvidia-smi           GPU utilisation and VRAM (comes with your NVIDIA driver)
  * llama-swap /running  loaded models (best effort; skipped if unavailable)
"""
import argparse
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.request
from collections import deque
from pathlib import Path

import psutil
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

STATUS_FILE = Path.home() / ".jas" / "status.json"
NO_WINDOW = 0x08000000 if os.name == "nt" else 0
SPARK = "▁▂▃▄▅▆▇█"

STATE_STYLE = {
    "idle": ("IDLE - waiting for you", "green"),
    "waiting": ("WAITING - model loading / reading the prompt", "yellow"),
    "thinking": ("THINKING (reasoning)", "magenta"),
    "generating": ("GENERATING", "cyan"),
    "tool": ("RUNNING A TOOL", "blue"),
    "approval": ("WAITING FOR YOUR APPROVAL", "bold yellow"),
    "offline": ("JAS IS NOT RUNNING", "red"),
}


# ---------------- helpers ----------------
def color_for(pct: float) -> str:
    return "green" if pct < 60 else "yellow" if pct < 85 else "red"


def bar(pct: float, width: int = 26) -> Text:
    pct = max(0.0, min(100.0, pct))
    filled = int(round(width * pct / 100))
    t = Text()
    t.append("█" * filled, style=color_for(pct))
    t.append("░" * (width - filled), style="grey35")
    return t


def spark(values, width=40) -> str:
    vals = list(values)[-width:]
    if not vals:
        return ""
    hi = max(vals) or 1.0
    return "".join(SPARK[min(7, int(v / hi * 7.999))] for v in vals)


def fmt_time(sec) -> str:
    if sec is None:
        return "?"
    sec = int(max(0, sec))
    if sec < 60:
        return f"{sec}s"
    m, s = divmod(sec, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


# ---------------- background samplers ----------------
class Sampler:
    """GPU (nvidia-smi), llama-swap /running and llama-server processes, polled off the UI thread."""

    def __init__(self, swap_url: str):
        self.gpu = None
        self.running = None
        self.llama = {"ram": 0, "cpu": 0.0, "n": 0}
        self.swap_url = swap_url.rstrip("/")
        self._procs = {}
        self._smi = shutil.which("nvidia-smi")
        threading.Thread(target=self._loop, daemon=True).start()

    def _poll_gpu(self):
        if not self._smi:
            return
        try:
            out = subprocess.run(
                [self._smi, "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3, creationflags=NO_WINDOW,
            ).stdout.strip().splitlines()[0]
            name, util, used, total, temp, power = [x.strip() for x in out.split(",")]

            def num(x):
                try:
                    return float(x)
                except ValueError:
                    return None
            self.gpu = {"name": name, "util": num(util), "used": num(used), "total": num(total),
                        "temp": num(temp), "power": num(power)}
        except Exception:
            self.gpu = None

    def _poll_swap(self):
        try:
            with urllib.request.urlopen(self.swap_url + "/running", timeout=1) as r:
                data = json.loads(r.read().decode())
            items = data.get("running", data) if isinstance(data, dict) else data
            self.running = [f"{i.get('model', '?')} ({i.get('state', 'ready')})" for i in items if isinstance(i, dict)]
        except Exception:
            self.running = None

    def _poll_llama(self):
        live = {}
        for p in psutil.process_iter(["pid", "name"]):
            if "llama-server" in (p.info["name"] or "").lower():
                live[p.info["pid"]] = self._procs.get(p.info["pid"]) or p
        self._procs = live
        ram, cpu = 0, 0.0
        for p in live.values():
            try:
                ram += p.memory_info().rss
                cpu += p.cpu_percent(None)
            except psutil.Error:
                pass
        self.llama = {"ram": ram, "cpu": cpu / (psutil.cpu_count() or 1), "n": len(live)}

    def _loop(self):
        tick = 0
        while True:
            self._poll_gpu()
            self._poll_llama()
            if tick % 2 == 0:
                self._poll_swap()
            tick += 1
            time.sleep(1.0)


# ---------------- status file ----------------
def read_status():
    try:
        d = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if d.get("state") != "offline" and not psutil.pid_exists(d.get("pid", -1)):
        d["state"] = "offline"
    return d


# ---------------- rendering ----------------
class Monitor:
    def __init__(self, sampler: Sampler):
        self.s = sampler
        self.speed_hist = deque(maxlen=60)       # instantaneous tok/s for the sparkline
        self.model_speed = {}                    # model -> last known average tok/s
        self._prev = (None, 0, 0.0)              # (reply_start, tokens, updated)
        psutil.cpu_percent(None)
        psutil.cpu_percent(None, percpu=True)

    def update_speed(self, st):
        key, tokens, upd = st.get("reply_start"), st.get("tokens", 0), st.get("updated", 0.0)
        pk, ptok, pupd = self._prev
        if key == pk and tokens > ptok and upd > pupd:
            self.speed_hist.append((tokens - ptok) / max(upd - pupd, 1e-3))
        elif key != pk:
            self.speed_hist.clear()
        self._prev = (key, tokens, upd)
        if st.get("tok_s", 0) > 0:
            self.model_speed[st.get("model")] = st["tok_s"]

    def header(self, st):
        label, style = STATE_STYLE.get(st["state"], (st["state"].upper(), "white"))
        t = Text()
        t.append("● ", style=style)
        t.append(label, style=style)
        if st["state"] == "tool" and st.get("tool"):
            t.append(f"  ({st['tool']})", style="blue")
        t.append("\nModel in use: ", style="grey70")
        t.append(str(st.get("model", "?")), style="bold white")
        if self.s.running:
            t.append("\nLoaded in llama-swap: ", style="grey70")
            t.append(", ".join(self.s.running), style="white")
        return t

    def task_block(self, st):
        g = Table.grid(padding=(0, 2))
        g.add_column(style="grey70", no_wrap=True)
        g.add_column()
        now = time.time()
        state = st["state"]
        if st.get("task_start") and state != "offline":
            g.add_row("Task elapsed", f"{fmt_time(now - st['task_start'])}   (step {st.get('step', 0)}/{st.get('max_steps', '?')})")
        elif st.get("last_task_s") is not None:
            g.add_row("Last task took", fmt_time(st["last_task_s"]))

        if st.get("ctx_limit") and st.get("ctx_used") and state != "offline":
            used, limit = st["ctx_used"], st["ctx_limit"]
            ctx = Text()
            ctx.append_text(bar(used / limit * 100, 22))
            ctx.append(f"  {used:,} / {limit:,} tokens")
            if st.get("notes") is not None:
                ctx.append(f"   |  {st['notes']} notes", style="grey62")
            g.add_row("Context", ctx)

        if state in ("waiting", "thinking", "generating") and st.get("reply_start"):
            tokens, expect = st.get("tokens", 0), max(st.get("expect_tokens", 250), 1)
            speed = st.get("tok_s", 0) or self.model_speed.get(st.get("model"), 0)
            now_speed = self.speed_hist[-1] if self.speed_hist else 0
            if state == "waiting":
                est = st.get("prompt_est")
                note = f"   (reading a prompt of ~{est:,} tokens)" if est else ""
                g.add_row("This reply", f"waiting for first token... {fmt_time(now - st['reply_start'])}{note}")
            else:
                pct = min(100.0, tokens / expect * 100)
                over = tokens > expect
                line = Text()
                line.append_text(bar(pct, 22))
                line.append(f"  {tokens} tok" + (" (longer than usual)" if over else f" / ~{expect} typical"))
                g.add_row("This reply", line)
                sp = Text(f"{st.get('tok_s', 0):.1f} tok/s avg", style="bold cyan")
                if now_speed:
                    sp.append(f"   {now_speed:.1f} now", style="grey70")
                g.add_row("Speed", sp)
                if self.speed_hist:
                    g.add_row("", Text(spark(self.speed_hist), style="cyan"))
                if speed > 0:
                    typical = None if over else (expect - tokens) / speed
                    worst = max(st.get("max_tokens", 4096) - tokens, 0) / speed
                    eta = Text()
                    eta.append(f"~{fmt_time(typical)} left" if typical is not None else "finishing (longer than typical)",
                               style="bold green" if typical is not None else "yellow")
                    eta.append(f"   (estimate; hard limit in {fmt_time(worst)})", style="grey62")
                    g.add_row("Time left", eta)
        elif state == "tool":
            g.add_row("Now", f"running tool '{st.get('tool', '')}' - the model resumes when it finishes")
        elif state == "approval":
            g.add_row("Now", "Jas is paused until you answer in the Jas window")

        lr = st.get("last_reply")
        if lr:
            t = Text()
            t.append(f"{lr.get('tokens', 0)} tokens", style="bold white")
            t.append("  |  ")
            t.append(f"{lr.get('tok_s', 0):.1f} tok/s", style="bold cyan")
            t.append(f"  |  {lr.get('model', '?')}", style="grey70")
            if lr.get("duration_s") is not None:
                t.append(f"  |  {fmt_time(lr['duration_s'])}", style="grey70")
            if lr.get("first_token_s") is not None:
                t.append(f"  |  1st token {lr['first_token_s']}s", style="grey50")
            g.add_row("Last request", t)
            if lr.get("prompt_tokens"):
                pt = Text()
                pt.append(f"{lr['prompt_tokens']:,} tokens", style="bold white")
                if lr.get("prompt_cached"):
                    pt.append(f" ({lr.get('prompt_new', 0):,} new, {lr['prompt_cached']:,} cached)", style="grey70")
                if lr.get("prompt_tok_s"):
                    pt.append(f"  |  read at {lr['prompt_tok_s']:.0f} tok/s", style="bold yellow")
                g.add_row("Last prompt", pt)
        return g

    def resources(self):
        g = Table.grid(padding=(0, 2))
        g.add_column(style="grey70", no_wrap=True)
        g.add_column(no_wrap=True)
        g.add_column(no_wrap=True)

        cpu = psutil.cpu_percent(None)
        cores = psutil.cpu_percent(None, percpu=True)
        g.add_row("CPU", bar(cpu), f"{cpu:4.0f}%   {spark(cores, 16)}")
        vm = psutil.virtual_memory()
        g.add_row("RAM", bar(vm.percent), f"{vm.used / 1e9:4.1f} / {vm.total / 1e9:.1f} GB")
        gpu = self.s.gpu
        if gpu and gpu["util"] is not None:
            g.add_row("GPU", bar(gpu["util"]), f"{gpu['util']:4.0f}%   {gpu['temp'] or 0:.0f}°C  {gpu['power'] or 0:.0f} W")
            if gpu["total"]:
                vp = gpu["used"] / gpu["total"] * 100
                g.add_row("VRAM", bar(vp), f"{gpu['used'] / 1024:4.2f} / {gpu['total'] / 1024:.2f} GB")
        else:
            g.add_row("GPU", Text("n/a (nvidia-smi not found)", style="grey50"), "")
        ll = self.s.llama
        if ll["n"]:
            g.add_row("llama-server", Text(f"{ll['n']} process(es)", style="white"),
                      f"{ll['ram'] / 1e9:.1f} GB RAM   {ll['cpu']:.0f}% CPU")
        else:
            g.add_row("llama-server", Text("not running (starts on first request)", style="grey50"), "")
        return g

    def render(self):
        st = read_status() or {"state": "offline", "model": "-"}
        self.update_speed(st) if st["state"] != "offline" else None
        parts = [self.header(st), Text(""), self.task_block(st), Text(""), self.resources()]
        return Panel(Group(*parts), title="[bold cyan]JAS MONITOR[/]", subtitle="[grey50]Ctrl+C to quit[/]",
                     border_style="cyan", padding=(1, 2), expand=False, width=84)


def main():
    global STATUS_FILE
    ap = argparse.ArgumentParser(description="Live monitor for Jas")
    ap.add_argument("--interval", type=float, default=0.5, help="refresh interval in seconds")
    ap.add_argument("--url", default=os.environ.get("JAS_URL", "http://127.0.0.1:8080/v1").removesuffix("/v1"),
                    help="llama-swap base URL (default: from JAS_URL or http://127.0.0.1:8080)")
    ap.add_argument("--file", default=str(Path.home() / ".jas" / "status.json"), help="status file written by jas.py")
    args = ap.parse_args()

    STATUS_FILE = Path(args.file)
    mon = Monitor(Sampler(args.url))
    try:
        with Live(mon.render(), refresh_per_second=8, screen=True) as live:
            while True:
                time.sleep(args.interval)
                live.update(mon.render())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()