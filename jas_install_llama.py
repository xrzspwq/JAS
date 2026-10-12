#!/usr/bin/env python3
"""
jas_install_llama.py - downloads and installs the right llama.cpp build for YOUR graphics card (Windows).

    python jas_install_llama.py --check          # only report: GPU, driver, CUDA/Vulkan/etc. status
    python jas_install_llama.py                  # detect, download, install, verify, patch your llama-swap config
    python jas_install_llama.py --backend vulkan # force a backend (auto|cuda|vulkan|sycl|rocm|cpu)
    python jas_install_llama.py --cuda 12        # NVIDIA: force the CUDA 12 build instead of 13
    python jas_install_llama.py --tag b11429     # pin a llama.cpp build instead of the newest
    python jas_install_llama.py --config C:/path/llama-swap.config.yaml   # config to patch (auto-detected otherwise)

What it does
  * NVIDIA -> CUDA build (+ the matching cudart zip: the CUDA runtime is BUNDLED, you do not need the CUDA
              Toolkit, only a recent NVIDIA driver).  Falls back to Vulkan if CUDA does not work.
  * AMD    -> Vulkan build (needs only the normal AMD driver).  ROCm is available with --backend rocm.
  * Intel  -> SYCL if oneAPI is installed, otherwise Vulkan.
  * other  -> Vulkan, then CPU.
  After installing, it runs `llama-server --list-devices` to PROVE the GPU backend works, and only then
  points your llama-swap config at it.  No pip packages needed (standard library only).

Install location: <this folder>/llama.cpp/current/llama-server.exe   (zips are cached in llama.cpp/downloads)
"""
import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime
from html import unescape
from pathlib import Path
from urllib.parse import unquote

REPO = "ggml-org/llama.cpp"
HERE = Path(__file__).resolve().parent
NO_WINDOW = 0x08000000 if os.name == "nt" else 0
UA = {"User-Agent": "jas-installer"}
API_HEADERS = {**UA, "Accept": "application/vnd.github+json"}
if os.environ.get("GITHUB_TOKEN"):
    API_HEADERS["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"

# backend key -> (regex for the main zip, with {tag}/{arch} filled in, family used for verification)
BACKENDS = {
    "cuda13": (r"llama-{tag}-bin-win-cuda-13\.\d+-{arch}\.zip", "cuda"),
    "cuda12": (r"llama-{tag}-bin-win-cuda-12\.\d+-{arch}\.zip", "cuda"),
    "vulkan": (r"llama-{tag}-bin-win-vulkan-{arch}\.zip", "vulkan"),
    "sycl": (r"llama-{tag}-bin-win-sycl-{arch}\.zip", "sycl"),
    "rocm": (r"llama-{tag}-bin-win-rocm-[\d.]+-{arch}\.zip", "rocm"),
    "cpu": (r"llama-{tag}-bin-win-cpu-{arch}\.zip", "cpu"),
}
DEVICE_RX = {"cuda": r"CUDA\d*\s*:", "vulkan": r"Vulkan\d*\s*:", "sycl": r"SYCL\d*\s*:",
             "rocm": r"(ROCm|HIP|CUDA)\d*\s*:", "cpu": None}


# ------------------------------------------------------------------ small helpers
def run(cmd, timeout=20):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, creationflags=NO_WINDOW,
                           encoding="utf-8", errors="replace")
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, str(e)


def http_get(url, headers=UA, timeout=30) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
        return r.read()


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def natural_key(s: str):
    return [int(x) for x in re.findall(r"\d+", s)]


# ------------------------------------------------------------------ system detection
def vendor_of(name: str, compat: str) -> str:
    s = f"{name} {compat}".lower()
    if any(k in s for k in ("nvidia", "geforce", "quadro")):
        return "nvidia"
    if any(k in s for k in ("amd", "radeon", "advanced micro")):
        return "amd"
    if "intel" in s:
        return "intel"
    if any(k in s for k in ("qualcomm", "adreno")):
        return "qualcomm"
    return "other"


def find_nvidia_smi():
    for c in (shutil.which("nvidia-smi"),
              r"C:\Windows\System32\nvidia-smi.exe",
              r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"):
        if c and Path(c).exists():
            return c
    return None


def detect_system() -> dict:
    info = {"arch": "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64",
            "gpus": [], "nvidia": None, "vulkan_loader": False, "cuda_toolkit": None,
            "hip": bool(os.environ.get("HIP_PATH")), "oneapi": bool(os.environ.get("ONEAPI_ROOT"))}
    ps = ("Get-CimInstance Win32_VideoController | "
          "Select-Object Name,AdapterCompatibility,DriverVersion | ConvertTo-Json -Compress")
    rc, out = run(["powershell", "-NoProfile", "-Command", ps], timeout=30)
    try:
        data = json.loads(out)
        for g in ([data] if isinstance(data, dict) else data):
            info["gpus"].append({"name": g.get("Name") or "?", "driver": g.get("DriverVersion") or "?",
                                 "vendor": vendor_of(g.get("Name") or "", g.get("AdapterCompatibility") or "")})
    except (ValueError, TypeError, AttributeError):
        pass

    smi = find_nvidia_smi()
    if smi:
        text = ""
        for args in ([smi], [smi, "-q"]):          # plain header, then the detailed report
            text += "\n" + run(args, timeout=30)[1]
        m_cuda = re.search(r"CUDA Version\s*:?\s*(\d+)\.(\d+)", text)
        m_drv = re.search(r"Driver Version\s*:?\s*(\d+\.\d+)", text)
        driver = m_drv.group(1) if m_drv else None
        if not driver:
            rc, dv = run([smi, "--query-gpu=driver_version", "--format=csv,noheader"])
            driver = dv.strip().splitlines()[0].strip() if rc == 0 and dv.strip() else None
        if not driver:                              # Windows "32.0.16.1742" -> NVIDIA "617.42"
            for g in info["gpus"]:
                if g["vendor"] == "nvidia":
                    digits = "".join(g["driver"].split(".")[-2:])[-5:]
                    driver = f"{digits[:3]}.{digits[3:]}" if len(digits) == 5 else None
        nv = {"smi": smi, "driver": driver or "?",
              "cuda": (int(m_cuda.group(1)), int(m_cuda.group(2))) if m_cuda else None, "cc": None}
        rc, out = run([smi, "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"])
        if rc == 0 and out.strip():
            parts = [p.strip() for p in out.strip().splitlines()[0].split(",")]
            try:
                nv["cc"] = float(parts[1])
            except (IndexError, ValueError):
                pass
        info["nvidia"] = nv
    sysroot = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    info["vulkan_loader"] = (sysroot / "System32" / "vulkan-1.dll").exists()
    info["cuda_toolkit"] = os.environ.get("CUDA_PATH") or shutil.which("nvcc")
    return info


def print_report(info: dict):
    print("== System check ==")
    print(f"CPU architecture : {info['arch']}")
    if info["gpus"]:
        for g in info["gpus"]:
            print(f"Graphics adapter : {g['name']}  [{g['vendor']}]  driver {g['driver']}")
    else:
        print("Graphics adapter : none detected")
    vendors = {g["vendor"] for g in info["gpus"]}
    nv = info["nvidia"]
    if "nvidia" in vendors or nv:
        if nv:
            cc = f", compute capability {nv['cc']}" if nv["cc"] else ""
            if nv["cuda"]:
                print(f"NVIDIA driver    : OK ({nv['driver']}), supports CUDA up to {nv['cuda'][0]}.{nv['cuda'][1]}{cc}")
            else:
                print(f"NVIDIA driver    : found ({nv['driver']}){cc}. Could not read its CUDA version from nvidia-smi;")
                print("                   CUDA builds will be tried anyway and verified against your GPU.")
        else:
            print("NVIDIA driver    : NOT FOUND (nvidia-smi missing). Install the driver from nvidia.com/drivers")
        print("CUDA Toolkit     : " + (f"found at {info['cuda_toolkit']} (not required)" if info["cuda_toolkit"]
                                       else "not installed - that is fine, the llama.cpp CUDA download bundles the runtime"))
    if "amd" in vendors:
        print("AMD ROCm/HIP SDK : " + ("found (HIP_PATH set)" if info["hip"] else "not found (Vulkan will be used; ROCm is optional)"))
    if "intel" in vendors:
        print("Intel oneAPI     : " + ("found" if info["oneapi"] else "not found (Vulkan will be used)"))
    print("Vulkan loader    : " + ("found" if info["vulkan_loader"] else "NOT FOUND (update your graphics driver)"))


def make_plan(info: dict, force_backend: str = "auto", force_cuda: str = None) -> list:
    """Ordered list of backends to try; the first one that installs AND verifies wins."""
    vendors = [g["vendor"] for g in info["gpus"]]
    nv = info["nvidia"]
    cuda = []
    if nv or "nvidia" in vendors:
        cc_ok_13 = not (nv and nv["cc"]) or nv["cc"] >= 7.5     # CUDA 13 dropped Maxwell/Pascal/Volta
        if nv and nv["cuda"]:
            major = nv["cuda"][0]
            if major >= 13 and cc_ok_13:
                cuda.append("cuda13")
            if major >= 12:
                cuda.append("cuda12")
        else:                                                   # version unknown: try both, verification decides
            cuda = (["cuda13"] if cc_ok_13 else []) + ["cuda12"]
    if force_cuda:
        cuda = [c for c in (f"cuda{force_cuda}",) if f"cuda{force_cuda}" in BACKENDS] or cuda

    if force_backend and force_backend != "auto":
        if force_backend == "cuda":
            return cuda or ["cuda12"]
        return [force_backend]

    plan = []
    if "nvidia" in vendors or nv:
        plan += cuda
        if info["vulkan_loader"]:
            plan.append("vulkan")
    elif "amd" in vendors:
        plan.append("vulkan")
    elif "intel" in vendors:
        plan += (["sycl"] if info["oneapi"] else []) + ["vulkan"]
    else:
        plan.append("vulkan")
    plan.append("cpu")
    return list(dict.fromkeys(plan))


# ------------------------------------------------------------------ finding releases / assets
def parse_assets_html(html: str) -> dict:
    assets = {}
    for row in html.split("<li ")[1:]:
        m = re.search(r'href="([^"]*/releases/download/[^"]+)"', row)
        if not m:
            continue
        url = unescape(m.group(1))
        url = "https://github.com" + url if url.startswith("/") else url
        sha = re.search(r"sha256:([0-9a-f]{64})", row)
        assets[unquote(url.rsplit("/", 1)[-1])] = {"url": url, "sha256": sha.group(1) if sha else None}
    return assets


def html_assets(tag: str) -> dict:
    return parse_assets_html(http_get(f"https://github.com/{REPO}/releases/expanded_assets/{tag}").decode("utf-8", "replace"))


class Releases:
    """Newest-first (tag, assets) pairs. Uses the GitHub API, falls back to scraping the releases page."""

    def __init__(self, pinned=None):
        self.pinned, self.seen, self._gen = pinned, [], self._make()

    def _make(self):
        if self.pinned:
            yield self.pinned, html_assets(self.pinned)
            return
        try:
            items = json.loads(http_get(f"https://api.github.com/repos/{REPO}/releases?per_page=20", API_HEADERS))
            got = False
            for rel in items:
                tag = rel.get("tag_name", "")
                if not re.fullmatch(r"b\d+", tag):
                    continue
                assets = {a["name"]: {"url": a["browser_download_url"],
                                      "sha256": (a.get("digest") or "").replace("sha256:", "") or None}
                          for a in rel.get("assets", [])}
                got = True
                yield tag, assets
            if got:
                return
        except (urllib.error.URLError, ValueError, KeyError, TypeError):
            pass
        html = http_get(f"https://github.com/{REPO}/releases").decode("utf-8", "replace")
        tags = sorted({int(t) for t in re.findall(r"/releases/tag/b(\d+)", html)}, reverse=True)[:12]
        for t in tags:
            yield f"b{t}", html_assets(f"b{t}")

    def __iter__(self):
        i = 0
        while True:
            if i < len(self.seen):
                yield self.seen[i]
                i += 1
                continue
            try:
                self.seen.append(next(self._gen))
            except StopIteration:
                return


def find_assets(tag: str, assets: dict, backend: str, arch: str):
    """-> (main_name, cudart_name_or_None) if this release has everything needed for the backend."""
    pattern, family = BACKENDS[backend]
    rx = re.compile(pattern.format(tag=re.escape(tag), arch=arch))
    hits = sorted((n for n in assets if rx.fullmatch(n)), key=natural_key)
    if not hits:
        return None
    main = hits[-1]
    if family != "cuda":
        return main, None
    ver = re.search(r"cuda-(\d+\.\d+)", main).group(1)
    cudart = f"cudart-llama-bin-win-cuda-{ver}-{arch}.zip"
    return (main, cudart) if cudart in assets else None


# ------------------------------------------------------------------ download / install / verify
def download(url: str, dest: Path, sha256: str = None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (not sha256 or sha256_of(dest) == sha256):
        print(f"  {dest.name}: already downloaded")
        return
    part = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r, open(part, "wb") as f:
        total, done, t0 = int(r.headers.get("Content-Length") or 0), 0, time.time()
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
            done += len(block)
            pct = f"{done * 100 // total}%" if total else ""
            print(f"\r  {dest.name}: {done / 1e6:6.0f}/{total / 1e6:.0f} MB {pct}  {done / 1e6 / max(time.time() - t0, 1e-3):.0f} MB/s ",
                  end="", flush=True)
    print()
    if sha256 and sha256_of(part) != sha256:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {dest.name} (corrupt download)")
    os.replace(part, dest)


def llama_running() -> bool:
    if os.name != "nt":
        return False
    rc, out = run(["tasklist", "/FI", "IMAGENAME eq llama-server.exe"])
    return "llama-server.exe" in out.lower()


def install(root: Path, main_zip: Path, cudart_zip: Path = None) -> Path:
    tmp, cur = root / "current.new", root / "current"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(main_zip) as z:
        z.extractall(tmp)
    exe = next(tmp.rglob("llama-server.exe"), None)
    if exe is None:
        raise RuntimeError("llama-server.exe not found inside the zip")
    if exe.parent != tmp:                      # flatten if the zip had a sub-folder
        for item in list(exe.parent.iterdir()):
            shutil.move(str(item), str(tmp / item.name))
    if cudart_zip:
        scratch = root / "cudart.tmp"
        shutil.rmtree(scratch, ignore_errors=True)
        with zipfile.ZipFile(cudart_zip) as z:
            z.extractall(scratch)
        for dll in scratch.rglob("*.dll"):
            shutil.copy2(dll, tmp / dll.name)
        shutil.rmtree(scratch, ignore_errors=True)
    if cur.exists():
        try:
            shutil.rmtree(cur)
        except PermissionError:
            raise RuntimeError("could not replace the old install: stop llama-swap / llama-server first")
    tmp.rename(cur)
    return cur / "llama-server.exe"


def verify(exe: Path, backend: str):
    """Run llama-server --list-devices. -> (ok, device_lines, raw_output)."""
    family = BACKENDS[backend][1]
    rx = DEVICE_RX[family]
    rc, out = run([str(exe), "--list-devices" if rx else "--version"], timeout=90)
    devices = [l.strip() for l in out.splitlines() if re.match(r"^\s*[A-Za-z]+\d+\s*:\s", l)]
    ok = rc == 0 and (rx is None or re.search(rx, out) is not None)
    return ok, devices, out


# ------------------------------------------------------------------ llama-swap config patching
CMD_RE = re.compile(r"^(\s*cmd:\s*)(.*?)\s*$")
TOKEN_RE = re.compile(r'^(?:"[^"]*"|\S+)')


def patch_cmd_line(line: str, exe: str):
    """Point the program at the start of a llama-swap `cmd:` line to `exe`. Returns the new line or None."""
    m = CMD_RE.match(line)
    if not m:
        return None
    prefix, val = m.groups()
    if val.startswith('"'):
        return None
    wrapped = len(val) >= 2 and val[0] == "'" and val[-1] == "'"
    inner = val[1:-1].replace("''", "'") if wrapped else val
    t = TOKEN_RE.match(inner)
    if not t:
        return None
    base = t.group(0).strip('"').replace("\\", "/").rsplit("/", 1)[-1].lower()
    if base not in ("llama-server", "llama-server.exe"):
        return None
    new_inner = (f'"{exe}"' if " " in exe else exe) + inner[t.end():]
    new = prefix + ("'" + new_inner.replace("'", "''") + "'" if (wrapped or " " in exe) else new_inner)
    return None if new == line.rstrip("\n") else new


def patch_config(path: Path, exe: str) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    changed = 0
    for i, line in enumerate(lines):
        new = patch_cmd_line(line, exe)
        if new is not None:
            lines[i] = new
            changed += 1
    if changed:
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return changed


def find_configs(explicit):
    if explicit:
        return [Path(explicit)]
    names = ("llama-swap.config.yaml", "llama-swap_config.yaml", "llama-swap.yaml", "config.yaml")
    dirs = [Path.cwd() / "models", Path.cwd(), HERE / "models", HERE]
    found = []
    for d in dirs:
        for n in names:
            p = d / n
            if p.exists() and p.resolve() not in [f.resolve() for f in found]:
                if "models:" in p.read_text(encoding="utf-8", errors="replace"):
                    found.append(p)
    return found


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Install the right llama.cpp for your GPU")
    ap.add_argument("--check", action="store_true", help="only report GPU / driver / backend status")
    ap.add_argument("--backend", default="auto", choices=["auto", "cuda", "vulkan", "sycl", "rocm", "cpu"])
    ap.add_argument("--cuda", choices=["12", "13"], help="NVIDIA: force the CUDA 12 or 13 build")
    ap.add_argument("--tag", help="pin a llama.cpp build tag such as b11429 (default: newest complete build)")
    ap.add_argument("--dir", default=str(HERE / "llama.cpp"), help="install folder")
    ap.add_argument("--config", help="llama-swap config to patch (default: auto-detect)")
    ap.add_argument("--yes", action="store_true", help="don't ask before downloading")
    args = ap.parse_args()

    if sys.platform != "win32" and not args.check:
        sys.exit("This installer is for Windows (it downloads the win-x64 builds).")

    info = detect_system()
    print_report(info)
    plan = make_plan(info, args.backend, args.cuda)
    print(f"\nBackend plan     : {' -> '.join(plan)}   (next one is tried if the previous fails to verify)")
    if args.check:
        return
    if llama_running():
        sys.exit("\nllama-server is running. Stop llama-swap (Ctrl+C in its window) and run this again.")

    root = Path(args.dir)
    releases = Releases(args.tag)
    for backend in plan:
        choice = None
        try:
            for tag, assets in releases:
                found = find_assets(tag, assets, backend, info["arch"])
                if found:
                    choice = (tag, assets, *found)
                    break
        except (urllib.error.URLError, OSError) as e:
            sys.exit(f"\nCould not reach GitHub: {e}")
        if not choice:
            print(f"\n[{backend}] no recent llama.cpp release has this build for {info['arch']}; skipping.")
            continue
        tag, assets, main_name, cudart_name = choice
        print(f"\n== {backend}: llama.cpp {tag} ==\n  {main_name}" + (f"\n  {cudart_name}  (CUDA runtime)" if cudart_name else ""))
        if not args.yes and input("Download and install? [Y/n] ").strip().lower() in ("n", "no"):
            sys.exit("Cancelled.")
        try:
            dl = root / "downloads"
            download(assets[main_name]["url"], dl / main_name, assets[main_name]["sha256"])
            if cudart_name:
                download(assets[cudart_name]["url"], dl / cudart_name, assets[cudart_name]["sha256"])
            exe = install(root, dl / main_name, (dl / cudart_name) if cudart_name else None)
        except (RuntimeError, OSError, urllib.error.URLError) as e:
            print(f"  install failed: {e}")
            continue
        print("  verifying (llama-server --list-devices)...")
        ok, devices, out = verify(exe, backend)
        for d in devices:
            print(f"    {d}")
        if not ok:
            print(f"  [{backend}] did NOT verify. Output tail:")
            print("    " + "\n    ".join(out.strip().splitlines()[-8:]))
            if "0xc0000135" in out.lower() or "3221225781" in out:
                print("    (a DLL is missing - install the Microsoft Visual C++ Redistributable 2015-2022 x64)")
            continue

        exe_posix = exe.as_posix()
        (root / "install.json").write_text(json.dumps({
            "server": exe_posix, "tag": tag, "backend": backend, "devices": devices,
            "installed": datetime.now().isoformat(timespec="seconds")}, indent=2), encoding="utf-8")
        print(f"\nOK: llama.cpp {tag} ({backend}) installed -> {exe_posix}")
        configs = find_configs(args.config)
        for cfg in configs:
            n = patch_config(cfg, exe_posix)
            print(f"Config {cfg}: {n} command(s) updated" + (f" (backup: {cfg.name}.bak)" if n else " (already up to date)"))
        if not configs:
            print("No llama-swap config found. Use this program path in every `cmd:` line:\n  " + exe_posix)
        print("\nNext: restart llama-swap, then re-tune VRAM (the CUDA build uses memory differently):")
        print("  llama-swap --config <your config> --listen 127.0.0.1:8080")
        return
    sys.exit("\nNo backend could be installed and verified. Run with --check and send me the report.")


if __name__ == "__main__":
    main()