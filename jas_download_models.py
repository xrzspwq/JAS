#!/usr/bin/env python3
"""
jas_download_models.py
Downloads the GGUF models for the "Jas" local agent (abliterated/uncensored,
preferring imatrix "i1" quants for better quality per bit) and generates a
llama-swap config that wires them together.

Usage (Windows PowerShell):
    pip install -U huggingface_hub hf_transfer
    python jas_download_models.py --list                 # preview, download nothing
    python jas_download_models.py                        # download everything
    python jas_download_models.py --quality high         # Q5_K_M for the 30B models (~22 GB each)
    python jas_download_models.py --only fast main       # subset
    python jas_download_models.py --with-optional        # also gpt-oss-20b
    python jas_download_models.py --dir D:/jas/models --yes

Repos are tried in order; if none work, the Hub is searched (imatrix repos
preferred). ALWAYS read the model card of whatever got picked.
Set HF_TOKEN in your environment if you hit rate limits.
"""
import argparse
import os
import re
import shutil
import sys
from pathlib import Path

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    sys.exit("Missing dependency. Run:  pip install -U huggingface_hub hf_transfer")

Q4 = ["Q4_K_M", "Q4_K_S", "IQ4_XS", "Q4_0"]
MOE_FLAGS = "-ngl 99 --n-cpu-moe 43 -c 16384 -fa on --cache-type-k q8_0 --cache-type-v q8_0 -t 6 --jinja"

MODELS = {
    "fast": {
        "desc": "Router / quick replies (CPU, always loaded; high-quality quant is cheap at 4B)",
        "repos": [
            "mradermacher/Huihui-Qwen3-4B-Instruct-2507-abliterated-i1-GGUF",
            "mradermacher/Huihui-Qwen3-4B-Instruct-2507-abliterated-GGUF",
            "huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated-GGUF",
        ],
        "search": "Huihui Qwen3 4B Instruct 2507 abliterated GGUF",
        "quants": ["Q6_K", "Q5_K_M"] + Q4,
        "flags": "-ngl 0 -t 2 -c 8192 --jinja",
    },
    "main": {
        "desc": "Orchestrator, PC actions, tool calling (MoE, experts in RAM)",
        "moe": True,
        "repos": [
            "mradermacher/Huihui-Qwen3-30B-A3B-Instruct-2507-abliterated-i1-GGUF",
            "mradermacher/Huihui-Qwen3-30B-A3B-Instruct-2507-abliterated-GGUF",
            "Sowkwndms/Huihui-Qwen3-30B-A3B-Instruct-2507-abliterated-Q4_K_M-GGUF",
        ],
        "search": "Huihui Qwen3 30B A3B Instruct 2507 abliterated GGUF",
        "quants": Q4,
        "flags": MOE_FLAGS,
    },
    "coder": {
        "desc": "Coding specialist (MoE)",
        "moe": True,
        "repos": [
            "mradermacher/Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated-i1-GGUF",
            "mradermacher/Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated-GGUF",
            "huihui-ai/Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated-GGUF",
        ],
        "search": "Huihui Qwen3 Coder 30B A3B Instruct abliterated GGUF",
        "quants": Q4,
        "flags": MOE_FLAGS,
    },
    "thinker": {
        "desc": "Research synthesis / hard reasoning (MoE, thinking)",
        "moe": True,
        "repos": [
            "mradermacher/Huihui-Qwen3-30B-A3B-Thinking-2507-abliterated-i1-GGUF",
            "mradermacher/Huihui-Qwen3-30B-A3B-Thinking-2507-abliterated-GGUF",
            "huihui-ai/Huihui-Qwen3-30B-A3B-Thinking-2507-abliterated-GGUF",
        ],
        "search": "Huihui Qwen3 30B A3B Thinking 2507 abliterated GGUF",
        "quants": Q4,
        "flags": MOE_FLAGS,
    },
    "thinker-lite": {
        "desc": "OPTIONAL lighter reasoning model (gpt-oss-20b abliterated, ~12 GB)",
        "optional": True,
        "repos": ["drwhall/gpt-oss-20b-abliterated-gguf"],
        "search": "gpt-oss-20b abliterated GGUF",
        "quants": ["MXFP4", "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q8_0"],
        "flags": "-ngl 99 --n-cpu-moe 24 -c 16384 -fa on -t 6 --jinja",
    },
    "embed": {
        "desc": "Embeddings for long-term memory (CPU)",
        "repos": ["Qwen/Qwen3-Embedding-0.6B-GGUF"],
        "search": "Qwen3 Embedding 0.6B GGUF",
        "quants": ["Q8_0", "Q4_K_M"],
        "flags": "-ngl 0 -t 2 --embeddings -c 8192",
    },
}

SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)


def pick_files(api, repo, quants):
    """Return the GGUF file path(s) (all shards) to download from a repo, or None."""
    try:
        files = api.list_repo_files(repo)
    except Exception:
        return None
    ggufs = [f for f in files if f.lower().endswith(".gguf")
             and "mmproj" not in f.lower() and "imatrix" not in f.lower()]
    if not ggufs:
        return None
    for q in quants:
        matches = sorted(f for f in ggufs if q.lower() in f.lower())
        if not matches:
            continue
        first = matches[0]
        if SHARD_RE.search(first):
            base = SHARD_RE.sub("", first)
            return sorted(f for f in matches if SHARD_RE.sub("", f) == base)
        return [first]
    if len(ggufs) == 1:  # single-quant repos sometimes omit the tag in the filename
        return ggufs
    return None


def resolve(api, name, spec):
    for repo in spec["repos"]:
        files = pick_files(api, repo, spec["quants"])
        if files:
            return repo, files
    print(f"  [{name}] known repos failed, searching the Hub...")
    try:
        results = list(api.list_models(search=spec["search"], sort="downloads", limit=25))
        results.sort(key=lambda m: not any(t in m.id.lower() for t in ("i1", "imatrix", "unsloth")))
        for m in results:
            if "gguf" not in m.id.lower():
                continue
            files = pick_files(api, m.id, spec["quants"])
            if files:
                return m.id, files
    except Exception as e:
        print(f"  search failed: {e}")
    return None, None


def total_size(api, repo, files):
    try:
        return sum((i.size or 0) for i in api.get_paths_info(repo, files))
    except Exception:
        return 0


def gb(n):
    return f"{n / 1e9:.1f} GB" if n else "? GB"


def write_config(out_dir, results):
    """Generate a llama-swap config.yaml for whatever was downloaded."""
    lines = ["# Generated by jas_download_models.py", "models:"]
    for name, info in results.items():
        lines.append(f'  "{name}":')
        lines.append(f'    cmd: llama-server --port ${{PORT}} -m "{info["path"].as_posix()}" {MODELS[name]["flags"]}')
        if name not in ("fast", "embed"):
            lines.append("    ttl: 900")
    small = [n for n in results if n in ("fast", "embed")]
    big = [n for n in results if n not in ("fast", "embed")]
    lines += ["", "groups:"]
    if small:
        lines += ["  resident:", "    swap: false", "    exclusive: false",
                  "    members: [" + ", ".join(f'"{n}"' for n in small) + "]"]
    if big:
        lines += ["  heavy:", "    swap: true", "    exclusive: false",
                  "    members: [" + ", ".join(f'"{n}"' for n in big) + "]"]
    cfg = out_dir / "llama-swap.config.yaml"
    cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return cfg


def main():
    ap = argparse.ArgumentParser(description="Download Jas models")
    ap.add_argument("--dir", default=str(Path.cwd() / "models"))
    ap.add_argument("--only", nargs="*", help="roles to fetch (default: all non-optional)")
    ap.add_argument("--with-optional", action="store_true")
    ap.add_argument("--quality", choices=["balanced", "high"], default="balanced",
                    help="high = Q5_K_M for the 30B MoE models (~22 GB each, slower, needs more free RAM)")
    ap.add_argument("--list", action="store_true", help="resolve and show the plan only")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    if args.quality == "high":
        for spec in MODELS.values():
            if spec.get("moe"):
                spec["quants"] = ["Q5_K_M", "Q5_K_S"] + Q4

    out_dir = Path(args.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi()

    wanted = args.only or [n for n, s in MODELS.items() if args.with_optional or not s.get("optional")]
    bad = [n for n in wanted if n not in MODELS]
    if bad:
        sys.exit(f"Unknown role(s): {bad}. Valid: {list(MODELS)}")

    print(f"Resolving {len(wanted)} model(s)...\n")
    plan, grand_total = {}, 0
    for name in wanted:
        spec = MODELS[name]
        repo, files = resolve(api, name, spec)
        if not repo:
            print(f"  [{name}] NOT FOUND. Search the Hub manually and edit MODELS in this script.")
            continue
        size = total_size(api, repo, files)
        grand_total += size
        plan[name] = {"repo": repo, "files": files}
        print(f"  [{name}] {spec['desc']}\n      {repo}\n      {', '.join(files)}  ({gb(size)})")

    if not plan:
        sys.exit("Nothing to download.")

    free = shutil.disk_usage(out_dir).free
    print(f"\nTotal: {gb(grand_total)}   Free disk: {gb(free)}   Target: {out_dir}")
    if grand_total and grand_total > free * 0.95:
        sys.exit("Not enough free disk space.")
    if args.list:
        return
    if not args.yes and input("Download now? [y/N] ").strip().lower() != "y":
        return

    results = {}
    for name, p in plan.items():
        print(f"\n== Downloading {name} ==")
        local = None
        for f in p["files"]:
            try:
                path = hf_hub_download(repo_id=p["repo"], filename=f, local_dir=str(out_dir))
            except Exception as e:
                print(f"  FAILED {f}: {e}")
                local = None
                break
            if local is None:  # llama-server needs the FIRST shard
                local = Path(path)
        if local:
            results[name] = {"path": local, "repo": p["repo"]}

    if results:
        cfg = write_config(out_dir, results)
        print("\nDone. Config written to:", cfg)
        print("\nModels used (read each model card):")
        for name, r in results.items():
            print(f"  {name:13s} https://huggingface.co/{r['repo']}")
        print("\nNext: llama-swap --config", cfg, "--listen 127.0.0.1:8080")


if __name__ == "__main__":
    main()