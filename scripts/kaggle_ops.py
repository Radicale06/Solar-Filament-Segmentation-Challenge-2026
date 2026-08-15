"""One-command Kaggle operations: code upload, kernel push, monitoring, submission.

The full automated loop (no Kaggle UI needed):

    python scripts/kaggle_ops.py whoami                      # resolve username
    python scripts/kaggle_ops.py push-code                   # src+scripts+configs -> private dataset filseg-code
    python scripts/kaggle_ops.py push-kernel [--max-hours H] [--resume] [--fold N]
                                                             # push notebook; Kaggle runs it automatically
    python scripts/kaggle_ops.py status                      # kernel run state
    python scripts/kaggle_ops.py fetch [-p DIR]              # download output (checkpoints + submission.csv)
    python scripts/kaggle_ops.py push-ckpt -p DIR            # fetched checkpoints -> private dataset filseg-ckpt
    python scripts/kaggle_ops.py submit -f FILE -m MSG       # submit a CSV to the competition

Resume flow across sessions: fetch -> push-ckpt -> push-kernel --resume
(--resume attaches filseg-ckpt; the notebook auto-detects it and continues training).

Auth: reads KAGGLE_API_TOKEN from the repo .env (never printed).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STAGE = REPO / "data" / "kaggle_stage"
COMP = "filament-segmentation-2026"
CODE_SLUG = "filseg-code"
CKPT_SLUG = "filseg-ckpt"
KERNEL_SLUG = "filseg-train"
SWEEP_SLUG = "filseg-sweep"

sys.path.insert(0, str(REPO / "src"))


def _env() -> dict:
    import os

    from dotenv import load_dotenv

    load_dotenv(REPO / ".env")
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"          # kernel logs contain unicode; Windows default cp1252 breaks the CLI
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def kaggle(*args: str, capture: bool = True) -> str:
    exe = REPO / "venv" / "Scripts" / "kaggle.exe"
    cmd = [str(exe)] + list(args)
    r = subprocess.run(cmd, capture_output=capture, text=True, env=_env(),
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} failed:\n{(r.stderr or r.stdout or '')[:2000]}")
    return r.stdout or ""


def whoami() -> str:
    cache = STAGE / "username.txt"
    if cache.exists():
        return cache.read_text().strip()
    out = kaggle("config", "view")
    m = re.search(r"username[:=]\s*([A-Za-z0-9_-]+)", out)
    if not m or m.group(1).lower() in ("none", "null"):
        # fallback: parse the ref of any of our kernels/datasets
        out = kaggle("kernels", "list", "-m", "--page-size", "1", "--csv")
        lines = [ln for ln in out.strip().splitlines() if "/" in ln]
        if not lines:
            raise RuntimeError("could not resolve Kaggle username; run any kernel/dataset once or set it manually in data/kaggle_stage/username.txt")
        m = re.match(r"([A-Za-z0-9_-]+)/", lines[-1])
    user = m.group(1)
    STAGE.mkdir(parents=True, exist_ok=True)
    cache.write_text(user)
    return user


def _copy_code(dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for sub in ("src", "scripts", "configs"):
        shutil.copytree(REPO / sub, dst / sub,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def _dataset_exists(user: str, slug: str) -> bool:
    try:
        out = kaggle("datasets", "list", "-m", "-s", slug, "--csv")
    except RuntimeError:
        return False
    return f"{user}/{slug}" in out


def push_dataset(user: str, slug: str, title: str, fill: callable, message: str) -> None:
    d = STAGE / slug
    fill(d)
    meta = {"title": title, "id": f"{user}/{slug}", "licenses": [{"name": "CC0-1.0"}]}
    (d / "dataset-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    if _dataset_exists(user, slug):
        print(kaggle("datasets", "version", "-p", str(d), "-m", message, "--dir-mode", "zip").strip())
    else:
        print(kaggle("datasets", "create", "-p", str(d), "--dir-mode", "zip").strip())
    print(f"dataset {user}/{slug} pushed")


def push_code(user: str) -> None:
    push_dataset(user, CODE_SLUG, CODE_SLUG, _copy_code, "code update")


def push_ckpt(user: str, src_dir: Path, best_only: bool = False) -> None:
    ck = src_dir / "checkpoints"
    if not ck.exists():
        raise SystemExit(f"no checkpoints/ under {src_dir}")

    def fill(d: Path) -> None:
        if d.exists():
            shutil.rmtree(d)
        if best_only:
            for f in ck.rglob("best.pth"):
                dest = d / "checkpoints" / f.relative_to(ck)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
        else:
            shutil.copytree(ck, d / "checkpoints")

    push_dataset(user, CKPT_SLUG, CKPT_SLUG, fill, "checkpoint update")


def _wait_dataset_ready(user: str, slug: str, timeout_s: int = 600) -> None:
    """A kernel pushed while its dataset is still processing starts without the
    mount — block until Kaggle reports the dataset ready."""
    import time

    t0 = time.time()
    while True:
        st = kaggle("datasets", "status", f"{user}/{slug}").strip().lower()
        if "ready" in st:
            return
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"dataset {user}/{slug} not ready after {timeout_s}s (status: {st})")
        print(f"  waiting for dataset {slug} ({st})...")
        time.sleep(15)


def push_kernel(user: str, fold: int, max_hours: float, resume: bool) -> None:
    _wait_dataset_ready(user, CODE_SLUG)
    if resume:
        _wait_dataset_ready(user, CKPT_SLUG)
    d = STAGE / "kernel"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    nb_src = (REPO / "notebooks" / "kaggle_train.ipynb").read_text(encoding="utf-8")
    nb_src = nb_src.replace("MAX_HOURS = 10.5", f"MAX_HOURS = {max_hours}")
    nb_src = nb_src.replace("FOLD = 0", f"FOLD = {fold}")
    (d / "filseg-train.ipynb").write_text(nb_src, encoding="utf-8")

    sources = [f"{user}/{CODE_SLUG}"]
    if resume:
        sources.append(f"{user}/{CKPT_SLUG}")
    meta = {
        "id": f"{user}/{KERNEL_SLUG}",
        "title": KERNEL_SLUG,
        "code_file": "filseg-train.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        # P100s (cuda capability 6.0) are unsupported by the image's torch build
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": True,
        "dataset_sources": sources,
        "competition_sources": [COMP],
        "kernel_sources": [],
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(kaggle("kernels", "push", "-p", str(d)).strip())
    print(f"kernel {user}/{KERNEL_SLUG} pushed (fold={fold}, max_hours={max_hours}, resume={resume}) — Kaggle is now running it")


def push_sweep(user: str, fold: int) -> None:
    _wait_dataset_ready(user, CODE_SLUG)
    _wait_dataset_ready(user, CKPT_SLUG)
    d = STAGE / "sweep_kernel"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    nb_src = (REPO / "notebooks" / "kaggle_sweep.ipynb").read_text(encoding="utf-8")
    nb_src = nb_src.replace("FOLD = 0", f"FOLD = {fold}")
    (d / "filseg-sweep.ipynb").write_text(nb_src, encoding="utf-8")
    meta = {
        "id": f"{user}/{SWEEP_SLUG}",
        "title": SWEEP_SLUG,
        "code_file": "filseg-sweep.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": True,
        "dataset_sources": [f"{user}/{CODE_SLUG}", f"{user}/{CKPT_SLUG}"],
        "competition_sources": [COMP],
        "kernel_sources": [],
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(kaggle("kernels", "push", "-p", str(d)).strip())
    print(f"kernel {user}/{SWEEP_SLUG} pushed (fold={fold})")


def status(user: str, slug: str = None) -> None:
    print(kaggle("kernels", "status", f"{user}/{slug or KERNEL_SLUG}").strip())


def fetch(user: str, dest: Path, slug: str = None) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", f"{user}/{slug or KERNEL_SLUG}", "-p", str(dest)).strip())
    print(f"output -> {dest}")


def submit(file: Path, message: str) -> None:
    print(kaggle("competitions", "submit", COMP, "-f", str(file), "-m", message).strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("whoami")
    sub.add_parser("push-code")
    pk = sub.add_parser("push-kernel")
    pk.add_argument("--fold", type=int, default=0)
    pk.add_argument("--max-hours", type=float, default=10.5)
    pk.add_argument("--resume", action="store_true")
    ps = sub.add_parser("push-sweep")
    ps.add_argument("--fold", type=int, default=0)
    st = sub.add_parser("status")
    st.add_argument("--kernel", default=None)
    f = sub.add_parser("fetch")
    f.add_argument("-p", "--path", default=str(REPO / "data" / "kaggle_output"))
    f.add_argument("--kernel", default=None)
    pc = sub.add_parser("push-ckpt")
    pc.add_argument("-p", "--path", default=str(REPO / "data" / "kaggle_output"))
    pc.add_argument("--best-only", action="store_true")
    s = sub.add_parser("submit")
    s.add_argument("-f", "--file", required=True)
    s.add_argument("-m", "--message", required=True)
    args = ap.parse_args()

    if args.cmd == "submit":
        submit(Path(args.file), args.message)
        return
    user = whoami()
    if args.cmd == "whoami":
        print(user)
    elif args.cmd == "push-code":
        push_code(user)
    elif args.cmd == "push-kernel":
        push_kernel(user, args.fold, args.max_hours, args.resume)
    elif args.cmd == "push-sweep":
        push_sweep(user, args.fold)
    elif args.cmd == "status":
        status(user, args.kernel)
    elif args.cmd == "fetch":
        fetch(user, Path(args.path), args.kernel)
    elif args.cmd == "push-ckpt":
        push_ckpt(user, Path(args.path), args.best_only)


if __name__ == "__main__":
    main()
