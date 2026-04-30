"""
run_all_data_scripts.py

Orchestrates the full FX data preparation pipeline in order:

  Step 1 — Download  : Fetches raw FX data and saves to data/parquet
  Step 2 — Inspect   : Runs read-only diagnostics on raw data
  Step 3 — Clean     : Removes gaps, bad ticks, duplicates
  Step 4 — EDA       : Exploratory analysis, charts, distribution reports
  Step 5 — Features  : Engineers indicators and feature columns
  Step 6 — Labels    : Generates forward-return labels for ML targets
  Step 7 — Split     : Creates train / val / test dataset splits

After all steps complete, the system is ready for strategy backtesting.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  USAGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Full pipeline (first-time setup):
    python scripts/run_all_data_scripts.py

  Resume from a specific step (e.g. step 3):
    python scripts/run_all_data_scripts.py --from-step 3

  Skip steps whose outputs already exist:
    python scripts/run_all_data_scripts.py --skip-existing

  Force re-run even if outputs already exist:
    python scripts/run_all_data_scripts.py --force

  Preview what would run without executing anything:
    python scripts/run_all_data_scripts.py --dry-run

  Combine flags freely:
    python scripts/run_all_data_scripts.py --from-step 4 --force

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import subprocess
import sys
import os
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Pipeline definition ───────────────────────────────────────────────────────
#
# validates     : directories that must exist AND be non-empty after the step
# skip_if_exist : if ALL of these are non-empty, step is skipped with --skip-existing

STEPS = [
    {
        "number": 1,
        "name": "Download",
        "description": "Fetching raw FX tick data from source",
        "module": "scripts.download_fx_data",
        "script": "download_fx_data.py",
        "validates": ["data/parquet"],
        "skip_if_exist": ["data/parquet"],
    },
    {
        "number": 2,
        "name": "Inspect",
        "description": "Running diagnostics on raw data (read-only)",
        "module": "scripts.inspect_fx_data",
        "script": "inspect_fx_data.py",
        "validates": [],
        "skip_if_exist": [],        # read-only — never skipped
    },
    {
        "number": 3,
        "name": "Clean",
        "description": "Cleaning gaps, duplicates, and bad ticks",
        "module": "scripts.clean_fx_data",
        "script": "clean_fx_data.py",
        "validates": ["data/processed/cleaned", "data/processed/reports"],
        "skip_if_exist": ["data/processed/cleaned"],
    },
    {
        "number": 4,
        "name": "EDA",
        "description": "Running exploratory data analysis",
        "module": "scripts.eda_fx_data",
        "script": "eda_fx_data.py",
        "validates": ["eda/raw_snapshot", "eda/samples", "eda/reports"],
        "skip_if_exist": ["eda/reports"],
    },
    {
        "number": 5,
        "name": "Features",
        "description": "Engineering indicators and feature columns",
        "module": "scripts.features_fx_data",
        "script": "features_fx_data.py",
        "validates": ["features/pair", "features/reports"],
        "skip_if_exist": ["features/pair"],
    },
    {
        "number": 6,
        "name": "Labels",
        "description": "Generating forward-return labels",
        "module": "scripts.labels_fx_data",
        "script": "labels_fx_data.py",
        "validates": ["labels/pair", "labels/reports"],
        "skip_if_exist": ["labels/pair"],
    },
    {
        "number": 7,
        "name": "Split",
        "description": "Creating train / val / test dataset splits",
        "module": "scripts.split_fx_data",
        "script": "split_fx_data.py",
        "validates": ["datasets/train", "datasets/val", "datasets/test", "datasets/reports"],
        "skip_if_exist": ["datasets/train", "datasets/val", "datasets/test"],
    },
]

TOTAL_STEPS = len(STEPS)

# ── Dependency config ─────────────────────────────────────────────────────────
#
# To add or remove a required package, edit requirements.txt only.
# No changes needed here.
#
# MANUAL_INSTALL: packages that must NOT be auto-installed because the correct
# build is platform-dependent (e.g. CPU vs CUDA). The user is shown
# instructions instead.
#
# PIP_TO_IMPORT: pip name → import name, only where they differ.

MANUAL_INSTALL = {
    "torch": (
        "PyTorch requires a platform-specific install command.\n"
        "     Visit https://pytorch.org/get-started/locally/ to get the right\n"
        "     command for your OS / CUDA version, then re-run this script."
    ),
}

PIP_TO_IMPORT = {
    "scikit-learn":   "sklearn",
    "pyyaml":         "yaml",
    "python-dotenv":  "dotenv",
    "beautifulsoup4": "bs4",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_windows() -> bool:
    return sys.platform.startswith("win")


def sep(char="─", width=56) -> str:
    return char * width


def dir_nonempty(path: Path) -> bool:
    """True only if path is a directory containing at least one file."""
    return path.is_dir() and any(True for f in path.rglob("*") if f.is_file())


def can_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging():
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger, log_file


# ── Dependency checking ───────────────────────────────────────────────────────

def parse_requirements() -> list:
    """
    Read requirements.txt and return [(pip_name, import_name), ...].
    Skips blank lines, full-line comments, and commented-out optional packages.
    """
    req_file = PROJECT_ROOT / "requirements.txt"
    if not req_file.exists():
        return []

    packages = []
    for raw in req_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pip_name = line.split("==")[0].split(">=")[0].split("#")[0].strip()
        if not pip_name:
            continue
        import_name = PIP_TO_IMPORT.get(pip_name.lower(), pip_name.lower())
        packages.append((pip_name, import_name))
    return packages


def pip_install(args_str: str, logger: logging.Logger) -> bool:
    cmd = [sys.executable, "-m", "pip", "install"] + args_str.split() + ["--quiet"]
    logger.info(f"     pip install {args_str} ...")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            logger.error(r.stderr.strip())
            return False
        return True
    except Exception as exc:
        logger.error(f"     pip error: {exc}")
        return False


def check_venv(logger: logging.Logger) -> None:
    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    if not in_venv:
        logger.warning(
            "  ⚠  Not running inside a virtual environment.\n"
            "     PyCharm : File → Settings → Project → Python Interpreter\n"
            "     Terminal: source venv/bin/activate   (macOS/Linux)\n"
            "               venv\\Scripts\\activate       (Windows)"
        )
    else:
        logger.info(f"  Virtual environment: {sys.prefix}")


def ensure_dependencies(logger: logging.Logger) -> bool:
    check_venv(logger)
    logger.info("  Checking dependencies ...")

    packages = parse_requirements()
    if not packages:
        logger.warning("  requirements.txt not found — skipping dependency check.")
        return True

    req_file = PROJECT_ROOT / "requirements.txt"

    missing_auto   = [(p, i) for p, i in packages
                      if p.lower() not in MANUAL_INSTALL and not can_import(i)]
    missing_manual = [(p, i) for p, i in packages
                      if p.lower() in MANUAL_INSTALL and not can_import(i)]

    all_ok = True

    if missing_auto:
        logger.info(f"  Missing: {', '.join(p for p, _ in missing_auto)}")
        logger.info("  Installing from requirements.txt ...")
        if pip_install(f"-r {req_file}", logger):
            still = [p for p, i in missing_auto if not can_import(i)]
            if still:
                logger.error(
                    "  Installed but still unimportable: " + ", ".join(still)
                    + "\n  Run  pip install -r requirements.txt  manually."
                )
                all_ok = False
            else:
                logger.info("  All packages installed successfully  ✓")
        else:
            all_ok = False
    else:
        logger.info("  All dependencies present  ✓")

    for pip_name, _ in missing_manual:
        logger.error(
            f"\n  ✗  '{pip_name}' is not installed.\n"
            f"     {MANUAL_INSTALL[pip_name.lower()]}\n"
        )
        all_ok = False

    return all_ok


# ── Pre-flight ────────────────────────────────────────────────────────────────

def run_preflight(logger: logging.Logger) -> bool:
    logger.info(sep())
    logger.info("  Pre-flight checks")
    logger.info(sep())

    major, minor = sys.version_info[:2]
    if major < 3 or (major == 3 and minor < 8):
        logger.error(
            f"  Python 3.8+ required — you are running {major}.{minor}.\n"
            "  Download from https://www.python.org/downloads/"
        )
        return False
    logger.info(f"  Python {major}.{minor}  ✓")

    return ensure_dependencies(logger)


# ── Output validation ─────────────────────────────────────────────────────────

def validate_outputs(step: dict, logger: logging.Logger) -> bool:
    if not step["validates"]:
        return True
    problems = [
        f"    ✗  {'missing' if not (PROJECT_ROOT / p).exists() else 'empty'}:  {p}"
        for p in step["validates"]
        if not dir_nonempty(PROJECT_ROOT / p)
    ]
    if problems:
        logger.error(
            f"\n  Output validation failed (step {step['number']}):\n"
            + "\n".join(problems)
            + "\n  Script ran without errors but produced no output files."
        )
        return False
    return True


# ── Skip logic ────────────────────────────────────────────────────────────────

def should_skip(step: dict, logger: logging.Logger) -> bool:
    if not step["skip_if_exist"]:
        return False
    if all(dir_nonempty(PROJECT_ROOT / p) for p in step["skip_if_exist"]):
        logger.info(
            f"  ⏭  Step {step['number']}/{TOTAL_STEPS}: {step['name']}"
            " — outputs already exist, skipping."
        )
        return True
    return False


# ── Command builder ───────────────────────────────────────────────────────────

def build_command(step: dict) -> list:
    if is_windows():
        return [sys.executable, "-m", step["module"]]
    return [sys.executable, str(PROJECT_ROOT / "scripts" / step["script"])]


# ── Step runner ───────────────────────────────────────────────────────────────

def run_step(step: dict, logger: logging.Logger) -> bool:
    n, name = step["number"], step["name"]
    cmd = build_command(step)

    logger.info(sep())
    logger.info(f"  Step {n}/{TOTAL_STEPS}: {name}")
    logger.info(f"  {step['description']} ...")
    logger.debug(f"  Command: {' '.join(cmd)}")

    start = time.time()
    try:
        env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            cwd=str(PROJECT_ROOT), env=env,
        )
    except FileNotFoundError:
        logger.error(f"  Script not found: {cmd[-1]}")
        return False
    except Exception as exc:
        logger.error(f"  Failed to launch step {n}: {exc}")
        return False

    elapsed = time.time() - start

    if result.stdout:
        logger.debug(f"[stdout]\n{result.stdout.rstrip()}")
    if result.stderr:
        logger.debug(f"[stderr]\n{result.stderr.rstrip()}")

    if result.returncode != 0:
        if result.stderr:
            for line in result.stderr.strip().splitlines()[-25:]:
                logger.info(f"    {line}")
        logger.error(f"\n  Step {n} ({name}) FAILED (exit {result.returncode}, {elapsed:.1f}s)\n")
        return False

    if not validate_outputs(step, logger):
        return False

    logger.info(f"  Step {n}/{TOTAL_STEPS}: {name} — complete  ✓  ({elapsed:.1f}s)")
    return True


# ── Completion banner ─────────────────────────────────────────────────────────

def print_complete(logger: logging.Logger) -> None:
    logger.info("")
    logger.info(sep("═"))
    logger.info("  ✅  System initialised — all pipeline steps complete!")
    logger.info(sep("═"))
    logger.info("")

    guide_path = PROJECT_ROOT / "backtest" / "backtest_guide.txt"
    if guide_path.exists():
        content = guide_path.read_text(encoding="utf-8").strip()
        logger.info(content if content else "  (backtest_guide.txt is empty)")
    else:
        logger.info("  You can now test your strategies:")
        logger.info("    python scripts/backtest.py --strategy <name>")
        logger.info("    python scripts/backtest.py --list-strategies")
        logger.info("")
        logger.info("  Datasets:  datasets/train/   datasets/val/   datasets/test/")
        logger.info("  Tip: add your guide to  backtest/backtest_guide.txt")

    logger.info("")


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        prog="run_all_data_scripts.py",
        description="FX Data Pipeline — see module docstring for full usage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--from-step", type=int, default=1, metavar="N",
                        help="Start from step N (1–7). Default: 1.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip steps whose outputs already exist.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run all steps even if outputs exist. Overrides --skip-existing.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print steps and commands without executing anything.")
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    logger, log_file = setup_logging()

    logger.info("")
    logger.info(sep("═"))
    logger.info("  FX Data Pipeline")
    logger.info(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Log     : {log_file}")
    logger.info(sep("═"))
    logger.info(
        f"  --from-step {args.from_step}  |  --skip-existing {args.skip_existing}"
        f"  |  --force {args.force}  |  --dry-run {args.dry_run}"
    )
    logger.info("")

    if not (1 <= args.from_step <= TOTAL_STEPS):
        logger.error(f"  --from-step must be 1–{TOTAL_STEPS}, got {args.from_step}")
        sys.exit(1)

    if args.dry_run:
        logger.info("  DRY RUN — nothing will be executed.\n")
        for step in STEPS:
            logger.info(
                f"    Step {step['number']}: {step['name']:<12}"
                f"  →  {' '.join(build_command(step))}"
            )
        logger.info("")
        return

    if args.from_step == 1:
        if not run_preflight(logger):
            sys.exit(1)
        logger.info("")
    else:
        logger.info(f"  Resuming from step {args.from_step} — pre-flight skipped.\n")

    steps_to_run = [s for s in STEPS if s["number"] >= args.from_step]
    pipeline_start = time.time()
    executed = skipped = 0

    for step in steps_to_run:
        if args.skip_existing and not args.force and should_skip(step, logger):
            skipped += 1
            continue

        if not run_step(step, logger):
            logger.error(sep("═"))
            logger.error(f"  Pipeline stopped at step {step['number']}: {step['name']}")
            logger.error(
                f"  Resume:  python scripts/run_all_data_scripts.py"
                f" --from-step {step['number']}"
            )
            logger.error(sep("═"))
            sys.exit(1)

        executed += 1

    mins, secs = divmod(int(time.time() - pipeline_start), 60)
    logger.info(sep())
    logger.info(f"  Steps run: {executed}   Skipped: {skipped}   Total time: {mins}m {secs}s")
    logger.info(sep())

    print_complete(logger)


if __name__ == "__main__":
    main()