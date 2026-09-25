"""Run the whole pipeline in order. Step 01 is skipped if data/processed/sample.parquet exists
(pass --resample to force it). Step runtimes are written to outputs/tables/runtime.csv."""
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
RUNTIME = ROOT / "outputs" / "tables" / "runtime.csv"
STEPS = ["01_sample.py", "02_features.py", "03_cluster.py", "04_models.py", "05_explain.py",
         "06_k_sensitivity.py", "07_report.py", "08_results.py", "09_verify_models.py"]


def previous_01_seconds():
    """Last recorded 01_sample.py time: runtime.csv from an earlier run, else the earlier run_all.log."""
    if RUNTIME.exists():
        old = pd.read_csv(RUNTIME)
        hit = old[(old["step"] == "01_sample.py") & old["seconds"].notna()]
        if len(hit):
            return float(hit["seconds"].iloc[0]), hit["note"].iloc[0]
    log = ROOT / "data" / "processed" / "run_all.log"
    if log.exists():
        m = re.findall(r"== 01_sample.py done in (\d+)s", log.read_text(encoding="utf-8", errors="ignore"))
        if m:
            return float(m[-1]), "not rerun; time from an earlier --resample run (data/processed/run_all.log)"
    return None, "not rerun; no earlier timing found"


def write_runtime(rows, t_start):
    df = pd.DataFrame(rows + [{"step": "total (steps run in this invocation)",
                               "seconds": time.time() - t_start, "note": ""}])
    RUNTIME.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RUNTIME, index=False)


def main():
    resample = "--resample" in sys.argv
    t_start = time.time()
    rows = []
    for step in STEPS:
        if step == "01_sample.py" and not resample and (ROOT / "data" / "processed" / "sample.parquet").exists():
            secs, note = previous_01_seconds()
            if "not rerun" not in str(note):
                note = "not rerun; " + str(note)
            rows.append({"step": step, "seconds": secs, "note": note})
            print(f"== {step}: skipped (sample.parquet exists; use --resample to rebuild)")
            continue
        print(f"== {step}", flush=True)
        t0 = time.time()
        subprocess.run([sys.executable, step], cwd=SRC, check=True)
        rows.append({"step": step, "seconds": time.time() - t0, "note": "this run"})
        print(f"== {step} done in {time.time() - t0:.0f}s", flush=True)
        write_runtime(rows, t_start)
    write_runtime(rows, t_start)
    subprocess.run([sys.executable, "07_report.py", "--methods-only"], cwd=SRC, check=True)
    print(f"== all done in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
