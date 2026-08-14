"""Build consensus label targets from the competition train JSON."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg import labels  # noqa: E402

if __name__ == "__main__":
    out = labels.build_all()
    print(f"labels written to {out}")
