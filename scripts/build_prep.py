"""Precompute the preprocessed 3-channel image cache for all train+test images."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg import preprocessing  # noqa: E402

if __name__ == "__main__":
    preprocessing.build_prep_cache()
    print("prep cache complete")
