"""Historical Spec 103/108 entry point; canonical owner is harvester.v50.wdl_prior_infer.

Kept as a real, runnable script (not deleted) because
tests/spec103/test_wdl_prior_sanity.py subprocess-invokes this exact file path, and
scripts/evaluate_spec103_wdl_prior.py imports load_model/predict_rgb from this module name.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from harvester.v50.wdl_prior_infer import load_model, main, predict_rgb  # noqa: E402, F401, I001


if __name__ == "__main__":
    raise SystemExit(main())
