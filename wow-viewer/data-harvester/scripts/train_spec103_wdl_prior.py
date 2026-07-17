"""Historical Spec 103/108 entry point; canonical owner is harvester.v50.wdl_prior_train.

Kept as a thin shim because tests/spec103/test_wdl_prior_sanity.py imports
filter_deployable_rows from this module name directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from harvester.v50.wdl_prior_train import (  # noqa: E402, F401, I001
    PriorDataset,
    evaluate,
    filter_deployable_rows,
    main,
    wdl_loss,
)


if __name__ == "__main__":
    raise SystemExit(main())
