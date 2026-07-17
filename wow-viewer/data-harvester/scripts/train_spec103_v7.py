"""Historical Spec 103 entry point; canonical owner is harvester.v50.terrain_refiner_train.

Kept as a thin shim because tests/spec103/test_wdl_prior_sanity.py imports V7TileDataset
from this module name directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from harvester.v50.terrain_refiner_train import (  # noqa: E402, F401, I001
    V7TileDataset,
    compute_loss,
    evaluate,
    main,
    object_coverage_per_tile,
    render_preview,
    rgb_std_per_tile,
)


if __name__ == "__main__":
    raise SystemExit(main())
