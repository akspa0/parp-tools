# Evidence model — renderer fidelity refresh

Proposed evidence/policy contracts, not implemented APIs or a new runtime telemetry backend.

| Entity | Required fields / relationship | Rules |
|---|---|---|
| Reproduction | owner/criterion, revision, configured client build/fingerprint, map, asset/group/chunk, camera, settings, baseline output | identify exact inputs; no machine-local default root in source/portable fixtures |
| Source citation | era/build, source kind, configured location, revision/hash, file/symbol/range, excerpt/claim | distinguish observed, reference-confirmed, hypothesis and unknown; pointer alone is unverified |
| Ground-effect observation | candidate/generated/submitted counts, world position, distance/radius/fade, density, asset/material route, lighting source | separate placement density, distance rejection and opacity; do not infer instancing from grouping |
| Liquid trace | WMO/MH2O, raw IDs/nibbles/flags, exact row or miss, family, layout source, layer/merge choice, submitted material | group/material/family/layout IDs distinct; unknown/fallback explicit |
| Sky trace | active build/map/time, lighting rows/weights, skybox row/model, fallback, asset status, visibility gate, pass state, blank/default mode | schema is not actual row evidence; day/night must use same identified inputs |
| Correction decision | failed criterion, baseline evidence, diagnosis, approved write set, pure tests, operator gate | one concern; no reader modification without separate verified-format approval |
| Receipt | changed paths, exact commands/exit codes, criterion→real-output mapping, baseline/after paths, verdict | compilation proves no visual/FPS outcome; missing real evidence leaves criterion open |

State machine: source observation → identified reproduction → verified source/diagnosis → bounded
correction approval → code/test validation → operator comparison → accepted/rejected/blocked.
Rejected corrections retain baseline and evidence; do not silently broaden write scope.
