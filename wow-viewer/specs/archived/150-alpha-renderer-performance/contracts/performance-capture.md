# Contract: Production Renderer Performance Capture

The first capture surface is `WowViewer.Tool.ValidationCapture profile-render`.

Required report identity:

- Alpha build and configured client root label
- map input and target tile/route
- camera/projection and resolution
- detailed/retained/capture-preload residency policy
- warmup and measured frame counts
- source revision/dirty-tree note

Required timing classification:

- total CPU frame time
- each existing production stage
- GPU/driver time when an optional timer is available
- explicit unavailable/failed state when it is not
- client-read and deferred-load timing separately from settled frames

Required workload pressure:

- terrain visible/culled chunks and tile draw count
- WDL visible/hidden tiles
- WMO/MDX visible/culled counts
- opaque batch/fallback and transparent counts
- WMO group/liquid/doodad submissions
- draw, uniform, active-texture, and texture-bind counters where the owner exposes them
- pending terrain/assets/deferred work
- overlay owner timings and prepared/submitted/deferred counts

Comparison rules:

1. Baseline and candidate use the same control-scene identity.
2. Warmup frames are excluded from the measured aggregate.
3. The report names the dominant owner and the evidence supporting that choice.
4. FPS is not derived from CPU time when GPU/driver timing is unknown.
5. A report cannot close real-client visual or FPS proof by itself.
