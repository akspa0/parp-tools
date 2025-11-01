# Product Context

Why: Early WoW interiors were iterated in Q3 tooling. v14 WMO is close to Q3 data. This tool bridges WMO → Q3 for research and iteration.

How it should work:
- CLI input: WMO file, output dir, verbosity, texture extraction.
- Writes: .bsp, .map, optional textures and shader script.
- Minimal but correct BSP so Q3/ioquake3 can load or editors can compile.

User experience:
- Clear progress logs.
- Deterministic outputs.
- Errors explain missing chunks or unsupported cases.
