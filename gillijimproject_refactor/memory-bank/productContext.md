# Product Context

## Users
- **Technical Artists & Engineers**: Maintaining Alpha-to-Modern and Modern-to-Alpha asset pipelines.
- **Modders/Researchers**: Analyzing the history of WoW data formats.

## Problem
- **Format Obscurity**: The Alpha 0.5.3 format is monolithic, uses legacy structures (MCLQ liquids, reversed FourCCs), and lacks official documentation.
- **Data Loss**: Naive conversion attempts often lose critical metadata (liquids, holes, sound emitters, object placements) or cause client crashes.
- **Guesswork**: Previous efforts relied on trial-and-error, leading to unstable tools.

## Desired Experience
- **Reliable Pipeline**: A "one-stop shop" toolchain that can read any Alpha asset and write it back perfectly (roundtrip).
- **Definitive Specs**: No guessing. All format details are documented in `memory-bank/specs/Alpha-0.5.3-Format.md` and enforced by code.
- **Diagnostics**: When things go wrong, the tools explain *why* (e.g., "MCLQ offset 0", "WMO count mismatch") via clear CSV reports and logs.
- **Automation**: Commands like `roundtrip` and `pack-alpha` automate the complex grunt work of offset patching and monolithic file assembly.

