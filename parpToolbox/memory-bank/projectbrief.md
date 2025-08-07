# Project Brief: The parpToolbox Project

## Overview
This project, `parpToolbox`, is a new, clean-room implementation of a command-line tool for parsing, converting, and exporting World of Warcraft file formats, with an initial focus on PM4, PD4, and WMO files. It is being built from the ground up to provide a modern, maintainable, and robust foundation for WoW tooling.

The project uses the `wow.tools.local` library as a core dependency for handling low-level file formats. The legacy `WoWToolbox` project and its associated files are being preserved as a valuable source of reference logic and implementation details, but all new development will occur within `parpToolbox`.

## Core Requirements
- Build a new console application (`parpToolbox`) for file processing.
- Add support for PM4, PD4, and WMO parsing, conversion, and exporting (WMO to OBJ export is implemented).
- Use `wow.tools.local` strictly as an external library—never modify its codebase.
- Ensure all outputs, tests, and new features are cleanly architected within the new project.
- Port relevant and correct logic from the legacy `WoWToolbox` project as needed.

## Goals
- A stable, modern, and maintainable tool for WoW file formats.
- High-quality, verifiable data exports (e.g., OBJ models).
- Clean separation of concerns between `parpToolbox` and its dependencies.
- Full documentation and test coverage for all new functionality.
