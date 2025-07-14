# Product Context: The parpToolbox Project

## Why This Project Exists
The World of Warcraft research and tooling community requires modern, reliable tools to work with complex game file formats like PM4, PD4, and WMO. While powerful libraries like `wow.tools.local` exist, there is a need for a dedicated, high-level application that can orchestrate complex tasks like model exporting and data conversion.

## Problems Solved
- The lack of a single, focused tool for common conversion and export tasks, such as converting WMO models to the widely-supported OBJ format.
- The architectural complexity and technical debt of the legacy `WoWToolbox` project, which hindered new development.
- The need for a clean, maintainable codebase that can be easily extended in the future.

## How It Should Work
- Users will interact with a command-line application (`parpToolbox`) to perform operations on WoW files.
- The tool will leverage `wow.tools.local` for underlying file parsing, ensuring accuracy and compatibility.
- All generated output (e.g., OBJ files) will be saved to a structured, timestamped output directory, preserving the integrity of the source data.

## User Experience Goals
- A simple and intuitive command-line interface.
- Fast and accurate file processing.
- Clear, actionable error messages.
