
# 📏 Project Coding Standards & consistency

## 1. File Format Handling (Little-Endian & Reversed Signatures)
The WoW ADT/WDT format uses little-endian data, but chunk signatures (FourCC) are stored reversed on disk relative to their readable ASCII representation.

**RULE:** Eliminate "double-negative" confusion by enforcing **Readable Signatures Internally**.

### ❌ DO NOT:
- Define constants as reversed strings: `const string SIG_MTEX = "XETM";`
- Use logic that checks for reversed strings: `if (sig == "XETM") ...`
- Mix and match: `if (sig == "MTEX" || sig == "XETM")` (messy)

### ✅ DO:
- Define constants as **Readable**: `const string SIG_MTEX = "MTEX";`
- **Normalize on Read**: Immediately reverse the 4 bytes read from disk into a readable string before storing or processing.
- **Reverse on Write**: Reverse the readable string back to disk format (e.g. "MTEX" -> "XETM") only at the moment of writing to the stream.
- **Log Readable**: All logs should say "Found MTEX", never "Found XETM".

## 2. Code Style (Warcraft.NET Alignment)
- **Namespaces**: Use file-scoped namespaces (`namespace My.Namespace;`) to reduce nesting.
- **Var**: Use `var` when the type is obvious from the right-hand side (`var dict = new Dictionary<...>()`). Use explicit types when it's not (`int count = GetCount()`).
- **Properties**: Use auto-properties (`public int Id { get; set; }`) over fields where possible.
- **Comments**: Use XML comments (`/// <summary>`) for public APIs and complex logic.

## 3. Project Structure
- **Cli Tools**: Each tool should have its own project in the solution (e.g. `WoWRollback.Cli`, `AlphaWdtInspector`).
- **Shared Logic**: Common parsing logic belongs in `WoWRollback.Core` or specific library projects, not duplicated in CLI tools.
  - *Audit Note*: `AdtPatcher.cs` in `PM4Module` duplicates logic from `Core`. This must be merged.

## 4. Memory Bank Maintenance
- **Update Frequently**: Update `activeContext.md` after every significant logical change or discovery.
- **Audit Plans**: Keep a "Next Steps" section that acts as a backlog for technical debt (like this audit).
