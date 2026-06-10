---
trigger: always_off
---

# 🌊 Windsurf Rules — C++ → C# Port (Target: net9.0)

## 1. Philosophy
- **Code is the memory bank** — all relevant context, design notes, and rationale live as intelligent comments *in the code itself*.
- **Single source of truth** — this file defines the rules. No extra “memory bank” files, no redundant documentation.
- **Reality only** — no logging of disproven or outdated findings. If it’s wrong, delete it.
- **Faithful port** — preserve all functionality from the C++ original, but adapt to idiomatic C# where it improves clarity, safety, or maintainability.

---

## 2. Project Scope
- **Goal**: Reimplement the existing C++ project in C#, file-by-file, targeting `net9.0`.
- **Approach**: Direct functional parity first, then incremental C#-specific refinements.
- **Exclusions**: No speculative features, no premature optimizations, no “future-proofing” beyond what’s already in scope.

---

## 3. File-by-File Porting Rules
1. **One C++ file → One C# file** (unless merging is clearly beneficial).
2. **Preserve structure**: Keep class/function names and signatures as close as possible until the port is complete.
3. **Inline documentation**:  
   - Use `///` XML doc comments for public APIs.  
   - Use `//` for quick rationale or porting notes.  
   - Tag porting notes with `// [PORT]` so they’re easy to find.
4. **Mark TODOs** inline with `// TODO(PORT): ...` — remove them once resolved.
5. **No external “memory bank”** — if it’s worth remembering, it goes in the code.

---

## 4. C# Adaptation Guidelines
- **Target Framework**: `net9.0`
- **Language Features**: Use modern C# features where they improve clarity (e.g., `using` declarations, pattern matching, `Span<T>` for perf-sensitive code).
- **Interop**: If native interop is required, isolate it in a dedicated `Interop` namespace.
- **Error Handling**: Replace C++ error codes with C# exceptions where appropriate.
- **Memory Management**: Use `IDisposable` and `using` instead of manual cleanup.

---

## 5. Testing & Verification
- **Unit Tests**: Mirror original C++ test coverage in C#.
- **Behavioral Parity**: For each ported file, verify output matches original C++ behavior.
- **Performance Checks**: Only after functional parity is achieved.

---

## 6. Commit Rules
- **One logical change per commit** — ideally one file per commit during porting.
- **Commit Messages**:  
  - `Port <filename>` for direct ports.  
  - `Refactor <filename>` for C#-specific improvements post-port.
- **No “WIP” commits** — finish the file before committing.

---

## 7. Maintenance
- **Update this file only when rules change** — not for every decision.
- **Delete outdated comments** — stale info is worse than no info.
- **Keep it lean** — if a rule doesn’t serve the project, remove it.

---

## 8. Example Inline Commenting Style
```csharp
// [PORT] Original C++ used raw pointers here.
// Adapted to Span<T> for safety and performance.
// TODO(PORT): Verify no unintended allocations in hot path.