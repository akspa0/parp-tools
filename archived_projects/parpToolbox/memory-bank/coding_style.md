# Coding Style Guidelines for PM4 / PD4 Readers

These conventions align our code with the style used in `wow.tools.local` (see `WoWFormatLib.FileReaders.WMOReader`). All future PM4/PD4 code must follow these rules.

## File & Class Layout
1. **One Reader per File** – Each chunk or top-level format reader resides in its own `.cs` file with the class name matching the file name.
2. **Namespace Hierarchy** – Use `ParpToolbox.Formats.P4.Chunks.Common` for shared readers, and `ParpToolbox.Formats.PM4.Chunks` / `ParpToolbox.Formats.PD4.Chunks` for format-specific ones.
3. **Ordering** – `using` statements, namespace declaration, public types, then private helpers, mirroring WMOReader’s structure.

## Chunk Reader Pattern
```csharp
public sealed class MspvChunk : IChunk
{
    public const string Signature = "MSPV";

    // Immutable payload model
    public IReadOnlyList<Vector3> Vertices { get; private set; } = Array.Empty<Vector3>();

    public void LoadBinaryData(ReadOnlySpan<byte> data)
    {
        using var ms = new MemoryStream(data);
        using var br = new BinaryReader(ms);
        // parse fields
    }
}
```
* Use `const string Signature` for the canonical FourCC.
* Provide immutable public properties populated via `LoadBinaryData`.
* Throw `InvalidDataException` for malformed payloads.

## Adapter Pattern
* Implement `IPm4Loader` / `IPd4Loader` interfaces.
* Use a streaming `BinaryReader` loop: read FourCC (with byte-reverse helper), size (UInt32LE), then payload bytes.
* Switch on canonical `Signature` constants.
* Seek to `payloadStart + size` to preserve alignment.

## FourCC Handling
* Always call `FourCc.Read(br)` to obtain canonical IDs.
* Shared constants remain in canonical order (docs/engine order) even though on-disk they’re reversed.

## Comments & XML Docs
* Public members require XML comments summarising purpose and parameters.
* Use triple-slash `///` style like `WoWFormatLib`.

## Exceptions & Validation
* Fail fast with `ArgumentException` / `InvalidDataException` when inputs are invalid.
* Never silently ignore missing required chunks.

## Formatting & Naming
* Follow Microsoft C# conventions (PascalCase for public, camelCase for locals). No Hungarian notation.
* Keep line length ≤ 120 chars.

---
Adhering to these guidelines ensures consistency with `wow.tools.local` and maintainability across formats.
