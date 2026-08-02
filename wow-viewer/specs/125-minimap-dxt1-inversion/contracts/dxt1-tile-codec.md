# Contract: DXT1 Tile Codec

**Library**: `WowViewer.Core.IO.Blp` | **File**: `Dxt1TileCodec.cs`

## Purpose

Reproduce the authored tile's lossy encode/decode cycle on a synthetic tile (FR-002), and verify our
encoder against authored bytes (FR-014).

## API

```csharp
namespace WowViewer.Core.IO.Blp;

/// <summary>
/// Applies the DXT1 encode/decode cycle to a synthetic tile, producing an image with equivalent
/// degradation characteristics to an authored DXT1 tile. In-memory only; never writes a BLP.
/// </summary>
public static class Dxt1TileCodec
{
    /// <summary>Encode to DXT1 raw blocks then decode back to RGBA. The parity cycle (FR-002, FR-015).</summary>
    public static Image<Rgba32> EncodeDecode(Image<Rgba32> source);

    /// <summary>
    /// Decode an authored BLP tile to RGBA. Wraps the existing BlpRgbReader path (FR-001).
    /// </summary>
    public static Image<Rgba32> DecodeAuthored(byte[] authoredBlp);

    /// <summary>
    /// Round-trip agreement: decode an authored tile, re-encode it, and measure block-level agreement
    /// with the authored bytes. Correctness check on OUR encoder (FR-014). Returns 0..1.
    /// </summary>
    public static float RoundTripAgreement(byte[] authoredBlp);
}
```

## Behaviour

- `EncodeDecode` uses `BCnEncoder` with `CompressionFormat.Bc1` (DXT1), matching the authored tiles'
  measured encoding. Handles both opaque and alpha-punchthrough modes.
- `RoundTripAgreement` returns the fraction of 4×4 blocks whose re-encoded bytes match the authored
  bytes. Target: ≥95% (SC-009).
- No BLP container is ever written (Out of Scope).

## Validation

- `Dxt1TileCodecTests`: round-trip agreement ≥95% on authored tiles; parity cycle produces a
  measurably different (blockier) image than the pristine input; undamaged input passes through the
  cycle with expected degradation.
