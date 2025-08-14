namespace ParpToolbox.Formats.P4.Chunks.Common;

// Back-compat aliases for older code paths that referenced legacy MSUR names.
// These provide a non-invasive bridge so downstream code compiles without
// rewriting call sites. Remove after migrating callers to new schema.

public sealed partial class MsurChunk
{
    public sealed partial class Entry
    {
        // Legacy alias: previously called SurfaceGroupKey
        public byte SurfaceGroupKey => GroupKey;

        // Legacy alias: various names that referenced the raw byte at offset 0x03
        public byte AttributeMask => Unknown03;
        public byte SurfaceAttributeMask => Unknown03;
        public byte Padding_0x03 => Unknown03;

        // Legacy alias: older code treated Float10 as Height
        public float Height => Float10;

        // Heuristic flags previously derived from an attribute mask.
        // NOTE: Bit mapping is speculative; adjust if authoritative mapping is known.
        public bool IsM2Bucket => (Unknown03 & 0x01) != 0;
        public bool IsLiquidCandidate => (Unknown03 & 0x02) != 0;
    }
}
