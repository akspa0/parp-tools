namespace MdxLTool.Formats.Mdx;

/// <summary>
/// MDX chunk FourCC identifiers.
/// Based on Ghidra analysis of WoW 0.5.3 client and wowdev.wiki/MDX.
/// </summary>
public static class MdxHeaders
{
    /// <summary>MDX magic: "MDLX" (0x584C444D little-endian)</summary>
    public const uint MAGIC = 0x584C444D;

    // Core chunks
    public const string VERS = "VERS"; // Version
    public const string MODL = "MODL"; // Model info
    public const string SEQS = "SEQS"; // Sequences (animations)
    public const string GLBS = "GLBS"; // Global sequences
    public const string MTLS = "MTLS"; // Materials
    public const string TEXS = "TEXS"; // Textures
    public const string GEOS = "GEOS"; // Geosets
    public const string GEOA = "GEOA"; // Geoset animations

    // Skeletal
    public const string BONE = "BONE"; // Bones
    public const string HELP = "HELP"; // Helpers
    public const string PIVT = "PIVT"; // Pivot points
    public const string ATCH = "ATCH"; // Attachments

    // Effects
    public const string LITE = "LITE"; // Lights
    public const string PREM = "PREM"; // Particle emitters (v1)
    public const string PRE2 = "PRE2"; // Particle emitters (v2)
    public const string RIBB = "RIBB"; // Ribbon emitters
    public const string EVTS = "EVTS"; // Events

    // Other
    public const string CAMS = "CAMS"; // Cameras
    public const string CLID = "CLID"; // Collision
    public const string HTST = "HTST"; // Hit test shapes
    public const string TXAN = "TXAN"; // Texture animations
    public const string CORN = "CORN"; // PopcornFX emitters (Reforged)

    // Geoset sub-chunks
    public const string VRTX = "VRTX"; // Vertices
    public const string NRMS = "NRMS"; // Normals
    public const string PTYP = "PTYP"; // Primitive types
    public const string PCNT = "PCNT"; // Primitive counts
    public const string PVTX = "PVTX"; // Primitive vertices (indices)
    public const string GNDX = "GNDX"; // Group indices
    public const string MTGC = "MTGC"; // Matrix group counts
    public const string MATS = "MATS"; // Matrix indices
    public const string UVAS = "UVAS"; // UV set count
    public const string UVBS = "UVBS"; // UV coordinates
    public const string BIDX = "BIDX"; // Bone indices (skinning)
    public const string BWGT = "BWGT"; // Bone weights (skinning)
    public const string ATSQ = "ATSQ"; // Animation tracks? (found in 0.5.3)
}
