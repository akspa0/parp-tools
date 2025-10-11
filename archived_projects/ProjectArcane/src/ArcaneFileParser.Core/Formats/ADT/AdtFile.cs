using System.IO;
using ArcaneFileParser.Core.Common;
using ArcaneFileParser.Core.Formats.ADT.Chunks;

namespace ArcaneFileParser.Core.Formats.ADT;

/// <summary>
/// Handles parsing and writing of ADT (Terrain) files for both retail (v18) and experimental (v22/v23) formats.
/// </summary>
public class AdtFile
{
    private readonly ChunkParser _parser;
    private bool _isExperimentalFormat;

    /// <summary>
    /// Gets whether this ADT file uses the experimental format (v22/v23) from Cataclysm beta.
    /// </summary>
    public bool IsExperimentalFormat => _isExperimentalFormat;

    /// <summary>
    /// Gets the version chunk.
    /// </summary>
    public MverChunk? Mver { get; private set; }

    /// <summary>
    /// Gets the header chunk for retail format.
    /// </summary>
    public MhdrChunk? Mhdr { get; private set; }

    /// <summary>
    /// Gets the header chunk for experimental format.
    /// </summary>
    public AhdrChunk? Ahdr { get; private set; }

    /// <summary>
    /// Gets the chunk index array for retail format.
    /// </summary>
    public McinChunk? Mcin { get; private set; }

    /// <summary>
    /// Gets the texture filename list for retail format.
    /// </summary>
    public MtexChunk? Mtex { get; private set; }

    /// <summary>
    /// Gets the model filename list for retail format.
    /// </summary>
    public MmdxChunk? Mmdx { get; private set; }

    /// <summary>
    /// Gets the model filename offsets for retail format.
    /// </summary>
    public MmidChunk? Mmid { get; private set; }

    /// <summary>
    /// Gets the WMO filename list for retail format.
    /// </summary>
    public MwmoChunk? Mwmo { get; private set; }

    /// <summary>
    /// Gets the WMO filename offsets for retail format.
    /// </summary>
    public MwidChunk? Mwid { get; private set; }

    /// <summary>
    /// Gets the doodad placement information for retail format.
    /// </summary>
    public MddfChunk? Mddf { get; private set; }

    /// <summary>
    /// Gets the object placement information for retail format.
    /// </summary>
    public ModfChunk? Modf { get; private set; }

    /// <summary>
    /// Gets the map chunk data for retail format.
    /// </summary>
    public McnkChunk? Mcnk { get; private set; }

    /// <summary>
    /// Gets the vertex height data for experimental format.
    /// </summary>
    public AvtxChunk? Avtx { get; private set; }

    /// <summary>
    /// Gets the normal vectors for experimental format.
    /// </summary>
    public AnrmChunk? Anrm { get; private set; }

    /// <summary>
    /// Gets the texture filenames for experimental format.
    /// </summary>
    public AtexChunk? Atex { get; private set; }

    /// <summary>
    /// Gets the model filenames for experimental format.
    /// </summary>
    public AdooChunk? Adoo { get; private set; }

    /// <summary>
    /// Gets the map chunk data for experimental format.
    /// </summary>
    public AcnkChunk? Acnk { get; private set; }

    public AdtFile()
    {
        _parser = new ChunkParser();
        RegisterChunkHandlers();
    }

    private void RegisterChunkHandlers()
    {
        // Register version chunk handler
        _parser.RegisterHandler<MverChunk>("MVER");

        // Register retail format chunk handlers
        _parser.RegisterHandler<MhdrChunk>("MHDR");
        _parser.RegisterHandler<McinChunk>("MCIN");
        _parser.RegisterHandler<MtexChunk>("MTEX");
        _parser.RegisterHandler<MmdxChunk>("MMDX");
        _parser.RegisterHandler<MmidChunk>("MMID");
        _parser.RegisterHandler<MwmoChunk>("MWMO");
        _parser.RegisterHandler<MwidChunk>("MWID");
        _parser.RegisterHandler<MddfChunk>("MDDF");
        _parser.RegisterHandler<ModfChunk>("MODF");
        _parser.RegisterHandler<McnkChunk>("MCNK");

        // Register experimental format chunk handlers
        _parser.RegisterHandler<AhdrChunk>("AHDR");
        _parser.RegisterHandler<AvtxChunk>("AVTX");
        _parser.RegisterHandler<AnrmChunk>("ANRM");
        _parser.RegisterHandler<AtexChunk>("ATEX");
        _parser.RegisterHandler<AdooChunk>("ADOO");
        _parser.RegisterHandler<AcnkChunk>("ACNK");
    }

    public void Parse(string filePath)
    {
        foreach (var chunk in _parser.ReadFile(filePath))
        {
            // Process each chunk based on its type
            switch (chunk)
            {
                case MverChunk mver:
                    Mver = mver;
                    _isExperimentalFormat = mver.Version is 22 or 23;
                    break;

                // Retail format chunks
                case MhdrChunk mhdr when !_isExperimentalFormat:
                    Mhdr = mhdr;
                    break;
                case McinChunk mcin when !_isExperimentalFormat:
                    Mcin = mcin;
                    break;
                case MtexChunk mtex when !_isExperimentalFormat:
                    Mtex = mtex;
                    break;
                case MmdxChunk mmdx when !_isExperimentalFormat:
                    Mmdx = mmdx;
                    break;
                case MmidChunk mmid when !_isExperimentalFormat:
                    Mmid = mmid;
                    break;
                case MwmoChunk mwmo when !_isExperimentalFormat:
                    Mwmo = mwmo;
                    break;
                case MwidChunk mwid when !_isExperimentalFormat:
                    Mwid = mwid;
                    break;
                case MddfChunk mddf when !_isExperimentalFormat:
                    Mddf = mddf;
                    break;
                case ModfChunk modf when !_isExperimentalFormat:
                    Modf = modf;
                    break;
                case McnkChunk mcnk when !_isExperimentalFormat:
                    Mcnk = mcnk;
                    break;

                // Experimental format chunks
                case AhdrChunk ahdr when _isExperimentalFormat:
                    Ahdr = ahdr;
                    break;
                case AvtxChunk avtx when _isExperimentalFormat:
                    Avtx = avtx;
                    break;
                case AnrmChunk anrm when _isExperimentalFormat:
                    Anrm = anrm;
                    break;
                case AtexChunk atex when _isExperimentalFormat:
                    Atex = atex;
                    break;
                case AdooChunk adoo when _isExperimentalFormat:
                    Adoo = adoo;
                    break;
                case AcnkChunk acnk when _isExperimentalFormat:
                    Acnk = acnk;
                    break;
            }
        }
    }

    public string CreateReport(string filePath)
    {
        return _parser.CreateReadableReport(filePath);
    }
} 