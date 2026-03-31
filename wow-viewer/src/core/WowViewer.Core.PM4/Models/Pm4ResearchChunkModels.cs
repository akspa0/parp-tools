using System.Numerics;

namespace WowViewer.Core.PM4.Models;

public sealed record Pm4ChunkRecord(
    string Signature,
    int HeaderOffset,
    int DataOffset,
    uint Size,
    byte[] Payload);

public sealed record Pm4Bounds3(
    Vector3 Min,
    Vector3 Max)
{
    public Vector3 Span => Max - Min;
}

public sealed record Pm4MshdHeader(
    uint Field00,
    uint Field04,
    uint Field08,
    uint Field0C,
    uint Field10,
    uint Field14,
    uint Field18,
    uint Field1C);

public sealed record Pm4MslkEntry(
    byte TypeFlags,
    byte Subtype,
    ushort Padding,
    uint GroupObjectId,
    int MspiFirstIndex,
    byte MspiIndexCount,
    uint LinkId,
    ushort RefIndex,
    ushort SystemFlag)
{
    public uint _0x04 => GroupObjectId;
}

public sealed record Pm4MsurEntry(
    byte GroupKey,
    byte IndexCount,
    byte AttributeMask,
    byte Padding,
    Vector3 Normal,
    float Height,
    uint MsviFirstIndex,
    uint MdosIndex,
    uint PackedParams)
{
    public byte _0x00 => GroupKey;

    public byte _0x02 => AttributeMask;

    public float PlaneDistance => Height;

    public uint _0x18 => MdosIndex;

    public uint _0x1C => PackedParams;

    public uint Ck24 => (PackedParams >> 8) & 0x00FF_FFFF;

    public byte Ck24Type => (byte)((PackedParams >> 24) & 0xFF);

    public ushort Ck24ObjectId => (ushort)(Ck24 & 0xFFFF);
}

public sealed record Pm4MprlEntry(
    ushort Unk00,
    short Unk02,
    ushort Unk04,
    ushort Unk06,
    Vector3 Position,
    short Unk14,
    ushort Unk16);

public sealed record Pm4MprrEntry(
    ushort Value1,
    ushort Value2)
{
    public bool IsSentinel => Value1 == 0xFFFF;
}

public sealed record Pm4MdbhEntry(uint DestructibleBuildingCount);

public sealed record Pm4MdbiEntry(uint DestructibleBuildingIndex);

public sealed record Pm4MdbfEntry(string Filename, int RawLength);

public sealed record Pm4MdosEntry(uint DestructibleBuildingIndex, uint DestructionState);

public sealed record Pm4MdsfEntry(uint MsurIndex, uint MdosIndex);

public sealed record Pm4KnownChunkSet(
    Pm4MshdHeader? Mshd,
    IReadOnlyList<Pm4MslkEntry> Mslk,
    IReadOnlyList<Vector3> Mspv,
    IReadOnlyList<uint> Mspi,
    IReadOnlyList<Vector3> Msvt,
    IReadOnlyList<uint> Msvi,
    IReadOnlyList<Pm4MsurEntry> Msur,
    IReadOnlyList<Vector3> Mscn,
    IReadOnlyList<Pm4MprlEntry> Mprl,
    IReadOnlyList<Pm4MprrEntry> Mprr,
    Pm4MdbhEntry? Mdbh,
    IReadOnlyList<Pm4MdbiEntry> Mdbi,
    IReadOnlyList<Pm4MdbfEntry> Mdbf,
    IReadOnlyList<Pm4MdosEntry> Mdos,
    IReadOnlyList<Pm4MdsfEntry> Mdsf);

public sealed record Pm4ExplorationSnapshot(
    uint Version,
    int ChunkCount,
    int MslkCount,
    int MspvCount,
    int MspiCount,
    int MsvtCount,
    int MsviCount,
    int MsurCount,
    int MscnCount,
    int MprlCount,
    int MprrCount,
    Pm4Bounds3? MsvtBounds,
    Pm4Bounds3? MscnBounds,
    Pm4Bounds3? MprlBounds,
    IReadOnlyList<string> Diagnostics);

public sealed record Pm4QuadrantSummary(
    string Plane,
    float MidA,
    float MidB,
    int LowLow,
    int LowHigh,
    int HighLow,
    int HighHigh);

public sealed record Pm4VectorSetSummary(
    string Name,
    int Count,
    Pm4Bounds3? Bounds,
    Vector3? Centroid,
    IReadOnlyList<Pm4QuadrantSummary> Quadrants);

public sealed record Pm4MprlSummary(
    int TotalCount,
    int NormalCount,
    int TerminatorCount,
    short? FloorMin,
    short? FloorMax,
    float? RotationMinDegrees,
    float? RotationMaxDegrees);

public sealed record Pm4Ck24Summary(
    uint Ck24,
    byte Ck24Type,
    ushort Ck24ObjectId,
    int SurfaceCount,
    int TotalIndexCount,
    float AverageHeight,
    int DistinctMdosCount);

public sealed record Pm4ChunkSummary(
    string Signature,
    uint Size);

public sealed record Pm4TerminologyEntry(
    string RawField,
    string? LocalAlias,
    string Confidence,
    string Notes);

public sealed record Pm4AnalysisReport(
    string? SourcePath,
    uint Version,
    IReadOnlyList<Pm4ChunkSummary> ChunkOrder,
    IReadOnlyList<string> UnknownChunks,
    Pm4VectorSetSummary Mspv,
    Pm4VectorSetSummary Msvt,
    Pm4VectorSetSummary Mscn,
    Pm4VectorSetSummary MprlPositions,
    Pm4MprlSummary Mprl,
    IReadOnlyList<Pm4TerminologyEntry> Terminology,
    IReadOnlyList<Pm4Ck24Summary> TopCk24Groups,
    IReadOnlyList<string> ResearchNotes,
    IReadOnlyList<string> Diagnostics);

public static class Pm4TerminologyCatalog
{
    private static readonly IReadOnlyList<Pm4TerminologyEntry> InspectEntries =
    [
        new("MSUR._0x00", "GroupKey", "low", "Local research alias only; wowdev does not currently give this byte a settled semantic name."),
        new("MSUR._0x02", "AttributeMask", "low", "This replaced an older 'unknown byte 2' label locally, but the bit meanings are still open."),
        new("MSUR._0x10", "Height", "medium", "Current corpus evidence says this float behaves like a signed plane-distance term, not a generic vertical height."),
        new("MSUR._0x18", "MdosIndex", "medium", "The linkage is useful, but raw-name-first wording is still safer than presenting this as official format terminology."),
        new("MSUR._0x1C", "PackedParams; derived CK24/Ck24Type/Ck24ObjectId", "medium", "The 24-bit key and low16 slice are local derived identities, not format-native field names."),
        new("MSLK._0x04", "GroupObjectId", "low", "Local alias only; current evidence does not justify treating it as a confirmed full-object identity field.")
    ];

    public static IReadOnlyList<Pm4TerminologyEntry> ForInspectReport() => InspectEntries;

    public static IReadOnlyList<Pm4TerminologyEntry> ForCk24Forensics() => InspectEntries;
}

public sealed record Pm4ChunkDecodeAudit(
    string Signature,
    int ChunkCount,
    ulong TotalBytes,
    uint MinChunkSize,
    uint MaxChunkSize,
    int DistinctSizeCount,
    int EntryCount,
    bool HasMeaningfulData,
    int StrideRemainderCount,
    IReadOnlyList<uint> ExampleSizes);

public sealed record Pm4ReferenceAudit(
    string Name,
    int TotalCount,
    int ValidCount,
    int InvalidCount,
    IReadOnlyList<string> Examples);

public sealed record Pm4DecodeAuditReport(
    string? SourcePath,
    uint Version,
    int ChunkCount,
    int RecognizedChunkCount,
    int UnknownChunkCount,
    bool HasTrailingBytesDiagnostic,
    bool HasOverrunDiagnostic,
    IReadOnlyList<Pm4ChunkDecodeAudit> ChunkAudits,
    IReadOnlyList<Pm4ReferenceAudit> ReferenceAudits,
    IReadOnlyList<string> UnknownChunkSignatures,
    IReadOnlyList<string> Diagnostics);

public sealed record Pm4CorpusChunkAudit(
    string Signature,
    int FileCount,
    int DataFileCount,
    int TotalChunkCount,
    ulong TotalBytes,
    uint MinChunkSize,
    uint MaxChunkSize,
    int DistinctSizeCount,
    int TotalEntryCount,
    int FilesWithStrideRemainders,
    bool IsCommon,
    IReadOnlyList<uint> ExampleSizes);

public sealed record Pm4CorpusReferenceAudit(
    string Name,
    int TotalCount,
    int InvalidCount,
    IReadOnlyList<string> ExampleFailures);

public sealed record Pm4CorpusAuditReport(
    string InputDirectory,
    int FileCount,
    int FilesWithDiagnostics,
    int FilesWithUnknownChunks,
    IReadOnlyList<Pm4CorpusChunkAudit> ChunkAudits,
    IReadOnlyList<Pm4CorpusReferenceAudit> ReferenceAudits,
    IReadOnlyList<string> UnknownChunkSignatures,
    IReadOnlyList<string> Diagnostics);

public sealed record Pm4ValueFrequency(
    string Value,
    int Count);

public sealed record Pm4FieldDistribution(
    string Field,
    int TotalCount,
    int DistinctCount,
    string? Range,
    IReadOnlyList<Pm4ValueFrequency> TopValues,
    string? Notes);

public sealed record Pm4RelationshipEdgeSummary(
    string Edge,
    string Status,
    int Fits,
    int Misses,
    string Evidence,
    string NextStep);

public sealed record Pm4Ck24ObjectIdReuseCase(
    string? SourcePath,
    int? TileX,
    int? TileY,
    ushort Ck24ObjectId,
    int DistinctCk24Count,
    int DistinctTypeCount,
    int SurfaceCount,
    IReadOnlyList<Pm4ValueFrequency> TopCk24Values);

public sealed record Pm4LinkageIdentitySummary(
    int DistinctCk24Count,
    int DistinctCk24ObjectIdCount,
    int ObjectIdGroupsAnalyzed,
    int ReusedObjectIdGroupCount,
    int ReusedAcrossTypeGroupCount,
    IReadOnlyList<Pm4Ck24ObjectIdReuseCase> TopReuseCases);

public sealed record Pm4LinkageMismatchFamily(
    string FamilyKey,
    int FileCount,
    int EntryCount,
    int DistinctGroupObjectIdCount,
    int DistinctLow16ObjectIdCount,
    int DistinctRefIndexCount,
    int MatchingCk24ObjectIdEntryCount,
    int MatchingFullCk24EntryCount,
    int EntriesInFilesWithMscn,
    int EntriesInFilesWithBadMdos,
    IReadOnlyList<Pm4ValueFrequency> CandidateDomains,
    IReadOnlyList<Pm4ValueFrequency> TopLow16ObjectIds,
    IReadOnlyList<Pm4LinkageMismatchExample> TopExamples);

public sealed record Pm4LinkageMismatchExample(
    string? SourcePath,
    int? TileX,
    int? TileY,
    ushort RefIndex,
    uint GroupObjectId,
    ushort GroupObjectLow16,
    uint GroupObjectLow24,
    uint LinkId,
    byte TypeFlags,
    byte Subtype,
    ushort SystemFlag,
    IReadOnlyList<string> CandidateDomains,
    bool Low16MatchesCk24ObjectId,
    bool Low24MatchesCk24,
    bool FileHasMscn,
    bool FileHasBadMdos);

public sealed record Pm4BadMdosCluster(
    string? SourcePath,
    int? TileX,
    int? TileY,
    uint Ck24,
    byte Ck24Type,
    ushort Ck24ObjectId,
    int SurfaceCount,
    int InvalidMdosCount,
    int ValidMdosCount,
    int DistinctInvalidMdosCount,
    int DistinctValidMdosCount,
    int MeshVertexCount);

public sealed record Pm4LinkageReport(
    string InputDirectory,
    int FileCount,
    int FilesWithRefIndexMismatches,
    int FilesWithBadMdos,
    int TotalRefIndexMismatchCount,
    IReadOnlyList<Pm4RelationshipEdgeSummary> Relationships,
    Pm4LinkageIdentitySummary IdentitySummary,
    IReadOnlyList<Pm4FieldDistribution> Distributions,
    IReadOnlyList<Pm4LinkageMismatchFamily> TopMismatchFamilies,
    IReadOnlyList<Pm4BadMdosCluster> TopBadMdosClusters,
    IReadOnlyList<string> Notes);

public sealed record Pm4MscnCoordinateSummary(
    int TotalPointCount,
    int SwappedWorldTileFitCount,
    int RawWorldTileFitCount,
    int AmbiguousWorldTileFitCount,
    int TileLocalLikeCount,
    int NeitherFitCount,
    int FilesSwappedDominant,
    int FilesRawDominant,
    int FilesTileLocalDominant,
    int FilesNoDominant);

public sealed record Pm4MscnClusterExample(
    string? SourcePath,
    int? TileX,
    int? TileY,
    uint Ck24,
    byte Ck24Type,
    ushort Ck24ObjectId,
    int SurfaceCount,
    int ValidMdosRefCount,
    int DistinctMdosCount,
    int InvalidMdosRefCount,
    int MeshVertexCount,
    string AlignmentMode);

public sealed record Pm4MscnRelationshipReport(
    string InputDirectory,
    int FileCount,
    int FilesWithMscn,
    int FilesWithTileCoordinates,
    int TotalMscnPointCount,
    IReadOnlyList<Pm4RelationshipEdgeSummary> Relationships,
    Pm4MscnCoordinateSummary CoordinateSpace,
    IReadOnlyList<Pm4FieldDistribution> ClusterDistributions,
    IReadOnlyList<Pm4MscnClusterExample> TopNonZeroClusters,
    IReadOnlyList<Pm4MscnClusterExample> TopZeroClusters,
    IReadOnlyList<Pm4MscnClusterExample> TopInvalidMdosClusters,
    IReadOnlyList<string> Notes);

public sealed record Pm4MspiInterpretationSummary(
    int ActiveLinkCount,
    int IndicesModeOnlyCount,
    int TrianglesModeOnlyCount,
    int BothModesCount,
    int NeitherModeCount);

public sealed record Pm4LinkIdPatternSummary(
    int TotalCount,
    int SentinelTileLinkCount,
    int ZeroCount,
    int OtherCount,
    IReadOnlyList<Pm4ValueFrequency> TopDecodedTiles,
    IReadOnlyList<Pm4ValueFrequency> TopOtherValues);

public sealed record Pm4UnknownFinding(
    string Name,
    string Status,
    string Evidence,
    string NextStep);

public sealed record Pm4MslkFamilySummary(
    string FamilyKey,
    int FileCount,
    int EntryCount,
    int DirectMsurFitCount,
    int DirectMprlFitCount,
    int NonZeroGroupObjectIdCount,
    int GroupObjectIdMatchesMprlKeyCount,
    int ZeroLinkIdCount,
    int SentinelTileLinkCount,
    int OtherLinkIdCount,
    int ActiveMspiWindowCount,
    int DistinctRefIndexCount,
    int DistinctGroupObjectIdCount,
    IReadOnlyList<Pm4ValueFrequency> TopMismatchDomains,
    IReadOnlyList<Pm4ValueFrequency> TopDecodedTiles);

public sealed record Pm4MsurFamilySummary(
    string FamilyKey,
    int FileCount,
    int SurfaceCount,
    int DistinctCk24Count,
    int DistinctCk24TypeCount,
    int DistinctCk24ObjectIdCount,
    int DistinctMdosIndexCount,
    int IncomingMslkCount,
    int DistinctIncomingMslkFamilyCount,
    double AverageIndexCount,
    double AveragePlaneDistance,
    double AverageNormalZ,
    IReadOnlyList<Pm4ValueFrequency> TopCk24Types,
    IReadOnlyList<Pm4ValueFrequency> TopCk24Values);

public sealed record Pm4UnknownsReport(
    string InputDirectory,
    int FileCount,
    int NonEmptyFileCount,
    IReadOnlyList<Pm4CorpusChunkAudit> ChunkPopulation,
    IReadOnlyList<Pm4RelationshipEdgeSummary> Relationships,
    IReadOnlyList<Pm4FieldDistribution> FieldDistributions,
    IReadOnlyList<Pm4MslkFamilySummary> TopMslkFamilies,
    IReadOnlyList<Pm4MsurFamilySummary> TopMsurFamilies,
    Pm4LinkIdPatternSummary LinkIdPatterns,
    Pm4MspiInterpretationSummary MspiInterpretation,
    IReadOnlyList<Pm4UnknownFinding> Unknowns,
    IReadOnlyList<string> Notes);

public sealed record Pm4MshdMetricCorrelation(
    string Metric,
    int ExactMatchCount,
    int WithinOneCount,
    double PearsonCorrelation);

public sealed record Pm4MshdFieldSummary(
    string Field,
    int DistinctCount,
    int ZeroCount,
    int NonZeroCount,
    IReadOnlyList<Pm4ValueFrequency> TopValues,
    IReadOnlyList<Pm4MshdMetricCorrelation> MetricCorrelations);

public sealed record Pm4MshdRelationshipSummary(
    string Relationship,
    int MatchCount,
    int FileCount,
    string Notes);

public sealed record Pm4MshdReport(
    string InputDirectory,
    int FileCount,
    int FilesWithMshd,
    IReadOnlyList<Pm4MshdFieldSummary> Fields,
    IReadOnlyList<Pm4MshdRelationshipSummary> Relationships,
    IReadOnlyList<string> Notes);