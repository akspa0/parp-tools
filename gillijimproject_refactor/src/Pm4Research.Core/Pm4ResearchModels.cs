using System.Numerics;

namespace Pm4Research.Core;

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
    ushort SystemFlag);

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

public sealed record Pm4MdbhEntry(
    uint DestructibleBuildingCount);

public sealed record Pm4MdbiEntry(
    uint DestructibleBuildingIndex);

public sealed record Pm4MdbfEntry(
    string Filename,
    int RawLength);

public sealed record Pm4MdosEntry(
    uint DestructibleBuildingIndex,
    uint DestructionState);

public sealed record Pm4MdsfEntry(
    uint MsurIndex,
    uint MdosIndex);

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

public sealed record Pm4ResearchFile(
    string? SourcePath,
    uint Version,
    IReadOnlyList<Pm4ChunkRecord> Chunks,
    Pm4KnownChunkSet KnownChunks,
    IReadOnlyList<string> Diagnostics);

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
    IReadOnlyList<string> TopDiagnostics);

public sealed record Pm4MslkRefIndexDomainFit(
    string Domain,
    int Count,
    bool Fits);

public sealed record Pm4MslkRefIndexMismatch(
    int MslkIndex,
    ushort RefIndex,
    uint GroupObjectId,
    uint LinkId,
    byte TypeFlags,
    byte Subtype,
    int MspiFirstIndex,
    byte MspiIndexCount,
    IReadOnlyList<Pm4MslkRefIndexDomainFit> DomainFits);

public sealed record Pm4MslkRefIndexFileAudit(
    string? SourcePath,
    int? TileX,
    int? TileY,
    uint Version,
    int MslkCount,
    int MsurCount,
    int InvalidRefIndexCount,
    IReadOnlyList<Pm4MslkRefIndexMismatch> Mismatches);

public sealed record Pm4MslkRefIndexDomainSummary(
    string Domain,
    int MatchingMismatchCount,
    int NonMatchingMismatchCount);

public sealed record Pm4MslkRefIndexCorpusAudit(
    string InputDirectory,
    int FileCount,
    int FilesWithMismatches,
    int TotalMismatchCount,
    IReadOnlyList<Pm4MslkRefIndexDomainSummary> DomainSummaries,
    IReadOnlyList<Pm4MslkRefIndexFileAudit> TopFiles);

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

public sealed record Pm4UnknownsReport(
    string InputDirectory,
    int FileCount,
    int NonEmptyFileCount,
    IReadOnlyList<Pm4CorpusChunkAudit> ChunkPopulation,
    IReadOnlyList<Pm4RelationshipEdgeSummary> Relationships,
    IReadOnlyList<Pm4FieldDistribution> FieldDistributions,
    Pm4LinkIdPatternSummary LinkIdPatterns,
    Pm4MspiInterpretationSummary MspiInterpretation,
    IReadOnlyList<Pm4UnknownFinding> Unknowns,
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
    IReadOnlyList<Pm4ValueFrequency> TopLow16ObjectIds);

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

public sealed record Pm4ConfidenceSummary(
    int HighLayoutChunkCount,
    int MediumLayoutChunkCount,
    int LowLayoutChunkCount,
    int HighSemanticFieldCount,
    int MediumSemanticFieldCount,
    int LowSemanticFieldCount,
    int VeryLowSemanticFieldCount,
    int ConflictCount);

public sealed record Pm4ChunkStructureConfidence(
    string Signature,
    int? Stride,
    int FileCount,
    int DataFileCount,
    int TotalEntryCount,
    int FilesWithStrideRemainders,
    string LayoutConfidence,
    string SemanticConfidence,
    string HallucinationRisk,
    string Evidence,
    string NextStep);

public sealed record Pm4FieldConfidenceFinding(
    string Field,
    string Classification,
    string LayoutConfidence,
    string SemanticConfidence,
    string HallucinationRisk,
    string Evidence,
    string NextStep);

public sealed record Pm4StructureConflict(
    string Field,
    string LegacyClaim,
    string CurrentDecode,
    string Conflict,
    string Evidence,
    string NextStep);

public sealed record Pm4StructureConfidenceReport(
    string InputDirectory,
    int FileCount,
    Pm4ConfidenceSummary Summary,
    IReadOnlyList<Pm4ChunkStructureConfidence> ChunkConfidence,
    IReadOnlyList<Pm4FieldConfidenceFinding> FieldConfidence,
    IReadOnlyList<Pm4StructureConflict> Conflicts,
    IReadOnlyList<string> Notes);

public sealed record Pm4HeightCandidateSummary(
    string Candidate,
    float MeanAbsoluteError,
    int FitsWithinPointOne,
    int FitsWithinOne,
    int FitsWithinFour);

public sealed record Pm4MsurGeometryExample(
    string? SourcePath,
    int? TileX,
    int? TileY,
    int SurfaceIndex,
    uint Ck24,
    byte Ck24Type,
    ushort Ck24ObjectId,
    byte IndexCount,
    float StoredNormalMagnitude,
    float SignedDot,
    float AbsoluteDot,
    float Height,
    float GeometricPlaneDistance,
    float StoredPlaneDistance,
    Vector3 Centroid);

public sealed record Pm4MsurGeometryReport(
    string InputDirectory,
    int FileCount,
    int AnalyzedSurfaceCount,
    int DegenerateSurfaceCount,
    int UnitLikeStoredNormalCount,
    int StrongAlignmentCount,
    int ModerateAlignmentCount,
    int WeakAlignmentCount,
    int PositiveAlignmentCount,
    int NegativeAlignmentCount,
    float AverageStoredNormalMagnitude,
    float AverageAbsoluteDot,
    IReadOnlyList<Pm4HeightCandidateSummary> HeightCandidates,
    IReadOnlyList<Pm4FieldDistribution> Distributions,
    IReadOnlyList<Pm4MsurGeometryExample> BestAlignedExamples,
    IReadOnlyList<Pm4MsurGeometryExample> WorstAlignedExamples,
    IReadOnlyList<string> Notes);

public sealed record Pm4RefIndexDomainBaseline(
    string Domain,
    int FitCount,
    float Coverage);

public sealed record Pm4RefIndexFamilyDomainScore(
    string Domain,
    int FitCount,
    float Coverage,
    float BaselineCoverage,
    float CoverageDelta,
    float Lift);

public sealed record Pm4RefIndexClassifiedFamily(
    string FamilyKey,
    string Classification,
    string Confidence,
    int FileCount,
    int EntryCount,
    ushort MinRefIndex,
    ushort MaxRefIndex,
    IReadOnlyList<Pm4RefIndexFamilyDomainScore> DomainScores,
    IReadOnlyList<Pm4ValueFrequency> TopRefIndices);

public sealed record Pm4RefIndexClassificationSummary(
    int ResolvedFamilyCount,
    int AmbiguousFamilyCount,
    int ResolvedEntryCount,
    IReadOnlyList<Pm4ValueFrequency> ClassificationCounts);

public sealed record Pm4RefIndexClassifierReport(
    string InputDirectory,
    int FileCount,
    int FilesWithMismatches,
    int TotalMismatchCount,
    IReadOnlyList<Pm4RefIndexDomainBaseline> DomainBaselines,
    Pm4RefIndexClassificationSummary Summary,
    IReadOnlyList<Pm4RefIndexClassifiedFamily> TopFamilies,
    IReadOnlyList<string> Notes);

public sealed record Pm4MprlFootprintSummary(
    int TileRefCount,
    int LinkedRefCount,
    int LinkedNormalCount,
    int LinkedTerminatorCount,
    int TileInBoundsCount,
    int TileNearBoundsCount,
    int LinkedInBoundsCount,
    int LinkedNearBoundsCount,
    short? LinkedFloorMin,
    short? LinkedFloorMax);

public sealed record Pm4AnalysisReport(
    string? SourcePath,
    uint Version,
    IReadOnlyList<Pm4ChunkSummary> ChunkOrder,
    IReadOnlyList<string> UnrecognizedChunks,
    Pm4VectorSetSummary Mspv,
    Pm4VectorSetSummary Msvt,
    Pm4VectorSetSummary Mscn,
    Pm4VectorSetSummary MprlPositions,
    Pm4MprlSummary Mprl,
    IReadOnlyList<Pm4Ck24Summary> TopCk24Groups,
    IReadOnlyList<string> Diagnostics);

public sealed record Pm4ObjectHypothesis(
    string Family,
    int FamilyObjectIndex,
    uint Ck24,
    byte Ck24Type,
    ushort Ck24ObjectId,
    int SurfaceCount,
    int TotalIndexCount,
    IReadOnlyList<int> SurfaceIndices,
    IReadOnlyList<uint> MdosIndices,
    IReadOnlyList<byte> GroupKeys,
    IReadOnlyList<uint> MslkGroupObjectIds,
    IReadOnlyList<ushort> MslkRefIndices,
    Pm4Bounds3? Bounds,
    Pm4MprlFootprintSummary MprlFootprint);

public sealed record Pm4TileObjectHypothesisReport(
    string? SourcePath,
    int? TileX,
    int? TileY,
    uint Version,
    int Ck24GroupCount,
    int TotalHypothesisCount,
    IReadOnlyList<Pm4ObjectHypothesis> Objects,
    IReadOnlyList<string> Diagnostics);