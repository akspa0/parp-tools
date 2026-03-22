namespace Pm4Research.Core;

public static class Pm4ResearchStructureConfidenceAnalyzer
{
    public static Pm4StructureConfidenceReport AnalyzeDirectory(string inputDirectory)
    {
        Pm4CorpusAuditReport audit = Pm4ResearchAuditAnalyzer.AnalyzeDirectory(inputDirectory);
        Pm4UnknownsReport unknowns = Pm4ResearchUnknownsAnalyzer.AnalyzeDirectory(inputDirectory);
        Pm4MscnRelationshipReport mscn = Pm4ResearchMscnAnalyzer.AnalyzeDirectory(inputDirectory);
        Pm4LinkageReport linkage = Pm4ResearchLinkageAnalyzer.AnalyzeDirectory(inputDirectory);
        Pm4MsurGeometryReport msurGeometry = Pm4ResearchMsurGeometryAnalyzer.AnalyzeDirectory(inputDirectory);
        Pm4RefIndexClassifierReport refIndexClassifier = Pm4ResearchMslkRefIndexClassifier.AnalyzeDirectory(inputDirectory);

        Dictionary<string, Pm4CorpusChunkAudit> chunks = audit.ChunkAudits.ToDictionary(static chunk => chunk.Signature, StringComparer.Ordinal);
        Dictionary<string, Pm4RelationshipEdgeSummary> unknownRelationships = unknowns.Relationships.ToDictionary(static relation => relation.Edge, StringComparer.Ordinal);
        Dictionary<string, Pm4RelationshipEdgeSummary> mscnRelationships = mscn.Relationships.ToDictionary(static relation => relation.Edge, StringComparer.Ordinal);
        Dictionary<string, Pm4RelationshipEdgeSummary> linkageRelationships = linkage.Relationships.ToDictionary(static relation => relation.Edge, StringComparer.Ordinal);
        Dictionary<string, Pm4FieldDistribution> distributions = unknowns.FieldDistributions.ToDictionary(static distribution => distribution.Field, StringComparer.Ordinal);

        List<Pm4ChunkStructureConfidence> chunkConfidence =
        [
            BuildChunk("MSPV", 12, chunks, "high", "high", "low", "Vector stream with zero stride remainders and fully verified MSPI->MSPV references.", "Treat this as one of the stable anchor chunks for future semantic work."),
            BuildChunk("MSPI", 4, chunks, "high", "high", "low", BuildEdgeEvidence(unknownRelationships, "MSPI -> MSPV"), "Keep using it as the verified path-index stream while other link semantics remain open."),
            BuildChunk("MSVT", 12, chunks, "high", "high", "low", "Vector stream with zero stride remainders and fully verified MSVI->MSVT references.", "Use this as a stable geometry baseline before applying viewer/world transforms."),
            BuildChunk("MSVI", 4, chunks, "high", "high", "low", BuildEdgeEvidence(unknownRelationships, "MSVI -> MSVT"), "Any future miss here should be treated as a real layout regression."),
            BuildChunk("MSLK", 20, chunks, "high", "low", "high", $"{BuildEdgeEvidence(unknownRelationships, "MSLK.RefIndex -> MSUR")} classifier: resolvedFamilies={refIndexClassifier.Summary.ResolvedFamilyCount}, resolvedEntries={refIndexClassifier.Summary.ResolvedEntryCount}.", "Keep the 20-byte record layout, but treat RefIndex and GroupObjectId semantics as provisional until the ambiguous family set shrinks further."),
            BuildChunk("MSUR", 32, chunks, "high", "medium", "medium", $"{BuildEdgeEvidence(unknownRelationships, "MSUR.Msvi window -> MSVI")} {BuildEdgeEvidence(mscnRelationships, "MSUR.MdosIndex -> MSCN")} geometry: strongNormals={msurGeometry.StrongAlignmentCount}/{msurGeometry.AnalyzedSurfaceCount}.", "Bytes 4..19 now look geometry-real; the remaining semantic risk is mostly in MdosIndex and PackedParams interpretation."),
            BuildChunk("MSCN", 12, chunks, "high", "medium", "medium", $"{BuildEdgeEvidence(mscnRelationships, "MSUR.MdosIndex -> MSCN")} {BuildEdgeEvidence(mscnRelationships, "CK24 MSCN(raw) bounds -> MSVT bounds")}", "Treat MSCN as a real linked Vector3 stream, but keep coordinate-space claims separate from layout certainty."),
            BuildChunk("MPRL", 24, chunks, "high", "low", "high", $"{DescribeChunk(chunks, "MPRL")} {DescribeDistribution(distributions, "MPRL.Unk14")}", "Keep the 24-byte layout pinned and resist renaming the unknown fields without ADT/object correlation."),
            BuildChunk("MPRR", 4, chunks, "high", "very-low", "very-high", $"{BuildEdgeEvidence(unknownRelationships, "MPRR.Value1 -> MPRL")} {BuildEdgeEvidence(unknownRelationships, "MPRR.Value1 -> MSVT")}", "Treat MPRR as structurally real but semantically open until clustered against tile/object truth."),
            BuildChunk("MSHD", 32, chunks, "high", "very-low", "medium", $"{DescribeChunk(chunks, "MSHD")} {DescribeDistribution(distributions, "MSHD.Field00")}", "Do not assign domain names to MSHD fields before a dedicated correlation pass."),
            BuildChunk("MDBH", 4, chunks, "high", "low", "medium", $"{DescribeChunk(chunks, "MDBH")} {BuildEdgeEvidence(unknownRelationships, "MDOS.buildingIndex -> MDBH")}", "Treat the count field as structurally safe, but keep building-index semantics open."),
            BuildChunk("MDOS", 8, chunks, "high", "very-low", "very-high", $"{DescribeChunk(chunks, "MDOS")} {BuildEdgeEvidence(unknownRelationships, "MDOS.buildingIndex -> MDBH")}", "Sparse population means MDOS semantics should stay tentative until Wintergrasp/destructible truth is checked."),
            BuildChunk("MDSF", 8, chunks, "high", "medium", "medium", $"{BuildEdgeEvidence(unknownRelationships, "MDSF.MsurIndex -> MSUR")} {BuildEdgeEvidence(unknownRelationships, "MDSF.MdosIndex -> MDOS")}", "This linkage is structurally strong, but only one tile currently gives it real semantic coverage."),
        ];

        Pm4FieldDistribution? mslkTypeFlags = FindDistribution(distributions, "MSLK.TypeFlags");
        Pm4FieldDistribution? mslkSubtype = FindDistribution(distributions, "MSLK.Subtype");
        Pm4FieldDistribution? mslkSystemFlag = FindDistribution(distributions, "MSLK.SystemFlag");
        Pm4FieldDistribution? msurGroupKey = FindDistribution(distributions, "MSUR.GroupKey");
        Pm4FieldDistribution? msurAttributeMask = FindDistribution(distributions, "MSUR.AttributeMask");
        Pm4FieldDistribution? mprlUnk14 = FindDistribution(distributions, "MPRL.Unk14");
        Pm4FieldDistribution? mprlUnk16 = FindDistribution(distributions, "MPRL.Unk16");

        List<Pm4FieldConfidenceFinding> fieldConfidence =
        [
            BuildField("MSLK.TypeFlags", "named-guess", "high", "low", "high", DescribeDistribution(mslkTypeFlags), "Correlate values with mismatch families, CK24 types, and linked MPRL populations before keeping the name."),
            BuildField("MSLK.Subtype", "named-guess", "high", "low", "high", DescribeDistribution(mslkSubtype), "Treat it as a compact classifier only; do not call it floor/layer without proof."),
            BuildField("MSLK.SystemFlag", "constant-field", "high", "low", "medium", DescribeDistribution(mslkSystemFlag), "Its constancy is real, but its purpose is still open."),
            BuildField("MSLK.GroupObjectId", "conflicted-identity", "high", "low", "high", BuildEdgeEvidence(linkageRelationships, "RefIndex mismatch low24(GroupObjectId) -> CK24"), "Keep it out of any direct ownership/hierarchy claims until it matches a target domain consistently."),
            BuildField("MSLK.LinkId", "partially-validated", "high", "medium", "medium", $"sentinelTile={unknowns.LinkIdPatterns.SentinelTileLinkCount}/{unknowns.LinkIdPatterns.TotalCount}, zero={unknowns.LinkIdPatterns.ZeroCount}, other={unknowns.LinkIdPatterns.OtherCount}", "Treat it as a tile-link envelope today, not a generic object identifier."),
            BuildField("MSLK.RefIndex", "conflicted-reference", "high", "low", "very-high", $"{BuildEdgeEvidence(unknownRelationships, "MSLK.RefIndex -> MSUR")} classifier: resolvedFamilies={refIndexClassifier.Summary.ResolvedFamilyCount}, ambiguousFamilies={refIndexClassifier.Summary.AmbiguousFamilyCount}.", "Keep the field byte range, but stop assuming every value is an MSUR slot; use the family classifier before promoting any replacement semantics."),
            BuildField("MSUR.GroupKey", "named-guess", "high", "low", "high", DescribeDistribution(msurGroupKey), "Correlate it with CK24 families and scene roles before keeping the name."),
            BuildField("MSUR.AttributeMask", "named-guess", "high", "low", "high", DescribeDistribution(msurAttributeMask), "Do not map bits to walkability/liquid/collision without tile-truth checks."),
            BuildField("MSUR.Normal + Height", "geometry-validated-plane", "high", "high", "low", $"normals: strong={msurGeometry.StrongAlignmentCount}/{msurGeometry.AnalyzedSurfaceCount}, avg|dot|={msurGeometry.AverageAbsoluteDot:F4}; height best candidate=storedPlane.- meanAbsErr={msurGeometry.HeightCandidates.First().MeanAbsoluteError:F4}", "Treat bytes 4..15 as proven surface normals; treat the final float as a signed plane-distance term rather than a generic vertical height."),
            BuildField("MSUR.MsviFirstIndex + IndexCount", "verified-reference", "high", "high", "low", BuildEdgeEvidence(unknownRelationships, "MSUR.Msvi window -> MSVI"), "This is one of the strongest current structural seams; treat misses as a decoder bug."),
            BuildField("MSUR.MdosIndex", "partially-validated", "high", "medium", "high", BuildEdgeEvidence(mscnRelationships, "MSUR.MdosIndex -> MSCN"), "Treat it as a strong but incomplete MSCN linkage, not a closed scene-node semantic."),
            BuildField("MSUR.PackedParams -> CK24/Ck24Type/Ck24ObjectId", "derived-bit-slice", "high", "medium", "medium", $"distinctCK24={linkage.IdentitySummary.DistinctCk24Count}, distinctObjectId={linkage.IdentitySummary.DistinctCk24ObjectIdCount}, reusedObjectIdGroups={linkage.IdentitySummary.ReusedObjectIdGroupCount}", "Keep this as a verified derived identity slice; do not over-promote low16 object id into a full hierarchy key."),
            BuildField("MSCN.Vector3 coordinates", "conflicted-frame", "high", "medium", "medium", $"rawOverlap={GetEdgeCounts(mscnRelationships, "CK24 MSCN(raw) bounds -> MSVT bounds")}, swappedOverlap={GetEdgeCounts(mscnRelationships, "CK24 MSCN(swapped XY) bounds -> MSVT bounds")}", "Treat the 12-byte XYZ layout as stable while keeping coordinate-space transforms explicitly open."),
            BuildField("MPRL.Unk04", "unknown-field", "high", "very-low", "high", "Current standalone reports still do not prove that bytes 4..5 encode absolute yaw or a stable rotation basis.", "Validate against trusted object placements before renaming it."),
            BuildField("MPRL.Unk14", "unknown-field", "high", "low", "high", DescribeDistribution(mprlUnk14), "Floor/level is plausible, but still needs object-truth validation."),
            BuildField("MPRL.Unk16", "unknown-field", "high", "low", "high", DescribeDistribution(mprlUnk16), "Treat the two-value split as a class marker only until the classes are proven."),
            BuildField("MPRR.Value1", "conflicted-reference", "high", "very-low", "very-high", $"{BuildEdgeEvidence(unknownRelationships, "MPRR.Value1 -> MPRL")} {BuildEdgeEvidence(unknownRelationships, "MPRR.Value1 -> MSVT")}", "Do not collapse it to one target domain until cluster-level evidence breaks the tie."),
            BuildField("MSHD.Field00..Field1C", "unknown-header", "high", "very-low", "medium", $"{DescribeDistribution(distributions, "MSHD.Field00")} {DescribeDistribution(distributions, "MSHD.Field04")}", "Keep the fields readable by offset only until a real header correlation pass exists."),
            BuildField("MDOS.DestructibleBuildingIndex", "sparse-reference", "high", "very-low", "very-high", BuildEdgeEvidence(unknownRelationships, "MDOS.buildingIndex -> MDBH"), "Do not call this an MDBH slot index yet; one populated tile is not enough proof."),
        ];

        List<Pm4StructureConflict> conflicts =
        [
            BuildConflict("MSLK.LinkId", "Always 0xFFFFFFFF.", "Tile-link envelope that varies across the corpus.", "The field is not constant in PM4 data, even if some PD4-era notes described it that way.", $"{unknowns.LinkIdPatterns.SentinelTileLinkCount} sentinel-style links out of {unknowns.LinkIdPatterns.TotalCount} total; zero={unknowns.LinkIdPatterns.ZeroCount}; other={unknowns.LinkIdPatterns.OtherCount}.", "Keep any constant-field assumption out of the decoder and treat LinkId as data-bearing."),
            BuildConflict("MSLK.RefIndex", "Always an MSUR surface index.", "Frequently points outside MSUR in real PM4 data.", "The record layout survives, but the semantic target is not closed.", BuildEdgeEvidence(unknownRelationships, "MSLK.RefIndex -> MSUR"), "Cluster mismatch families before assigning a replacement target domain."),
            BuildConflict("MSUR.Normal + Height", "Bytes 4..19 are just a plausible normal/height guess.", "The vector is geometry-validated and the final float behaves like negative plane distance.", "The old risk was semantic uncertainty; the new risk is misnaming the last float as generic height when it behaves like a plane term.", $"strongNormals={msurGeometry.StrongAlignmentCount}/{msurGeometry.AnalyzedSurfaceCount}; bestHeightCandidate={msurGeometry.HeightCandidates.First().Candidate} meanAbsErr={msurGeometry.HeightCandidates.First().MeanAbsoluteError:F4}", "Keep the normal naming, but rename/re-describe the last float as a signed plane-distance term in future decoder cleanups."),
            BuildConflict("MSUR.MdosIndex", "Direct closed-form scene-node index.", "Strong MSCN linkage with a real miss population.", "The field is probably important, but not fully solved.", BuildEdgeEvidence(mscnRelationships, "MSUR.MdosIndex -> MSCN"), "Break misses down by CK24 family and compare with alternate scene-node hypotheses."),
            BuildConflict("MSCN coordinate frame", "Standalone MSCN needs blanket XY swap to align with PM4 geometry.", "Raw overlap beats swapped overlap in the standalone corpus.", "Older transform lore does not hold as a universal rule in the fresh reader path.", $"raw={GetEdgeCounts(mscnRelationships, "CK24 MSCN(raw) bounds -> MSVT bounds")}; swapped={GetEdgeCounts(mscnRelationships, "CK24 MSCN(swapped XY) bounds -> MSVT bounds")}", "Keep coordinate transforms as explicit hypotheses, not baked decoder truth."),
            BuildConflict("MPRR.Value1", "Pure MPRL reference or pure geometry reference.", "Fits multiple target domains heavily.", "The field is still overloaded or misidentified.", $"{BuildEdgeEvidence(unknownRelationships, "MPRR.Value1 -> MPRL")} {BuildEdgeEvidence(unknownRelationships, "MPRR.Value1 -> MSVT")}", "Resolve this with tile-family clustering instead of a global rename."),
        ];

        Pm4ConfidenceSummary summary = new(
            Count(chunkConfidence, static item => item.LayoutConfidence == "high"),
            Count(chunkConfidence, static item => item.LayoutConfidence == "medium"),
            Count(chunkConfidence, static item => item.LayoutConfidence == "low"),
            Count(fieldConfidence, static item => item.SemanticConfidence == "high"),
            Count(fieldConfidence, static item => item.SemanticConfidence == "medium"),
            Count(fieldConfidence, static item => item.SemanticConfidence == "low"),
            Count(fieldConfidence, static item => item.SemanticConfidence == "very-low"),
            conflicts.Count);

        List<string> notes =
        [
            "This report is intentionally conservative: byte-layout confidence and field-meaning confidence are tracked separately.",
            "High layout confidence does not mean the existing property names are trustworthy.",
            "Current strongest byte/semantic anchors are the vector/index streams (MSPV, MSPI, MSVT, MSVI), MSUR normal/plane geometry, MSUR->MSVI, and MDSF->{MSUR,MDOS} linkage.",
            "Current highest hallucination-risk zones are MSLK.RefIndex semantics, MPRL unknown fields, MPRR, and sparse destructible-building integration.",
        ];

        return new Pm4StructureConfidenceReport(
            inputDirectory,
            audit.FileCount,
            summary,
            chunkConfidence,
            fieldConfidence,
            conflicts,
            notes);
    }

    private static Pm4ChunkStructureConfidence BuildChunk(
        string signature,
        int? stride,
        IReadOnlyDictionary<string, Pm4CorpusChunkAudit> chunks,
        string layoutConfidence,
        string semanticConfidence,
        string hallucinationRisk,
        string evidence,
        string nextStep)
    {
        chunks.TryGetValue(signature, out Pm4CorpusChunkAudit? chunk);
        return new Pm4ChunkStructureConfidence(
            signature,
            stride,
            chunk?.FileCount ?? 0,
            chunk?.DataFileCount ?? 0,
            chunk?.TotalEntryCount ?? 0,
            chunk?.FilesWithStrideRemainders ?? 0,
            layoutConfidence,
            semanticConfidence,
            hallucinationRisk,
            evidence,
            nextStep);
    }

    private static Pm4FieldConfidenceFinding BuildField(
        string field,
        string classification,
        string layoutConfidence,
        string semanticConfidence,
        string hallucinationRisk,
        string evidence,
        string nextStep)
    {
        return new Pm4FieldConfidenceFinding(
            field,
            classification,
            layoutConfidence,
            semanticConfidence,
            hallucinationRisk,
            evidence,
            nextStep);
    }

    private static Pm4StructureConflict BuildConflict(
        string field,
        string legacyClaim,
        string currentDecode,
        string conflict,
        string evidence,
        string nextStep)
    {
        return new Pm4StructureConflict(field, legacyClaim, currentDecode, conflict, evidence, nextStep);
    }

    private static string DescribeChunk(IReadOnlyDictionary<string, Pm4CorpusChunkAudit> chunks, string signature)
    {
        if (!chunks.TryGetValue(signature, out Pm4CorpusChunkAudit? chunk))
            return $"{signature}: no corpus chunk audit found.";

        return $"{signature}: files={chunk.FileCount}, dataFiles={chunk.DataFileCount}, entries={chunk.TotalEntryCount}, remainderFiles={chunk.FilesWithStrideRemainders}, sizes={string.Join(",", chunk.ExampleSizes)}.";
    }

    private static string BuildEdgeEvidence(IReadOnlyDictionary<string, Pm4RelationshipEdgeSummary> relationships, string edge)
    {
        if (!relationships.TryGetValue(edge, out Pm4RelationshipEdgeSummary? relation))
            return $"{edge}: no relationship evidence found.";

        return $"{edge}: status={relation.Status}, fits={relation.Fits}, misses={relation.Misses}.";
    }

    private static string GetEdgeCounts(IReadOnlyDictionary<string, Pm4RelationshipEdgeSummary> relationships, string edge)
    {
        if (!relationships.TryGetValue(edge, out Pm4RelationshipEdgeSummary? relation))
            return "missing";

        return $"{relation.Fits}/{relation.Misses}";
    }

    private static Pm4FieldDistribution? FindDistribution(IReadOnlyDictionary<string, Pm4FieldDistribution> distributions, string field)
    {
        distributions.TryGetValue(field, out Pm4FieldDistribution? distribution);
        return distribution;
    }

    private static string DescribeDistribution(IReadOnlyDictionary<string, Pm4FieldDistribution> distributions, string field)
    {
        return DescribeDistribution(FindDistribution(distributions, field));
    }

    private static string DescribeDistribution(Pm4FieldDistribution? distribution)
    {
        if (distribution is null)
            return "No distribution evidence found.";

        string range = string.IsNullOrWhiteSpace(distribution.Range) ? string.Empty : $", range={distribution.Range}";
        string topValues = distribution.TopValues.Count == 0
            ? "none"
            : string.Join(", ", distribution.TopValues.Take(3).Select(static value => $"{value.Value}->{value.Count}"));
        return $"total={distribution.TotalCount}, distinct={distribution.DistinctCount}{range}, top={topValues}.";
    }

    private static int Count<T>(IEnumerable<T> items, Func<T, bool> predicate)
    {
        int count = 0;
        foreach (T item in items)
        {
            if (predicate(item))
                count++;
        }

        return count;
    }
}