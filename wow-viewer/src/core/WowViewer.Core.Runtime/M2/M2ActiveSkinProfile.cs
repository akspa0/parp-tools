using WowViewer.Core.M2;

namespace WowViewer.Core.Runtime.M2;

public sealed class M2ActiveSkinProfile
{
    public M2ActiveSkinProfile(
        M2ModelDocument model,
        M2SkinProfileSelection selection,
        M2SkinDocument skin,
        bool usesCompatibilityFallback)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(selection);
        ArgumentNullException.ThrowIfNull(skin);

        Model = model;
        Selection = selection;
        Skin = skin;
        UsesCompatibilityFallback = usesCompatibilityFallback;

        Dictionary<int, List<M2ActiveSkinBatch>> batchesBySection = new();
        int unmatchedBatchCount = 0;
        for (int batchIndex = 0; batchIndex < skin.Batches.Count; batchIndex++)
        {
            M2SkinBatch batch = skin.Batches[batchIndex];
            if (batch.SkinSectionIndex >= skin.SubmeshCount)
            {
                unmatchedBatchCount++;
                continue;
            }

            if (!batchesBySection.TryGetValue(batch.SkinSectionIndex, out List<M2ActiveSkinBatch>? sectionBatches))
            {
                sectionBatches = [];
                batchesBySection.Add(batch.SkinSectionIndex, sectionBatches);
            }

            sectionBatches.Add(new M2ActiveSkinBatch(batchIndex, batch));
        }

        List<M2ActiveSkinSection> activeSections = new(skin.SubmeshCount);
        int sectionsWithBatches = 0;
        for (int sectionIndex = 0; sectionIndex < skin.Submeshes.Count; sectionIndex++)
        {
            IReadOnlyList<M2ActiveSkinBatch> sectionBatches = batchesBySection.TryGetValue(sectionIndex, out List<M2ActiveSkinBatch>? matchedBatches)
                ? matchedBatches
                : [];
            if (sectionBatches.Count > 0)
                sectionsWithBatches++;

            activeSections.Add(new M2ActiveSkinSection(sectionIndex, skin.Submeshes[sectionIndex], sectionBatches));
        }

        ActiveSections = activeSections;
        SectionsWithBatchesCount = sectionsWithBatches;
        UnmatchedBatchCount = unmatchedBatchCount;
    }

    public M2ModelDocument Model { get; }

    public M2SkinProfileSelection Selection { get; }

    public M2SkinDocument Skin { get; }

    public bool UsesCompatibilityFallback { get; }

    public IReadOnlyList<M2ActiveSkinSection> ActiveSections { get; }

    public int ActiveSubmeshCount => Skin.SubmeshCount;

    public int ActiveSectionCount => ActiveSections.Count;

    public int SectionsWithBatchesCount { get; }

    public int ActiveBatchCount => Skin.BatchCount;

    public int UnmatchedBatchCount { get; }

    public int ActiveVertexLookupCount => Skin.VertexLookupCount;

    public int ActiveTriangleIndexCount => Skin.TriangleIndexCount;
}