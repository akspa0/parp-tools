namespace WowViewer.Core.Editor.Eras;

/// <summary>
/// An inclusive/exclusive build-version interval that a plugin or handler declares. A null bound is
/// unbounded on that side, so <see cref="All"/> covers every build.
/// </summary>
public readonly record struct EditorBuildEraRange(EditorBuildVersion? MinInclusive, EditorBuildVersion? MaxExclusive, string? Label = null)
{
    public static EditorBuildEraRange All => new(null, null);

    public static EditorBuildEraRange AtLeast(EditorBuildVersion min, string? label = null)
        => new(min, null, label);

    public static EditorBuildEraRange UpTo(EditorBuildVersion maxExclusive, string? label = null)
        => new(null, maxExclusive, label);

    public static EditorBuildEraRange Between(EditorBuildVersion minInclusive, EditorBuildVersion maxExclusive, string? label = null)
    {
        if (minInclusive >= maxExclusive)
            throw new ArgumentOutOfRangeException(nameof(maxExclusive), "A build era range must have a max exclusive bound strictly greater than its min inclusive bound.");

        return new EditorBuildEraRange(minInclusive, maxExclusive, label);
    }

    public bool Contains(EditorBuildVersion build)
        => (MinInclusive is null || build >= MinInclusive.Value)
           && (MaxExclusive is null || build < MaxExclusive.Value);

    /// <summary>
    /// How constrained this range is: 2 for a fully bounded range, 1 for a half-open range, 0 for
    /// <see cref="All"/>. More specific ranges win resolution.
    /// </summary>
    public int Specificity => (MinInclusive is null ? 0 : 1) + (MaxExclusive is null ? 0 : 1);

    public override string ToString()
        => Label is not null
            ? Label
            : $"[{Format(MinInclusive)}, {Format(MaxExclusive)})";

    private static string Format(EditorBuildVersion? bound)
        => bound is null ? "∞" : bound.Value.Original;
}