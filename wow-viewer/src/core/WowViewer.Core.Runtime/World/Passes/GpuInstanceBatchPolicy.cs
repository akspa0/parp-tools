namespace WowViewer.Core.Runtime.World.Passes;

/// <summary>Which instanced batch a submission belongs to.</summary>
public enum GpuInstanceBatchKind
{
    /// <summary>Fully opaque: blending off, depth write on.</summary>
    Opaque = 0,

    /// <summary>Distance-faded: blending on, depth write off, drawn after the opaque batch.</summary>
    Faded = 1,

    /// <summary>Contributes nothing; must not be submitted at all.</summary>
    Skip = 2,
}

/// <summary>
/// Decides which instanced batch a model instance belongs to (spec 207 US1).
/// </summary>
/// <remarks>
/// <para>
/// Instancing previously admitted only <c>fade &gt;= 0.999</c>, so every distance-faded instance was
/// drawn one draw call at a time. Fade begins at 80% of cull distance, so by area the fade band is
/// <c>1 - 0.8^2</c> = <b>36% of the visible disc</b> -- roughly a third of visible doodads, and
/// exactly the population instancing helps most: distant, numerous, repeated.
/// </para>
/// <para>
/// The gate could not simply be deleted: one instanced draw shares one blend state, so faded
/// geometry riding in the opaque batch would render fully opaque. Splitting the batch by fade state
/// keeps both correct, at two draws per model instead of one draw per instance.
/// </para>
/// <para>
/// This lives in Core rather than the renderer because <b>no test project references the viewer</b>,
/// so a decision that only exists in <c>ModelRenderer</c> cannot be tested at all.
/// </para>
/// </remarks>
public static class GpuInstanceBatchPolicy
{
    /// <summary>At or above this an instance is opaque enough to need no blending.</summary>
    public const float OpaqueFadeThreshold = 0.999f;

    /// <summary>
    /// Below this an instance cannot affect an 8-bit target, so drawing it is pure cost.
    /// </summary>
    public const float MinimumVisibleFade = 1.0f / 255.0f;

    public static GpuInstanceBatchKind Classify(float fadeAlpha)
    {
        if (float.IsNaN(fadeAlpha) || fadeAlpha < MinimumVisibleFade)
            return GpuInstanceBatchKind.Skip;

        return fadeAlpha >= OpaqueFadeThreshold
            ? GpuInstanceBatchKind.Opaque
            : GpuInstanceBatchKind.Faded;
    }

    /// <summary>
    /// Draw calls for one model, given how its instances classify.
    /// </summary>
    /// <remarks>
    /// The point of the whole exercise: this is bounded by 2 regardless of instance count, where the
    /// previous behaviour was <c>1 + fadedCount</c> — one instanced draw for the opaque instances
    /// plus one unbatched draw for every faded one.
    /// </remarks>
    public static int DrawCallsForModel(int opaqueInstances, int fadedInstances)
        => (opaqueInstances > 0 ? 1 : 0) + (fadedInstances > 0 ? 1 : 0);

    /// <summary>Draw calls the pre-spec-207 behaviour would have issued for the same instances.</summary>
    public static int LegacyDrawCallsForModel(int opaqueInstances, int fadedInstances)
        => (opaqueInstances > 0 ? 1 : 0) + fadedInstances;
}
