namespace WowViewer.Core.Editor.Eras;

/// <summary>A single era-scoped value paired with the build interval it covers.</summary>
public readonly record struct EraHandler<T>(EditorBuildEraRange Range, T Handler, int RegistrationOrder);

/// <summary>
/// Deterministically resolves the most specific era-scoped handler covering a build. This is the
/// implementation behind editor era handlers (Spec 166 FR-009) and any other place a plugin wants a
/// build-range fallback chain rather than a single hardcoded default.
/// </summary>
public static class EraHandlerResolver
{
    /// <summary>
    /// Picks the handler whose range contains <paramref name="build"/> and is most specific (most
    /// bounded). Ties are broken by <paramref name="registryOrder"/>, i.e. the first registered
    /// handler wins, so resolution is stable across runs. A <paramref name="fallback"/> (an
    /// <see cref="EditorBuildEraRange.All"/> handler) is returned only when nothing more specific
    /// matches or no range matches at all.
    /// </summary>
    public static EraHandler<T>? Resolve<T>(EditorBuildVersion build, IReadOnlyList<EraHandler<T>> handlers, EraHandler<T>? fallback = null)
    {
        ArgumentNullException.ThrowIfNull(handlers);

        EraHandler<T>? best = null;
        int bestSpecificity = int.MinValue;
        int bestOrder = int.MaxValue;

        for (int index = 0; index < handlers.Count; index++)
        {
            EraHandler<T> candidate = handlers[index];
            if (!candidate.Range.Contains(build))
                continue;

            int specificity = candidate.Range.Specificity;
            int order = candidate.RegistrationOrder >= 0 ? candidate.RegistrationOrder : index;

            if (IsBetter(specificity, order, bestSpecificity, bestOrder))
            {
                best = candidate;
                bestSpecificity = specificity;
                bestOrder = order;
            }
        }

        if (best is not null)
            return best;

        if (fallback is not null && fallback.Value.Range.Contains(build))
            return fallback;

        return null;
    }

    private static bool IsBetter(int specificity, int order, int bestSpecificity, int bestOrder)
    {
        if (specificity != bestSpecificity)
            return specificity > bestSpecificity;

        return order < bestOrder;
    }
}