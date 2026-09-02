namespace WowViewer.Core.Maps;

/// <summary>
/// Interpolates a liquid surface across a quad where only some corners carry liquid.
/// </summary>
/// <remarks>
/// <para>
/// <b>The defect this replaces.</b> The MCLQ 129x129 -&gt; 257x257 upsample in
/// <c>BuildUnifiedLiquid</c> admitted a destination pixel when <em>any</em> of the four source
/// corners had presence, then bilinear-interpolated using <b>all four heights unconditionally</b> —
/// including corners with no presence, whose stored height is not a surface value and is plausibly
/// zero. Blending a real water height against such a corner drags the surface toward it.
/// </para>
/// <para>
/// Partial-presence quads occur exactly at the <b>edge of a water body</b>, so the error is
/// concentrated on shorelines and is invisible in open water — which is why the reported symptom was
/// "the coast is the worst area".
/// </para>
/// <para>
/// The fix is presence-weighted interpolation: zero the weight of every absent corner and
/// renormalise, so the surface is interpolated only among corners that actually carry liquid. Where
/// all four are present this is identical to plain bilinear, so open water is unchanged.
/// </para>
/// </remarks>
public static class LiquidSurfaceInterpolation
{
    /// <summary>
    /// Bilinear interpolation over the corners that carry liquid, renormalised.
    /// </summary>
    /// <param name="topLeft">Height at (row, col).</param>
    /// <param name="topRight">Height at (row, col+1).</param>
    /// <param name="bottomLeft">Height at (row+1, col).</param>
    /// <param name="bottomRight">Height at (row+1, col+1).</param>
    /// <param name="topLeftPresent">Whether (row, col) carries liquid.</param>
    /// <param name="topRightPresent">Whether (row, col+1) carries liquid.</param>
    /// <param name="bottomLeftPresent">Whether (row+1, col) carries liquid.</param>
    /// <param name="bottomRightPresent">Whether (row+1, col+1) carries liquid.</param>
    /// <param name="fx">Horizontal fraction in [0,1].</param>
    /// <param name="fy">Vertical fraction in [0,1].</param>
    /// <param name="height">The interpolated surface height, when any corner is present.</param>
    /// <returns>False when no corner carries liquid; the caller must not write a surface.</returns>
    public static bool TryInterpolate(
        float topLeft,
        float topRight,
        float bottomLeft,
        float bottomRight,
        bool topLeftPresent,
        bool topRightPresent,
        bool bottomLeftPresent,
        bool bottomRightPresent,
        float fx,
        float fy,
        out float height)
    {
        height = 0f;

        float weightTopLeft = topLeftPresent ? (1f - fx) * (1f - fy) : 0f;
        float weightTopRight = topRightPresent ? fx * (1f - fy) : 0f;
        float weightBottomLeft = bottomLeftPresent ? (1f - fx) * fy : 0f;
        float weightBottomRight = bottomRightPresent ? fx * fy : 0f;

        float weightSum = weightTopLeft + weightTopRight + weightBottomLeft + weightBottomRight;

        // Every present corner can still contribute zero weight when the sample sits exactly on an
        // absent corner. Falling back to an unweighted mean of the present corners keeps the surface
        // continuous there instead of leaving a one-pixel hole at the quad edge.
        if (weightSum <= 0f)
        {
            int presentCount = 0;
            float sum = 0f;
            if (topLeftPresent) { sum += topLeft; presentCount++; }
            if (topRightPresent) { sum += topRight; presentCount++; }
            if (bottomLeftPresent) { sum += bottomLeft; presentCount++; }
            if (bottomRightPresent) { sum += bottomRight; presentCount++; }

            if (presentCount == 0)
                return false;

            height = sum / presentCount;
            return true;
        }

        height =
            ((topLeftPresent ? topLeft * weightTopLeft : 0f)
             + (topRightPresent ? topRight * weightTopRight : 0f)
             + (bottomLeftPresent ? bottomLeft * weightBottomLeft : 0f)
             + (bottomRightPresent ? bottomRight * weightBottomRight : 0f))
            / weightSum;

        return true;
    }

    /// <summary>True when a quad carries liquid on some corners but not all.</summary>
    /// <remarks>
    /// The population the defect above lived in. Counting it is how the fix's reach is measured
    /// rather than assumed.
    /// </remarks>
    public static bool IsPartiallyPresent(bool topLeft, bool topRight, bool bottomLeft, bool bottomRight)
    {
        int present = (topLeft ? 1 : 0) + (topRight ? 1 : 0) + (bottomLeft ? 1 : 0) + (bottomRight ? 1 : 0);
        return present is > 0 and < 4;
    }
}
