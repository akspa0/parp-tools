using System.Text;
using WoWViewer.Logging;

namespace WoWViewer.Rendering;

/// <summary>
/// Single formatter utility for M2 route diagnostics.
/// All probe and runtime M2 route logs go through this class.
/// </summary>
internal static class M2RouteDiagnostics
{
    private const string Prefix = "[M2-ROUTE]";

    public static string FormatRouteDecision(M2RouteDecision decision)
    {
        var sb = new StringBuilder();
        sb.Append($"{Prefix} modelPath={decision.ModelPath}");
        sb.Append($" build={decision.BuildProfileId}");
        sb.Append($" primary={decision.PrimaryRoute}");
        sb.Append($" applied={decision.AppliedRoute}");
        if (decision.SelectedSkinPath != null)
            sb.Append($" skin={decision.SelectedSkinPath}");
        if (decision.FallbackReason != null)
            sb.Append($" fallback={decision.FallbackReason}");
        return sb.ToString();
    }

    public static string FormatMaterialPassProfile(M2MaterialPassProfile profile)
    {
        var sb = new StringBuilder();
        sb.Append($"{Prefix} modelPath={profile.ModelPath}");
        sb.Append($" section={profile.SectionIndex}");
        sb.Append($" material={profile.MaterialIndex}");
        sb.Append($" layer={profile.LayerIndex}");
        sb.Append($" blend={profile.BlendDeclaration}");
        sb.Append($" passClass={profile.PassClass}");
        sb.Append($" depthWrite={profile.DepthWrite}");
        sb.Append($" blendEnabled={profile.BlendEnabled}");
        if (profile.AlphaThreshold.HasValue)
            sb.Append($" alphaThreshold={profile.AlphaThreshold.Value:F3}");
        return sb.ToString();
    }

    public static void LogRouteDecision(M2RouteDecision decision)
    {
        ViewerLog.Info(ViewerLog.Category.Mdx, FormatRouteDecision(decision));
    }

    public static void LogMaterialPassProfile(M2MaterialPassProfile profile)
    {
        ViewerLog.Info(ViewerLog.Category.Mdx, FormatMaterialPassProfile(profile));
    }
}