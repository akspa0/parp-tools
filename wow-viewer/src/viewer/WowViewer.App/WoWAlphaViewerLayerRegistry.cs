namespace WowViewer.App;

internal static class WoWAlphaViewerLayerRegistry
{
    public static IReadOnlyList<IWoWAlphaViewerLayerModule> GetRegisteredModules()
    {
        return
        [
            new FoundationContractsModule(),
        ];
    }

    private sealed class FoundationContractsModule : IWoWAlphaViewerLayerModule
    {
        public WoWAlphaViewerLayer Layer => WoWAlphaViewerLayer.FoundationContracts;

        public string Name => "Foundation Contracts";

        public bool IsReady()
        {
            return true;
        }

        public string DescribeStatus()
        {
            return "Layer 0 contracts are active: module registry + readiness reporting baseline.";
        }
    }
}

