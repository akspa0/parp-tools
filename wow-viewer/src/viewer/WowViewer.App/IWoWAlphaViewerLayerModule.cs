namespace WowViewer.App;

internal interface IWoWAlphaViewerLayerModule
{
    WoWAlphaViewerLayer Layer { get; }

    string Name { get; }

    bool IsReady();

    string DescribeStatus();
}

