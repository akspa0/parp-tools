namespace WowViewer.Tools.Shared;

public sealed record ToolHostDescriptor(string HostName, string Responsibility);

public static class ToolHosts
{
	public static readonly ToolHostDescriptor[] Planned =
	[
		new("WowViewer.App", "interactive shell and panel workflows"),
		new("WowViewer.Tool.Converter", "headless conversion and repair workflows"),
		new("WowViewer.Tool.Inspect", "headless inspection and forensics workflows")
	];
}
