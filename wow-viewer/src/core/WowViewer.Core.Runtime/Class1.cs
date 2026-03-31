namespace WowViewer.Core.Runtime;

public sealed record RuntimeBoundary(string Name, string Responsibility);

public static class RuntimeBoundaries
{
	public static readonly RuntimeBoundary[] All =
	[
		new("data-sources", "game roots, overlays, and file access contracts"),
		new("dbc-listfile", "build-aware metadata and lookup services"),
		new("sql-scene", "runtime population and external data integration"),
		new("world-render-telemetry", "stable world render frame contracts and optimization guidance"),
		new("viewer-services", "scene-facing services consumed by the app shell")
	];
}
