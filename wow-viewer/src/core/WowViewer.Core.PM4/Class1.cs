namespace WowViewer.Core.PM4;

public static class Pm4Boundary
{
	public const string CanonicalOwner = "WowViewer.Core.PM4";
	public const string LibrarySeed = "Pm4Research";
	public const string LegacyReference = "MdxViewer";
	public const string Contract = "Core.PM4 owns decode, runtime placement contracts, grouping, transforms, correlation, and reports.";
}

public static class Pm4PortStatus
{
	public static readonly string[] FirstSlice =
	[
		"typed research chunk models",
		"research document model",
		"binary PM4 reader",
		"exploration snapshot builder",
		"decode audit analyzers",
		"runtime placement contracts",
		"linkage analyzer",
		"placement math helpers",
		"mscn analyzer",
		"unknowns analyzer",
		"axis normal scoring helpers"
	];
}
