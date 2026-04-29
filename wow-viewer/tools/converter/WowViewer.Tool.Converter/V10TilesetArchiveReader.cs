using WowViewer.Core.IO.Files;

namespace WowViewer.Tool.Converter;

internal static class V10TilesetArchiveReader
{
	public static ArchiveCatalogSession GetOrCreateSession(string clientRoot)
	{
		ArgumentException.ThrowIfNullOrWhiteSpace(clientRoot);

		return ArchiveCatalogSessionCache.GetOrCreate(
			BuildArchiveRoots(clientRoot),
			new ArchiveCatalogBootstrapOptions(ExternalListfilePath: ResolveLegacyListfilePath()));
	}

	public static byte[]? TryReadVirtualFile(ArchiveCatalogSession session, string virtualPath)
	{
		ArgumentNullException.ThrowIfNull(session);
		ArgumentException.ThrowIfNullOrWhiteSpace(virtualPath);

		return session.ReadFile(virtualPath)
			?? session.TryReadFileFromDisk(virtualPath);
	}

	private static IReadOnlyList<string> BuildArchiveRoots(string clientRoot)
	{
		List<string> roots = [];
		string dataRoot = Path.Combine(clientRoot, "Data");
		if (Directory.Exists(dataRoot))
			roots.Add(dataRoot);
		if (!string.Equals(clientRoot, dataRoot, StringComparison.OrdinalIgnoreCase))
			roots.Add(clientRoot);
		return roots.Count > 0 ? roots : [clientRoot];
	}

	private static string? ResolveLegacyListfilePath()
	{
		string[] candidates =
		[
			Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "MdxViewer", "community-listfile-withcapitals.csv"),
			Path.Combine(AppContext.BaseDirectory, "community-listfile-withcapitals.csv"),
			"community-listfile-withcapitals.csv",
			"listfile.csv",
		];
		foreach (string candidate in candidates)
		{
			if (File.Exists(candidate))
				return candidate;
		}
		return null;
	}
}