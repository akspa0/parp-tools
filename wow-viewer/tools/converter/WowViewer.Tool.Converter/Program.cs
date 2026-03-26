using WowViewer.Core.Files;
using WowViewer.Core.IO;
using WowViewer.Core.IO.Files;
using WowViewer.Core.PM4;
using WowViewer.Tools.Shared;

if (args.Length == 0 || args.Contains("--help") || args.Contains("-h"))
{
	ShowUsage();
	return;
}

string command = args[0].ToLowerInvariant();
string[] tail = args.Skip(1).ToArray();

switch (command)
{
	case "detect":
		RunDetect(tail);
		break;
	default:
		Console.Error.WriteLine($"Unknown converter command '{command}'.");
		ShowUsage();
		Environment.ExitCode = 1;
		break;
}

static void RunDetect(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input file is required.");
		Environment.ExitCode = 1;
		return;
	}

	WowFileDetection detection = WowFileDetector.Detect(input);
	Console.WriteLine("WowViewer.Tool.Converter detect report");
	Console.WriteLine($"Input: {detection.SourcePath}");
	Console.WriteLine($"Kind: {detection.Kind}");
	Console.WriteLine($"Version: {detection.Version?.ToString() ?? "n/a"}");
	Console.WriteLine($"Owns families: {string.Join(", ", IoBoundaries.OwnedFamilies)}");
	Console.WriteLine($"PM4 source-of-truth: canonical={Pm4Boundary.CanonicalOwner}, seed={Pm4Boundary.LibrarySeed}, legacy={Pm4Boundary.LegacyReference}");
	Console.WriteLine($"Planned hosts: {ToolHosts.Planned.Length}");
}

static string? GetOption(string[] args, string longName, string shortName)
{
	for (int index = 0; index < args.Length - 1; index++)
	{
		if (string.Equals(args[index], longName, StringComparison.OrdinalIgnoreCase)
			|| string.Equals(args[index], shortName, StringComparison.OrdinalIgnoreCase))
		{
			return args[index + 1];
		}
	}

	return null;
}

static void ShowUsage()
{
	Console.WriteLine("WowViewer.Tool.Converter");
	Console.WriteLine("Usage:");
	Console.WriteLine("  wowviewer-converter detect --input <file>");
}
