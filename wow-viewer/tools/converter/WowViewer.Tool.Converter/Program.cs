using System.Text.Json;
using System.Text.Json.Serialization;
using WowViewer.Core.Files;
using WowViewer.Core.IO;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Maps;
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
	case "export-tex-json":
		RunExportTexJson(tail);
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

static void RunExportTexJson(string[] args)
{
	string? input = GetOption(args, "--input", "-i") ?? args.FirstOrDefault(static arg => !arg.StartsWith('-'));
	string? output = GetOption(args, "--output", "-o");
	if (string.IsNullOrWhiteSpace(input))
	{
		Console.Error.WriteLine("Error: input root ADT or _tex0.adt file is required.");
		Environment.ExitCode = 1;
		return;
	}

	WowFileDetection detection = WowFileDetector.Detect(input);
	if (detection.Kind is not (WowFileKind.Adt or WowFileKind.AdtTex))
	{
		Console.Error.WriteLine($"Error: export-tex-json requires a root ADT or _tex0.adt input, but detected {detection.Kind}.");
		Environment.ExitCode = 1;
		return;
	}

	string json = JsonSerializer.Serialize(
		AdtTextureReader.Read(input),
		CreateJsonOptions());

	if (!string.IsNullOrWhiteSpace(output))
	{
		string outputPath = Path.GetFullPath(output);
		string? directory = Path.GetDirectoryName(outputPath);
		if (!string.IsNullOrWhiteSpace(directory))
			Directory.CreateDirectory(directory);

		File.WriteAllText(outputPath, json);
		Console.WriteLine($"Wrote {outputPath}");
		return;
	}

	Console.WriteLine(json);
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

static JsonSerializerOptions CreateJsonOptions()
{
	JsonSerializerOptions options = new()
	{
		WriteIndented = true,
	};
	options.Converters.Add(new JsonStringEnumConverter());
	return options;
}

static void ShowUsage()
{
	Console.WriteLine("WowViewer.Tool.Converter");
	Console.WriteLine("Usage:");
	Console.WriteLine("  wowviewer-converter detect --input <file>");
	Console.WriteLine("  wowviewer-converter export-tex-json --input <file.adt|file_tex0.adt> [--output <report.json>]");
}
