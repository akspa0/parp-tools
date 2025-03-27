using CommandLine;

namespace WowToolSuite.Liquid
{
    public class CommandLineOptions
    {
        [Option("wlw-dir", Required = true, HelpText = "Directory containing WLW files")]
        public string WlwDirectory { get; set; } = string.Empty;

        [Option("adt-dir", Required = false, HelpText = "Directory containing ADT files")]
        public string AdtDirectory { get; set; } = string.Empty;

        [Option("output-dir", Required = true, HelpText = "Directory to output files")]
        public string OutputDirectory { get; set; } = string.Empty;

        [Option("patched-adt-dir", Required = false, HelpText = "Directory to output patched ADT files")]
        public string PatchedAdtDirectory { get; set; } = string.Empty;

        [Option("verbose", Default = false, HelpText = "Enable verbose output")]
        public bool Verbose { get; set; }

        [Option("export-obj", Default = false, HelpText = "Export water blocks as OBJ files")]
        public bool ExportObj { get; set; }

        [Option("patch-all-adts", Default = false, HelpText = "Copy all ADT files even if they don't have liquid")]
        public bool PatchAllAdts { get; set; }

        [Option("yaml-output", Default = false, HelpText = "Generate human-readable YAML output of intermediate data")]
        public bool YamlOutput { get; set; }
    }
} 