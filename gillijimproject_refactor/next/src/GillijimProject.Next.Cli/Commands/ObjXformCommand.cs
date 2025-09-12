using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using GillijimProject.Next.Core.Utils.Obj;

namespace GillijimProject.Next.Cli.Commands;

public static class ObjXformCommand
{
    public static int Run(string[] args)
    {
        if (args.Length == 0 || args.Contains("-h") || args.Contains("--help"))
        {
            PrintHelp();
            return 0;
        }

        string? inPath = null; string outDir = "out/obj_xform";
        bool flipX = false, flipY = false, flipZ = false;
        int rotZ = 0;
        double tx = 0, ty = 0, tz = 0;
        bool autoWinding = true;
        bool presetAdtToWdl = false;

        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--in": inPath = args[++i]; break;
                case "--out-dir": outDir = args[++i]; break;
                case "--flip-x": flipX = true; break;
                case "--flip-y": flipY = true; break;
                case "--flip-z": flipZ = true; break;
                case "--rotate-z": rotZ = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
                case "--translate-x": tx = double.Parse(args[++i], CultureInfo.InvariantCulture); break;
                case "--translate-y": ty = double.Parse(args[++i], CultureInfo.InvariantCulture); break;
                case "--translate-z": tz = double.Parse(args[++i], CultureInfo.InvariantCulture); break;
                case "--no-auto-winding": autoWinding = false; break;
                case "--preset":
                    var p = args[++i].ToLowerInvariant();
                    if (p == "adt-to-wdl") presetAdtToWdl = true;
                    break;
            }
        }

        if (string.IsNullOrWhiteSpace(inPath))
        {
            Console.Error.WriteLine("obj-xform: --in <path|glob> is required");
            return 2;
        }

        if (presetAdtToWdl)
        {
            // Server-origin subtraction measured by user; no rotation by default
            tx += -17066.666016; ty += -17066.666016; tz += -0.209080;
        }

        var files = Enumerate(inPath!).ToList();
        if (files.Count == 0)
        {
            Console.Error.WriteLine("obj-xform: no files matched input pattern");
            return 1;
        }

        // Determine base directory to preserve subfolder structure in output
        string baseDir = DetermineBaseDir(inPath!);
        Directory.CreateDirectory(outDir);
        var opts = new TransformOptions(flipX, flipY, flipZ, rotZ, tx, ty, tz, autoWinding);

        foreach (var f in files)
        {
            var doc = ObjIo.Read(f);
            var xdoc = ObjTransformer.Transform(doc, opts, out var swap);

            string rel;
            try { rel = Path.GetRelativePath(baseDir, f); }
            catch { rel = Path.GetFileName(f); }
            if (rel.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal) || rel.StartsWith(".." + Path.AltDirectorySeparatorChar, StringComparison.Ordinal))
                rel = Path.GetFileName(f);

            var dst = Path.Combine(outDir, rel);
            var dstDir = Path.GetDirectoryName(dst);
            if (!string.IsNullOrEmpty(dstDir)) Directory.CreateDirectory(dstDir);

            ObjIo.Write(dst, xdoc, swap);
            Console.WriteLine($"obj-xform: wrote {dst} (swapWinding={swap})");
        }

        return 0;
    }

    private static IEnumerable<string> Enumerate(string pattern)
    {
        // Single file
        if (File.Exists(pattern)) { yield return Path.GetFullPath(pattern); yield break; }

        // Directory input → recurse for *.obj
        if (Directory.Exists(pattern))
        {
            foreach (var f in Directory.EnumerateFiles(pattern, "*.obj", SearchOption.AllDirectories))
                yield return f;
            yield break;
        }

        // Support recursive glob like: C:\\path\\objects\\**\\*.obj
        if (pattern.Contains("**"))
        {
            var sep = Path.DirectorySeparatorChar;
            var alt = Path.AltDirectorySeparatorChar;
            int idx = pattern.IndexOf("**", StringComparison.Ordinal);
            string baseDir = pattern.Substring(0, idx).TrimEnd(sep, alt);
            if (string.IsNullOrEmpty(baseDir)) baseDir = Environment.CurrentDirectory;
            string tail = pattern.Substring(idx + 2).TrimStart(sep, alt);
            string tailNorm = tail.Replace(alt, sep);

            foreach (var f in Directory.EnumerateFiles(baseDir, "*", SearchOption.AllDirectories))
            {
                string rel = Path.GetRelativePath(baseDir, f).Replace(alt, sep);
                if (WildcardMatch(rel, tailNorm)) yield return f;
            }
            yield break;
        }

        // Flat glob in a specific directory
        var dir = Path.GetDirectoryName(pattern);
        var glob = Path.GetFileName(pattern);
        if (string.IsNullOrWhiteSpace(dir)) dir = Environment.CurrentDirectory;
        if (!Directory.Exists(dir)) yield break;
        foreach (var f in Directory.EnumerateFiles(dir!, glob, SearchOption.TopDirectoryOnly))
            yield return f;
    }

    private static string DetermineBaseDir(string pattern)
    {
        if (File.Exists(pattern))
        {
            var p = Path.GetFullPath(pattern);
            return Path.GetDirectoryName(p) ?? Environment.CurrentDirectory;
        }
        if (Directory.Exists(pattern))
        {
            return Path.GetFullPath(pattern);
        }
        if (pattern.Contains("**"))
        {
            var sep = Path.DirectorySeparatorChar;
            var alt = Path.AltDirectorySeparatorChar;
            int idx = pattern.IndexOf("**", StringComparison.Ordinal);
            string baseDir = pattern.Substring(0, idx).TrimEnd(sep, alt);
            if (string.IsNullOrEmpty(baseDir)) baseDir = Environment.CurrentDirectory;
            return Path.GetFullPath(baseDir);
        }
        var d = Path.GetDirectoryName(pattern);
        if (string.IsNullOrWhiteSpace(d)) d = Environment.CurrentDirectory;
        return Path.GetFullPath(d!);
    }

    private static bool WildcardMatch(string text, string pattern)
    {
        // Convert wildcard pattern (*, ?) into regex
        string RegexEscape(string s) => Regex.Escape(s).Replace("\\*", ".*").Replace("\\?", ".");
        var rx = new Regex("^" + RegexEscape(pattern) + "$", RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
        return rx.IsMatch(text);
    }

    private static void PrintHelp()
    {
        Console.WriteLine("obj-xform: transform OBJ with flips/rotation/translate and preserve winding");
        Console.WriteLine("  --in <path|glob|dir>   input file, directory (recurse *.obj), or glob (supports **)");
        Console.WriteLine("  --out-dir <dir>        output directory (subfolder structure is preserved)");
        Console.WriteLine("  --flip-x --flip-y --flip-z");
        Console.WriteLine("  --rotate-z <deg> (0|90|180|270)");
        Console.WriteLine("  --translate-x <v> --translate-y <v> --translate-z <v>");
        Console.WriteLine("  --no-auto-winding");
        Console.WriteLine("  --preset adt-to-wdl");
    }
}
