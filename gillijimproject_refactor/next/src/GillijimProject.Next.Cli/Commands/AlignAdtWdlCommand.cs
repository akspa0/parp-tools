using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using GillijimProject.Next.Core.Utils.Obj;
using GillijimProject.Next.Core.Alignment;

namespace GillijimProject.Next.Cli.Commands;

public static class AlignAdtWdlCommand
{
    public static int Run(string[] args)
    {
        if (args.Length == 0 || args.Contains("-h") || args.Contains("--help"))
        {
            PrintHelp();
            return 0;
        }

        string? wdlObj = null, adtObj = null, outDir = null;
        bool writeObj = false;
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--wdl-obj": wdlObj = args[++i]; break;
                case "--adt-obj": adtObj = args[++i]; break;
                case "--out-dir": outDir = args[++i]; break;
                case "--write-obj": writeObj = true; break;
            }
        }
        if (string.IsNullOrWhiteSpace(wdlObj) || string.IsNullOrWhiteSpace(adtObj))
        {
            Console.Error.WriteLine("align-adt-wdl: --wdl-obj and --adt-obj are required");
            return 2;
        }

        var wdl = ObjIo.Read(wdlObj!);
        var adt = ObjIo.Read(adtObj!);

        var wVerts = wdl.Vertices().Select(v => (v.X, v.Y, v.Z)).ToArray();
        var aVerts = adt.Vertices().Select(v => (v.X, v.Y, v.Z)).ToArray();
        if (wVerts.Length == 0 || aVerts.Length == 0) { Console.Error.WriteLine("align-adt-wdl: empty OBJ"); return 2; }

        (double minX, double minY, double maxX, double maxY) = BoundsXY3(wVerts);
        var srcXY = aVerts.Select(v => (v.X, v.Y)).ToList();

        var res = AlignmentSolver.SolveToTileBounds(srcXY, (minX, minY, maxX, maxY));

        Console.WriteLine($"align-adt-wdl: best transform -> {res}");
        Console.WriteLine("Suggested obj-xform invocation:");
        Console.WriteLine($"  obj-xform --in \"{adtObj}\" --out-dir aligned {(res.FlipX?"--flip-x ":string.Empty)}{(res.FlipY?"--flip-y ":string.Empty)}--rotate-z {res.RotZDeg} --translate-x {res.TranslateX.ToString(CultureInfo.InvariantCulture)} --translate-y {res.TranslateY.ToString(CultureInfo.InvariantCulture)}");

        if (writeObj && !string.IsNullOrWhiteSpace(outDir))
        {
            Directory.CreateDirectory(outDir!);
            var xdoc = ObjTransformer.Transform(adt, new TransformOptions(res.FlipX, res.FlipY, false, res.RotZDeg, res.TranslateX, res.TranslateY, 0, true), out var swap);
            var dst = Path.Combine(outDir!, Path.GetFileNameWithoutExtension(adtObj)) + "_aligned.obj";
            ObjIo.Write(dst, xdoc, swap);
            Console.WriteLine($"align-adt-wdl: wrote {dst} (swapWinding={swap})");
        }

        return 0;
    }

    private static (double minX, double minY, double maxX, double maxY) BoundsXY3(IEnumerable<(double X,double Y,double Z)> pts)
    {
        double minX = double.PositiveInfinity, minY = double.PositiveInfinity, maxX = double.NegativeInfinity, maxY = double.NegativeInfinity;
        foreach (var (x,y,_) in pts)
        {
            if (x < minX) minX = x; if (x > maxX) maxX = x;
            if (y < minY) minY = y; if (y > maxY) maxY = y;
        }
        return (minX, minY, maxX, maxY);
    }

    private static void PrintHelp()
    {
        Console.WriteLine("align-adt-wdl: discover orientation+translation to align ADT OBJ to WDL OBJ");
        Console.WriteLine("  --wdl-obj <path>  WDL tile OBJ (17x17 grid)\n  --adt-obj <path>  ADT tile OBJ (145x145 grid)\n  --write-obj --out-dir <dir>  write aligned ADT OBJ");
    }
}
