using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Numerics;
using WmoBspConverter.Wmo;
using WmoBspConverter.Bsp;

namespace WmoBspConverter
{
    class Program
    {
        static async Task<int> Main(string[] args)
        {
            // Simple command-line parsing since System.CommandLine version has compatibility issues
            if (args.Length == 0)
            {
                ShowUsage();
                return 1;
            }

            bool emitCube = false;
            string inputFile = args[0];
            string? outputFile = null;
            bool extractTextures = true;
            string? outputDir = null;
            bool verbose = false;
            string? objPath = null;
            bool allowFallback = false;
            bool includeNonRender = false;
            int? matOnly = null;
            int? groupIndex = null;
            string mopyPair = "a"; // a or b
            string mapping = "auto"; // auto|mopy|moba

            // Parse simple command line arguments
            for (int i = 1; i < args.Length; i++)
            {
                switch (args[i].ToLowerInvariant())
                {
                    case "--emit-cube":
                    case "-c":
                        emitCube = true;
                        break;
                    case "--output":
                    case "-o":
                        if (i + 1 < args.Length)
                        {
                            outputFile = args[++i];
                        }
                        break;
                    case "--extract-textures":
                    case "-t":
                        if (i + 1 < args.Length && bool.TryParse(args[i + 1], out bool value))
                        {
                            extractTextures = value;
                            i++;
                        }
                        else
                        {
                            extractTextures = true;
                        }
                        break;
                    case "--output-dir":
                    case "-d":
                        if (i + 1 < args.Length)
                        {
                            outputDir = args[++i];
                        }
                        break;
                    case "--verbose":
                    case "-v":
                        verbose = true;
                        break;
                    case "--obj":
                        if (i + 1 < args.Length)
                        {
                            objPath = args[++i];
                        }
                        break;
                    case "--mat-only":
                        if (i + 1 < args.Length && int.TryParse(args[i + 1], out var mo))
                        {
                            matOnly = mo;
                            i++;
                        }
                        break;
                    case "--group":
                        if (i + 1 < args.Length && int.TryParse(args[i + 1], out var gi))
                        {
                            groupIndex = gi; i++;
                        }
                        break;
                    case "--mopy-pair":
                        if (i + 1 < args.Length)
                        {
                            var val = args[++i].ToLowerInvariant();
                            if (val == "a" || val == "b") mopyPair = val;
                        }
                        break;
                    case "--mapping":
                        if (i + 1 < args.Length)
                        {
                            var val = args[++i].ToLowerInvariant();
                            if (val == "auto" || val == "mopy" || val == "moba") mapping = val;
                        }
                        break;
                    case "--allow-fallback":
                        allowFallback = true;
                        break;
                    case "--include-nonrender":
                        includeNonRender = true;
                        break;
                    case "--help":
                    case "-h":
                        ShowUsage();
                        return 0;
                }
            }

            try
            {
                if (emitCube)
                {
                    await CubeEmitter.EmitCubeAsync(outputFile, outputDir, verbose);
                }
                else if (!string.IsNullOrEmpty(objPath))
                {
                    await ExportObjAsync(inputFile, objPath!, allowFallback, includeNonRender, extractTextures, verbose, matOnly, groupIndex, mopyPair, mapping);
                }
                else
                {
                    await ConvertAsync(inputFile, outputFile, extractTextures, outputDir, verbose);
                }
                return 0;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"✗ Error: {ex.Message}");
                if (verbose)
                {
                    Console.WriteLine($"Stack trace: {ex.StackTrace}");
                }
                return 1;
            }
        }

        static void ShowUsage()
        {
            Console.WriteLine("WMO v14 → Quake 3 BSP Converter");
            Console.WriteLine("Usage: WmoBspConverter <input.wmo> [options] | --emit-cube [options] | <input.wmo> --obj <file> [--allow-fallback]");
            Console.WriteLine();
            Console.WriteLine("Options:");
            Console.WriteLine("  --emit-cube, -c           Emit a golden cube BSP (test) without WMO input");
            Console.WriteLine("  --output, -o <file>       Output BSP file path");
            Console.WriteLine("  --extract-textures, -t    Extract and convert BLP textures to PNG (default: true)");
            Console.WriteLine("  --output-dir, -d <dir>    Output directory for textures and shaders");
            Console.WriteLine("  --verbose, -v             Enable verbose logging");
            Console.WriteLine("  --obj <file>              Export OBJ (raw WMO coords). Skips BSP/.map path");
            Console.WriteLine("  --allow-fallback          When MOVI is absent, emit sequential-triple faces");
            Console.WriteLine("  --include-nonrender      Include non-render/collision/portal faces (from MOPY flags)");
            Console.WriteLine("  --mat-only <id>          Emit only faces with the given material ID (diagnostic)");
            Console.WriteLine("  --group <i>              Emit only a single MOGP group (diagnostic)");
            Console.WriteLine("  --mopy-pair <a|b>        For MOPYx2, prefer 'a' or 'b' entry when both are renderable (default: a)");
            Console.WriteLine("  --mapping <auto|mopy|moba>  Force mapping source (default: auto)");
            Console.WriteLine("  --help, -h                Show this help message");
            Console.WriteLine();
            Console.WriteLine("Examples:");
            Console.WriteLine("  WmoBspConverter building.wmo");
            Console.WriteLine("  WmoBspConverter dungeon.wmo --output maps/dungeon.bsp --extract-textures");
            Console.WriteLine("  WmoBspConverter tower.wmo -o tower.bsp -d ./output -v");
            Console.WriteLine("  WmoBspConverter --emit-cube -d ./output -v");
            Console.WriteLine("  WmoBspConverter model.wmo --obj ./out/model.obj -v");
            Console.WriteLine("  WmoBspConverter test.wmo --obj ./out/test.obj --allow-fallback");
        }

        static async Task ConvertAsync(string inputFile, string? outputFile, bool extractTextures, string? outputDir, bool verbose)
        {
            // Validate input file
            if (!File.Exists(inputFile))
            {
                throw new FileNotFoundException($"Input file not found: {inputFile}");
            }

            // Establish output directory first - NEVER use input directory!
            var outputDirectory = outputDir ?? 
                                 (string.IsNullOrEmpty(outputFile) ? Path.Combine(Directory.GetCurrentDirectory(), "output") : Path.GetDirectoryName(outputFile)) 
                                 ?? Path.Combine(Directory.GetCurrentDirectory(), "output");

            // Set output file path: if not specified, place BSP under outputDirectory
            if (string.IsNullOrEmpty(outputFile))
            {
                var name = Path.GetFileNameWithoutExtension(inputFile) + ".bsp";
                outputFile = Path.Combine(outputDirectory, name);
            }

            // Ensure output directory exists
            Directory.CreateDirectory(outputDirectory);
            var textureOutputDir = Path.Combine(outputDirectory, "textures");
            
            Console.WriteLine("WMO v14 → Quake 3 BSP Converter (.NET 9)");
            Console.WriteLine("==========================================");
            Console.WriteLine($"Input: {Path.GetFullPath(inputFile)}");
            Console.WriteLine($"Output: {Path.GetFullPath(outputFile)}");
            Console.WriteLine($"Texture extraction: {(extractTextures ? "Enabled" : "Disabled")}");
            Console.WriteLine($"Output directory: {Path.GetFullPath(outputDirectory)}");
            Console.WriteLine();

            if (verbose)
            {
                Console.WriteLine("[VERBOSE] Verbose mode enabled");
            }

            try
            {
                // Create converter with enhanced features
                var converter = new WmoV14ToBspConverter(outputDirectory, extractTextures);

                // Run enhanced conversion (uses v14 parser path)
                var result = await converter.ConvertAsync(inputFile, outputDirectory, System.Threading.CancellationToken.None);

                if (!result.Success)
                {
                    Console.WriteLine("✗ Conversion failed. See details above.");
                    Environment.ExitCode = 1;
                    return;
                }

                // Display conversion statistics (success)
                Console.WriteLine("✓ Conversion completed successfully!");
                Console.WriteLine($"  ⏱️  Time: {result.ConversionTime.TotalMilliseconds:N0}ms");
                Console.WriteLine($"  📐 Vertices: {result.TotalVertices:N0}");
                Console.WriteLine($"  🔺 Faces: {result.TotalFaces:N0}");
                Console.WriteLine($"  🎨 Textures: {result.TextureCount:N0}");
                Console.WriteLine($"  📦 Models: 1");
                Console.WriteLine($"  🏠 Leaves: 1");
                Console.WriteLine($"  ⚙️  Entities: 1");
                var outBsp = System.IO.Path.Combine(outputDirectory, System.IO.Path.GetFileNameWithoutExtension(inputFile) + ".bsp");
                if (File.Exists(outBsp))
                {
                    Console.WriteLine($"  💾 File size: {new FileInfo(outBsp).Length:N0} bytes");
                }

                // Show texture extraction info
                if (extractTextures && Directory.Exists(textureOutputDir))
                {
                    var textureFiles = Directory.GetFiles(textureOutputDir, "*.tga", SearchOption.AllDirectories);
                    Console.WriteLine($"  🖼️  Extracted textures: {textureFiles.Length}");
                    
                    var shaderFiles = Directory.GetFiles(textureOutputDir, "*.shader", SearchOption.AllDirectories);
                    if (shaderFiles.Length > 0)
                    {
                        Console.WriteLine($"  📝 Shader scripts: {shaderFiles.Length}");
                    }
                }

                // Success message for Quake 3 usage
                Console.WriteLine();
                Console.WriteLine("💡 The BSP file is ready for use in Quake 3 mapping tools:");
                Console.WriteLine($"   • GtkRadiant: Load {Path.GetFileName(outBsp)}");
                Console.WriteLine($"   • ioquake3: Copy to baseq3/maps/ directory");
                
                if (extractTextures)
                {
                    Console.WriteLine($"   • Textures: Copy {Path.GetFileName(textureOutputDir)}/ to baseq3/textures/");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"✗ Conversion failed: {ex.Message}");
                if (verbose)
                {
                    Console.WriteLine();
                    Console.WriteLine("Stack trace:");
                    Console.WriteLine(ex.StackTrace);
                }
                throw;
            }
        }
        private static async Task ExportObjAsync(string inputFile, string objPath, bool allowFallback, bool includeNonRender, bool extractTextures, bool verbose, int? matOnly, int? groupIndex, string mopyPair, string mapping)
        {
            var parser = new WmoV14Parser();
            var data = parser.ParseWmoV14(inputFile);
            var exporter = new Wmo.WmoObjExporter();
            bool preferSecond = mopyPair == "b";
            bool forceMopy = mapping == "mopy";
            bool forceMoba = mapping == "moba";
            exporter.Export(objPath, data, allowFallback, includeNonRender, extractTextures, matOnly, groupIndex, preferSecond, forceMopy, forceMoba);
            if (verbose) Console.WriteLine($"[OBJ] Wrote {Path.GetFullPath(objPath)}");
            await Task.CompletedTask;
        }
    }

    internal static class CubeEmitter
    {
        internal static async Task EmitCubeAsync(string? outputFile, string? outputDir, bool verbose)
        {
            var outDir = outputDir ?? (string.IsNullOrEmpty(outputFile) ? Directory.GetCurrentDirectory() : Path.GetDirectoryName(outputFile)) ?? ".";
            Directory.CreateDirectory(outDir);
            var outPath = outputFile ?? Path.Combine(outDir, "cube.bsp");

            var bsp = new BspFile();

            // Texture
            bsp.Textures.Add(new BspTexture { Name = "textures/common/caulk", Flags = 0, Contents = 0 });

            // Simple cube geometry (12 tris)
            float s = 64f;
            Vector3[] verts = new[]
            {
                new Vector3(-s, -s, -s), new Vector3(s, -s, -s), new Vector3(s, s, -s), new Vector3(-s, s, -s),
                new Vector3(-s, -s,  s), new Vector3(s, -s,  s), new Vector3(s, s,  s), new Vector3(-s, s,  s)
            };
            int[][] tris = new[]
            {
                new[]{0,1,2}, new[]{0,2,3},
                new[]{4,6,5}, new[]{4,7,6},
                new[]{4,5,1}, new[]{4,1,0},
                new[]{5,6,2}, new[]{5,2,1},
                new[]{6,7,3}, new[]{6,3,2},
                new[]{7,4,0}, new[]{7,0,3},
            };

            foreach (var t in tris)
            {
                var p0 = verts[t[0]]; var p1 = verts[t[1]]; var p2 = verts[t[2]];
                var e1 = p1 - p0; var e2 = p2 - p0; var n = Vector3.Cross(e1, e2);
                var len = n.Length(); if (len < 1e-6f) continue; n /= len;

                int start = bsp.Vertices.Count;
                bsp.Vertices.Add(new BspVertex { Position = p0, TextureCoordinate = Vector2.Zero, LightmapCoordinate = Vector2.Zero, Normal = n, Color = new byte[]{255,255,255,255} });
                bsp.Vertices.Add(new BspVertex { Position = p1, TextureCoordinate = Vector2.Zero, LightmapCoordinate = Vector2.Zero, Normal = n, Color = new byte[]{255,255,255,255} });
                bsp.Vertices.Add(new BspVertex { Position = p2, TextureCoordinate = Vector2.Zero, LightmapCoordinate = Vector2.Zero, Normal = n, Color = new byte[]{255,255,255,255} });

                int mstart = bsp.MeshVertices.Count; bsp.MeshVertices.Add(0); bsp.MeshVertices.Add(1); bsp.MeshVertices.Add(2);

                bsp.Faces.Add(new BspFace
                {
                    Texture = 0,
                    Effect = -1,
                    Type = 3,
                    FirstVertex = start,
                    NumVertices = 3,
                    FirstMeshVertex = mstart,
                    NumMeshVertices = 3,
                    Lightmap = -1,
                    Normal = n
                });
            }

            // Planes from faces
            foreach (var f in bsp.Faces)
            {
                var v0 = bsp.Vertices[f.FirstVertex].Position;
                var v1 = bsp.Vertices[f.FirstVertex+1].Position;
                var v2 = bsp.Vertices[f.FirstVertex+2].Position;
                var n = Vector3.Normalize(Vector3.Cross(v1 - v0, v2 - v0));
                var d = Vector3.Dot(n, v0);
                bsp.Planes.Add(new BspPlane { Normal = n, Distance = d });
            }

            // Bounds / model / nodes / leaves
            Vector3 min = new Vector3(float.MaxValue), max = new Vector3(float.MinValue);
            foreach (var v in bsp.Vertices)
            {
                var p = v.Position;
                min = new Vector3(Math.Min(min.X, p.X), Math.Min(min.Y, p.Y), Math.Min(min.Z, p.Z));
                max = new Vector3(Math.Max(max.X, p.X), Math.Max(max.Y, p.Y), Math.Max(max.Z, p.Z));
            }
            bsp.Models.Add(new BspModel { Min = min, Max = max, FirstFace = 0, NumFaces = bsp.Faces.Count });
            bsp.Nodes.Add(new BspNode { Plane = 0, Children = new[]{-1,-1}, Min = min, Max = max });
            bsp.Leaves.Add(new BspLeaf { Min = min, Max = max, Cluster = 0, Area = 0, FirstFace = 0, NumFaces = bsp.Faces.Count });
            for (int i=0;i<bsp.Faces.Count;i++) bsp.LeafFaces.Add(i);

            // Entities (worldspawn + spawn)
            bsp.Entities.Add("{\n  \"classname\" \"worldspawn\"\n}");
            var center = (min + max) * 0.5f;
            bsp.Entities.Add($"{{\n  \"classname\" \"info_player_deathmatch\"\n  \"origin\" \"{center.X:F1} {center.Y:F1} {max.Z + 32:F1}\"\n  \"angle\" \"0\"\n}}");

            // Save and verify
            bsp.Save(outPath);
            if (verbose) VerifyBspHeader(outPath);
            Console.WriteLine($"[CUBE] Wrote golden cube BSP: {outPath}");
            await Task.CompletedTask;
        }

        private static void VerifyBspHeader(string path)
        {
            using var fs = File.OpenRead(path);
            using var br = new BinaryReader(fs);
            int magic = br.ReadInt32();
            int ver = br.ReadInt32();
            Console.WriteLine($"[VERIFY] Magic=0x{magic:X8} Version={ver}");
            if (magic != BspHeader.Magic || ver != BspHeader.Version)
            {
                Console.WriteLine("[VERIFY] Invalid IBSP header");
            }
        }
    }
}