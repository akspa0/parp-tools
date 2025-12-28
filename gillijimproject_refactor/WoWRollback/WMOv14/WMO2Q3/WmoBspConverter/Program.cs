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
            string inputFile = args[0].StartsWith("-") ? "" : args[0];
            string? outputFile = null;
            bool extractTextures = true;
            string? outputDir = null;
            bool verbose = false;
            bool splitGroups = false;
            string? objPath = null;
            bool allowFallback = false;
            bool includeNonRender = false;
            int? matOnly = null;
            int? groupIndex = null;
            string mopyPair = "a"; // a or b
            string mapping = "auto"; // auto|mopy|moba
            bool convertToV17 = false;
            bool useQ3V2 = false;
            bool outputMap = false;

            // Parse simple command line arguments (start from 0 if first arg is a flag)
            int startIdx = args[0].StartsWith("-") ? 0 : 1;
            for (int i = startIdx; i < args.Length; i++)
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
                    case "--split-groups":
                    case "-s":
                        splitGroups = true;
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
                    case "--to-v17":
                        convertToV17 = true;
                        break;
                    case "--q3v2":
                    case "--q3-v2":
                        useQ3V2 = true;
                        break;
                    case "--map":
                        outputMap = true;
                        break;
                    case "--help":
                    case "-h":
                        ShowUsage();
                        return 0;
                }
            }

            try
            {
                if (args.Contains("--list-textures"))
                {
                    await ListTexturesAsync(inputFile);
                }
                else if (args.Contains("--verify-roundtrip"))
                {
                    await Verification.WmoRoundTripVerifier.VerifyAsync(verbose);
                }
                else if (emitCube)
                {
                    await CubeEmitter.EmitCubeAsync(outputFile, outputDir, verbose);
                }
                else if (convertToV17)
                {
                    await ConvertToV17Async(inputFile, outputFile, outputDir, verbose);
                }
                else if (!string.IsNullOrEmpty(objPath))
                {
                    await ExportObjAsync(inputFile, objPath!, allowFallback, includeNonRender, extractTextures, verbose, matOnly, groupIndex, mopyPair, mapping);
                }
                else if (outputMap)
                {
                    await ConvertToMapAsync(inputFile, outputFile, outputDir, verbose, splitGroups);
                }
                else if (useQ3V2)
                {
                    await ConvertQ3V2Async(inputFile, outputFile, outputDir, verbose);
                }
                else
                {
                    await ConvertAsync(inputFile, outputFile, extractTextures, outputDir, verbose, splitGroups);
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
            Console.WriteLine("  --split-groups, -s        Export each WMO group as separate .map file (for large WMOs)");
            Console.WriteLine("  --verbose, -v             Enable verbose logging");
            Console.WriteLine("  --obj <file>              Export OBJ (raw WMO coords). Skips BSP/.map path");
            Console.WriteLine("  --allow-fallback          When MOVI is absent, emit sequential-triple faces");
            Console.WriteLine("  --include-nonrender      Include non-render/collision/portal faces (from MOPY flags)");
            Console.WriteLine("  --mat-only <id>          Emit only faces with the given material ID (diagnostic)");
            Console.WriteLine("  --group <i>              Emit only a single MOGP group (diagnostic)");
            Console.WriteLine("  --mopy-pair <a|b>        For MOPYx2, prefer 'a' or 'b' entry when both are renderable (default: a)");
            Console.WriteLine("  --mapping <auto|mopy|moba>  Force mapping source (default: auto)");
            Console.WriteLine("  --to-v17                  Convert WMO v14 to v17 format (for use with wow.tools exporters)");
            Console.WriteLine("  --q3v2                    Use V2 BSP converter with proper BSP tree/visibility");
            Console.WriteLine("  --map                     Output .map file for GtkRadiant (instead of .bsp)");
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

        static async Task ListTexturesAsync(string inputFile)
        {
            if (!File.Exists(inputFile)) throw new FileNotFoundException("WMO not found", inputFile);
            var parser = new WmoV14Parser();
            var data = parser.ParseWmoV14(inputFile);
            Console.WriteLine($"Textures in {Path.GetFileName(inputFile)}:");
            foreach (var t in data.Textures)
            {
                Console.WriteLine($"  {t}");
            }
            await Task.CompletedTask;
        }

        static async Task ConvertAsync(string inputFile, string? outputFile, bool extractTextures, string? outputDir, bool verbose, bool splitGroups = false)
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
                var converter = new WmoV14ToBspConverter(outputDirectory, extractTextures, splitGroups);

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

        static async Task ConvertToV17Async(string inputFile, string? outputFile, string? outputDir, bool verbose)
        {
            // Validate input
            if (!File.Exists(inputFile))
            {
                throw new FileNotFoundException($"Input file not found: {inputFile}");
            }

            // Establish output path
            var outputDirectory = outputDir ?? Path.Combine(Directory.GetCurrentDirectory(), "output");
            Directory.CreateDirectory(outputDirectory);
            
            if (string.IsNullOrEmpty(outputFile))
            {
                var name = Path.GetFileNameWithoutExtension(inputFile) + "_v17.wmo";
                outputFile = Path.Combine(outputDirectory, name);
            }

            Console.WriteLine("WMO v14 → v17 Converter");
            Console.WriteLine("======================");
            Console.WriteLine($"Input: {Path.GetFullPath(inputFile)}");
            Console.WriteLine($"Output: {Path.GetFullPath(outputFile)}");
            Console.WriteLine();

            try
            {
                // Parse v14 WMO
                var parser = new WmoV14Parser();
                var v14Data = parser.ParseWmoV14(inputFile);
                
                Console.WriteLine($"[INFO] Parsed v14 WMO: {v14Data.Groups.Count} groups, {v14Data.Materials.Count} materials");

                // Convert to v17
                var converter = new WmoV14ToV17Converter();
                converter.ConvertAndWrite(v14Data, outputFile);

                Console.WriteLine();
                Console.WriteLine("✓ Conversion completed successfully!");
                Console.WriteLine();
                Console.WriteLine("💡 Next steps:");
                Console.WriteLine("   1. Use wow.tools.local or WoWFormatLib to export this v17 WMO");
                Console.WriteLine("   2. The v17 format preserves your MOBA-based material assignments");
                Console.WriteLine("   3. wow.tools exporters will handle portals, collision, and other features");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"✗ Conversion failed: {ex.Message}");
                if (verbose)
                {
                    Console.WriteLine($"Stack trace: {ex.StackTrace}");
                }
                throw;
            }

            await Task.CompletedTask;
        }

        static async Task ConvertQ3V2Async(string inputFile, string? outputFile, string? outputDir, bool verbose)
        {
            if (!File.Exists(inputFile))
                throw new FileNotFoundException($"Input file not found: {inputFile}");

            var outputDirectory = outputDir ?? Path.Combine(Directory.GetCurrentDirectory(), "output");
            Directory.CreateDirectory(outputDirectory);

            if (string.IsNullOrEmpty(outputFile))
            {
                outputFile = Path.Combine(outputDirectory, Path.GetFileNameWithoutExtension(inputFile) + ".bsp");
            }

            Console.WriteLine("WMO v14 → Quake 3 BSP V2 Converter");
            Console.WriteLine("===================================");
            Console.WriteLine($"Input:  {Path.GetFullPath(inputFile)}");
            Console.WriteLine($"Output: {Path.GetFullPath(outputFile)}");
            Console.WriteLine();

            // Parse WMO v14
            var parser = new WmoV14Parser();
            var wmoData = parser.ParseWmoV14(inputFile);

            Console.WriteLine($"[INFO] Parsed WMO v{wmoData.Version}: {wmoData.Groups.Count} groups, {wmoData.Materials.Count} materials");

            // Convert using V2 converter with proper BSP tree
            var converter = new Quake3.WmoToQ3ConverterV2();
            var q3bsp = converter.Convert(wmoData);

            // Write BSP file
            var writer = new Quake3.Q3BspWriter(q3bsp);
            writer.Write(outputFile);

            Console.WriteLine();
            Console.WriteLine("✓ Conversion completed!");
            Console.WriteLine($"  Vertices: {q3bsp.Vertices.Count:N0}");
            Console.WriteLine($"  Faces: {q3bsp.Faces.Count:N0}");
            Console.WriteLine($"  Textures: {q3bsp.Textures.Count:N0}");
            Console.WriteLine($"  Nodes: {q3bsp.Nodes.Count:N0}");
            Console.WriteLine($"  Leaves: {q3bsp.Leaves.Count:N0}");
            Console.WriteLine($"  Brushes: {q3bsp.Brushes.Count:N0}");
            Console.WriteLine($"  File size: {new FileInfo(outputFile).Length:N0} bytes");
            Console.WriteLine();
            Console.WriteLine("💡 To test in Quake 3:");
            Console.WriteLine($"   1. Copy {Path.GetFileName(outputFile)} to baseq3/maps/");
            Console.WriteLine("   2. Run: /devmap " + Path.GetFileNameWithoutExtension(outputFile));

            await Task.CompletedTask;
        }

        static async Task ConvertToMapAsync(string inputFile, string? outputFile, string? outputDir, bool verbose, bool splitGroups)
        {
            if (!File.Exists(inputFile))
                throw new FileNotFoundException($"Input file not found: {inputFile}");

            var outputDirectory = outputDir ?? Path.Combine(Directory.GetCurrentDirectory(), "output");
            Directory.CreateDirectory(outputDirectory);

            if (string.IsNullOrEmpty(outputFile))
            {
                outputFile = Path.Combine(outputDirectory, Path.GetFileNameWithoutExtension(inputFile) + ".map");
            }

            Console.WriteLine("WMO v14 → Quake 3 .MAP Converter");
            Console.WriteLine("================================");
            Console.WriteLine($"Input:  {Path.GetFullPath(inputFile)}");
            Console.WriteLine($"Output: {Path.GetFullPath(outputFile)}");
            Console.WriteLine();

            // Parse WMO v14
            var parser = new WmoV14Parser();
            var wmoData = parser.ParseWmoV14(inputFile);

            Console.WriteLine($"[INFO] Parsed WMO v{wmoData.Version}: {wmoData.Groups.Count} groups, {wmoData.Materials.Count} materials");

            // Create BspFile with actual geometry for the map generator
            var bspFile = new BspFile();
            
            // Add textures
            foreach (var tex in wmoData.Textures)
            {
                var cleanName = Path.GetFileNameWithoutExtension(tex).ToLowerInvariant();
                bspFile.Textures.Add(new BspTexture { Name = $"textures/wmo/{cleanName}" });
            }
            if (bspFile.Textures.Count == 0)
            {
                bspFile.Textures.Add(new BspTexture { Name = "textures/common/caulk" });
            }
            
            // Add geometry from all groups
            foreach (var group in wmoData.Groups)
            {
                int vertexBase = bspFile.Vertices.Count;
                
                // Add vertices
                foreach (var v in group.Vertices)
                {
                    bspFile.Vertices.Add(new BspVertex { Position = v });
                }
                
                // Add faces (triangles)
                for (int i = 0; i + 2 < group.Indices.Count; i += 3)
                {
                    var i0 = group.Indices[i];
                    var i1 = group.Indices[i + 1];
                    var i2 = group.Indices[i + 2];
                    int triIndex = i / 3;
                    int matId = triIndex < group.FaceMaterials.Count ? group.FaceMaterials[triIndex] : 0;
                    
                    bspFile.Faces.Add(new BspFace
                    {
                        FirstVertex = vertexBase + i0,
                        NumVertices = 3,
                        Texture = matId
                    });
                }
            }
            
            Console.WriteLine($"[INFO] Prepared {bspFile.Vertices.Count} vertices, {bspFile.Faces.Count} faces for .map generation");

            // Generate .map file
            var mapGen = new WmoMapGenerator();
            if (splitGroups)
            {
                mapGen.GenerateMapFilePerGroup(outputFile, wmoData, bspFile);
            }
            else
            {
                mapGen.GenerateMapFile(outputFile, wmoData, bspFile);
            }

            Console.WriteLine();
            Console.WriteLine("✓ Conversion completed!");
            Console.WriteLine($"  File: {Path.GetFullPath(outputFile)}");
            Console.WriteLine();
            Console.WriteLine("💡 To use in GtkRadiant:");
            Console.WriteLine("   1. Open GtkRadiant");
            Console.WriteLine($"   2. File → Open → {Path.GetFileName(outputFile)}");
            Console.WriteLine("   3. Build → BSP to compile");

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
            
            // Expand bounds slightly
            min -= new Vector3(64, 64, 64);
            max += new Vector3(64, 64, 64);
            
            // Create world model
            bsp.Models.Add(new BspModel { Min = min, Max = max, FirstFace = 0, NumFaces = bsp.Faces.Count });
            
            // Create BSP tree: root node with 2 leaf children
            // In Q3, negative child = -(leafIndex + 1), so -1 = leaf 0, -2 = leaf 1
            bsp.Nodes.Add(new BspNode { Plane = 0, Children = new[]{-1, -2}, Min = min, Max = max });
            
            // Leaf 0: contains all faces (inside)
            for (int i = 0; i < bsp.Faces.Count; i++) bsp.LeafFaces.Add(i);
            bsp.Leaves.Add(new BspLeaf { 
                Min = min, Max = max, 
                Cluster = 0, Area = 0, 
                FirstFace = 0, NumFaces = bsp.Faces.Count,
                FirstBrush = 0, NumBrushes = 0
            });
            
            // Leaf 1: empty (outside)
            bsp.Leaves.Add(new BspLeaf { 
                Min = min, Max = max, 
                Cluster = -1, Area = 0, 
                FirstFace = 0, NumFaces = 0,
                FirstBrush = 0, NumBrushes = 0
            });
            
            // Setup visibility: 1 cluster, all visible
            bsp.NumClusters = 1;
            bsp.BytesPerCluster = 1;
            bsp.VisData = new byte[] { 0x01 }; // Cluster 0 can see cluster 0

            // Entities (worldspawn + spawn)
            bsp.Entities.Add("{\n\"classname\" \"worldspawn\"\n}");
            var center = (min + max) * 0.5f;
            bsp.Entities.Add($"{{\n\"classname\" \"info_player_deathmatch\"\n\"origin\" \"{center.X:F0} {center.Y:F0} {max.Z + 32:F0}\"\n\"angle\" \"0\"\n}}");

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