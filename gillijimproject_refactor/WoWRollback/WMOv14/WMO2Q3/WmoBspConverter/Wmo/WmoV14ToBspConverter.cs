using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using WoWFormatLib.FileReaders;
using WoWFormatLib.Structs.WMO;
using WoWFormatLib.FileProviders;
using WmoBspConverter.Bsp;
using WmoBspConverter.Textures;

namespace WmoBspConverter.Wmo
{
    /// <summary>
    /// Main converter class for transforming WMO v14 files to Quake 3 BSP format.
    /// Uses the proven WoWFormatLib approach from old_sources for reliability.
    /// </summary>
    public class WmoV14ToBspConverter
    {
        private readonly TextureProcessor _textureProcessor;
        private readonly bool _extractTextures;
        private readonly WmoMapGenerator _mapGenerator;

        public WmoV14ToBspConverter(string? outputDir = null, bool extractTextures = true)
        {
            _textureProcessor = new TextureProcessor(outputDir ?? Path.GetTempPath(), extractTextures);
            _extractTextures = extractTextures;
            _mapGenerator = new WmoMapGenerator();
        }

        /// <summary>
        /// Convert WMO v14 to BSP using our custom WMO v14 parser
        /// </summary>
        public Task<BspFile> ConvertAsync(string wmoFilePath, string? outputDir = null)
        {
            if (!File.Exists(wmoFilePath))
                throw new FileNotFoundException($"WMO file not found: {wmoFilePath}");

            Console.WriteLine($"[INFO] Starting conversion of {Path.GetFileName(wmoFilePath)}");
            
            // Use our custom WMO v14 parser that handles the MOMO container structure
            var parser = new WmoV14Parser();
            var wmoData = parser.ParseWmoV14(wmoFilePath);

            var bspFile = parser.ConvertToBsp(wmoData);

            // Step 1: Process textures
            ConvertTexturesAsync(wmoData, bspFile);

            // Step 2: Convert geometry from all groups
            ConvertGeometry(wmoData, bspFile);

            // Step 3: Add basic entities
            ConvertEntities(wmoData, bspFile);

            // Step 4: Create basic BSP structure
            CreateBasicBspStructure(bspFile);

            // Step 5: Generate .map file (primary output format)
            var mapFileName = Path.GetFileNameWithoutExtension(wmoFilePath) + ".map";
            var mapFilePath = outputDir != null
                ? Path.Combine(outputDir, mapFileName)
                : Path.Combine(Path.GetDirectoryName(wmoFilePath) ?? "", mapFileName);
                
            _mapGenerator.GenerateMapFile(mapFilePath, wmoData, bspFile);

            Console.WriteLine($"[INFO] Conversion completed successfully!");
            Console.WriteLine($"[STATS] Vertices: {bspFile.Vertices.Count}, Faces: {bspFile.Faces.Count}, Textures: {bspFile.Textures.Count}");
            Console.WriteLine($"[INFO] 🎯 PRIMARY OUTPUT: .map file for Quake 3 editing");
            Console.WriteLine($"[INFO] 📁 {mapFilePath}");
            Console.WriteLine($"[INFO] 🔧 Next steps: 1) Open .map in GtkRadiant, 2) Compile with Q3Map2");

            return Task.FromResult(bspFile);
        }

        private void ConvertTexturesAsync(WmoV14Parser.WmoV14Data wmoData, BspFile bspFile)
        {
            Console.WriteLine($"[INFO] Processing textures...");
            
            foreach (var textureName in wmoData.Textures)
            {
                var bspTexture = new BspTexture
                {
                    Name = textureName,
                    Flags = 0
                };
                
                bspFile.Textures.Add(bspTexture);
            }
            
            Console.WriteLine($"[DEBUG] Added {bspFile.Textures.Count} textures to BSP");
        }

        private void ConvertGeometry(WmoV14Parser.WmoV14Data wmoData, BspFile bspFile)
        {
            Console.WriteLine($"[INFO] Converting geometry...");
            
            // The geometry was already added by the parser's ConvertToBsp method
            // This method is here for additional processing if needed
            
            int totalVertices = bspFile.Vertices.Count;
            int totalFaces = bspFile.Faces.Count;
            
            Console.WriteLine($"[DEBUG] Using {totalVertices} vertices and {totalFaces} faces from parser");
        }

        private BspFile BuildBspFromWmo(WoWFormatLib.Structs.WMO.WMO wmo)
        {
            var bsp = new BspFile();

            // Textures
            if (wmo.textures != null)
            {
                foreach (var t in wmo.textures)
                {
                    if (!string.IsNullOrEmpty(t.filename))
                    {
                        bsp.Textures.Add(new BspTexture { Name = t.filename, Flags = 0 });
                    }
                }
            }
            if (bsp.Textures.Count == 0)
            {
                bsp.Textures.Add(new BspTexture { Name = "textures/common/caulk", Flags = 0 });
            }

            // Geometry from groups
            if (wmo.group != null)
            {
                for (int gi = 0; gi < wmo.group.Length; gi++)
                {
                    var g = wmo.group[gi].mogp;
                    if (g.vertices == null || g.indices == null) continue;

                    var verts = g.vertices; // MOVT[] with Vector3 'vector'
                    var idx = g.indices;    // ushort[]

                    for (int i = 0; i + 2 < idx.Length; i += 3)
                    {
                        var i0 = idx[i];
                        var i1 = idx[i + 1];
                        var i2 = idx[i + 2];
                        if (i0 >= verts.Length || i1 >= verts.Length || i2 >= verts.Length)
                            continue;

                        var start = bsp.Vertices.Count;
                        var p0 = verts[i0].vector;
                        var p1 = verts[i1].vector;
                        var p2 = verts[i2].vector;

                        // Duplicate three vertices
                        bsp.Vertices.Add(new BspVertex { Position = p0, TextureCoordinate = Vector2.Zero, LightmapCoordinate = Vector2.Zero, Normal = Vector3.UnitY, Color = new byte[] { 255, 255, 255, 255 } });
                        bsp.Vertices.Add(new BspVertex { Position = p1, TextureCoordinate = Vector2.Zero, LightmapCoordinate = Vector2.Zero, Normal = Vector3.UnitY, Color = new byte[] { 255, 255, 255, 255 } });
                        bsp.Vertices.Add(new BspVertex { Position = p2, TextureCoordinate = Vector2.Zero, LightmapCoordinate = Vector2.Zero, Normal = Vector3.UnitY, Color = new byte[] { 255, 255, 255, 255 } });

                        // Face
                        var face = new BspFace
                        {
                            Texture = 0,
                            Effect = -1,
                            Type = 1,
                            FirstVertex = start,
                            NumVertices = 3,
                            FirstMeshVertex = 0,
                            NumMeshVertices = 0,
                            Lightmap = -1
                        };
                        bsp.Faces.Add(face);
                    }
                }
            }

            // Minimal entity
            var entity = new StringBuilder();
            entity.AppendLine("{");
            entity.AppendLine("  \"classname\" \"worldspawn\"");
            entity.AppendLine("}");
            bsp.Entities.Add(entity.ToString());

            // Basic structure
            GeneratePlanesFromFaces(bsp);
            if (bsp.Lightmaps.Count == 0) bsp.Lightmaps.Add(Array.Empty<byte>());
            if (bsp.Models.Count == 0)
            {
                bsp.Models.Add(new BspModel
                {
                    Min = new Vector3(-1000, -1000, -1000),
                    Max = new Vector3(1000, 1000, 1000),
                    FirstFace = 0,
                    NumFaces = bsp.Faces.Count
                });
            }
            if (bsp.Nodes.Count == 0)
            {
                bsp.Nodes.Add(new BspNode
                {
                    Plane = 0,
                    Children = new[] { -1, -1 },
                    Min = new Vector3(-1000, -1000, -1000),
                    Max = new Vector3(1000, 1000, 1000)
                });
            }
            if (bsp.Leaves.Count == 0)
            {
                bsp.Leaves.Add(new BspLeaf
                {
                    Min = new Vector3(-1000, -1000, -1000),
                    Max = new Vector3(1000, 1000, 1000),
                    Cluster = 0,
                    Area = 0,
                    FirstFace = 0,
                    NumFaces = bsp.Faces.Count
                });
            }
            bsp.LeafFaces.Clear();
            for (int i = 0; i < bsp.Faces.Count; i++) bsp.LeafFaces.Add(i);

            return bsp;
        }

        private void ConvertEntities(WmoV14Parser.WmoV14Data wmoData, BspFile bspFile)
        {
            Console.WriteLine($"[INFO] Converting entities...");
            
            // Add WMO info entity
            var entity = new StringBuilder();
            entity.AppendLine("{");
            entity.AppendLine("  \"classname\" \"worldspawn\"");
            entity.AppendLine($"  \"message\" \"WMO v{wmoData.Version}\"");
            entity.AppendLine($"  \"wmo_groups\" \"{wmoData.Groups.Count}\"");
            entity.AppendLine($"  \"wmo_textures\" \"{wmoData.Textures.Count}\"");
            entity.AppendLine("}");
            
            bspFile.Entities.Add(entity.ToString());
        }

        private void CreateBasicBspStructure(BspFile bspFile)
        {
            Console.WriteLine($"[INFO] Creating basic BSP structure...");
            
            // Generate planes from actual geometry - CRITICAL for Quake 3!
            GeneratePlanesFromFaces(bspFile);
            
            // Add one lightmap
            bspFile.Lightmaps.Add(Array.Empty<byte>());
            
            // Create one basic model
            var model = new BspModel
            {
                Min = new Vector3(-1000, -1000, -1000),
                Max = new Vector3(1000, 1000, 1000),
                FirstFace = 0,
                NumFaces = bspFile.Faces.Count
            };
            
            bspFile.Models.Add(model);
            
            // Create BSP nodes - CRITICAL for Quake 3!
            GenerateBasicBspNodes(bspFile);
            
            // Create one basic leaf
            var leaf = new BspLeaf
            {
                Min = new Vector3(-1000, -1000, -1000),
                Max = new Vector3(1000, 1000, 1000),
                Cluster = 0,
                Area = 0,
                FirstFace = 0,
                NumFaces = bspFile.Faces.Count
            };
            
            bspFile.Leaves.Add(leaf);
            
            // Create leaf faces mapping
            for (int i = 0; i < bspFile.Faces.Count; i++)
            {
                bspFile.LeafFaces.Add(i);
            }
            
            Console.WriteLine($"[SUCCESS] Basic BSP structure created with {bspFile.Planes.Count} planes, {bspFile.Nodes.Count} nodes");
        }

        private void GenerateBasicBspNodes(BspFile bspFile)
        {
            // Create a simple BSP tree with root node
            if (bspFile.Planes.Count == 0)
            {
                // If no planes, create a default one
                bspFile.Planes.Add(new BspPlane
                {
                    Normal = Vector3.UnitZ,
                    Distance = 0
                });
            }
            
            // Create root node
            var rootNode = new BspNode
            {
                Plane = 0, // Reference first plane
                Children = new int[] { -1, -1 }, // both children point to leaf 0 for a degenerate but safe tree
                Min = new Vector3(-1000, -1000, -1000),
                Max = new Vector3(1000, 1000, 1000)
            };
            
            bspFile.Nodes.Add(rootNode);
            
            Console.WriteLine($"[DEBUG] Generated {bspFile.Nodes.Count} BSP nodes");
        }

        private void GeneratePlanesFromFaces(BspFile bspFile)
        {
            Console.WriteLine($"[INFO] Generating planes from {bspFile.Faces.Count} faces...");
            
            foreach (var face in bspFile.Faces)
            {
                // Get the three vertices of this face
                if (face.FirstVertex + 2 >= bspFile.Vertices.Count)
                    continue;
                    
                var v0 = bspFile.Vertices[face.FirstVertex].Position;
                var v1 = bspFile.Vertices[face.FirstVertex + 1].Position;
                var v2 = bspFile.Vertices[face.FirstVertex + 2].Position;
                
                // Calculate plane normal from the three points
                var edge1 = v1 - v0;
                var edge2 = v2 - v0;
                var normal = Vector3.Cross(edge1, edge2);
                
                // Normalize the normal
                if (normal.Length() > 0.0001f)
                {
                    normal = Vector3.Normalize(normal);
                    
                    // Calculate plane distance from origin
                    var distance = -Vector3.Dot(normal, v0);
                    
                    // Create the plane
                    var plane = new BspPlane
                    {
                        Normal = normal,
                        Distance = distance
                    };
                    
                    bspFile.Planes.Add(plane);
                    
                    // Store the computed normal on the face
                    face.Normal = normal;
                }
            }
            
            Console.WriteLine($"[DEBUG] Generated {bspFile.Planes.Count} planes from faces");
        }

        /// <summary>
        /// Initialize FileProvider for local file access (required by WoWFormatLib)
        /// </summary>
        private void InitializeFileProvider(string wmoFilePath)
        {
            try
            {
                var wmoDirectory = Path.GetDirectoryName(wmoFilePath) ?? Directory.GetCurrentDirectory();
                var localProvider = new LocalFileProvider(wmoDirectory);
                FileProvider.SetProvider(localProvider, "local");
                FileProvider.SetDefaultBuild("local");
                Console.WriteLine($"[DEBUG] Initialized FileProvider for local directory: {wmoDirectory}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[WARNING] Failed to initialize FileProvider: {ex.Message}");
            }
        }

        /// <summary>
        /// Enhanced conversion with comprehensive output and error handling
        /// </summary>
        public async Task<ConversionResult> ConvertAsync(string inputFilePath, string outputDirectory, CancellationToken cancellationToken = default)
        {
            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            var startTime = DateTime.UtcNow;

            // Validate input
            if (!File.Exists(inputFilePath))
                throw new FileNotFoundException($"WMO file not found: {inputFilePath}", inputFilePath);

            Console.WriteLine($"🚀 Starting WMO v14 → Quake 3 BSP conversion");
            Console.WriteLine($"📁 Input: {Path.GetFileName(inputFilePath)}");
            Console.WriteLine($"📁 Output: {Path.GetFullPath(outputDirectory)}");
            Console.WriteLine($"🕒 Started at: {startTime:yyyy-MM-dd HH:mm:ss}");

            try
            {
                // Initialize FileProvider for local file access (required by WoWFormatLib)
                InitializeFileProvider(inputFilePath);
                
                // Use the proven WoWFormatLib WMOReader approach
                var reader = new WMOReader();
                var wmo = reader.LoadWMO(inputFilePath);

                var result = new ConversionResult();

                if (wmo.textures != null)
                {
                    result.TextureCount = wmo.textures.Length;
                    Console.WriteLine($"🎨 Found {result.TextureCount} textures");
                }

                if (wmo.group != null)
                {
                    result.GroupCount = wmo.group.Length;
                    Console.WriteLine($"🏗️ Found {result.GroupCount} WMO groups");

                    // Process each group and count geometry
                    foreach (var group in wmo.group)
                    {
                        if (group.mogp.vertices != null && group.mogp.indices != null)
                        {
                            result.TotalVertices += group.mogp.vertices.Length;
                            result.TotalFaces += group.mogp.indices.Length / 3;
                        }
                    }

                    Console.WriteLine($"📊 Total geometry: {result.TotalVertices} vertices, {result.TotalFaces} faces");
                }

                // Build BSP from WoWFormatLib geometry (handles separate group files)
                var bspFile = BuildBspFromWmo(wmo);
                
                // Ensure output directory exists
                Directory.CreateDirectory(outputDirectory);
                
                var bspFileName = Path.Combine(outputDirectory, Path.GetFileNameWithoutExtension(inputFilePath) + ".bsp");
                var shaderFileName = Path.Combine(outputDirectory, "scripts", Path.GetFileNameWithoutExtension(inputFilePath) + ".shader");
                
                // Create scripts directory
                var scriptsDir = Path.Combine(outputDirectory, "scripts");
                Directory.CreateDirectory(scriptsDir);

                // Write BSP file
                bspFile.Save(bspFileName);
                Console.WriteLine($"💾 BSP file written: {Path.GetFileName(bspFileName)}");

                // Generate shader scripts
                await GenerateShaderScriptsAsync(wmo, scriptsDir);
                Console.WriteLine($"🎨 Shader scripts written: {Path.Combine("scripts", Path.GetFileNameWithoutExtension(inputFilePath) + ".shader")}");

                // Update results
                result.Success = true;
                result.InputFile = inputFilePath;
                result.OutputFile = bspFileName;
                result.OutputDirectory = outputDirectory;
                result.ConversionTime = stopwatch.Elapsed;
                result.EndTime = DateTime.UtcNow;
                
                Console.WriteLine($"✅ Conversion completed successfully in {result.ConversionTime:mm\\:ss}");
                Console.WriteLine($"🎯 Results:");
                Console.WriteLine($"   - BSP file: {Path.GetFileName(bspFileName)}");
                Console.WriteLine($"   - Groups processed: {result.GroupCount}");
                Console.WriteLine($"   - Vertices converted: {result.TotalVertices}");
                Console.WriteLine($"   - Faces converted: {result.TotalFaces}");
                Console.WriteLine($"   - Textures processed: {result.TextureCount}");
                Console.WriteLine($"   - Conversion time: {result.ConversionTime:mm\\:ss}");

                return result;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Conversion failed: {ex.Message}");
                Console.WriteLine($"🔍 Details: {ex.StackTrace}");
                
                return new ConversionResult
                {
                    Success = false,
                    ErrorMessage = ex.Message,
                    InputFile = inputFilePath,
                    OutputDirectory = outputDirectory,
                    ConversionTime = stopwatch.Elapsed,
                    EndTime = DateTime.UtcNow
                };
            }
        }

        private async Task GenerateShaderScriptsAsync(dynamic wmo, string scriptsDirectory)
        {
            try
            {
                if (wmo.textures == null) return;

                var shaderContent = new StringBuilder();
                shaderContent.AppendLine("// WMO Material Shader Scripts");
                shaderContent.AppendLine("// Generated by WMO v14 to Quake 3 BSP Converter");
                shaderContent.AppendLine();

                foreach (var texture in wmo.textures)
                {
                    if (!string.IsNullOrEmpty(texture.filename))
                    {
                        var shaderName = $"wmo_{Path.GetFileNameWithoutExtension(texture.filename)}";
                        shaderContent.AppendLine($"textures/{shaderName}");
                        shaderContent.AppendLine("{");
                        shaderContent.AppendLine($"    qer_editorImage textures/{texture.filename}");
                        shaderContent.AppendLine("    {");
                        shaderContent.AppendLine($"        map textures/{texture.filename}");
                        shaderContent.AppendLine("        rgbGen vertex");
                        shaderContent.AppendLine("    }");
                        shaderContent.AppendLine("}");
                        shaderContent.AppendLine();
                    }
                }

                var shaderFileName = Path.Combine(scriptsDirectory, "wmo_materials.shader");
                await File.WriteAllTextAsync(shaderFileName, shaderContent.ToString());
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[WARNING] Failed to generate shader scripts: {ex.Message}");
            }
        }
    }

    /// <summary>
    /// Enhanced conversion result with comprehensive statistics
    /// </summary>
    public class ConversionResult
    {
        public bool Success { get; set; }
        public string? InputFile { get; set; }
        public string? OutputFile { get; set; }
        public string? OutputDirectory { get; set; }
        public string? ErrorMessage { get; set; }
        public TimeSpan ConversionTime { get; set; }
        public DateTime StartTime { get; set; }
        public DateTime EndTime { get; set; }
        
        public int TextureCount { get; set; }
        public int GroupCount { get; set; }
        public int TotalVertices { get; set; }
        public int TotalFaces { get; set; }
        
        public override string ToString()
        {
            if (Success)
            {
                return $"""
                ✅ Conversion Successful
                📁 Input: {Path.GetFileName(InputFile ?? "Unknown")}
                📁 Output: {Path.GetFileName(OutputFile ?? "Unknown")}
                ⏱️ Time: {ConversionTime:mm\:ss}
                📊 Stats: {GroupCount} groups, {TotalVertices} vertices, {TotalFaces} faces, {TextureCount} textures
                """;
            }
            else
            {
                return $"""
                ❌ Conversion Failed
                📁 Input: {Path.GetFileName(InputFile ?? "Unknown")}
                💥 Error: {ErrorMessage}
                ⏱️ Time: {ConversionTime:mm\:ss}
                """;
            }
        }
    }
}