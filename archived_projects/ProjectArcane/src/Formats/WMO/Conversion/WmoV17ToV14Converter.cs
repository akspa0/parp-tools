using System;
using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using System.Linq;
using ArcaneFileParser.Core.Chunks;
using ArcaneFileParser.Core.Formats.WMO.Chunks;
using ArcaneFileParser.Core.Types;

namespace ArcaneFileParser.Core.Formats.WMO.Conversion
{
    /// <summary>
    /// Converts WMO files from v17+ format to v14 (alpha) format.
    /// This involves:
    /// 1. Removing post-v14 chunks
    /// 2. Converting modern lighting to legacy lighting
    /// 3. Removing volume system chunks
    /// 4. Updating version numbers and flags
    /// 5. Converting asset paths to v14 format
    /// </summary>
    public class WmoV17ToV14Converter
    {
        private readonly Dictionary<string, IChunk> _sourceChunks;
        private readonly Dictionary<string, IChunk> _convertedChunks;
        private readonly Dictionary<string, string> _pathConversionCache;

        // Common path prefixes in modern WMO files
        private static readonly Dictionary<string, string> PathPrefixes = new()
        {
            { @"interface\", "Interface\\" },
            { @"textures\", "Textures\\" },
            { @"models\", "World\\Wmo\\" },
            { @"world\wmo\", "World\\Wmo\\" },
            { @"world\maps\", "World\\Maps\\" }
        };

        public WmoV17ToV14Converter()
        {
            _sourceChunks = new Dictionary<string, IChunk>();
            _convertedChunks = new Dictionary<string, IChunk>();
            _pathConversionCache = new Dictionary<string, string>();
        }

        /// <summary>
        /// Converts a v17+ WMO file to v14 format.
        /// </summary>
        /// <param name="sourceFile">Path to the source WMO file.</param>
        /// <param name="targetFile">Path where the converted file should be saved.</param>
        /// <returns>True if conversion was successful, false otherwise.</returns>
        public bool ConvertFile(string sourceFile, string targetFile)
        {
            try
            {
                // Read source file
                using (var reader = new BinaryReader(File.OpenRead(sourceFile)))
                {
                    ReadChunks(reader);
                }

                // Perform conversion
                ConvertToV14();

                // Write converted file
                using (var writer = new BinaryWriter(File.Create(targetFile)))
                {
                    WriteChunks(writer);
                }

                // Report any path conversion warnings
                ReportPathConversionWarnings();

                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error converting file: {ex.Message}");
                return false;
            }
        }

        private void ReadChunks(BinaryReader reader)
        {
            while (reader.BaseStream.Position < reader.BaseStream.Length)
            {
                var chunkName = new string(reader.ReadChars(4));
                var chunkSize = reader.ReadUInt32();
                var chunkData = reader.ReadBytes((int)chunkSize);

                using (var chunkReader = new BinaryReader(new MemoryStream(chunkData)))
                {
                    var chunk = CreateChunk(chunkName);
                    if (chunk != null)
                    {
                        chunk.Read(chunkReader);
                        _sourceChunks[chunkName] = chunk;
                    }
                }
            }
        }

        private void ConvertToV14()
        {
            // Update version to v14
            if (_sourceChunks.TryGetValue("MVER", out var mverChunk))
            {
                var mver = (MVER)mverChunk;
                mver.Version = 14;
                _convertedChunks["MVER"] = mver;
            }

            // Convert header
            if (_sourceChunks.TryGetValue("MOHD", out var mohdChunk))
            {
                var mohd = (MOHD)mohdChunk;
                // Clear post-v14 flags
                mohd.Flags &= 0x7; // Keep only first 3 bits
                _convertedChunks["MOHD"] = mohd;
            }

            // Convert modern volumes to v14 compatible formats
            ConvertVolumes();

            // Convert modern lighting to legacy lighting
            ConvertLighting();

            // Copy v14-compatible chunks
            foreach (var chunk in _sourceChunks)
            {
                if (IsV14CompatibleChunk(chunk.Key))
                {
                    _convertedChunks[chunk.Key] = chunk.Value;
                }
            }
        }

        private void ConvertVolumes()
        {
            // Convert box volumes to visible blocks
            if (_sourceChunks.TryGetValue("MBVD", out var mbvdChunk))
            {
                var mbvd = (MBVD)mbvdChunk;
                var movb = new MOVB();
                var movv = new MOVV(); // We'll need this for vertex storage

                foreach (var boxVolume in mbvd.AmbientBoxVolumes)
                {
                    // Create vertices for the box volume using the planes
                    var vertices = CreateBoxVolumeVertices(boxVolume.Planes);
                    
                    // Add vertices to MOVV
                    var startIndex = movv.VisibleVertices.Count;
                    movv.VisibleVertices.AddRange(vertices);

                    // Create visible block
                    var block = new MOVB.VisibleBlock
                    {
                        FirstVertex = (ushort)startIndex,
                        Count = (ushort)vertices.Count
                    };
                    movb.VisibleBlocks.Add(block);
                }

                _convertedChunks["MOVB"] = movb;
                _convertedChunks["MOVV"] = movv;
            }

            // Convert ambient volumes to ambient lights
            if (_sourceChunks.TryGetValue("MAVD", out var mavdChunk))
            {
                var mavd = (MAVD)mavdChunk;
                var molt = _convertedChunks.TryGetValue("MOLT", out var existingMolt) 
                    ? (MOLT)existingMolt 
                    : new MOLT();

                foreach (var ambientVolume in mavd.AmbientVolumes)
                {
                    var light = new MOLT.Light
                    {
                        Type = 3, // AMBIENT_LGT
                        UseAttenuation = 1,
                        Position = ambientVolume.Position,
                        Color = ambientVolume.Color1,
                        Intensity = 1.0f,
                        AttenuationStart = ambientVolume.Start,
                        AttenuationEnd = ambientVolume.End
                    };

                    molt.Lights.Add(light);
                }

                _convertedChunks["MOLT"] = molt;
            }

            // Convert global ambient volumes to ambient lights
            if (_sourceChunks.TryGetValue("MAVG", out var mavgChunk))
            {
                var mavg = (MAVG)mavgChunk;
                var molt = _convertedChunks.TryGetValue("MOLT", out var existingMolt) 
                    ? (MOLT)existingMolt 
                    : new MOLT();

                foreach (var globalVolume in mavg.GlobalAmbientVolumes)
                {
                    var light = new MOLT.Light
                    {
                        Type = 3, // AMBIENT_LGT
                        UseAttenuation = 0, // Global ambient has no attenuation
                        Position = new C3Vector(), // At origin
                        Color = globalVolume.Color1,
                        Intensity = 1.0f,
                        AttenuationStart = 0,
                        AttenuationEnd = 0
                    };

                    molt.Lights.Add(light);
                }

                _convertedChunks["MOLT"] = molt;
            }
        }

        private List<C3Vector> CreateBoxVolumeVertices(C4Plane[] planes)
        {
            var vertices = new List<C3Vector>();
            
            // Find intersections of every combination of three planes
            for (int i = 0; i < 4; i++)
            {
                for (int j = i + 1; j < 5; j++)
                {
                    for (int k = j + 1; k < 6; k++)
                    {
                        var vertex = IntersectThreePlanes(planes[i], planes[j], planes[k]);
                        if (vertex != null && IsPointInsideAllPlanes(vertex.Value, planes))
                        {
                            vertices.Add(vertex.Value);
                        }
                    }
                }
            }

            return vertices;
        }

        private C3Vector? IntersectThreePlanes(C4Plane p1, C4Plane p2, C4Plane p3)
        {
            // Create vectors from plane normals
            var n1 = new C3Vector { X = p1.A, Y = p1.B, Z = p1.C };
            var n2 = new C3Vector { X = p2.A, Y = p2.B, Z = p2.C };
            var n3 = new C3Vector { X = p3.A, Y = p3.B, Z = p3.C };

            // Calculate determinant
            var det = n1.X * (n2.Y * n3.Z - n2.Z * n3.Y) -
                     n1.Y * (n2.X * n3.Z - n2.Z * n3.X) +
                     n1.Z * (n2.X * n3.Y - n2.Y * n3.X);

            if (Math.Abs(det) < 1e-6f)
                return null; // Planes are parallel or nearly parallel

            // Calculate intersection point
            var point = new C3Vector
            {
                X = (-p1.D * (n2.Y * n3.Z - n2.Z * n3.Y) +
                     -p2.D * (n3.Y * n1.Z - n3.Z * n1.Y) +
                     -p3.D * (n1.Y * n2.Z - n1.Z * n2.Y)) / det,
                Y = (-p1.D * (n2.Z * n3.X - n2.X * n3.Z) +
                     -p2.D * (n3.Z * n1.X - n3.X * n1.Z) +
                     -p3.D * (n1.Z * n2.X - n1.X * n2.Z)) / det,
                Z = (-p1.D * (n2.X * n3.Y - n2.Y * n3.X) +
                     -p2.D * (n3.X * n1.Y - n3.Y * n1.X) +
                     -p3.D * (n1.X * n2.Y - n1.Y * n2.X)) / det
            };

            return point;
        }

        private bool IsPointInsideAllPlanes(C3Vector point, C4Plane[] planes)
        {
            foreach (var plane in planes)
            {
                var distance = plane.A * point.X + plane.B * point.Y + plane.C * point.Z + plane.D;
                if (distance > 1e-6f) // Point is in front of plane
                    return false;
            }
            return true;
        }

        private bool IsV14CompatibleChunk(string chunkName)
        {
            // List of chunks that existed in v14
            var v14Chunks = new HashSet<string>
            {
                "MVER", "MOHD", "MOTX", "MOMT", "MOGN", "MOGI", "MOSB", 
                "MOPV", "MOPT", "MOPR", "MOVV", "MOVB", "MOLT", "MODS",
                "MODN", "MODD", "MFOG", "MCVP", "MOLD", "MOLM", "MOIN"
            };

            return v14Chunks.Contains(chunkName);
        }

        private void ConvertTexturePaths(MOTX motx)
        {
            var convertedPaths = new List<string>();
            foreach (var path in motx.TexturePaths)
            {
                convertedPaths.Add(ConvertAssetPath(path, "Textures\\"));
            }
            motx.TexturePaths.Clear();
            motx.TexturePaths.AddRange(convertedPaths);
        }

        private void ConvertModelPaths(MODN modn)
        {
            var convertedPaths = new List<string>();
            foreach (var path in modn.ModelPaths)
            {
                convertedPaths.Add(ConvertAssetPath(path, "World\\Wmo\\"));
            }
            modn.ModelPaths.Clear();
            modn.ModelPaths.AddRange(convertedPaths);
        }

        private string ConvertAssetPath(string path, string defaultPrefix)
        {
            if (string.IsNullOrEmpty(path))
                return path;

            // Check cache first
            if (_pathConversionCache.TryGetValue(path, out var cachedPath))
                return cachedPath;

            // Convert path to lowercase for comparison
            var lowerPath = path.ToLowerInvariant();

            // Remove any BLP/M2/WMO extension for processing
            var extension = Path.GetExtension(lowerPath);
            var pathWithoutExt = extension != "" ? path[..^extension.Length] : path;

            // Convert path separators
            var normalizedPath = pathWithoutExt.Replace('/', '\\');

            // Check if path starts with any known prefix
            var hasKnownPrefix = false;
            foreach (var prefix in PathPrefixes)
            {
                if (lowerPath.StartsWith(prefix.Key))
                {
                    normalizedPath = prefix.Value + normalizedPath[prefix.Key.Length..];
                    hasKnownPrefix = true;
                    break;
                }
            }

            // If no known prefix found, add default prefix
            if (!hasKnownPrefix)
            {
                normalizedPath = defaultPrefix + normalizedPath;
            }

            // Add extension back
            var convertedPath = normalizedPath + extension;

            // Cache the conversion
            _pathConversionCache[path] = convertedPath;

            return convertedPath;
        }

        private void ReportPathConversionWarnings()
        {
            if (_pathConversionCache.Count == 0)
                return;

            Console.WriteLine("\nPath Conversion Report:");
            Console.WriteLine("----------------------");
            
            foreach (var conversion in _pathConversionCache)
            {
                if (conversion.Key != conversion.Value)
                {
                    Console.WriteLine($"Converted: {conversion.Key}");
                    Console.WriteLine($"      To: {conversion.Value}\n");
                }
            }
        }

        private void ConvertLighting()
        {
            // Create legacy lighting chunks if modern lighting exists
            if (_sourceChunks.ContainsKey("MOLS") || _sourceChunks.ContainsKey("MOLP"))
            {
                var molv = new MOLV();
                var moin = new MOIN();
                var moma = new MOMA();

                // Convert spot lights
                if (_sourceChunks.TryGetValue("MOLS", out var molsChunk))
                {
                    var mols = (MOLS)molsChunk;
                    ConvertSpotLightsToLegacy(mols, molv, moin, moma);
                }

                // Convert point lights
                if (_sourceChunks.TryGetValue("MOLP", out var molpChunk))
                {
                    var molp = (MOLP)molpChunk;
                    ConvertPointLightsToLegacy(molp, molv, moin, moma);
                }

                // Add converted chunks
                _convertedChunks["MOLV"] = molv;
                _convertedChunks["MOIN"] = moin;
                _convertedChunks["MOMA"] = moma;
            }
        }

        private void ConvertSpotLightsToLegacy(MOLS mols, MOLV molv, MOIN moin, MOMA moma)
        {
            foreach (var spotLight in mols.SpotLights)
            {
                if (!spotLight.IsUsed)
                    continue;

                // Convert spot light to lightmap vertex
                var vertex = new MOLV.LightmapVertex
                {
                    Position = spotLight.Position,
                    Color = spotLight.Color,
                    // Calculate UV coordinates based on light properties
                    U = Math.Min(1.0f, spotLight.Intensity / 255.0f),
                    V = Math.Min(1.0f, spotLight.EndRange / 255.0f)
                };

                molv.LightmapVertices.Add(vertex);

                // Add indices
                moin.Indices.Add((ushort)(molv.LightmapVertices.Count - 1));

                // Add material attributes
                var materialAttr = new MOMA.MaterialAttribute
                {
                    Flags = (uint)(spotLight.Flags & 0xFF), // Keep only relevant flags
                    Transparency = Math.Min(1.0f, spotLight.AttenuationEnd / 255.0f)
                };
                moma.MaterialAttributes.Add(materialAttr);
            }
        }

        private void ConvertPointLightsToLegacy(MOLP molp, MOLV molv, MOIN moin, MOMA moma)
        {
            foreach (var pointLight in molp.LightPoints)
            {
                if (!pointLight.IsUsed)
                    continue;

                // Convert point light to lightmap vertex
                var vertex = new MOLV.LightmapVertex
                {
                    Position = pointLight.Position,
                    Color = pointLight.Color,
                    // Calculate UV coordinates based on light properties
                    U = Math.Min(1.0f, pointLight.Intensity / 255.0f),
                    V = Math.Min(1.0f, pointLight.AttenuationEnd / 255.0f)
                };

                molv.LightmapVertices.Add(vertex);

                // Add indices
                moin.Indices.Add((ushort)(molv.LightmapVertices.Count - 1));

                // Add material attributes
                var materialAttr = new MOMA.MaterialAttribute
                {
                    Flags = (uint)(pointLight.Info & 0xFF), // Keep only relevant flags
                    Transparency = Math.Min(1.0f, pointLight.AttenuationEnd / 255.0f)
                };
                moma.MaterialAttributes.Add(materialAttr);
            }
        }

        private void WriteChunks(BinaryWriter writer)
        {
            // Write chunks in correct order
            var chunkOrder = new[]
            {
                "MVER", "MOHD", "MOTX", "MOMT", "MOGN", "MOGI", "MOSB", "MOPV", "MOPT",
                "MOPR", "MOVV", "MOVB", "MOLT", "MODS", "MODN", "MODD", "MFOG", "MCVP",
                "MOPY", "MOVI", "MOVT", "MONR", "MOTV", "MOLV", "MOIN", "MOMA", "MOBA"
            };

            foreach (var chunkName in chunkOrder)
            {
                if (_convertedChunks.TryGetValue(chunkName, out var chunk))
                {
                    // Write chunk header
                    writer.Write(chunkName.ToCharArray());
                    
                    // Get chunk data
                    using (var ms = new MemoryStream())
                    using (var chunkWriter = new BinaryWriter(ms))
                    {
                        chunk.Write(chunkWriter);
                        var chunkData = ms.ToArray();
                        
                        // Write chunk size and data
                        writer.Write((uint)chunkData.Length);
                        writer.Write(chunkData);
                    }
                }
            }
        }

        private IChunk CreateChunk(string chunkName)
        {
            return chunkName switch
            {
                "MVER" => new MVER(),
                "MOHD" => new MOHD(),
                "MOTX" => new MOTX(),
                "MOMT" => new MOMT(),
                "MOGN" => new MOGN(),
                "MOGI" => new MOGI(),
                "MOSB" => new MOSB(),
                "MOPV" => new MOPV(),
                "MOPT" => new MOPT(),
                "MOPR" => new MOPR(),
                "MOVV" => new MOVV(),
                "MOVB" => new MOVB(),
                "MOLT" => new MOLT(),
                "MODS" => new MODS(),
                "MODN" => new MODN(),
                "MODD" => new MODD(),
                "MFOG" => new MFOG(),
                "MCVP" => new MCVP(),
                "MOPY" => new MOPY(),
                "MOVI" => new MOVI(),
                "MOVT" => new MOVT(),
                "MONR" => new MONR(),
                "MOTV" => new MOTV(),
                "MOLS" => new MOLS(),
                "MOLP" => new MOLP(),
                "MOLR" => new MOLR(),
                "MOLV" => new MOLV(),
                "MOIN" => new MOIN(),
                "MOMA" => new MOMA(),
                "MOBA" => new MOBA(),
                _ => null
            };
        }
    }
} 