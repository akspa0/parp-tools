using System;
using System.Collections.Generic;
using System.IO;
using ArcaneFileParser.Core.Chunks;
using ArcaneFileParser.Core.Formats.WMO.Chunks;
using ArcaneFileParser.Core.Formats.WMO.Structures;
using ArcaneFileParser.Core.Formats.WMO.Validation;

namespace ArcaneFileParser.Core.Formats.WMO.Converters
{
    /// <summary>
    /// Handles conversion of WMO files from v14 (Alpha) to v17 format
    /// </summary>
    public class WmoV14ToV17Converter
    {
        private readonly WMOFile sourceWmo;
        private readonly WMOFile targetWmo;
        private readonly List<string> conversionErrors;

        // Lightmap constants
        private const int LIGHTMAP_WIDTH = 256;
        private const int LIGHTMAP_HEIGHT = 256;

        // Flag conversion mappings
        private static readonly Dictionary<uint, uint> MohdFlagMap = new Dictionary<uint, uint>
        {
            { 0x1, 0x1 },     // DO_NOT_ATTENUATE_VERTICES_BASED_ON_DISTANCE_TO_PORTAL
            { 0x2, 0x2 },     // USE_UNIFIED_RENDER_PATH
            { 0x4, 0x4 },     // USE_LIQUID_TYPE_DBC_ID
            { 0x8, 0x8 },     // DO_NOT_FIX_VERTEX_COLOR_ALPHA
            { 0x10, 0x0 },    // LIGHTMAP (v14) -> removed in v17
            { 0x20, 0x0 },    // EXTERIOR_LIT (v14) -> handled differently in v17
            { 0x40, 0x40 }    // AMBIENT_LIGHT
        };

        private static readonly Dictionary<uint, uint> MogpFlagMap = new Dictionary<uint, uint>
        {
            { 0x1, 0x1 },     // HAS_BSP
            { 0x2, 0x0 },     // LIGHTMAP (v14) -> removed in v17
            { 0x4, 0x4 },     // HAS_VERTEX_COLORS
            { 0x8, 0x8 },     // EXTERIOR
            { 0x10, 0x0 },    // UNK_WOD (v14) -> removed in v17
            { 0x20, 0x20 },   // NO_COLLISION
            { 0x40, 0x40 },   // EXTERIOR_LIT
            { 0x80, 0x80 },   // UNREACHABLE
            { 0x100, 0x100 }, // SHOW_SKY
            { 0x200, 0x200 }, // HAS_LIGHTS
            { 0x400, 0x0 },   // LIGHT_CULLED (v14) -> removed in v17
            { 0x800, 0x800 }, // HAS_DOODADS
            { 0x1000, 0x1000 }, // HAS_WATER
            { 0x2000, 0x2000 }, // INDOOR
            { 0x8000, 0x8000 }, // HAS_BSP2
            { 0x10000, 0x10000 } // ALWAYS_DRAW
        };

        public WmoV14ToV17Converter(WMOFile source)
        {
            if (source.Version != 14)
                throw new ArgumentException("Source WMO must be version 14");

            sourceWmo = source;
            targetWmo = new WMOFile();
            conversionErrors = new List<string>();
        }

        public WMOFile Convert()
        {
            try
            {
                // Validate source WMO collision integrity
                var sourceValidator = new WmoCollisionValidator(sourceWmo);
                if (!sourceValidator.Validate())
                {
                    foreach (var error in sourceValidator.GetValidationErrors())
                    {
                        conversionErrors.Add($"Source validation: {error}");
                    }
                    throw new InvalidDataException("Source WMO failed collision validation");
                }

                // Convert root chunks
                ConvertRootChunks();

                // Convert group chunks
                ConvertGroupChunks();

                // Validate target WMO collision integrity
                var targetValidator = new WmoCollisionValidator(targetWmo);
                if (!targetValidator.Validate())
                {
                    foreach (var error in targetValidator.GetValidationErrors())
                    {
                        conversionErrors.Add($"Target validation: {error}");
                    }
                    throw new InvalidDataException("Converted WMO failed collision validation");
                }

                return targetWmo;
            }
            catch (Exception ex)
            {
                conversionErrors.Add($"Conversion error: {ex.Message}");
                throw;
            }
        }

        public IReadOnlyList<string> GetConversionErrors() => conversionErrors;

        private void ConvertRootChunks()
        {
            // Set version to 17
            var mver = new MVER { Version = 17 };
            targetWmo.Chunks["MVER"] = mver;

            // Convert MOHD
            ConvertMOHD();

            // Copy unchanged chunks
            CopyChunk("MOTX");
            CopyChunk("MOMT");
            CopyChunk("MOGN");
            CopyChunk("MOGI");
            CopyChunk("MOSB");

            // Convert portal chunks
            ConvertPortalChunks();

            // Convert lighting
            ConvertLighting();

            // Convert doodads
            ConvertDoodadChunks();

            // Copy fog
            CopyChunk("MFOG");
        }

        private void ConvertMOHD()
        {
            var sourceMohd = sourceWmo.GetChunk<MOHD>("MOHD");
            var targetMohd = new MOHD
            {
                TextureCount = sourceMohd.TextureCount,
                GroupCount = sourceMohd.GroupCount,
                PortalCount = sourceMohd.PortalCount,
                LightCount = 0, // Reset light count as we'll convert to vertex lighting
                DoodadNameCount = sourceMohd.DoodadNameCount,
                DoodadDefinitionCount = sourceMohd.DoodadDefinitionCount,
                DoodadSetCount = sourceMohd.DoodadSetCount,
                AmbientColor = sourceMohd.AmbientColor,
                WmoId = sourceMohd.WmoId,
                BoundingBox = sourceMohd.BoundingBox.Clone(),
                Flags = ConvertMohdFlags(sourceMohd.Flags)
            };

            targetWmo.Chunks["MOHD"] = targetMohd;
        }

        private uint ConvertMohdFlags(uint sourceFlags)
        {
            uint targetFlags = 0;
            foreach (var flagPair in MohdFlagMap)
            {
                if ((sourceFlags & flagPair.Key) != 0)
                {
                    targetFlags |= flagPair.Value;
                }
            }
            return targetFlags;
        }

        private void ConvertPortalChunks()
        {
            // Convert MOPT (Portal definitions)
            var sourceMopt = sourceWmo.GetChunk<MOPT>("MOPT");
            if (sourceMopt != null)
            {
                var targetMopt = new MOPT();
                foreach (var portal in sourceMopt.Portals)
                {
                    targetMopt.AddPortal(new Portal
                    {
                        BasePosition = portal.BasePosition.Clone(),
                        Normal = portal.Normal.Clone(),
                        VertexCount = portal.VertexCount
                    });
                }
                targetWmo.Chunks["MOPT"] = targetMopt;
            }

            // Convert MOPV (Portal vertices)
            CopyChunk("MOPV");

            // Convert MOPR (Portal references)
            var sourceMopr = sourceWmo.GetChunk<MOPR>("MOPR");
            if (sourceMopr != null)
            {
                var targetMopr = new MOPR();
                foreach (var reference in sourceMopr.References)
                {
                    // Convert 32-bit portal indices to 16-bit
                    if (reference.PortalIndex > ushort.MaxValue)
                        throw new InvalidDataException("Portal index too large for v17 format");

                    targetMopr.AddReference(new PortalReference
                    {
                        PortalIndex = (ushort)reference.PortalIndex,
                        GroupIndex = (ushort)reference.GroupIndex,
                        Side = reference.Side
                    });
                }
                targetWmo.Chunks["MOPR"] = targetMopr;
            }
        }

        private void ConvertLighting()
        {
            var molm = sourceWmo.GetChunk<MOLM>("MOLM");
            var mold = sourceWmo.GetChunk<MOLD>("MOLD");

            if (molm != null && mold != null)
            {
                // Convert lightmap data to vertex colors
                foreach (var group in sourceWmo.Groups)
                {
                    var mocv = new MOCV();
                    var vertices = group.GetChunk<MOVT>("MOVT").Vertices;
                    var colors = new List<Color>();

                    // Get group's lightmap entry
                    var lightmapIndex = GetGroupLightmapIndex(group);
                    if (lightmapIndex >= 0 && lightmapIndex < molm.Lightmaps.Count)
                    {
                        var lightmapEntry = molm.Lightmaps[lightmapIndex];
                        var lightmapData = mold.Textures[lightmapIndex];

                        // Calculate lightmap coordinates for each vertex
                        for (int i = 0; i < vertices.Count; i++)
                        {
                            var vertex = vertices[i];
                            var lightmapUV = GetLightmapCoordinates(vertex, group, lightmapEntry);
                            var color = SampleLightmapTexture(lightmapUV, lightmapEntry, lightmapData);
                            colors.Add(color);
                        }
                    }
                    else
                    {
                        // No lightmap for this group, use default lighting
                        for (int i = 0; i < vertices.Count; i++)
                        {
                            colors.Add(new Color(255, 255, 255, 255));
                        }
                    }

                    mocv.Colors = colors;
                    group.Chunks["MOCV"] = mocv;
                }
            }
        }

        private int GetGroupLightmapIndex(WMOGroup group)
        {
            var mogp = group.GetChunk<MOGP>("MOGP");
            if ((mogp.Flags & 0x2) != 0) // Has lightmap flag in v14
            {
                // The lightmap index is typically stored in the group data
                // This is a simplified version - you might need to adjust based on actual format
                return (int)mogp.UniqueId % sourceWmo.GetChunk<MOLD>("MOLD").Textures.Count;
            }
            return -1;
        }

        private Vector2 GetLightmapCoordinates(Vector3 vertex, WMOGroup group, MOLM.LightmapEntry entry)
        {
            // Get group's bounding box
            var mogp = group.GetChunk<MOGP>("MOGP");
            var bbox = mogp.BoundingBox;

            // Calculate relative position within bounding box
            float u = (vertex.X - bbox.Min.X) / (bbox.Max.X - bbox.Min.X);
            float v = (vertex.Z - bbox.Min.Z) / (bbox.Max.Z - bbox.Min.Z);

            // Map to lightmap coordinates
            u = entry.X / (float)LIGHTMAP_WIDTH + (u * entry.Width / (float)LIGHTMAP_WIDTH);
            v = entry.Y / (float)LIGHTMAP_HEIGHT + (v * entry.Height / (float)LIGHTMAP_HEIGHT);

            return new Vector2(u, v);
        }

        private Color SampleLightmapTexture(Vector2 uv, MOLM.LightmapEntry entry, MOLD.LightmapTexture texture)
        {
            // Sample the DXT1 texture using our decoder
            return DXT1Decoder.SampleTexture(
                texture.Texels,
                LIGHTMAP_WIDTH,
                LIGHTMAP_HEIGHT,
                uv.X,
                uv.Y
            );
        }

        private void ConvertDoodadChunks()
        {
            // Copy doodad chunks if they exist
            CopyChunk("MODS");
            CopyChunk("MODN");
            CopyChunk("MODD");
        }

        private void CopyChunk(string chunkId)
        {
            if (sourceWmo.Chunks.TryGetValue(chunkId, out var chunk))
            {
                targetWmo.Chunks[chunkId] = chunk;
            }
        }

        private void ConvertGroupChunks()
        {
            foreach (var sourceGroup in sourceWmo.Groups)
            {
                var targetGroup = new WMOGroup();

                // Set version to 17
                targetGroup.Chunks["MVER"] = new MVER { Version = 17 };

                // Convert MOGP
                var sourceMogp = sourceGroup.GetChunk<MOGP>("MOGP");
                sourceMogp.ConvertToV17();
                targetGroup.Chunks["MOGP"] = sourceMogp;

                // Copy geometry chunks
                CopyGroupChunk(sourceGroup, targetGroup, "MOPY");
                CopyGroupChunk(sourceGroup, targetGroup, "MOVI");
                CopyGroupChunk(sourceGroup, targetGroup, "MOVT");
                CopyGroupChunk(sourceGroup, targetGroup, "MONR");
                CopyGroupChunk(sourceGroup, targetGroup, "MOTV");

                // Convert batches
                ConvertGroupBatches(sourceGroup, targetGroup);

                // Copy BSP tree
                CopyGroupChunk(sourceGroup, targetGroup, "MOBN");
                CopyGroupChunk(sourceGroup, targetGroup, "MOBR");

                targetWmo.Groups.Add(targetGroup);
            }
        }

        private void CopyGroupChunk(WMOGroup source, WMOGroup target, string chunkId)
        {
            if (source.Chunks.TryGetValue(chunkId, out var chunk))
            {
                target.Chunks[chunkId] = chunk;
            }
        }

        private void ConvertGroupBatches(WMOGroup source, WMOGroup target)
        {
            var sourceMogp = source.GetChunk<MOGP>("MOGP");
            
            // Create MOBA chunk from v14 batch data
            var moba = new MOBA();
            
            // Convert interior batches
            foreach (var batch in sourceMogp.InteriorBatches)
            {
                if (batch.Count > 0)
                {
                    moba.AddBatch(new MOBABatch
                    {
                        StartVertex = batch.StartVertex,
                        VertexCount = batch.Count,
                        MinIndex = batch.MinIndex,
                        MaxIndex = batch.MaxIndex,
                        MaterialId = (byte)batch.MaterialId,
                        Flags = 0 // Interior batch
                    });
                }
            }

            // Convert exterior batches
            foreach (var batch in sourceMogp.ExteriorBatches)
            {
                if (batch.Count > 0)
                {
                    moba.AddBatch(new MOBABatch
                    {
                        StartVertex = batch.StartVertex,
                        VertexCount = batch.Count,
                        MinIndex = batch.MinIndex,
                        MaxIndex = batch.MaxIndex,
                        MaterialId = (byte)batch.MaterialId,
                        Flags = 1 // Exterior batch
                    });
                }
            }

            if (moba.Batches.Count > 0)
            {
                target.Chunks["MOBA"] = moba;
            }
        }
    }
} 