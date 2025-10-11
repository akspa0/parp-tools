using System.Collections.Generic;
using System.IO;
using ArcaneFileParser.Core.Chunks;
using ArcaneFileParser.Core.Interfaces;

namespace ArcaneFileParser.Core.Formats.WMO.Chunks
{
    /// <summary>
    /// MOLD chunk - Lightmap texture data for v14 (Alpha) WMO
    /// Contains DXT1 compressed color palette data
    /// </summary>
    public class MOLD : ChunkBase
    {
        public override string ChunkId => "MOLD";

        public class LightmapTexture
        {
            public const int TEXEL_SIZE = 32768; // Size of DXT1 compressed texture data
            public byte[] Texels { get; set; }
            public uint InMemoryPadding { get; set; } // Always 0 in file
        }

        public List<LightmapTexture> Textures { get; private set; }

        public MOLD()
        {
            Textures = new List<LightmapTexture>();
        }

        public override void Read(BinaryReader reader, uint size)
        {
            int numTextures = (int)(size / (LightmapTexture.TEXEL_SIZE + 4)); // Each texture is 32768 bytes + 4 bytes padding
            Textures.Clear();

            for (int i = 0; i < numTextures; i++)
            {
                var texture = new LightmapTexture
                {
                    Texels = reader.ReadBytes(LightmapTexture.TEXEL_SIZE),
                    InMemoryPadding = reader.ReadUInt32() // Should always be 0 in file
                };
                Textures.Add(texture);
            }
        }
    }
} 