using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace WoWRollback.LkToAlphaModule.Liquids;

internal static class Mh2oSerializer
{
    private const int HeaderCount = 256;
    private const int HeaderSize = 12;
    private const int InstanceStructSize = 24;

    public static byte[] Build(Mh2oChunk?[] chunks)
    {
        if (chunks.Length != HeaderCount)
            throw new ArgumentException($"Expected {HeaderCount} MH2O entries", nameof(chunks));

        bool hasLayers = chunks.Any(c => c is { IsEmpty: false });
        if (!hasLayers)
            return Array.Empty<byte>();

        var infos = new HeaderSerializationInfo[HeaderCount];
        for (int i = 0; i < HeaderCount; i++)
        {
            var chunk = chunks[i];
            infos[i] = PrepareHeader(chunk);
        }

        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        // Reserve header table
        writer.Write(new byte[HeaderCount * HeaderSize]);

        // Stage 1: allocate instance tables
        foreach (ref var info in infos.AsSpan())
        {
            if (info.InstanceCount == 0)
                continue;

            info.OffsetInstances = (uint)ms.Position;
            writer.Write(new byte[info.InstanceCount * InstanceStructSize]);
        }

        // Stage 2: write attributes and instance payloads
        foreach (ref var info in infos.AsSpan())
        {
            if (info.InstanceCount == 0)
                continue;

            if (info.AttributeBytes is not null)
            {
                info.OffsetAttributes = (uint)ms.Position;
                writer.Write(info.AttributeBytes);
            }
            else
            {
                info.OffsetAttributes = 0;
            }

            foreach (var instance in info.Instances)
            {
                if (instance.ExistsBitmap?.Length > 0)
                {
                    instance.OffsetExistsBitmap = (uint)ms.Position;
                    writer.Write(instance.ExistsBitmap);
                }
                else
                {
                    instance.OffsetExistsBitmap = 0;
                }

                if (instance.VertexBytes.Length > 0)
                {
                    instance.OffsetVertexData = (uint)ms.Position;
                    writer.Write(instance.VertexBytes);
                }
                else
                {
                    instance.OffsetVertexData = 0;
                }
            }
        }

        // Stage 3: fill instance tables
        foreach (ref var info in infos.AsSpan())
        {
            if (info.InstanceCount == 0)
                continue;

            ms.Position = info.OffsetInstances;
            foreach (var instance in info.Instances)
            {
                writer.Write(instance.LiquidTypeId);
                writer.Write((ushort)instance.VertexFormat);
                writer.Write(instance.MinHeightLevel);
                writer.Write(instance.MaxHeightLevel);
                writer.Write(instance.XOffset);
                writer.Write(instance.YOffset);
                writer.Write(instance.Width);
                writer.Write(instance.Height);
                writer.Write(instance.OffsetExistsBitmap);
                writer.Write(instance.OffsetVertexData);
            }
        }

        // Stage 4: write headers
        ms.Position = 0;
        for (int i = 0; i < HeaderCount; i++)
        {
            var info = infos[i];
            writer.Write(info.OffsetInstances);
            writer.Write((uint)info.InstanceCount);
            writer.Write(info.OffsetAttributes);
        }

        return ms.ToArray();
    }

    private static HeaderSerializationInfo PrepareHeader(Mh2oChunk? chunk)
    {
        if (chunk is null || chunk.IsEmpty)
            return HeaderSerializationInfo.Empty;

        var info = new HeaderSerializationInfo(chunk.Instances.Count);

        if (chunk.Attributes is not null)
            info.AttributeBytes = BuildAttributes(chunk.Attributes);

        for (int i = 0; i < chunk.Instances.Count; i++)
        {
            var src = chunk.Instances[i];
            var instData = new InstanceSerializationData
            {
                LiquidTypeId = src.LiquidTypeId,
                VertexFormat = src.Lvf,
                MinHeightLevel = src.MinHeightLevel,
                MaxHeightLevel = src.MaxHeightLevel,
                XOffset = src.XOffset,
                YOffset = src.YOffset,
                Width = src.Width,
                Height = src.Height,
                ExistsBitmap = src.ExistsBitmap,
                VertexBytes = BuildVertexBytes(src)
            };
            info.Instances[i] = instData;
        }

        return info;
    }

    private static byte[]? BuildAttributes(Mh2oAttributes attrs)
    {
        Span<byte> buffer = stackalloc byte[16];
        for (int y = 0; y < 8; y++)
        {
            byte fishRow = 0;
            byte deepRow = 0;
            for (int x = 0; x < 8; x++)
            {
                int bitIndex = y * 8 + x;
                if (((attrs.FishableMask >> bitIndex) & 1UL) != 0)
                    fishRow |= (byte)(1 << x);
                if (((attrs.DeepMask >> bitIndex) & 1UL) != 0)
                    deepRow |= (byte)(1 << x);
            }
            buffer[y] = fishRow;
            buffer[8 + y] = deepRow;
        }

        return buffer.ToArray();
    }

    private static byte[] BuildVertexBytes(Mh2oInstance instance)
    {
        int vertexCount = instance.VertexCount;
        switch (instance.Lvf)
        {
            case LiquidVertexFormat.HeightDepth:
            {
                if (instance.HeightMap is null || instance.DepthMap is null)
                    throw new InvalidOperationException("HeightDepth format requires height and depth maps.");

                using var ms = new MemoryStream(vertexCount * (sizeof(float) + sizeof(byte)));
                using var writer = new BinaryWriter(ms);
                for (int i = 0; i < vertexCount; i++)
                    writer.Write(instance.HeightMap[i]);
                for (int i = 0; i < vertexCount; i++)
                    writer.Write(instance.DepthMap[i]);
                return ms.ToArray();
            }
            case LiquidVertexFormat.DepthOnly:
            {
                if (instance.DepthMap is null)
                    throw new InvalidOperationException("DepthOnly format requires depth map.");
                return (byte[])instance.DepthMap.Clone();
            }
            default:
                throw new NotSupportedException($"Liquid vertex format '{instance.Lvf}' is not supported.");
        }
    }

    private sealed class HeaderSerializationInfo
    {
        public static readonly HeaderSerializationInfo Empty = new(0);

        public HeaderSerializationInfo(int instanceCount)
        {
            InstanceCount = instanceCount;
            if (instanceCount > 0)
                Instances = new InstanceSerializationData[instanceCount];
            else
                Instances = Array.Empty<InstanceSerializationData>();
        }

        public int InstanceCount { get; }
        public InstanceSerializationData[] Instances { get; }
        public byte[]? AttributeBytes { get; set; }
        public uint OffsetInstances { get; set; }
        public uint OffsetAttributes { get; set; }
    }

    private sealed class InstanceSerializationData
    {
        public ushort LiquidTypeId { get; set; }
        public LiquidVertexFormat VertexFormat { get; set; }
        public float MinHeightLevel { get; set; }
        public float MaxHeightLevel { get; set; }
        public byte XOffset { get; set; }
        public byte YOffset { get; set; }
        public byte Width { get; set; }
        public byte Height { get; set; }
        public byte[]? ExistsBitmap { get; set; }
        public byte[] VertexBytes { get; set; } = Array.Empty<byte>();
        public uint OffsetExistsBitmap { get; set; }
        public uint OffsetVertexData { get; set; }
    }
}
