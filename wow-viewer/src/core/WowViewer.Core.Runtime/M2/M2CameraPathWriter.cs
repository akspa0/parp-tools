using System.Buffers.Binary;
using System.Numerics;
using System.Text;

namespace WowViewer.Core.Runtime.M2;

public static class M2CameraPathWriter
{
    private const uint Version = 0x108u;
    private const int HeaderSize = 0x130;
    private const int SequenceStride = 0x40;
    private const int CameraStride = 0x64;
    private const int TrackStride = 0x14;

    public static void Write(string outputPath, M2CameraPathDocument path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);
        ArgumentNullException.ThrowIfNull(path);
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outputPath)) ?? ".");
        File.WriteAllBytes(outputPath, Build(path));
    }

    public static byte[] Build(M2CameraPathDocument path)
    {
        ArgumentNullException.ThrowIfNull(path);
        M2CameraPathEvaluator.NormalizeAndValidate(path);
        if (path.Keyframes.Count == 0)
            throw new InvalidDataException("Cannot export an empty camera path to M2.");

        string modelName = string.IsNullOrWhiteSpace(path.Name) ? "WoWViewerCamera" : path.Name;
        byte[] modelNameBytes = Encoding.UTF8.GetBytes(modelName + "\0");
        int cursor = HeaderSize;
        int nameOffset = cursor;
        cursor += modelNameBytes.Length;
        int sequenceOffset = Align(cursor, 0x10);
        cursor = sequenceOffset + SequenceStride;
        int sequenceLookupOffset = Align(cursor, 2);
        cursor = sequenceLookupOffset + 2;
        int cameraOffset = Align(cursor, 0x10);
        cursor = cameraOffset + CameraStride;

        int positionTrackOffset = Align(cursor, 4);
        cursor = positionTrackOffset + TrackStride;
        int targetTrackOffset = Align(cursor, 4);
        cursor = targetTrackOffset + TrackStride;
        int rollTrackOffset = Align(cursor, 4);
        cursor = rollTrackOffset + TrackStride;

        int positionReferencesOffset = Align(cursor, 4);
        cursor = positionReferencesOffset + 16;
        int targetReferencesOffset = Align(cursor, 4);
        cursor = targetReferencesOffset + 16;
        int rollReferencesOffset = Align(cursor, 4);
        cursor = rollReferencesOffset + 16;

        int keyCount = path.Keyframes.Count;
        int positionTimesOffset = Align(cursor, 4);
        cursor = positionTimesOffset + keyCount * sizeof(uint);
        int targetTimesOffset = Align(cursor, 4);
        cursor = targetTimesOffset + keyCount * sizeof(uint);
        int rollTimesOffset = Align(cursor, 4);
        cursor = rollTimesOffset + keyCount * sizeof(uint);
        int positionValuesOffset = Align(cursor, 4);
        cursor = positionValuesOffset + keyCount * sizeof(float) * 3;
        int targetValuesOffset = Align(cursor, 4);
        cursor = targetValuesOffset + keyCount * sizeof(float) * 3;
        int rollValuesOffset = Align(cursor, 4);
        cursor = rollValuesOffset + keyCount * sizeof(float);

        byte[] data = new byte[cursor];
        Encoding.ASCII.GetBytes("MD20").CopyTo(data, 0);
        WriteUInt32(data, 0x04, Version);
        WriteUInt32(data, 0x08, (uint)modelNameBytes.Length);
        WriteUInt32(data, 0x0C, (uint)nameOffset);
        WriteUInt32(data, 0x1C, 1u);
        WriteUInt32(data, 0x20, (uint)sequenceOffset);
        WriteUInt32(data, 0x24, 1u);
        WriteUInt32(data, 0x28, (uint)sequenceLookupOffset);
        WriteVector3(data, 0xA0, ComputeBounds(path, true));
        WriteVector3(data, 0xAC, ComputeBounds(path, false));
        WriteSingle(data, 0xB8, ComputeRadius(path));
        WriteUInt32(data, 0x110, 1u);
        WriteUInt32(data, 0x114, (uint)cameraOffset);
        modelNameBytes.CopyTo(data, nameOffset);
        WriteSequence(data, sequenceOffset, path.DurationMs, ComputeBounds(path, true), ComputeBounds(path, false), ComputeRadius(path));
        WriteInt16(data, sequenceLookupOffset, 0);

        WriteUInt32(data, cameraOffset + 0x00, unchecked((uint)-1));
        WriteSingle(data, cameraOffset + 0x04, MathF.Max(0.01f, path.Keyframes[0].FovDegrees * MathF.PI / 180f));
        WriteSingle(data, cameraOffset + 0x08, 20000f);
        WriteSingle(data, cameraOffset + 0x0C, 0.1f);
        WriteTrackHeader(data, cameraOffset + 0x10, positionReferencesOffset);
        WriteVector3(data, cameraOffset + 0x24, Vector3.Zero);
        WriteTrackHeader(data, cameraOffset + 0x30, targetReferencesOffset);
        WriteVector3(data, cameraOffset + 0x44, Vector3.Zero);
        WriteTrackHeader(data, cameraOffset + 0x50, rollReferencesOffset);

        WriteTrackReferences(data, positionReferencesOffset, keyCount, positionTimesOffset, positionValuesOffset);
        WriteTrackReferences(data, targetReferencesOffset, keyCount, targetTimesOffset, targetValuesOffset);
        WriteTrackReferences(data, rollReferencesOffset, keyCount, rollTimesOffset, rollValuesOffset);

        for (int index = 0; index < keyCount; index++)
        {
            M2CameraPathKeyframe key = path.Keyframes[index];
            WriteUInt32(data, positionTimesOffset + index * 4, (uint)Math.Max(0, key.TimeMs));
            WriteUInt32(data, targetTimesOffset + index * 4, (uint)Math.Max(0, key.TimeMs));
            WriteUInt32(data, rollTimesOffset + index * 4, (uint)Math.Max(0, key.TimeMs));
            WriteVector3(data, positionValuesOffset + index * 12, key.Position);
            WriteVector3(data, targetValuesOffset + index * 12, key.Target);
            WriteSingle(data, rollValuesOffset + index * 4, key.RollDegrees * MathF.PI / 180f);
        }

        return data;
    }

    private static void WriteSequence(byte[] data, int offset, int durationMs, Vector3 boundsMin, Vector3 boundsMax, float radius)
    {
        WriteUInt16(data, offset + 0x00, 0);
        WriteUInt16(data, offset + 0x02, 0);
        WriteUInt32(data, offset + 0x04, (uint)Math.Clamp(durationMs + 1, 1, int.MaxValue));
        WriteSingle(data, offset + 0x08, 0f);
        WriteUInt32(data, offset + 0x0C, 0u);
        WriteInt16(data, offset + 0x10, 0);
        WriteUInt32(data, offset + 0x14, 0u);
        WriteUInt32(data, offset + 0x18, (uint)Math.Clamp(durationMs + 1, 1, int.MaxValue));
        WriteVector3(data, offset + 0x20, boundsMin);
        WriteVector3(data, offset + 0x2C, boundsMax);
        WriteSingle(data, offset + 0x38, radius);
        WriteInt16(data, offset + 0x3C, -1);
        WriteUInt16(data, offset + 0x3E, ushort.MaxValue);
    }

    private static void WriteTrackHeader(byte[] data, int offset, int referencesOffset)
    {
        WriteUInt16(data, offset + 0x00, 1);
        WriteUInt16(data, offset + 0x02, ushort.MaxValue);
        WriteUInt32(data, offset + 0x04, 1u);
        WriteUInt32(data, offset + 0x08, (uint)referencesOffset);
        WriteUInt32(data, offset + 0x0C, 1u);
        WriteUInt32(data, offset + 0x10, (uint)referencesOffset + 8u);
    }

    private static void WriteTrackReferences(byte[] data, int offset, int count, int timesOffset, int valuesOffset)
    {
        WriteUInt32(data, offset, (uint)count);
        WriteUInt32(data, offset + 4, (uint)timesOffset);
        WriteUInt32(data, offset + 8, (uint)count);
        WriteUInt32(data, offset + 12, (uint)valuesOffset);
    }

    private static Vector3 ComputeBounds(M2CameraPathDocument path, bool min)
    {
        Vector3 value = min ? new Vector3(float.PositiveInfinity) : new Vector3(float.NegativeInfinity);
        foreach (M2CameraPathKeyframe key in path.Keyframes)
        {
            value = min ? Vector3.Min(value, key.Position) : Vector3.Max(value, key.Position);
            value = min ? Vector3.Min(value, key.Target) : Vector3.Max(value, key.Target);
        }
        return value;
    }

    private static float ComputeRadius(M2CameraPathDocument path)
    {
        Vector3 min = ComputeBounds(path, true);
        Vector3 max = ComputeBounds(path, false);
        return MathF.Max(1f, Vector3.Distance(min, max) * 0.5f);
    }

    private static int Align(int value, int alignment)
    {
        int remainder = value % alignment;
        return remainder == 0 ? value : value + alignment - remainder;
    }

    private static void WriteUInt32(byte[] data, int offset, uint value) => BinaryPrimitives.WriteUInt32LittleEndian(data.AsSpan(offset, 4), value);
    private static void WriteUInt16(byte[] data, int offset, ushort value) => BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan(offset, 2), value);
    private static void WriteInt16(byte[] data, int offset, short value) => BinaryPrimitives.WriteInt16LittleEndian(data.AsSpan(offset, 2), value);
    private static void WriteSingle(byte[] data, int offset, float value) => WriteUInt32(data, offset, unchecked((uint)BitConverter.SingleToInt32Bits(value)));

    private static void WriteVector3(byte[] data, int offset, Vector3 value)
    {
        WriteSingle(data, offset + 0x00, value.X);
        WriteSingle(data, offset + 0x04, value.Y);
        WriteSingle(data, offset + 0x08, value.Z);
    }
}
