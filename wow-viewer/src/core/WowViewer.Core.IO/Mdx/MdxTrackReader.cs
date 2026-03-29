using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.Mdx;

namespace WowViewer.Core.IO.Mdx;

internal static class MdxTrackReader
{
    public static MdxVector3NodeTrack ReadVector3Track(Stream stream, long limit, string tag, string contextLabel, string overrunMessage)
    {
        uint keyCount = ReadUInt32(stream);
        if (keyCount > 100000)
            throw new InvalidDataException($"{contextLabel}: invalid {tag} key count {keyCount}.");

        uint interpolationType = ReadUInt32(stream);
        int globalSequenceId = ReadInt32(stream);
        List<MdxVector3Keyframe> keys = new(checked((int)keyCount));

        for (uint keyIndex = 0; keyIndex < keyCount; keyIndex++)
        {
            int time = ReadInt32(stream);
            Vector3 value = ReadVector3(stream);
            Vector3? inTangent = null;
            Vector3? outTangent = null;
            if (TrackUsesTangents(interpolationType))
            {
                inTangent = ReadVector3(stream);
                outTangent = ReadVector3(stream);
            }

            keys.Add(new MdxVector3Keyframe(time, value, inTangent, outTangent));
        }

        if (stream.Position > limit)
            throw new InvalidDataException(overrunMessage);

        return new MdxVector3NodeTrack(tag, (MdxTrackInterpolationType)interpolationType, globalSequenceId, keys);
    }

    public static MdxQuaternionNodeTrack ReadQuaternionTrack(Stream stream, long limit, string tag, string contextLabel, string overrunMessage)
    {
        uint keyCount = ReadUInt32(stream);
        if (keyCount > 100000)
            throw new InvalidDataException($"{contextLabel}: invalid {tag} key count {keyCount}.");

        uint interpolationType = ReadUInt32(stream);
        int globalSequenceId = ReadInt32(stream);
        List<MdxQuaternionKeyframe> keys = new(checked((int)keyCount));

        for (uint keyIndex = 0; keyIndex < keyCount; keyIndex++)
        {
            int time = ReadInt32(stream);
            Quaternion value = ReadCompressedQuaternion(stream);
            Quaternion? inTangent = null;
            Quaternion? outTangent = null;
            if (TrackUsesTangents(interpolationType))
            {
                inTangent = ReadCompressedQuaternion(stream);
                outTangent = ReadCompressedQuaternion(stream);
            }

            keys.Add(new MdxQuaternionKeyframe(time, value, inTangent, outTangent));
        }

        if (stream.Position > limit)
            throw new InvalidDataException(overrunMessage);

        return new MdxQuaternionNodeTrack(tag, (MdxTrackInterpolationType)interpolationType, globalSequenceId, keys);
    }

    public static Quaternion ReadCompressedQuaternion(Stream stream)
    {
        uint data0 = ReadUInt32(stream);
        uint data1 = ReadUInt32(stream);

        int xq = ((int)data1) >> 10;
        int yq = ((int)((data1 << 22) | (data0 >> 10))) >> 11;
        int zq = ((int)(data0 << 11)) >> 11;

        const float scaleX = 1.0f / (1 << 21);
        const float scaleYZ = 1.0f / (1 << 20);

        float x = xq * scaleX;
        float y = yq * scaleYZ;
        float z = zq * scaleYZ;
        float s = x * x + y * y + z * z;
        float w = MathF.Abs(s - 1.0f) < scaleYZ ? 0.0f : MathF.Sqrt(MathF.Max(0.0f, 1.0f - s));
        return new Quaternion(x, y, z, w);
    }

    private static bool TrackUsesTangents(uint interpolationType)
    {
        return interpolationType >= 2u;
    }

    private static uint ReadUInt32(Stream stream)
    {
        Span<byte> bytes = stackalloc byte[sizeof(uint)];
        stream.ReadExactly(bytes);
        return BinaryPrimitives.ReadUInt32LittleEndian(bytes);
    }

    private static int ReadInt32(Stream stream)
    {
        return unchecked((int)ReadUInt32(stream));
    }

    private static float ReadSingle(Stream stream)
    {
        Span<byte> bytes = stackalloc byte[sizeof(float)];
        stream.ReadExactly(bytes);
        return BinaryPrimitives.ReadSingleLittleEndian(bytes);
    }

    private static Vector3 ReadVector3(Stream stream)
    {
        return new Vector3(ReadSingle(stream), ReadSingle(stream), ReadSingle(stream));
    }
}