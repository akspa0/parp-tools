using System;
using System.Collections.Generic;
using System.IO;
using GillijimProject.WowFiles;
using WoWRollback.LkToAlphaModule.Models;

namespace WoWRollback.LkToAlphaModule.Writers;

public static class McseWriterLk
{
    public static Chunk BuildMcseChunk(IReadOnlyList<AlphaMcseEntry> entries, bool prefer76Byte = true)
    {
        if (entries == null || entries.Count == 0)
            return new Chunk("MCSE", 0, Array.Empty<byte>());

        if (prefer76Byte)
        {
            var data = new byte[entries.Count * 76];
            int pos = 0;
            foreach (var e in entries)
            {
                WriteUInt32(data, ref pos, e.SoundPointId);
                WriteUInt32(data, ref pos, e.SoundNameId);
                WriteSingle(data, ref pos, e.PosX);
                WriteSingle(data, ref pos, e.PosY);
                WriteSingle(data, ref pos, e.PosZ);
                WriteSingle(data, ref pos, e.MinDistance);
                WriteSingle(data, ref pos, e.MaxDistance);
                WriteSingle(data, ref pos, e.CutoffDistance);
                WriteUInt32(data, ref pos, e.StartTime);
                WriteUInt32(data, ref pos, e.EndTime);
                WriteUInt32(data, ref pos, e.Mode);
                WriteUInt32(data, ref pos, e.GroupSilenceMin);
                WriteUInt32(data, ref pos, e.GroupSilenceMax);
                WriteUInt32(data, ref pos, e.PlayInstancesMin);
                WriteUInt32(data, ref pos, e.PlayInstancesMax);
                WriteUInt32(data, ref pos, e.LoopCountMin);
                WriteUInt32(data, ref pos, e.LoopCountMax);
                WriteUInt32(data, ref pos, e.InterSoundGapMin);
                WriteUInt32(data, ref pos, e.InterSoundGapMax);
            }
            return new Chunk("MCSE", data.Length, data);
        }
        else
        {
            var data = new byte[entries.Count * 52];
            int pos = 0;
            foreach (var e in entries)
            {
                WriteUInt32(data, ref pos, e.SoundPointId);
                WriteUInt32(data, ref pos, e.SoundNameId);
                WriteSingle(data, ref pos, e.PosX);
                WriteSingle(data, ref pos, e.PosY);
                WriteSingle(data, ref pos, e.PosZ);
                WriteSingle(data, ref pos, e.MinDistance);
                WriteSingle(data, ref pos, e.MaxDistance);
                WriteSingle(data, ref pos, e.CutoffDistance);
                WriteUInt16(data, ref pos, (ushort)Math.Min(ushort.MaxValue, e.StartTime));
                WriteUInt16(data, ref pos, (ushort)Math.Min(ushort.MaxValue, e.EndTime));
            }
            return new Chunk("MCSE", data.Length, data);
        }
    }

    private static void WriteUInt32(byte[] buf, ref int pos, uint v)
    {
        BitConverter.GetBytes(v).CopyTo(buf, pos);
        pos += 4;
    }
    private static void WriteUInt16(byte[] buf, ref int pos, ushort v)
    {
        BitConverter.GetBytes(v).CopyTo(buf, pos);
        pos += 2;
    }
    private static void WriteSingle(byte[] buf, ref int pos, float v)
    {
        BitConverter.GetBytes(v).CopyTo(buf, pos);
        pos += 4;
    }
}
