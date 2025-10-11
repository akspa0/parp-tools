using System.IO;
using System.Text;

namespace WoWToolbox.Core.v2.Foundation.Helpers
{
    /// <summary>
    /// Helper utilities for reading and writing null-terminated strings in binary data.
    /// These are minimal shims added to unblock compilation of chunk classes that
    /// were originally referencing lingering helper methods from prior code.
    /// </summary>
    public static class StringReadHelper
    {
        public static string ReadNullTerminatedString(BinaryReader reader, Encoding encoding)
        {
            var bytes = new List<byte>();
            byte b;
            while ((b = reader.ReadByte()) != 0)
                bytes.Add(b);
            return encoding.GetString(bytes.ToArray());
        }
    }

    public static class StringWriteHelper
    {
        public static void WriteNullTerminatedString(BinaryWriter writer, string value, Encoding encoding)
        {
            var bytes = encoding.GetBytes(value);
            writer.Write(bytes);
            writer.Write((byte)0); // null terminator
        }
    }
}
