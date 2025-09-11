using System;
using System.Runtime.InteropServices;

namespace DBCTool.Mpq
{
    internal static class StormLib
    {
        private const string DllName = "StormLib.dll";

        [DllImport(DllName, CharSet = CharSet.Unicode, SetLastError = true)]
        public static extern bool SFileOpenArchive(string szArchiveName, uint dwPriority, uint dwFlags, out IntPtr phArchive);

        [DllImport(DllName, CharSet = CharSet.Unicode, SetLastError = true)]
        public static extern bool SFileOpenPatchArchive(IntPtr hBaseArchive, string szPatchMpqName, string szPatchPathPrefix, uint dwFlags);

        [DllImport(DllName, SetLastError = true)]
        public static extern bool SFileCloseArchive(IntPtr hArchive);

        [DllImport(DllName, CharSet = CharSet.Unicode, SetLastError = true)]
        public static extern bool SFileOpenFileEx(IntPtr hMpq, string szFileName, uint dwSearchScope, out IntPtr phFile);

        [DllImport(DllName, SetLastError = true)]
        public static extern uint SFileGetFileSize(IntPtr hFile, out uint pdwFileSizeHigh);

        [DllImport(DllName, SetLastError = true)]
        public static extern bool SFileReadFile(IntPtr hFile, IntPtr lpBuffer, uint dwToRead, out uint pdwRead, IntPtr lpOverlapped);

        [DllImport(DllName, SetLastError = true)]
        public static extern bool SFileCloseFile(IntPtr hFile);

        [DllImport(DllName, CharSet = CharSet.Unicode, SetLastError = true)]
        public static extern bool SFileHasFile(IntPtr hMpq, string szFileName);

        // Common flags
        public const uint SFILE_OPEN_HARD_DISK_FILE = 0x00000000;
        public const uint SFILE_OPEN_FROM_MPQ = 0x00000000;
        public const uint SFILE_OPEN_PATCHED_FILE = 0x00000001; // For SFileOpenFileEx search scope
    }
}
