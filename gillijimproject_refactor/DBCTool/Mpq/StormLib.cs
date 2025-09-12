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
        public static extern bool SFileOpenPatchArchive(IntPtr hBaseArchive, string szPatchMpqName, string? szPatchPathPrefix, uint dwFlags);

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

        // File enumeration
        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        public struct SFILE_FIND_DATA
        {
            // Note: StormLib uses TCHAR arrays. Use generous buffer sizes for Unicode.
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 1024)]
            public string cFileName;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 1024)]
            public string szPlainName;
            public uint dwHashIndex;
            public uint dwBlockIndex;
            public uint dwFileSize;
            public uint dwCompressedSize;
            public uint dwFlags;
            public uint dwFileTimeLow;
            public uint dwFileTimeHigh;
            public uint lcLocale;
            public uint wPlatform;
        }

        [DllImport(DllName, CharSet = CharSet.Unicode, SetLastError = true)]
        public static extern IntPtr SFileFindFirstFile(IntPtr hMpq, string szMask, out SFILE_FIND_DATA lpFindFile, string? szListFile);

        [DllImport(DllName, CharSet = CharSet.Unicode, SetLastError = true)]
        public static extern bool SFileFindNextFile(IntPtr hFind, out SFILE_FIND_DATA lpFindFile);

        [DllImport(DllName, SetLastError = true)]
        public static extern bool SFileFindClose(IntPtr hFind);

        // Common flags for SFileOpenFileEx search scope
        public const uint SFILE_OPEN_FROM_MPQ = 0x00000000;
        public const uint SFILE_OPEN_CHECK_EXISTS = 0x00000001;
        public const uint SFILE_OPEN_PATCHED_FILE = 0x00000002;

        // Back-compat names used in code (BASE_FILE isn't a StormLib scope; map it to CHECK_EXISTS semantics)
        public const uint SFILE_OPEN_BASE_FILE = SFILE_OPEN_CHECK_EXISTS;
        public const uint SFILE_OPEN_HARD_DISK_FILE = SFILE_OPEN_FROM_MPQ;
    }
}
