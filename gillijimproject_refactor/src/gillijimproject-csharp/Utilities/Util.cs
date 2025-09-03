using System;

namespace GillijimProject.Utilities
{
    public static class Util
    {
        public static int FindChunkOffset(byte[] fileData, string chunkName, int startOffset)
        {
            byte[] chunkBytes = System.Text.Encoding.ASCII.GetBytes(chunkName);
            Array.Reverse(chunkBytes);

            for (int i = startOffset; i <= fileData.Length - 4; i++)
            {
                if (fileData[i] == chunkBytes[0] && fileData[i + 1] == chunkBytes[1] && fileData[i + 2] == chunkBytes[2] && fileData[i + 3] == chunkBytes[3])
                {
                    return i;
                }
            }
            return -1; // Not found
        }
    }
}
