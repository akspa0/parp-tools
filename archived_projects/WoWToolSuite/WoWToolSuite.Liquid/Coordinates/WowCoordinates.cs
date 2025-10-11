using System;
using System.Numerics;

namespace WowToolSuite.Liquid.Coordinates
{
    public class ChunkCoordinate
    {
        public int X { get; }
        public int Y { get; }

        public ChunkCoordinate(int x, int y)
        {
            X = x;
            Y = y;
        }
    }

    public class WowCoordinates
    {
        public const float ADT_SIZE = 533.33333f;
        public const int CHUNKS_PER_ADT = 16;
        public const float CHUNK_SIZE = ADT_SIZE / CHUNKS_PER_ADT;
        public const int TILES_PER_MAP_SIDE = 64;
        public const int CENTER_ADT = 32; // World origin is at ADT (32,32)
        public const float WORLD_HALF_SIZE = (TILES_PER_MAP_SIDE / 2) * ADT_SIZE; // ~17066.66656 yards

        public float X { get; }
        public float Y { get; }
        public float Z { get; }

        public AdtCoordinate AdtCoordinates { get; }
        public ChunkCoordinate ChunkCoordinates { get; }
        public Vector2 ChunkLocalPosition { get; }

        public WowCoordinates(float x, float y, float z)
        {
            X = x;
            Y = y;
            Z = z;

            // Convert world coordinates to ADT grid coordinates
            float adtX = (x + WORLD_HALF_SIZE) / ADT_SIZE;
            float adtY = (y + WORLD_HALF_SIZE) / ADT_SIZE;

            // Get ADT coordinates (0-63)
            int adtGridX = (int)Math.Floor(adtX);
            int adtGridY = (int)Math.Floor(adtY);

            AdtCoordinates = new AdtCoordinate(adtGridX, adtGridY);

            // Calculate chunk coordinates within ADT (0-15)
            float localX = (x + WORLD_HALF_SIZE) % ADT_SIZE;
            float localY = (y + WORLD_HALF_SIZE) % ADT_SIZE;

            int chunkX = (int)(localX / CHUNK_SIZE);
            int chunkY = (int)(localY / CHUNK_SIZE);

            ChunkCoordinates = new ChunkCoordinate(chunkX, chunkY);

            // Calculate local position within chunk (0-1)
            float chunkLocalX = (localX % CHUNK_SIZE) / CHUNK_SIZE;
            float chunkLocalY = (localY % CHUNK_SIZE) / CHUNK_SIZE;

            ChunkLocalPosition = new Vector2(chunkLocalX, chunkLocalY);
        }

        public bool IsInValidMapBounds()
        {
            // Check if the ADT coordinates are within the 64x64 grid
            return AdtCoordinates.X >= 0 && AdtCoordinates.X < TILES_PER_MAP_SIDE &&
                   AdtCoordinates.Y >= 0 && AdtCoordinates.Y < TILES_PER_MAP_SIDE;
        }

        public string GetAdtFileName()
        {
            // Return the ADT filename in WoW's format (development_X_Y.adt)
            return $"development_{AdtCoordinates.X}_{AdtCoordinates.Y}.adt";
        }

        public override string ToString()
        {
            return $"Global: ({X:F2}, {Y:F2}, {Z:F2})\n" +
                   $"ADT: ({AdtCoordinates.X}, {AdtCoordinates.Y})\n" +
                   $"Chunk: ({ChunkCoordinates.X}, {ChunkCoordinates.Y})\n" +
                   $"ChunkLocal: ({ChunkLocalPosition.X:F2}, {ChunkLocalPosition.Y:F2})";
        }

        public static Vector3 LocalToGlobal(Vector3 localPosition, int adtX, int adtY)
        {
            // Convert ADT-local coordinates back to world coordinates
            return new Vector3(
                localPosition.X + ((adtX - CENTER_ADT) * ADT_SIZE),
                localPosition.Y + ((adtY - CENTER_ADT) * ADT_SIZE),
                localPosition.Z
            );
        }

        public static Vector3 GlobalToLocal(Vector3 globalPosition, int adtX, int adtY)
        {
            // Convert world coordinates to ADT-local coordinates
            return new Vector3(
                globalPosition.X - ((adtX - CENTER_ADT) * ADT_SIZE),
                globalPosition.Y - ((adtY - CENTER_ADT) * ADT_SIZE),
                globalPosition.Z
            );
        }
    }
} 