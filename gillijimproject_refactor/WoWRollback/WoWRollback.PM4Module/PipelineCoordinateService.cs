using System.Numerics;

namespace WoWRollback.PM4Module
{
    public static class PipelineCoordinateService
    {
        private const float MapExtent = 32f * 533.33333f; // 17066.66656

        /// <summary>
        /// Converts Server/World coordinates (X, Y, Z) to ADT Placement coordinates (X, Y, Z).
        /// Note: The mapping appears to be:
        /// Placement.X = MapExtent - World.Y
        /// Placement.Y = World.Z
        /// Placement.Z = MapExtent - World.X
        /// </summary>
        public static Vector3 ServerToAdtPosition(Vector3 serverPos)
        {
            float placementX = MapExtent - serverPos.Y;
            float placementY = serverPos.Z;
            float placementZ = MapExtent - serverPos.X;

            return new Vector3(placementX, placementY, placementZ);
        }
    }
}
