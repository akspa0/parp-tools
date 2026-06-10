namespace PM4NextExporter.Services
{
    /// <summary>
    /// Helper to decode a packed 32-bit ParentId into 16-bit ContainerId and 16-bit ObjectId.
    /// Default mapping: high 16 bits = ContainerId, low 16 bits = ObjectId.
    /// </summary>
    public static class ParentIdDecoder
    {
        /// <summary>
        /// Extract the 16-bit ContainerId from a 32-bit ParentId (high 16 bits).
        /// </summary>
        public static ushort GetContainerId(uint parentId)
        {
            return (ushort)((parentId >> 16) & 0xFFFF);
        }

        /// <summary>
        /// Extract the 16-bit ObjectId from a 32-bit ParentId (low 16 bits).
        /// </summary>
        public static ushort GetObjectId(uint parentId)
        {
            return (ushort)(parentId & 0xFFFF);
        }

        /// <summary>
        /// Compose a 32-bit ParentId from 16-bit ContainerId (high) and 16-bit ObjectId (low).
        /// </summary>
        public static uint Compose(ushort containerId, ushort objectId)
        {
            return ((uint)containerId << 16) | (uint)objectId;
        }

        /// <summary>
        /// Decode both ids in one call.
        /// </summary>
        public static (ushort ContainerId, ushort ObjectId) Decode(uint parentId)
        {
            return (GetContainerId(parentId), GetObjectId(parentId));
        }
    }
}
