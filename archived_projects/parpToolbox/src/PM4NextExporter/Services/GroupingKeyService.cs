namespace PM4NextExporter.Services
{
    public static class GroupingKeyService
    {
        public static string ParentIndexKey(uint parentIndex) => parentIndex.ToString();
        public static string MsurIndexCountKey(int indexCount) => indexCount.ToString();

        public static (ushort ContainerId, ushort ObjectId) Parent16(uint parentId, bool swap = false)
        {
            var container = ParentIdDecoder.GetContainerId(parentId);
            var obj = ParentIdDecoder.GetObjectId(parentId);
            return swap ? (obj, container) : (container, obj);
        }
    }
}
