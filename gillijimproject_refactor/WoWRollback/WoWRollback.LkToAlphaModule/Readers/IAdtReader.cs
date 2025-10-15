namespace WoWRollback.LkToAlphaModule.Readers;

public interface IAdtReader
{
    // Reads LK root/obj/tex ADT triple and returns an in-memory model
    Models.LkAdtData Read(string rootAdtPath, string objAdtPath, string texAdtPath);
}
