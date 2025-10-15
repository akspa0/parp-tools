namespace WoWRollback.LkToAlphaModule.Writers;

public interface IAdtWriter
{
    void WriteAlphaAdt(Models.AlphaAdtData data, string outFile);
}
