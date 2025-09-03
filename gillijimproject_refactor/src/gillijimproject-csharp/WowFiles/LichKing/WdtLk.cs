using System.IO;

namespace GillijimProject.WowFiles.LichKing
{
    public class WdtLk
    {
        private readonly string _wdtName;
        private readonly Chunk _cMver;
        private readonly Mphd _cMphd;
        private readonly Main _cMain;
        private readonly Chunk _cMwmo;
        private readonly Chunk _cModf;

        public WdtLk(string wdtName, Chunk cMver, Mphd cMphd, Main cMain, Chunk cMwmo, Chunk cModf)
        {
            _wdtName = wdtName;
            _cMver = cMver;
            _cMphd = cMphd;
            _cMain = cMain;
            _cMwmo = cMwmo;
            _cModf = cModf;
        }

        public void Write(string path)
        {
            using var fs = new FileStream(path, FileMode.Create);
            fs.Write(_cMver.GetWholeChunk());
            fs.Write(_cMphd.GetWholeChunk());
            fs.Write(_cMain.GetWholeChunk());
            fs.Write(_cMwmo.GetWholeChunk());
            fs.Write(_cModf.GetWholeChunk());
        }
    }
}
