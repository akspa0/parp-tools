using System;
using System.Threading.Tasks;

namespace ChunkTester
{
    class Program
    {
        static async Task<int> Main(string[] args)
        {
            Console.WriteLine("PM4/PD4 Chunk Tester");
            Console.WriteLine("====================");
            
            ChunkTesterSimple.Run(args);
            
            return 0;
        }
    }
} 