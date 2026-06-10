using System;
using System.IO;
using System.CommandLine;
using System.Threading.Tasks;

namespace WCAnalyzer.TestParser
{
    public class Program
    {
        public static int Main(string[] args)
        {
            // Create command line options
            var rootCommand = new RootCommand("WCAnalyzer Test Parser for PM4/PD4 Files");
            
            var fileOption = new Option<FileInfo>(
                name: "--file",
                description: "The PM4/PD4 file to parse"
            ) { IsRequired = true };
            fileOption.AddAlias("-f");
            rootCommand.AddOption(fileOption);
            
            // Set handler
            rootCommand.SetHandler((FileInfo file) => {
                if (!file.Exists)
                {
                    Console.WriteLine($"Error: File '{file.FullName}' not found.");
                    return;
                }
                
                SimpleTestParser.ParseFile(file.FullName);
            }, fileOption);
            
            // Parse and execute
            return rootCommand.Invoke(args);
        }
    }
} 