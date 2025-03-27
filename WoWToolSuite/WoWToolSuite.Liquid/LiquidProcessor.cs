using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using WowToolSuite.Liquid.Models;

namespace WowToolSuite.Liquid
{
    public class LiquidProcessor
    {
        /// <summary>
        /// Processes a WLW liquid file and extracts liquid blocks
        /// </summary>
        /// <param name="filePath">Path to the WLW file</param>
        /// <returns>List of parsed liquid blocks</returns>
        public List<LiquidBlock> ProcessLiquidFile(string filePath)
        {
            // This is a simplified version - you might need to adapt this
            // to match your actual WLW parsing implementation
            try
            {
                var liquidFile = LiquidParser.ParseWlwOrWlmFile(filePath, false, false);
                if (liquidFile == null)
                {
                    Console.WriteLine($"Failed to parse {filePath}");
                    return new List<LiquidBlock>();
                }

                // Set source file for each block
                foreach (var block in liquidFile.Blocks)
                {
                    block.SourceFile = filePath;
                }

                return liquidFile.Blocks;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error processing file {filePath}: {ex.Message}");
                return new List<LiquidBlock>();
            }
        }
    }
} 