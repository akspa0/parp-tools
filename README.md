# parp-tools
Tools for analyzing World of Warcraft files

gillijimproject_refactor - A C# port of mjollna/gillijimproject with many enhancements

 - WoWRollback - A complete toolbox for parsing Alpha World of Warcraft map files into 3.3.5-era ADT files, with a viewer tool for exploring data

 - gillijimproject-csharp - The pure c# port. Acts as a library to convert AlphaWDT's into LK ADTs (with improper AreaIDs)

 - AlphaWDTAnalysisTool - The proof of concept patch tool for AreaID patching. Rolled in to WoWRollback as a library.

PM4Tool - PM4 parsing tools

parpToolbox - Additional PM4 parsing tools
 - PM4FacesTool - current experimental toolkit for parsing PM4 files. Will be rolled into WoWRollback.

----

# Archived projects

parpToolbox/src/PM4Rebuilder - A complete tool for parsing PM4 files into exportable objects. Includes lots of analysis tooling too.

parpToolbox - Base library for PM4/PD4/WMO reading, with tons of tests and analysis tooling for understanding the PM4 file format.

PM4Tool/WoWToolbox - a PM4/PD4 file decoder with support for reading from ADT and WMO assets for identifying data in the PM4 or PD4 file formats. Currently a working proof of concept. Supports leaked development pm4 files and the cataclysm-era split ADT files for analysis. We use WotLK 3.3.5 assets (wmo/m2) for mesh comparison.

ProjectArcane - Extemely work-in-progress re-write, based on the wowdev wiki documentation

WCA\WCAnalyzer - A multi-tool built off ModernWoWTools/Warcraft.NET for parsing ADT files from version 0.6 through 11.x and dumping useful data from them, as well as parsing PM4 and PD4 files for useful data.



