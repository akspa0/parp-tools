# A wild idea
A tool to paint pictures on the terrain, using only textures we have access to - 
the textures need to be analyzed for composition, using a vlm and/or opencv2 -
to detect patterns, mainly, and correlate the patterns to the filenames in some way
then we need to utilize this data to paint chunks with up to 4 textures and 3 alphamsks for blending. If we target 3.3.5, we can also use MCCV vertex shaders. If we target >3.x, we can only paint with textures. Our input would be an image of any sort.
our outputs would be a freshly generated minimap (if possible - this is a sub-project for 2D minimap generation through MCNK reading and texture/alphamask blending - look at wow.export, since that's what it does!), map files for LK ADT and alphaWDT.

beyond this, we could do similar work with terrain data, for prefabs detection work.
With the combination of texture and terrain data as data inputs, we should be able to find all prefabs via progmatic means.

with all this data in a more manageable state, we should also be able to literally paint any image to the terrain, including images we've restored of pre-alpha world of warcraft. In the end, we will be able to restore, with some degree of certainty, texturing of terrain that no longer exists, without painting it by hand. We can lock things down by areaID, perhaps also offer a means of only exporting specific ADT's for a specified set of areaIDs in either source or target version, or by name alone. We have some of this functionality in some of our earlier work - AlphaWDTAnalysisTool, next/, and possibly somewhere in the WoWRollback.Orchestrator (The AreaID ADT exporter).

We should be able to pull this all together to help build WoWRollback as the perfect tool to take any work from Noggit projects back out, as well as paint new things in, from the outside of the map editor.

MCNK's are meant to be read and decoded somehow.

We could extend things out to WMO blp's too, so we can identify from any minimap, what objects are placed, and generate that data. We also need to think about the PM4 pathfinding data, and try to find a way to match it up with the WMO data that we have on disk. It's not the same thing, though - pathfinding data is just collision surfaces, and line-of-sight portaling stuff. We need to generate similarities between PM4 objects and WMO objects, and then try to find correlation within each object's data. We can gate things by expected asset path, which should help reduce incorrect matches by a high degree.

This is where WoWRollback as a project should go - intertwine the work we've been doing for over a year, into a single digital archeology toolkit, for every version of WoW. It's a well for content creators to draw from. That's the point.

----
Perhaps we should look at using AI agents to figure out some of the really confusing problems we've got at-hand. It makes little sense for me to be handling the tiny little ins and outs when we have ai agents that could do all the data analysis and comparisons to figure out what's wrong with our documented data - and why we have so many issues getting placements working. Why are we doing the grunt work when we have agentic AI workloads available to us?! Qwen3-Coder is a very capable model for such work, and we can set it up through something like openrouter, for our uses.