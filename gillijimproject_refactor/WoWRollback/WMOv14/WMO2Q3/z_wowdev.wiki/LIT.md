'''These files are obsolete!''' 

''LIT files have stored lighting-information until some patch. Today, lightning is stored in the following .[[:Category:DBC|DBC]]s:
*[[DB/Light]]
*[[DB/LightFloatBand]]
*[[DB/LightIntBand]]
*[[DB/LightParams]]
*[[DB/LightSkybox]]

For worlds that have terrain data, a corresponding LIT file includes information about the sky color, and possibly lighting conditions. They are stored in World\name\lights.lit

==Header==

 '''Offset 	Type 	Description'''
 0x00 	uint32 	Version number
 0x04 	uint32 	Light count - number of lights defined in this file

64 bytes per light:

 '''Offset 	Type 		Description'''
 0x00 	{{Template:Type|C2iVector}} 	(-1,-1) for the 'default' first record, (0,0) otherwise
 0x08 	int32 		-1 for the 'default' first record, 0 otherwise
 0x0C 	{{Template:Type|C3Vector}} 	Coordinates (X,Y,Z)
 0x18 	float 		Light radius
 0x1C 	float 		Light dropoff
 0x20 	char[32] 	Light name

The float values seem to be multiplied by 36. Dividing by 36 gives back the original scale (I think) In the case of "I think", I think that the game client uses the floats to perform some kind of Cube-Mapped LightMapping (hence the X, Y, Z, -X, -Y, -Z values). -DG

 struct LitFile
 {
  uint32_t_t versionNumber;              // 0x80000003 v8.3 for Undermine, 0x80000004 v8.4 in 0.5.3, 0x80000005 v8.5 starting from 0.11.0
  int32_t lightCount;                    // if -1, lightGroup contains a single partial entry
  LightListData lightList[lightCount];   // ignored if lightCount < 1
  [[#Light_Data|LightData]] lightGroup[abs(lightCount)];
  
  struct LightListData
  {
    {{Template:Type|C2iVector}} m_chunk;
    int32_t m_chunkRadius;
    {{Template:Type|C3Vector}} m_lightLocation;
    float m_lightRadius;
    float m_lightDropoff;
    char m_lightName[0x20];               // 0 padded, xFFFD padded in v8.3
  };
 }

==Light Data==

4 * 0x15F0 bytes per entry.

The first block of the four seems to have the sky colors, the second and fourth are usually all black, the third might be lighting colors or something else entirely.
 '''Offset 	Type 		Description'''
 0x0000 	18 * int32 	Lengths
 0x0048 	18 * 64 * int32	Color + time records
 0x1248 	32 * float 	Float values A
 0x12C8 	32 * float 	Float values B
 0x1348 	uint32 		Int value I
 0x134C 	32 * float 	Float values C
 0x13CC 	32 * float 	Float values D
 0x144C 	32 * float 	Float values E
 0x14CC 	32 * float 	Float values F
 0x154C 	uint32 		Int value J
 0x1550 	32 * float 	Float values G
 0x15D0 	8 * uint32 	Padding (all 0)

The color and time records are in the following format:

Each row of 64 integers contains 32 pairs of integers: the first value is the time in half-minutes (on a scale of 0 to 2880 from midnight to midnight), the second value is a BGRX color. The i-th row contains Lengths[ i ] records like that. I think the color values for intermediate times are interpolated based on the times given in this list.

So there are 18 time-based color rows described here, for the first block these are always the sky colors (well, the first 8 at least). WoWmapview is currently only drawing a very crude, fake sky globe - the colors may or may not match up ;)

Each light has 4 sets of light data which map to the "Params" of [[DB/Light#Structure|DB/Light]].

<b>Note:</b> If <code>header.lightCount == -1</code> there will only be data for <tt>m_lightdata</tt>, the other entries will be defaulted.

 struct LightData
 {
   DiskLightDataItem m_lightdata;       // ParamsClear 
  #if header.lightCount > -1
   DiskLightDataItem m_stormdata;       // ParamsStorm
   DiskLightDataItem m_lightdataWater;  // ParamsClearWat
   DiskLightDataItem m_stormdataWater;  // ParamsStormWat
  #endif
 }[LitFile.lightCount];
 
 struct DiskLightDataItem
 {
   const uint32_t numHighLights = (header.versionNumber == 0x80000003 ? 0xE : 0x12);
 
   int32_t m_highlightCount[numHighLights];
   LightMarker m_highlightMarker[numHighLights][0x20];  // the equivalent of [[DB/LightIntBand]]
   float m_fogEnd[0x20];                                // the following float arrays are the equivalent of [[DB/LightFloatBand]]
   float m_fogStartScaler[0x20];
   int32_t m_highlightSky;
   float m_skyData[4][0x20];
   int32_t m_cloudMask;
  #if header.versionNumber ≥ 0x80000005
   float m_paramData[4][0xA];                           // the equivalent of [[DB/LightParams]]'s alpha fields
  #endif
 };
  
 struct LightMarker
 {
   int32_t time;
   {{Template:Type|CImVector}} color;
 };


The 7 sets of floating-point values have to describe the arrangement of the sky colors somehow, but they're pretty difficult to interpret. They usually contain at most 8 values, the rest being 0.

So today I experimented with a custom .LIT file (red and blue skies are hilarious), so here are the meanings for the various color tracks:
 '''Number 	Description'''
 0 	Global diffuse light
 1 	Global ambient light
 2 	Sky color 0 (top)
 3 	Sky color 1 (middle)
 4 	Sky color 2 (middle to horizon)
 5 	Sky color 3 (above horizon)
 6 	Sky color 4 (horizon)
 7 	Fog color / background mountains color
 8 	 ?
 9 	Sun color + sun halo color
 10 	Sun larger halo color
 11 	 ?
 12 	Cloud color
 13 	 ?
 14 	 ?
 15 	Ground shadow color
 16 	Water color [light]
 17 	Water color [dark]

The below structure equates to each individual light in the file.

 struct CurrentLight
 {
   {{Template:Type|CImVector}} DirectColor;
   {{Template:Type|CImVector}} AmbientColor;
   {{Template:Type|CImVector}} SkyArray[6];
   {{Template:Type|CImVector}} CloudArray[5];
   {{Template:Type|CImVector}} WaterArray[4];
   float FogEnd;
   float FogStartScalar;
   {{Template:Type|CImVector}} ShadowOpacity;
   float Darkness;
   float CloudData[4];
 };



The different skies are interpolated based on distance.

The four sets of data are completely different. #0 is the default look. #1 and #3 are usually all black. #2 might be the 'ghost view' lighting for when you're dead, but I'm not sure.


[[Category:Format]]
