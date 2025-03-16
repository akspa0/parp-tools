# WLW

From wowdev

Jump to navigation Jump to search

These files are found for all of World of Warcraft's water however are not read by any version of the client. For each contained body of water in the game (except for the oceans) there exists one of these files which documents it's heightmap (Water Level Water). Since release, there is also a WLQ file located at the exact same file and pathname as the .wlw files. Little else is known about this file format...

## Structure

Below is a structure based on the known information of this file. This may not be correct.

```
struct WLW { char magic[4]; // LIQ* uint16_t version; // 0, 1, 2 uint16_t _unk06; // always 1, probably also part of version uint16_t liquidType; // version ≤ 1 LiquidTypeu, version 2 = DB/LiquidType ID uint16_t padding; // more likely that liquidType is a uint32_t uint32_t block_count; struct block { C3Vectori vertices[0x10]; C2Vectori coord; // internal coords uint16_t data[0x50]; }[block_count]; uint32_t block2_count; // only seen in 'world/maps/azeroth/test.wlm' struct block2 { C3Vectori _unk00; C2Vectori _unk0C; char _unk14[0x38]; // 4 floats then 0 filled in the above }[block2_count]; #if version ≥ 1 char unknown; // mostly 1 #endif enum LiquidType // appears to match MCLQ tiles { still = 0, ocean = 1, ? = 2, // used by 'Shadowmoon Pools 02.wlm' slow/river = 4, magma = 6, fast flowing = 8, } }
```

### Height Map

The blocks are made of 48 float's that make up 16 vertices ( z-up ) arranged in a grid that starts in the lower right corner.

```
15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0
```

When all the chunks are combined, they form a height map of the water, similar to those in the ADT's.