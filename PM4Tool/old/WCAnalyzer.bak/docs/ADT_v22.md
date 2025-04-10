ADT/v22 - wowdev

# ADT/v22

From wowdev

Jump to navigation Jump to search

Most likely temporary version. Do not bother implementing until final version is there. This is not used in current clients.

This document may not list all chunks.

## Contents

* 1 AHDR: HeaDeR
* 2 AVTX: VerTeX
* 3 ANRM: NoRMals
* 4 ATEX: TEXtures
* 5 ADOO: DOOdad
* 6 ACNK: ChuNK

  + 6.1 ALYR: LaYeR

    - 6.1.1 AMAP: alpha MAP
  + 6.2 ASHD: SHaDow map
  + 6.3 ACDO: Chunk - Definitions of Objects

## AHDR: HeaDeR

* size: 0x40

```
DWORD version // 22, mirrors MVER DWORD vertices_x // 129 DWORD vertices_y // 129 DWORD chunks_x // 16 DWORD chunks_y // 16 DWORD[11]
```

## AVTX: VerTeX

* size: may be variable. Its based on the header. header.vertices_x*header.vertices_y + (header.vertices_x-1)*(header.vertices_y-1) * 4 bytes (float).

A large collection of floats. Again with inner[(header.vertices_x-1)*(header.vertices_y-1)] and outer[header.vertices_x*header.vertices_y] vertices.

Important: These are NOT mixed as in the ADT/v18s, but have the 129*129 field first, then the 128*128 one.

```
float outer[129][129]; float inner[128][128];
```

for example. Same should apply for the normals.

## ANRM: NoRMals

* size: may be variable. Its based on the header. header.vertices_x*header.vertices_y + (header.vertices_x-1)*(header.vertices_y-1) * 3 bytes.

Like ADT/v18, these are triples of chars being a vector. 127 is 1.0, -127 is -1.0.

## ATEX: TEXtures

* size: variable

Used for texture filenames. This is an array of chunks. There are as many chunks as there are textures used in this ADT tile, which is limited to 128 textures per ADT.

## ADOO: DOOdad

* size: variable

Used for M2 and WMO filenames. Both are in here. Again, an array of chunks with a chunk for every filename.

## ACNK: ChuNK

* size: variable

This one is a bit like the MCNK one. There is an header for it, if the size is bigger than 0x40. If it has an header, its like this:

```
int indexX int indexY DWORD int areaId WORD DWORD[4] lowdetailtextureingmap WORD DWORD[7]
```

### ALYR: LaYeR

* size: 0x20 or more.

This will include the alpha map if flags&0x100.

```
int textureID // as in ATEX int flags DWORD[6]
```

#### AMAP: alpha MAP

* optional. size: variable (?)

either uncompressed 8bit (depending on WDT) or 4bit RLE

### ASHD: SHaDow map

* optional. size: 0x200 (?)

most likely like the old one.

### ACDO: Chunk - Definitions of Objects

* optional. size: 0x38

Doodads and WMOs are now both defined here.

```
int modelid // as in ADOO float position[3] float rotation[3] float scale[3] float uint uniqueId DWORD[] // name / doodadsets?
```

Retrieved from "https://wowdev.wiki/index.php?title=ADT/v22&oldid=26618"

Category:

* Format

## Navigation menu

* This page was last edited on 3 March 2019, at 16:01.

* Privacy policy
* From wowdev
* Disclaimers