# MDX

From wowdev

Jump to navigation Jump to search

This section only applies to versions < .

MDX files are chunked binary files that contain model objects. They are the predecessor of the M2 format.

First used in Warcraft 3, MDX was actively developed and used in WoW as the primary model format until patch (0.11.0.3925). Although obsolete, some DBCs still contain filenames with a .mdx extension.

Note: The majority of the below information has been taken from the (0.5.3.3368) client and is only truly compliant for version 1300 of the format.

## Contents

* 1 Structure
* 2 Common Types

  + 2.1 C3Color
  + 2.2 C4QuaternionCompressed
  + 2.3 CMdlBounds
  + 2.4 MDLKEYTRACK
  + 2.5 MDLSIMPLEKEYTRACK
  + 2.6 MDLGENOBJECT

    - 2.6.1 Flags
    - 2.6.2 KGTR
    - 2.6.3 KGRT
    - 2.6.4 KGSC
* 3 VERS
* 4 MODL
* 5 SEQS
* 6 GLBS
* 7 MTLS

  + 7.1 MTLS(Reforged)
  + 7.2 KMTE
  + 7.3 KMTA
  + 7.4 KMTF
* 8 TEXS
* 9 TXAN

  + 9.1 KTAT
  + 9.2 KTAR
  + 9.3 KTAS
* 10 GEOS

  + 10.1 GEOS (≤ v1400)
  + 极简版内容...
