# WLQ

From wowdev

Jump to navigation Jump to search

# WLQ Files

One of these files exist for each WLW file in the game. They are also located at the same path and filename as each's .wlw counterpart.

## Header

```
Type Name Descrition uint32 magic always 2QIL uint16 version uint16 unk06 always 1 char unk08[4] always 0 uint32? liquidType 0 = river, 1 = ocean, 2 = magma, 3 = slime (same as WMO liquid_basic_types) uint16[9] unk10 uint32 block_count same format as WLW blocks
```

## Data

Following the header are nChunks 360 byte blocks ( see WLW )