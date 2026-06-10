# PM4

From wowdev

Jump to navigation Jump to search

PM4 files are server side supplementary files to ADTs, just as PD4 are for WMOs. There is one file per root ADT. They are not supposed to be shipped to the client and are used by the server only.

## Contents

* 1 MVER
* 2 MSHD
* 3 MSPV
* 4 MSPI
* 5 MSCN
* 6 MSLK
* 7 MSVT
* 8 MSVI
* 9 MSUR
* 10 MPRL
* 11 MPRR
* 12 MDBH

  + 12.1 MDBF
  + 12.2 MDBI
* 13 MDOS
* 14 MDSF

# MVER

See PD4#MVER.

# MSHD

See PD4#MSHD.

# MSPV

See PD4#MSPV.

# MSPI

See PD4#MSPI.

# MSCN

See PD4#MSCN.

# MSLK

See PD4#MSLK.

# MSVT

See PD4#MSVT.

# MSVI

See PD4#MSVI.

# MSUR

See PD4#MSUR.

# MPRL

```
struct { uint16_t _0x00; // Always 0 in version_??.u int16_t _0x02; // Always -1 in version_??.u uint16_t _0x04; uint16_t _0x06; C3Vectori position; int16_t _0x14; uint16_t _0x16; } mprl[];
```

# MPRR

```
struct { uint16_t _0x00; uint16_t _0x02; } mprr[];
```

# MDBH

This chunk has MDBF and MDBI as sub chunks and reimplements chunk size with count.

```
uint32_t count; struct { CHUNK index; CHUNK filename[3]; } m_destructible_building_header[count];
```

## MDBF

```
char m_destructible_building_filename[];
```

## MDBI

```
uint32_t m_destructible_building_index;
```

# MDOS

```
struct { uint32_t _0x00; uint32_t _0x04; } mdos[];
```

# MDSF

```
struct { uint32_t _0x00; uint32_t _0x04; } mdsf[];