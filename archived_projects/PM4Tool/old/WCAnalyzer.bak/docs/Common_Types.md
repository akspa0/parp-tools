# Common Types

From wowdev

Jump to navigation Jump to search

This page lists types commonly used in WoW, but not specific to a file format.

## Contents

* 1 foreign_key

  + 1.1 foreign_key_mask
* 2 stringref

  + 2.1 langstringref
* 3 C2Vector
* 4 C2iVector
* 5 C3Vector
* 6 C3iVector
* 7 C4Vector
* 8 C4iVector
* 9 C33Matrix
* 10 C34Matrix
* 11 C44Matrix
* 12 C4Plane
* 13 C4Quaternion
* 14 CRange
* 15 CiRange
* 16 CAaBox
* 17 CAaSphere
* 18 CArgb
* 19 CImVector
* 20 C3sVector
* 21 C3Segment
* 22 CFacet
* 23 C3Ray
* 24 CRect
* 25 CiRect
* 26 fixed_point
* 27 fixed16

# foreign_key

A reference into a DBC or DB2 database file. E.g.

```
foreign_keyi<uint16_t, &LiquidTypeRec::m_spellID> liquid_spell_id; ^ type of reference ^ table referenced ^ variable referenced
```

## foreign_key_mask

The same as #foreign_key, but instead of directly containing the ID, this field contains a bit mask where the bit indexes set indicate what is referenced.

The value 0x5 would not reference ID 5, but IDs 1 and 4.

Note: It probably isn't consistent with being 1 or 0 based, and likely rarely is noted, sorry.

# stringref

A reference into the string block of a DBC or DB2 database file. See DBC#String_Block.

```
using stringref = uint32_t;
```

## langstringref

A field for localized strings, consisting of an array of stringrefs followed by a field of flags. The amount of language entries depends on the version, with Cataclysm dropping all but the client's language. See Localization for details.

```
struct langstringref { #if < stringref enUS; // also enGB stringref koKR; stringref frFR; stringref deDE; stringref enCN; // also zhCN stringref enTW; // also zhTW stringref esES; stringref esMX; //! \todo verify that pre->8 locales hadn't changed #if ≥ (2.1.0.6692) stringref ruRU; stringref jaJPu; // unknowns are still unused as of 8.0.1.26095 stringref ptPT; // also ptBR stringref itIT; stringref unknown_12; stringref unknown_13; stringref unknown_14; stringref unknown_15; #endif uint32_t flags; #else stringref client_locale; #endif };
```

# C2Vector

A two component float vector.

```
struct C2Vector { float x; float y; };
```

# C2iVector

A two component int vector.

```
struct C2iVector { int x; int y; };
```

# C3Vector

A three component float vector.

```
struct C3Vector { /*0x00*/ float x; /*0x04*/ float y; /*0x08*/ float z; };
```

# C3iVector

A three component int vector.

```
struct C3iVector { int x; int y; int z; };
```

# C4Vector

A four component float vector.

```
struct C4Vector { float x; float y; float z; float w; };
```

# C4iVector

A four component int vector.

```
struct C4iVector { int x; int y; int z; int w; };
```

# C33Matrix

A three by three matrix.

```
struct C33Matrix // todo: row or column? { C3Vectori columns[3]; };
```

# C34Matrix

A three by four matrix.

```
struct C34Matrix // todo: row or column? { C3Vectori columns[4]; };
```

# C44Matrix

A four by four column-major matrix.

```
struct C44Matrix { C4Vectori columns[4]; };
```

# C4Plane

A 3D plane defined by four floats.

```
struct C4Plane // todo: verify { C3Vectori normal; float distance; };
```

Those 4 floats are a, b, c, d variables from General form of the equation of a plane

# C4Quaternion

A quaternion.

```
struct C4Quaternion { float x; float y; float z; float w; // Unlike Quaternions elsewhere, the scalar part ('w') is the last element in the struct instead of the first };
```

# CRange

A one dimensional float range defined by the bounds.

```
struct Range { float min; float max; };
```

# CiRange

A one dimensional int range defined by the bounds.

```
struct Range { int min; int max; };
```

# CAaBox

An axis aligned box described by the minimum and maximum point.

```
struct CAaBox { /*0x00*/ C3Vectori min; /*0x0C*/ C3Vectori max; };
```

# CAaSphere

An axis aligned sphere described by position and radius.

```
struct CAaSphere { C3Vectori position; float radius; };
```

# CArgb

A color given in values of red, green, blue and alpha. Either

```
using CArgb = uint32_t;
```

or

```
struct CArgb // todo: verify, add CRgba, ..? { unsigned char r; unsigned char g; unsigned char b; unsigned char a; };
```

# CImVector

A color given in values of blue, green, red and alpha

```
struct CImVector { unsigned char b; unsigned char g; unsigned char r; unsigned char a; };
```

# C3sVector

A three component vector of shorts.

```
struct C3sVector { int16_t x; int16_t y; int16_t z; };
```

# C3Segment

```
struct C3Segment { C3Vector start; C3Vector end; };
```

# CFacet

```
struct CFacet { C4Plane plane; C3Vector vertices[3]; };
```

# C3Ray

A ray defined by an origin and direction.

```
struct C3Ray { C3Vector origin; C3Vector dir; };
```

# CRect

/\* in 0.5.3.3368 the members are wrapped in a union struct, where you can access the value as either "miny" or "top", "minx" or "left", "maxy" or "bottom", "maxx" or "right" \*/

```
struct CRect { float miny; // top float minx; // left float maxy; // bottom float maxx; // right };
```

# CiRect

/\* in 0.5.3.3368 the members are wrapped in a union struct, where you can access the value as either "miny" or "top", "minx" or "left", "maxy" or "bottom", "maxx" or "right" \*/

```
struct CiRect { int miny; // top int minx; // left int maxy; // bottom int maxx; // right };
```

# fixed_point

A fixed point real number, opposed to a floating point. A sign bit, a given number of integer_bits and decimal_bits, adding up to the bit count of the Base type.

For conversion, essentially ignore the sign and divide by 2^decimal_bits, then multiply with the sign again. In the inverse direction, again mask the sign, divide by the factor, and multiply the sign.

For the special case if there are no integer_bits, i.e. fixed16i, use the factor of (2^(decimal_bits + 1) - 1) instead in order to use the full range.

As follows, a generic C++11 implementation of this. Yes, this could be one single function, but it was written to be generic and to be used inside structs where `to_float()` is `operator float()` instead to allow for easy use.

```
template<typename Base, size_t integer_bits, size_t decimal_bits> struct fixed_point { static const float constexpr factor = integer_bits ? (1 << decimal_bits ) : (1 << (decimal_bits + 1)) - 1 ; union { struct { Base integer_and_decimal : integer_bits + decimal_bits; Base sign : 1; }; Base raw; }; constexpr operator float() const { return (sign ? -1.f : 1.f) * integer_and_decimal / factor; } constexpr fixed_point (float x) : sign (x < 0.f) , integer_and_decimal (std::abs (x) * factor) {} };
```

# fixed16

A fixed point number without integer part. Trivially implemented using

```
using fixed16 = int16_t; // divide by 0x7fff to get float value
```

or correctly implemented as

```
using fixed16 = fixed_pointi<uint16_t, 0, 15>;