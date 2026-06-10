# MDX Chunks — Build 0.5.3.3368

This folder documents the MDX/MDL chunk model used by **0.5.3.3368** from current Ghidra evidence.

## Core parser evidence
- MDLX magic check appears as little-endian `'XLDM'` assertion.
- Section-type symbols observed include:
  - `MDLSEQUENCESSECTION`, `MDLGLOBALSEQSECTION`
  - `MDLMATERIALSECTION`, `MDLTEXLAYER`, `MDLTEXTURESECTION`, `MDLTEXANIMSECTION`
  - `MDLGEOSETSECTION`, `MDLGEOSETANIMSECTION`
  - `MDLBONESECTION`, `MDLLIGHTSECTION`, `MDLATTACHMENTSECTION`
  - `MDLCAMERASECTION`, `MDLEVENTSECTION`

## Chunks documented
- `XLDM.md`
- `MDLSEQUENCESSECTION.md`
- `ANIMATION_SYSTEM.md`
- `MDLMATERIALSECTION.md`
- `MDLTEXTURESECTION.md`
- `MDLTEXLAYER.md`
- `MDLGEOSETSECTION.md`
- `MDLBONESECTION.md`
- `MDLCAMERASECTION.md`

## Notes
- 0.5.3 has strong evidence for section taxonomy but weaker direct function labels than 0.7.0.
- Field-by-field section payload schemas remain partially unresolved and are marked accordingly.
