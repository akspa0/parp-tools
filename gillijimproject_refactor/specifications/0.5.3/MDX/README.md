# MDX/MDL Sections — 0.5.3.3368

## Confirmed loader-side evidence
- MDLX magic check via `'XLDM'` comparison (`0x00834364`)
- Section type symbols present:
  - `MDLSEQUENCESSECTION` (`0x008344A4`)
  - `MDLGLOBALSEQSECTION` (`0x008344C8`)
  - `MDLMATERIALSECTION` (`0x008344EC`)
  - `MDLTEXLAYER` (`0x00834510`)
  - `MDLTEXTURESECTION` (`0x0083452C`)
  - `MDLTEXANIMSECTION` (`0x0083454C`)
  - `MDLGEOSETSECTION` (`0x008345D8`)
  - `MDLGEOSETANIMSECTION` (`0x0083466C`)
  - `MDLBONESECTION` (`0x008346F4`)
  - `MDLLIGHTSECTION` (`0x00834714`)
  - `MDLATTACHMENTSECTION` (`0x00834750`)
  - `MDLCAMERASECTION` (`0x00834798`)
  - `MDLEVENTSECTION` (`0x008347B8`)

## Notes
- This pass confirms section taxonomy and magic validation for 0.5.3.
- Exact section payload offsets/sizes require parser-function decompile + xref traversal in a second pass.

## Next Deep-Dive Targets
- Recover per-section entry width and field maps from loop bodies
- Correlate keyframe section types to concrete interpolation flags
- Map section ordering guarantees in 0.5.3 loader
