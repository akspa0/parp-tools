# Contract: Camera Follow Target

Shape of the camera's follow state, resolved fresh every frame — never cached across frames beyond the
target reference itself, since the whole point is tracking a moving bone.

```json
{
  "modelInstanceId": "m2:1234",
  "pipeline": "Modern",
  "boneReference": { "keyBoneId": "Head" },
  "resolved": true,
  "worldPosition": { "x": 100.2, "y": 55.0, "z": 12.8 },
  "worldOrientation": { "yawDegrees": 45.0, "pitchDegrees": -3.0, "rollDegrees": 0.0 }
}
```

When the target model is no longer present in the scene:

```json
{
  "modelInstanceId": "m2:1234",
  "pipeline": "Modern",
  "boneReference": { "keyBoneId": "Head" },
  "resolved": false,
  "worldPosition": null,
  "worldOrientation": null
}
```

**Contract rules**:

- `resolved: false` is a normal, expected outcome (FR-011's edge case), never an error condition on its
  own — the camera falls back to free-fly from its last known position when this happens, it does not
  freeze or throw.
- `pipeline` records which bone-matrix source produced the resolution (`Legacy` = `MdxAnimator.BoneMatrices`,
  `Modern` = `M2BonePoseState.Matrices`) so a consumer can tell which code path is actually live for a
  given followed model — useful for debugging, since the two pipelines have different data-availability
  characteristics (Spec 154's era-reading findings apply to the modern pipeline, not the legacy one).
