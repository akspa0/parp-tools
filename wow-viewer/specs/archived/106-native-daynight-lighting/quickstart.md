# Calibration Quickstart (User-Owned Capture)

This procedure prepares evidence only. The user runs every native/client or Capture render under AGENTS Rule 0; no agent-launched capture, harvest, or training run is part of this plan.

## 1. Select a control tile

Choose one outdoor 0.5.3.3368 terrain tile with visible directional relief and MCSH. Record its map/tile name, client build identity, and top-down camera framing. Keep geometry, assets, time, and camera identical in native and viewer captures.

## 2. Capture the lock time

Run the prepared Capture tool with a declared profile/time and sidecar output; substitute only your selected client root, tile, profile source, and output location:

```powershell
dotnet run --project tools/capture/WowViewer.Tool.Capture -- render `
  --client-root "<client-root>" `
  --tile-name "<map>_<x>_<y>" `
  --output "<viewer-lock.png>" `
  --resolution 512 `
  --game-time <normalized-time> `
  --lighting-source lit `
  --lit-virtual-path "<map-lights.lit>" `
  --lighting-metadata-output "<viewer-lock.sidecar.json>"
```

The Phase 2 implementation adds the required exact profile selection and will reject this command until its direction transform is calibrated. Capture the matching native game image at the same declared time.

## 3. Lock the transform

Compare terrain light/shadow orientation first; color differences are not a substitute for axis/sign evidence. Select one signed axis permutation only if it matches the native image. Record both images and hashes in the calibration artifact.

## 4. Hold out two times

Repeat at two distinct normalized times. Do not alter the transform. Promote it only if both held-out comparisons match orientation.

## 5. Separate follow-ups

Measure MCSH attenuation and sky-band altitude placement independently. Do not change the recovered direction model to compensate for either measurement.
