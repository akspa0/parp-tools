# Research: Renderer Marketing Capture Automation

## Decision 1: Reuse camera-path warmup and renderer-frame capture

- **Decision**: Feature tours start from the existing `ViewerApp_CameraPaths` warm/play path and `TryStartCurrentViewVideoRecording` encoder route.
- **Evidence**: `StartCameraPathVideoCapture` validates the path, calls `BeginCameraPathPreload`, waits until its bounded tiles and objects are resident, then calls `TryStartCurrentViewVideoRecording`. The encoder receives RGBA framebuffer data directly; it is not desktop capture.
- **Why**: A second playback/capture loop could drift from the renderer torture-test workflow and obscure the performance behavior the user wants to measure.
- **Rejected**: OS screen recording or a separate playback runner. Neither establishes renderer-frame behavior and both duplicate currently working capture/warmup semantics.

## Decision 2: Capture only the intended tour presentation

- **Decision**: A tour forces the full-frame capture tap and hides normal chrome. An owned overlay renderer draws only the active recipe callout before that tap; future registered feature controls may use the same presentation model.
- **Evidence**: The no-UI capture tap runs before ImGui, while the with-UI capture tap runs after `DrawUI`. The existing `Tab` chrome switch suppresses normal panels.
- **Why**: The recording can show a clean scene except for a deliberate feature callout without an operator clicking controls during playback.
- **Rejected**: Turning on the full options bar for every beat. It obscures the scene and couples a recipe to unstable dock/window layout.

## Decision 3: Treat the benchmark as evidence, not a quality claim

- **Decision**: The receipt serializes the bounded `WorldRenderFrameHistory` snapshot captured over the recorded run, including total CPU frame-time distribution, retained sample count, movement flag, hitch threshold/list, and recorder overhead. It labels a degraded or failed attempt explicitly.
- **Evidence**: `WorldScene.FrameHistory` is always-on, bounded, allocation-free on the render path, and exposes an on-demand `Snapshot`; its diagnostics explicitly retain periodic hitches and unaccounted time.
- **Why**: Existing known low-FPS dips must be visible in the evidence. An encoded video is not proof that a benchmark completed cleanly.
- **Rejected**: Reporting only the current FPS or a simple encoder-success bit. Both omit temporal hitches and do not characterize renderer health.

## Decision 4: Keep model and receipt policy library-owned

- **Decision**: Put recipe validation, attempt state, receipt construction, output-root policy, and handoff validation in `WowViewer.Core.Runtime/Marketing`. The viewer owns only capture lifecycle composition and ImGui drawing through a small overlay adapter.
- **Why**: The behavior is deterministic and testable without a GPU/window. The design follows the library-first rule and avoids adding a new `ViewerApp` field or partial class.
- **Rejected**: Putting JSON, timing math, and external handoff rules in `ViewerApp_CaptureAutomation.cs`. That would extend an already oversized UI owner and make deterministic tests difficult.

## Decision 5: Define a transport-neutral Comfy/MCP handoff now; do not call ComfyUI now

- **Decision**: A completed receipt can produce a versioned handoff descriptor containing only relative artifact references and selected provenance. Transport invocation remains an opt-in future adapter.
- **Evidence**: This session exposes no callable Comfy MCP tool. The operator reports a local ComfyUI instance, but no versioned server schema or workflow has been selected.
- **Why**: It creates a stable authoring boundary without guessing an HTTP endpoint, exposing client paths, or claiming generated promotional media.
- **Rejected**: Directly posting to `127.0.0.1:8199` or fabricating a Comfy workflow. Both would be undocumented, non-portable behavior with no available testable contract.

## Decision 6: README media remains an operator proof gate

- **Decision**: Add README links/embeds only after a real tour video and still are reviewed against their receipt. The feature supplies provenance and a capture location; it does not add placeholders.
- **Why**: A synthetic or unverified showcase would misrepresent the viewer and fail the repository's real-data evidence rules.

