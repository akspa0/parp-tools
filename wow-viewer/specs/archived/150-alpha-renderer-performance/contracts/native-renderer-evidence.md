# Contract: Native 0.5.3 Renderer Evidence Ledger

Each row in `memory-bank/workstream-alpha053-renderer-performance.md` MUST use this shape:

```text
Build: 0.5.3.3368
Anchor: <function/symbol/address/xref/data reference>
Area: world | terrain | object | resource | state | lod | unknown
Observation: <behavior seen in Ghidra>
Confidence: proven | inferred | unknown
Viewer implication: <bounded candidate constraint or experiment>
Evidence owner: primary agent | user Ghidra session
```

Rules:

- The anchor must identify where the observation came from.
- The observation describes behavior and data flow, not copied client code.
- `proven` is reserved for a directly observed 0.5.3 program behavior.
- Later-build or legacy-renderer evidence must name its build and be marked comparative.
- An unknown row is valid and prevents the related behavior from being presented as native parity.
