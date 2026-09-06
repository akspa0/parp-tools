# Tasks: Agent Governance — Scope Fidelity, Receipts & Spec Hygiene (Spec 224)

**Spec**: [spec.md](spec.md) · **Plan**: [plan.md](plan.md)

## Phase 1 — Rules and mechanism

- [x] 224-T101: Add `## 9. Governance` to repo-root `AGENTS.md` stating: scope freeze (FR-1),
      receipt gate (FR-2), write containment (FR-3), spec-sync (FR-5), context discipline (FR-6),
      and the cleanup ledger with `Last cleanup` / `Next cleanup due` timestamps.
      **Receipt**: AGENTS.md section committed in this change; see session summary.
- [x] 224-T102: Author `.roo/skills/speckit-cleanup/SKILL.md` (audit procedure, FR-4) and install it
      to the operator skill directory.
      **Receipt**: file created and copied; verify with `Get-Content C:\Users\akspa\.roo\skills\speckit-cleanup\SKILL.md`.
- [x] 224-T103: Apply the Spec 212 operator correction (strike decorative reticle/compass/bezel
      scope; re-aim at ImGui-panels-on-camera-frame-surfaces; default rig OFF).
      **Receipt**: `CameraHudRig.cs` rewritten without rejected elements, `Enabled = false`;
      viewer Debug build 0 errors; 212 tasks/spec/plan edited with dated correction notes.
- [x] 224-T104: Register Spec 224 in `specs/STATUS.md` and record the operator correction in the
      memory bank.
      **Receipt**: STATUS.md row added; `activeContext.md` and `progress.md` updated with the
      dated operator correction.
- [ ] **Gate 1**: Operator reads `AGENTS.md` §9 and the skill file and approves the rules as
      binding. No further implementation without that approval.

## Phase 2 — First cleanup run

- [ ] 224-T201: Execute `speckit-cleanup` across all active specs; produce the dated audit report
      under `specs/224-speckit-governance/evidence/`; un-check any checked task lacking a receipt.
- [ ] 224-T202: Archive implemented/closed specs and compress memory-bank landed narrative; update
      the AGENTS.md ledger timestamp.
- [ ] **Gate 2**: Operator confirms `activeContext.md` + `STATUS.md` describe reality with no stale
      lanes.

## Phase 3 — Steady state

- [ ] 224-T301: Recurring monthly execution (first session of each month, or operator-invoked);
      ledger timestamp updated on every run.
