{{ ... }}

## 4. Guiding Philosophy & Methodology

To address past frustrations and ensure this effort is successful, we will adhere to a strict philosophy built on three pillars. This is our contract to avoid wasted time.

1.  **Slow is Smooth, Smooth is Fast**: We will be deliberately methodical. Every step will be small, verifiable, and approved before moving to the next. There will be no "big bang" integrations.
2.  **Performance by Design**: Modern hardware will be exploited. All data processing tasks will be designed for parallel execution from the ground up. Multithreading is a requirement, not an optimization.
3.  **Audit-First Refactoring**: This is our core methodology. We will not write a line of implementation code until a complete audit of the source component is performed, documented, and approved by you.

### The Audit Process

For each component, the process is:
1.  **Complete Audit**: Systematically analyze the source codebase.
2.  **Create Audit Document**: Produce a detailed markdown document (e.g., `docs/audits/Viewer_Audit.md`) that captures the working pipeline, key algorithms, dependencies, and a proposed simplified structure.
3.  **Review and Approve**: You will review the audit document. No implementation begins until you approve.
4.  **Implement**: Based on the approved audit, implement the new, refactored component.

## 5. Phased Implementation roadmap

*The roadmap is updated to reflect our new philosophy.*

### Phase 1: Foundation & Viewer Audit

- **Goal**: Prepare the solution and conduct the first, most critical audit on the `WoWRollback` viewer.
- **Tasks**:
    1.  Define core interfaces and set up DI in `WoWRollback.Core`.
    2.  **Audit the `WoWRollback` Viewer**: Create `docs/audits/Viewer_Audit.md`, documenting its current state, the `04_Overlay_Plugin_Architecture.md` goals, and a clear path for the refactor. Get your approval.

### Phase 2: Viewer Refactor & Overlay Foundation

- **Goal**: Implement the findings from the viewer audit.
- **Tasks**:
    1.  Create the `WoWRollback.Viewer` project and move assets into it.
    2.  Implement the frontend **Overlay Manifest** loader and **Runtime Core** based on the approved audit.

### Phase 3: World Geometry Integration (Audit & Implement)

- **Goal**: Methodically integrate all world geometry formats.
- **Tasks**:
    1.  **Audit `AlphaWDTAnalysisTool`**: Create `docs/audits/AlphaWDT_Audit.md` and get approval.
    2.  **Implement `WoWRollback.Plugins.AlphaWdt`**.
    3.  **Audit `ADTPreFabTool.poc` & `next`**: Create audit docs for ADT and WDL.
    4.  **Implement `WoWRollback.Plugins.Adt` & `WoWRollback.Plugins.Wdl`**.

### Phase 4: PM4 Integration (Audit & Implement)

- **Goal**: Integrate the definitive PM4 handling logic.
- **Tasks**:
    1.  **Audit `PM4FacesTool`**: Conduct a complete audit, producing `docs/audits/PM4_Audit.md`.
    2.  **Implement `WoWRollback.Plugins.Pm4`** based on the approved audit.

*Phases 5 (Water/Lighting) and 6 (Polish) remain the same.*

{{ ... }}

## 7. Next Steps

- **Action**: Please approve this updated, audit-first plan.
- **Outcome**: Once approved, I will begin with **Phase 1**: setting up the solution structure and then immediately starting the **Viewer Audit**.
