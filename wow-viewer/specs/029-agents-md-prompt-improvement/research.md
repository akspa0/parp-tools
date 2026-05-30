# Research: AGENTS.md Prompt Engineering Patterns

**Date**: 2026-05-30
**Sources**: gepa (genetic-pareto), superx/caveman, Anbeeld/AGENTS.md, awesome-agent-md

---

## 1. gepa (Genetic-Pareto Optimization)

**Source**: https://github.com/gepa-ai/gepa

### Key Concept: Actionable Side Information (ASI)

GEPA's core insight is that traditional optimizers know *that* a candidate failed but not *why*. GEPA uses LLMs to read full execution traces — error messages, profiling data, reasoning logs — and diagnose failures. This diagnostic feedback (ASI) serves as the text-optimization analogue of a gradient.

**Relevance to AGENTS.md**: The ASI pattern can be applied to how agents self-improve. Instead of a binary pass/fail on a spec or plan, agents should capture *why* something failed and feed that back into prompt evolution. This maps to our skill-registry approach — each skill should accumulate ASI over time.

### Patterns to Adopt:

1. **Diagnostic feedback loop**: When a plan or spec task fails, capture the failure trace (error message, reasoning gap, data mismatch) and use it to refine the next iteration.
2. **Pareto-aware selection**: When optimizing prompts/instructions, maintain a set of candidates that excel on different subsets — don't converge on a single "best" variant that loses coverage on edge cases.
3. **System-aware merge**: Combine strengths of multiple instruction variants when they succeed on different task types.

---

## 2. superx (Caveman — Autonomous Multi-Agent Coding)

**Source**: https://github.com/randomittin/superx

### Key Concepts: Subagent Orchestration, Wave Planning, Token Efficiency

superx orchestrates 14 specialized agents with 10-parallel execution, achieving ~75% token savings via "caveman ultra mode" (token compression + model routing).

### Patterns to Adopt:

1. **Subagent rules**: Launch 2+ subagents or none. Never exactly 1 (which is sequential, not parallel). This is now the standard in Anbeeld/AGENTS.md and should be in ours too.
2. **Wave-based planning**: Tasks grouped by dependency into waves. Each wave runs in parallel with fresh context (200K token window). No context rot from prior waves.
3. **Acceptance criteria as blocking gates**: Every task has runnable checks (grep, curl, test). Tasks cannot advance until all criteria pass. If a task fails after 2 fix attempts, it's marked BLOCKED and escalates.
4. **Caveman compression**: ~75% token savings through:
   - Model routing (cheaper model for lint, mid for docs, expensive for code)
   - Token-efficient prompts (remove filler, be direct)
   - Fresh context per wave (no accumulated garbage)
5. **Question-mark protocol**: Agent only stops to ask when it actually needs user input. Surfaces questions with quick-pick buttons.

---

## 3. Anbeeld/AGENTS.md (Evidence-Driven Global Instructions)

**Source**: https://github.com/Anbeeld/AGENTS.md

### Key Concepts: Priority System, Evidence Protocol, Uncertainty Protocol

This is the most directly applicable reference — it's a complete, production-grade AGENTS.md designed specifically for LLM coding agents.

### Patterns to Adopt (Directly):

1. **Numbered Priority System**:
   ```
   1. Correctness
   2. Evidence
   3. Safety
   4. Minimal changes
   5. Consistency
   6. Performance
   ```
   If rules conflict, lower-numbered priority wins. This is critical because agents currently have no conflict resolution mechanism.

2. **Boundaries Section** with explicit NEVER directives:
   - NEVER fabricate paths, commits, APIs, config keys, env vars, test results, or capabilities
   - NEVER game verification by weakening assertions
   - NEVER expose secrets
   - NEVER run or suggest destructive commands without confirmation
   - Be direct. Avoid flattery, filler, and agreeing with incorrect premises.

3. **Uncertainty Protocol**:
   - Ask before acting when intent is materially ambiguous
   - Ask before choices that change behavior, API/UX, naming, persistence, auth, dependencies, config, or compatibility
   - Prefer one targeted question. When bundling, ensure each question can be answered independently.
   - Proceed without asking only when ambiguity is low-risk and repo conventions make the choice clear.

4. **Evidence Protocol**:
   - Gather evidence proportional to risk:
     - Trivial low-risk edit: inspect target file + adjacent context
     - Behavioral/API/dependency/infrastructure change: trace execution path, call sites, constraints, regression surface
   - Check local code, imports, config, types, tests before assuming behavior
   - If local dependency or generated code is unreadable, check upstream docs or source before guessing
   - Prefer external verification over self-review
   - State uncertainty when something cannot be confirmed

5. **Subagent Rules**:
   - Use 2+ subagents or none. NEVER launch exactly 1 subagent
   - Main agent is a builder, not a dispatcher. Work first, delegate second
   - Each subagent prompt must specify a concrete return format
   - Keep quick scoping and simple concurrent I/O in the main agent

6. **Testing Requirements**:
   - Preserve existing tests
   - Update tests when behavior changes
   - Do not silently change tested behavior
   - Scope validation proportionally
   - If relevant checks already fail, state that and don't attribute to your work
   - If full validation is impractical, run the narrowest relevant check

7. **Change Constraints**:
   - Do exactly what was asked. Do not expand scope without clear reason
   - Reuse existing abstractions, helpers, dependencies, style, naming, structure, error handling
   - Prefer the smallest viable change
   - Add dependencies only when necessary; choose the smallest viable option

8. **Response Format**:
   - Be concise and specific by default. No filler, intros, or restated requirements
   - Answer direct questions directly when possible

---

## 4. awesome-agent-md (AGENTS.md Template Ecosystem)

**Source**: https://github.com/rentprompts/awesome-agent-md

### Key Concepts: Templates, Real-World Samples, Tooling

An awesome list of AGENTS.md formats, already referenced by 20k+ open-source projects.

### Patterns to Adopt:

1. **Universal Sample Pattern**: Include concrete developer-environment tips (not abstract rules) — e.g., specific commands, paths, and patterns that agents execute.
2. **Testing Instructions**: Explicit testing commands per project, not generic "run tests" instructions.
3. **PR Instructions**: Title format, pre-commit checks, branch naming conventions.
4. **Templates**: The [Advanced Template](https://github.com/rentprompts/awesome-agent-md/blob/main/List/templates/advanced.md) includes safety rules, testing, and style — aligns with Anbeeld's approach.

### Key References:
- OpenAI Codex official AGENTS.md: https://github.com/openai/codex/blob/main/AGENTS.md
- Microsoft POML AGENTS.md: https://github.com/microsoft/poml/blob/main/AGENTS.md

---

## Consolidated Patterns for Adoption

| # | Pattern | Source | Current State in Our AGENTS.md | Action |
|---|---------|--------|-------------------------------|--------|
| 1 | **Priority-based rule system** | Anbeeld | No rule conflict resolution — rules are flat | Add new "## Priorities" section with numbered levels |
| 2 | **Uncertainty protocol** | Anbeeld | Only RULE 11A covers periodic context checks; no general uncertainty handling | Add new "## Uncertainty" section |
| 3 | **Evidence protocol** | Anbeeld | Rules have process guidance but no graduated evidence tiers | Add new "## Evidence" section |
| 4 | **Subagent rules (2+ or none)** | Anbeeld, superx | No subagent guidance at all | Add to "## Conventions" section |
| 5 | **Boundaries (NEVER directives)** | Anbeeld | Some implicit boundaries in RULE 8 (scope creep) and RULE 6 (training scripts); need explicit NEVER directives | Add to "## Conventions" section |
| 6 | **Testing requirements** | Anbeeld | Build & Validation section covers commands but not testing discipline | Enhance "## Build And Validation" section |
| 7 | **Change constraints** | Anbeeld | Scattered across rules; need consolidated guidance | Add to "## Conventions" section |
| 8 | **Response format directive** | Anbeeld, awesome-agent-md | No guidance on conciseness | Add to "## Conventions" section |
| 9 | **Acceptance criteria as gates** | superx, speckit specs | Plans have tasks but acceptance criteria are optional | Enhance plan template guidance |
| 10 | **GEPA ASI pattern** | gepa | No feedback loop for prompt/skill evolution | Add to skill registry or new section |
