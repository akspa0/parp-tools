# Feature Specification: AGENTS.md Prompt Engineering Improvement

**Feature Branch**: `029-agents-md-prompt-improvement`

**Created**: 2026-05-30

**Status**: Draft

**Input**: User description: "Improve AGENTS.md prompting guidance by researching and incorporating patterns from gepa (genetic-pareto prompt optimization), caveman/superx (autonomous multi-agent coding), Anbeeld/AGENTS.md (evidence-driven global instructions), and awesome-agent-md (AGENTS.md template ecosystem). Enhance prompt engineering for plans, specs, and coding agent instructions with priority-based rules, uncertainty protocols, subagent orchestration, and token-efficient patterns."

## User Scenarios & Testing

### User Story 1 - LLM Agent Follows Priority-Based Rule System (Priority: P1)

As an LLM coding agent reading our AGENTS.md, I need clear priority ordering of rules so that when rules conflict, I know which takes precedence without ambiguity.

**Why this priority**: Rule conflicts are the most common source of agent errors. When rules contradict (e.g., "minimal changes" vs "add tests"), the agent currently has no resolution mechanism.

**Independent Test**: Give the agent a task where "keep changes minimal" conflicts with "add tests for changed code." The agent should resolve the conflict by checking rule priority numbers and following the higher-priority rule.

**Acceptance Scenarios**:

1. **Given** the AGENTS.md has numbered priorities (1 = Correctness, 2 = Evidence, etc.), **When** a task involves adding tests that requires more than minimal changes, **Then** the agent should prioritize correctness/evidence over minimal changes.
2. **Given** the AGENTS.md rule priority section, **When** two rules apply to the same situation, **Then** the agent should explicitly reference which priority level governs the decision.

---

### User Story 2 - Agent Handles Uncertainty Explicitly (Priority: P1)

As an LLM coding agent, I need explicit protocols for handling uncertainty so I don't make dangerous assumptions or proceed with ambiguous instructions.

**Why this priority**: Undetected ambiguity is the primary cause of incorrect implementations and wasted iterations. A clear uncertainty protocol prevents agents from guessing when they should ask.

**Independent Test**: Give the agent an intentionally ambiguous instruction ("make it faster"). The agent should identify the ambiguity, ask a targeted clarifying question, and not proceed without resolution.

**Acceptance Scenarios**:

1. **Given** an ambiguous user request, **When** the agent cannot determine intent with high confidence, **Then** the agent should ask one targeted question before proceeding.
2. **Given** a choice that changes behavior, API/UX, naming, persistence, auth, dependencies, config, or compatibility, **When** the choice is not clearly specified, **Then** the agent should ask rather than guess.
3. **Given** a low-risk ambiguity, **When** repo conventions make the choice clear, **Then** the agent should state the assumption briefly and proceed.

---

### User Story 3 - Agent Gathers Evidence Proportional to Risk (Priority: P2)

As a developer relying on agent correctness, I need the agent to gather evidence proportional to the risk of the change, not waste time on trivial edits or skip critical investigation on complex ones.

**Why this priority**: Both over-investigation (slow) and under-investigation (buggy) hurt productivity. A graduated evidence protocol scales effort to risk.

**Independent Test**: Give the agent a trivial typo fix vs a behavioral API change. The first should require only local file inspection; the second should trace execution paths, call sites, and regression surface.

**Acceptance Scenarios**:

1. **Given** a trivial low-risk edit (typo, comment, formatting), **When** the agent starts the task, **Then** it should inspect only the target file and adjacent context.
2. **Given** a behavioral, API, dependency, or infrastructure change, **When** the agent starts the task, **Then** it should trace execution path, call sites, constraints, and regression surface before editing.
3. **Given** unreadable local code, **When** the agent needs to understand it, **Then** it should check upstream docs or source before guessing.

---

### User Story 4 - Agent Uses Subagents Correctly for Parallel Work (Priority: P2)

As a developer, I need the agent to parallelize independent work tracks efficiently without sequentializing parallel-friendly tasks or over-delegating trivial work.

**Why this priority**: Subagent misuse is the #1 performance antipattern. Agents either ignore parallelism entirely (slow) or create too many subagents (wasteful context).

**Independent Test**: Give the agent 3 independent files to read and summarize. The agent should launch 2+ subagents in parallel (or use parallel tool calls), not do them sequentially.

**Acceptance Scenarios**:

1. **Given** 2+ independent tracks of work, **When** each track can complete without the results of the others, **Then** the agent should launch all subagents in the same response.
2. **Given** only 1 clear track of work, **When** the work is dependent or sequential, **Then** the agent should stay in the main agent rather than creating a single subagent.
3. **Given** data already in main-agent context, **When** that data needs formatting or transformation, **Then** the agent should not hand it off to a subagent.

---

### User Story 5 - Plans and Specs Include Explicit Prompt Engineering Guidance (Priority: P3)

As an LLM model generating plans and specs, I need prompt engineering patterns baked into the document structure so downstream agents can execute them effectively.

**Why this priority**: Plans and specs are consumed by other LLM agents. If they lack explicit execution guidance (acceptance criteria, verification steps, dependency ordering), downstream agents produce inconsistent results.

**Independent Test**: Generate a plan using the speckit-plan skill. The output should include explicit acceptance criteria per task, verification steps, and dependency ordering.

**Acceptance Scenarios**:

1. **Given** a plan document with multiple tasks, **When** tasks have dependencies, **Then** the plan should explicitly group tasks into dependency waves.
2. **Given** a task in a plan, **When** the task is defined, **Then** it should include concrete acceptance_criteria (runnable checks, not prose) and verify steps.
3. **Given** a spec document, **When** it describes functional behavior, **Then** it should use clear GIVEN/WHEN/THEN scenario language that agents can parse.

---

### Edge Cases

- What happens when the LLM agent cannot find a matching priority level for a rule conflict?
- How does the agent handle uncertainty when the user is unresponsive?
- What if all subagent tracks depend on each other (no parallelism possible)?
- How does a plan handle a task with acceptance criteria that cannot be verified automatically?
- What if the GEPA optimization suggests a prompt change that contradicts a project rule?

## Requirements

### Functional Requirements

- **FR-001**: AGENTS.md MUST include a numbered priority system where lower numbers win when rules conflict (1 = highest priority).
- **FR-002**: AGENTS.md MUST define explicit boundaries using "NEVER" directives for: fabrication, verification gaming, secret exposure, destructive commands without confirmation.
- **FR-003**: AGENTS.md MUST include an uncertainty protocol specifying when to ask questions vs proceed with assumptions.
- **FR-004**: AGENTS.md MUST specify graduated evidence-gathering rules tied to change risk level.
- **FR-005**: AGENTS.md MUST specify subagent usage rules: launch 2+ or none, never exactly 1.
- **FR-006**: AGENTS.md MUST include testing requirements: preserve existing tests, scope validation proportionally, do not silently change tested behavior.
- **FR-007**: AGENTS.md MUST include change constraint rules: smallest viable change, reuse existing abstractions, do not expand scope without clear reason.
- **FR-008**: Plan documents MUST include dependency wave grouping when tasks have dependencies.
- **FR-009**: Plan task entries MUST include acceptance_criteria as runnable checks and verify steps.
- **FR-010**: Spec documents MUST use GIVEN/WHEN/THEN scenario format for functional behavior.
- **FR-011**: The codex skill registry in AGENTS.md MUST include references to gepa-style prompt optimization patterns for skill evolution.
- **FR-012**: AGENTS.md MUST specify a response format directive: be concise and specific, no filler or restated requirements.

### Key Entities

- **Priority Levels**: Numbered 1-N governing rule conflict resolution, where lower numbers win
- **Risk Levels**: Buckets for change risk (trivial low-risk, behavioral/API/infrastructure) determining evidence depth
- **Uncertainty Protocol**: Decision tree for when to ask vs. proceed with assumptions
- **Subagent Rule**: Constraint on parallel delegation (2+ or none)
- **Acceptance Criteria**: Runnable verification checks gating task completion in plans
- **Dependency Wave**: Grouping of plan tasks by dependency ordering for parallel execution
- **Actionable Side Information (ASI)**: Diagnostic feedback pattern from GEPA for reflective prompt improvement

## Success Criteria

### Measurable Outcomes

- **SC-001**: AGENTS.md rule conflicts are resolved deterministically by agents (measured: 0 ambiguity escalations per session from rule conflicts).
- **SC-002**: Agent task completion time on ambiguous instructions is reduced by 50% compared to guessing-and-correcting cycles.
- **SC-003**: Subagent misuse (exactly 1 subagent launches) drops to 0 occurrences per session.
- **SC-004**: Plan documents generated by speckit-plan include acceptance criteria for 100% of tasks.
- **SC-005**: Spec documents use GIVEN/WHEN/THEN format for 100% of behavioral scenarios.
- **SC-006**: Evidence-gathering overhead on trivial edits is under 3 tool calls.
- **SC-007**: Agents never fabricate paths, commits, APIs, test results, or capabilities.
- **SC-008**: Token efficiency improves by 20%+ through concise response directives and evidence proportionality.

## Assumptions

- The target users for AGENTS.md are LLM coding agents (Claude Code, GitHub Copilot, Codex CLI), not human developers primarily.
- The rule priority system is the primary conflict resolution mechanism; no additional arbitration layer is needed.
- Agents have access to subagent spawning capabilities (the platform supports parallelism).
- GEPA-style reflection is aspirational — we adopt the ASI pattern for skill evolution but not the full GEPA framework immediately.
- The AGENTS.md file will continue to live at the repository root and serve as the primary agent instruction file.
- Existing spec/plan/workflow surface in `.specify/` and `.codex/` will be preserved and augmented, not replaced.
- The Anbeeld/AGENTS.md priority numbering is a reference pattern; exact priority ordering will be tuned to this project's needs.
