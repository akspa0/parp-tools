---
name: speckit-analyze
description: Analyze a feature spec for completeness, risks, and gaps before planning. Use when the user says "analyze", "$speckit-analyze", or wants to stress-test a spec before committing to it.
---

# Speckit Analyze

Analyze a feature specification for completeness, risks, ambiguities, and gaps.

## When to Use

- User says "analyze", "$speckit-analyze", or "review the spec"
- After writing a spec, before generating the plan
- When the user wants a second opinion on a spec's quality

## Analysis Dimensions

### 1. Completeness Check

For each user story:
- Is the priority justified?
- Are acceptance scenarios specific enough to test?
- Are edge cases covered?
- Are error states defined?

### 2. Dependency Check

For each requirement:
- Does it depend on code that doesn't exist yet?
- Does it depend on another requirement being complete first?
- Does it depend on external systems or data?

### 3. Risk Assessment

For each phase:
- What could go wrong?
- What assumptions are we making?
- What is the blast radius if this breaks?
- Is there a simpler approach?

### 4. Constitution Compliance

Check every requirement against `wow-viewer/.specify/memory/constitution.md`:
- Repo independence: any cross-repo references?
- Library-first: does the design put logic in the right layer?
- Real-data validation: does every claim have a validation path?
- No game client path assumptions: any `H:\CLIENTS` references?

### 5. Gap Analysis

Compare the spec against:
- Existing code in `wow-viewer/src/`, `wow-viewer/tools/`
- Existing specs in `wow-viewer/docs/architecture/`
- The audit findings in `wow-viewer/docs/architecture/speckit-doc-audit-*.md`

Identify:
- Duplicated effort with existing features
- Missing integration points
- Stale assumptions from older docs

## Output

Write the analysis as a brief report (or discuss with the user):

```
## Spec Analysis: <feature name>

### Completeness
- [findings]

### Dependencies
- [findings]

### Risks
- [findings]

### Constitution Compliance
- [findings]

### Gaps
- [findings]

### Recommendation
[approve / revise / reject]
```
