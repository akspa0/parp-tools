# Alpha-Core SQL Scene Liveness Prompt

Use this prompt in a fresh planning chat when the goal is to use more of the Alpha-Core SQL payload for faithful NPC/gameobject presentation and possible scene liveness inside the active viewer.

## Prompt

Design a concrete implementation plan for expanding Alpha-Core SQL usage in `gillijimproject_refactor/src/MdxViewer` so the viewer can display SQL-driven NPCs and gameobjects more faithfully and evaluate future liveness features.

The plan must assume:

- the active viewer already supports optional SQL-driven world spawn injection from Alpha-Core
- current NPC presentation is too blunt, with all possible equipment/options appearing on character models instead of the correct set being respected
- the SQL payload likely contains more useful data than the current viewer consumes
- the user wants to explore animation-state handling, richer spawn fidelity, and eventually NPC pathing if the necessary data exists
- any pathing/system-liveness work must be grounded in real available data, not hand-wavy server fantasies

## What The Plan Must Produce

1. A first-pass audit plan for what Alpha-Core SQL and related DBC data actually provide.
2. A fidelity-improvement plan for NPCs and gameobjects.
3. A concrete proposal for equipment / option-set correctness.
4. A proposal for animation-state handling when the data supports it.
5. A staged pathing/liveness roadmap, including what should stay out of the first slice.
6. A performance risk analysis, since more live actors will worsen the current viewer bottlenecks if budgets stay loose.
7. A real-data validation plan.

## Required Constraints

- do not assume SQL contains pathing unless that is explicitly verified
- keep equipment correctness, animation-state correctness, and pathing as separate seams
- do not propose a fake always-moving NPC world just to make the viewer look lively
- if server-like behavior is proposed, identify the minimum local runtime subset needed instead of saying “build our own server” with no boundaries
- if PM4 or WMO data is proposed for navigation/pathing, treat that as a later research seam and call out what is actually known today versus guessed
- make the first slice about correctness of displayed actors before motion choreography

## Questions The Plan Must Answer

1. Which Alpha-Core SQL tables are the likely sources for spawn identity, display identity, equipment/options, and animation hints?
2. What does the current viewer already ingest, and what does it ignore?
3. What is the smallest slice that proves “SQL actor fidelity improved” without needing server emulation?
4. What kinds of pathing data might already exist, and what kinds are only hypothetical?
5. If pathing data does not exist, what is the most defensible later fallback: static idles, looped splines, PM4-informed navigation, or something else?
6. What hard performance budgets need to exist before large numbers of animated SQL actors are allowed in a loaded world?

## Suggested Deliverable Structure

1. Available data inventory
2. Fidelity-first implementation slice
3. Animation-state follow-up
4. Pathing/liveness roadmap
5. Performance and renderer constraints
6. Validation plan
7. Explicit non-goals

## Validation Rules

- do not describe SQL actor fidelity as correct without real-data runtime inspection on the fixed development dataset
- do not describe pathing as supported unless the data source and runtime behavior are both identified and implemented
- build-only validation is not enough for actor-correctness claims