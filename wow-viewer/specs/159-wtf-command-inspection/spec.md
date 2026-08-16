# Feature Specification: WTF Command Inspection

**Feature Branch**: `159-wtf-command-inspection`

**Created**: 2026-08-16

**Status**: Draft

**Input**: User description: "using speckit, write a spec to inspect WTF files and look for the sorts we
expect to see. 2.0.0 is the first client with any 'demo'-styled files, not named that way, but they act
as demonstration points in the game world that literally are tied to promotional screenshots they
released for the game, so I know these files exist and are real. I'm just trying to follow in the
breadcrumbs they left behind, and reconstruct the developer view of the game, in my viewer. That's what
this is all about. The wtf files are 'World of Warcraft Text Files'. They act as a surface for scripting
the client/renderer for development purposes."

## Scope Note — read before the user stories

Spec 158's earlier survey of WTF content stopped short in two ways this spec exists to correct. First, it
searched by filename assumption (only the `WTF\` folder, only a couple of builds), and even after being
corrected to search by content it still only read one file's contents in full (`Config.wtf`) and inferred
the rest from filenames. Second, and more importantly, it never checked 2.0.0 — the build the user
identifies directly as the first to carry demonstration-point content, tied to real promotional
screenshots Blizzard released. Spec 158's "no command content found" conclusion is therefore superseded
by this spec, not confirmed by it, and should be treated as open until this feature's sweep actually runs
against real 2.0.0 data.

This spec is deliberately scoped to **inspection only** — reading and classifying what is actually in
WTF files across the full staged client library, in the same spirit as Spec 155's full-corpus asset
sweep: sweep everything, don't hunt for one known instance, and use the known instance (2.0.0's
demonstration-point content) as a sanity check on the results once they exist, never as a precondition
for running the sweep or a reason to scope the sweep down to just that one build. Executing discovered
commands (worldport, teleport, camera placement) is Spec 158's job, once this spec's findings tell it
what real command syntax actually looks like — that work is out of scope here.

WTF ("World of Warcraft Text Files") is understood, per the user's direct correction, as a general
scripting surface the client's command interpreter reads and executes — not a settings-only format
distinct from commands. `SET name "value"` is the most common statement shape, not the only one the
format supports. This spec's job is to find out, from real data, what else is actually there.

**A second, more consequential correction landed after this spec's first draft.** The `.wtf` files this
project needs are not only loose files on disk — Blizzard shipped `.wtf` files packed inside the game's
own data MPQs (`misc.mpq` in 0.5.3, later `interface.mpq` and others), exactly like every other game
asset. Every search up to this point had only walked the loose filesystem, which cannot see archive-packed
content at all — the same blind spot Spec 155 already hit once and fixed for the WMO corpus. Using this
project's existing archive-catalog tooling (the same layer Spec 155's `AssetReferenceSweeper` already
uses) instead of the filesystem, `WTF\DefaultBindings.wtf` was found packed inside 0.5.3.3368's archives
and read directly:

```text
bind ALT-P TOGGLEPERFORMANCEDISPLAY
bind ALT-O TOGGLEPERFORMANCEVALUES
bind CTRL-P RESETPERFORMANCEVALUES
bind CTRL-R TOGGLEFPS
bind CTRL-Y TOGGLESTATS
bind CTRL-Q TOGGLETRIS
bind CTRL-W TOGGLEPORTALS
bind CTRL-E TOGGLECOLLISION
bind CTRL-T TOGGLECOLLISIONDISPLAY
bind ALT-B TOGGLEPLAYERBOUNDS
```

This is real, measured, direct confirmation of the exact Alt+P binding Spec 158 assumed from memory
(`TOGGLEPERFORMANCEDISPLAY`), plus a whole family of real dev-facing debug toggles (FPS, stats, triangle
count, portal visualization, collision, player bounds) neither spec had accounted for. `bind` statements
are a second real statement shape alongside `SET` — confirming the "general scripting surface" framing
above from actual data, not just the user's description of it.

The same check repeated identically against 2.0.0.5610 (using the corrected archive-based method) found
only the same `WTF\DefaultBindings.wtf` and `realmlist.wtf` — nothing else *named* in that build's
catalogued internal listfile. This does **not** mean the demonstration-point content isn't there. It means
a name-filtered search of an archive's already-catalogued listfile cannot find a file whose name was never
catalogued in the first place — structurally the identical trap Spec 155 solved once already (the 0.5.3
WMO corpus reporting "1 of 532" from its internal listfile alone). The user described the files as "not
named that way" — a non-obvious name is exactly the case a listfile-name search is blind to. Finding an
unlisted archive entry needs a different technique than listing what's already named, and that gap is
this spec's real remaining work, not a reason to conclude the file doesn't exist.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - See every distinct kind of line real WTF files actually contain (Priority: P1)

A user runs a sweep across a staged client's entire file tree and gets back, for every `.wtf` file found,
a classification of every line in it: recognized `SET name "value"` statements versus everything else,
with the "everything else" lines shown verbatim — not summarized, not counted only — so real command
syntax that has never been read by this project becomes visible for the first time.

**Why this priority**: This is the entire point of the feature. Without seeing real unrecognized lines
verbatim, nobody can tell what a worldport or teleport command (or anything else) actually looks like in
practice, and Spec 158's command-execution work would be building against guessed syntax instead of real
syntax.

**Independent Test**: Run the sweep against a build already known to be SET-only (0.5.3.3368 or
1.0.0.3980, per Spec 158's earlier reading of `Config.wtf`); confirm the report shows 100% recognized
lines for those files, zero unrecognized. Run it against 2.0.0.5610; confirm the report surfaces whatever
is actually there, whether that turns out to be more SET-only content, real command syntax, or something
else entirely — the tool's job is done once the real content is visible, regardless of which of those
three it turns out to be.

**Acceptance Scenarios**:

1. **Given** a `.wtf` file containing only `SET name "value"` lines, **When** it is swept, **Then** every
   line is classified as a recognized SET statement and none are reported as unrecognized.
2. **Given** a `.wtf` file containing a line that does not match the `SET name "value"` pattern, **When**
   it is swept, **Then** that line is classified as unrecognized and its exact original text is included
   in the report, not paraphrased or dropped.
3. **Given** a build's `.wtf` files are swept, **When** the report is produced, **Then** it states, per
   file and in aggregate, how many lines were recognized versus unrecognized — a build cannot show as
   "clean" without that being an explicit, checkable count.
4. **Given** the same sweep is run twice against the same build, **When** compared, **Then** the results
   are identical — this is a read-only survey with no side effects on the source files or the build.

---

### User Story 2 - Sweep archive-packed WTF files, not just loose ones (Priority: P1)

A user runs the sweep against a build and every `.wtf` file the build's data actually contains is found
and included — whether it sits loose on disk or is packed inside one of the build's own data archives.
(`WTF\DefaultBindings.wtf` is packed inside 0.5.3.3368's archives and does not exist as a loose file
anywhere in that build's staged directory at all — a filesystem-only search cannot see it, and did not,
across two earlier passes at this spec.)

**Why this priority**: Tied with Story 1 as foundational, and the single most consequential lesson learned
while drafting this spec — every WTF search up to this point found nothing until an archive-based search
was actually run, because real content genuinely is not all loose. A sweep that only walks the filesystem
would miss content already confirmed to exist, in the very first build checked this way.

**Independent Test**: Run the sweep against 0.5.3.3368; confirm `WTF\DefaultBindings.wtf` appears in the
results with its real content (including its `bind ALT-P TOGGLEPERFORMANCEDISPLAY` line) correctly read
and classified, even though the file does not exist as a loose file anywhere in that build's staged
directory tree.

**Acceptance Scenarios**:

1. **Given** a build whose `.wtf` content is packed inside its data archives, **When** it is swept,
   **Then** those files are found and read, not only files loose on disk.
2. **Given** a `.wtf` file that exists loose (e.g. a user-generated `Config.wtf`) and one that only exists
   packed inside an archive (e.g. `DefaultBindings.wtf`), **When** the build is swept, **Then** both are
   reported, each correctly attributed to where it actually came from.
3. **Given** a build whose file tree the sweep has not been told to expect any particular structure for,
   **When** it is swept, **Then** it still finds every `.wtf` file present — the sweep does not assume a
   fixed folder layout.

---

### User Story 3 - Sweep every staged build, with 2.0.0 as a sanity check, not a gate (Priority: P2)

A user runs the sweep across every staged client in the library — not only 2.0.0 — and gets one
comparable report per build. The build the user identifies as most likely to show real findings (2.0.0)
is where the results get the closest look first, but every other build is swept identically and reported
identically, so an unexpected finding elsewhere is not missed just because it wasn't expected there.

**Why this priority**: Mirrors this project's own established discipline (Spec 155): a full,
un-gated sweep is how a real, previously-unknown finding gets found — narrowing the sweep to only the
build a person expects the answer to already be in is exactly the mistake that discipline exists to
prevent, even when — especially when — that expectation comes from someone who knows the data well.

**Independent Test**: Run the sweep across all staged builds in one pass; confirm every build produces a
report, including builds with no unrecognized lines at all (that is itself a reportable, comparable
result, not an omission).

**Acceptance Scenarios**:

1. **Given** the full staged client library, **When** the sweep is run across all of it, **Then** every
   build produces a report, whether or not it contains any unrecognized lines.
2. **Given** 2.0.0.5610 is swept alongside every other build, **When** results are reviewed, **Then**
   2.0.0's findings are compared against the earlier (now superseded) conclusion drawn only from
   0.5.3.3368/1.0.0.3980, so the correction is visible and evidenced, not just asserted.
3. **Given** a build the sweep cannot fully read (a locked file, a permissions error), **When** that
   happens, **Then** it is reported as unreadable, not silently reported as "no unrecognized lines found"
   — the same could-not-check-versus-nothing-found distinction this project already applies elsewhere.

---

### User Story 4 - Test a candidate name against a build's archives directly (Priority: P2)

A user supplies one or more candidate file names — a guess at what a piece of content might actually be
called — and finds out immediately whether that name resolves to real content inside a build's archives,
independent of whether that name appears in any catalogued listfile at all.

**Why this priority**: The real, open gap this spec's own research surfaced, not a hypothetical one:
2.0.0.5610's catalogued internal listfile names only `DefaultBindings.wtf` and `realmlist.wtf`, yet the
user is confident real demonstration-point content exists there under some other, unknown name. A sweep
that only reports what a listfile already names would reproduce, for unnamed content, the exact "1 of
532" listfile-incompleteness trap Spec 155 already hit and fixed for WMO — except here the content's name
isn't merely under-catalogued, it may not be catalogued anywhere, so listing known files can never surface
it no matter how completely the corpus is enumerated. Testing a specific candidate name is a different
capability from sweeping known files, and does not block Stories 1–3.

**Independent Test**: Supply a candidate name already confirmed present (`WTF\DefaultBindings.wtf`) and
confirm it resolves. Supply a candidate name confirmed absent (a made-up name); confirm it correctly
reports as not found rather than a false positive. This is testable today without knowing any real
unlisted name in advance.

**Acceptance Scenarios**:

1. **Given** a candidate name that matches real content in a build's archives, **When** it is tested,
   **Then** the content is read and classified exactly as if it had been found through the normal sweep.
2. **Given** a candidate name that does not match anything in a build's archives, **When** it is tested,
   **Then** it is reported as not found — never a false match.
3. **Given** a list of many candidate names, **When** they are tested against a build, **Then** each
   result is reported individually, so a large batch of guesses (e.g. zone names, promotional-screenshot
   locations) can be worked through without hand-testing one at a time.

---

### Edge Cases

- A `.wtf` file that is empty, or contains only blank lines: reported as zero recognized and zero
  unrecognized lines, not treated as an error.
- A line that looks almost like a `SET` statement but is malformed (missing a closing quote, wrong
  argument count): classified as unrecognized rather than guessed into the recognized bucket — a near
  miss is still a real finding worth seeing verbatim, not silently normalized away.
- The same unrecognized line shape appears thousands of times across many files in one build: the report
  MUST make the distinct shapes visible without forcing a reader to scroll through every literal
  repetition — deduplication for readability must never come at the cost of hiding a shape that only
  appears once.
- A `.wtf` file encoded differently than plain ASCII/UTF-8 (some legacy client files use other
  encodings): reported as unreadable-with-reason rather than silently misread as garbage recognized lines.
- A build whose entire `.wtf` surface is confirmed empty of any file at all: reported as zero files found
  for that build, distinct from a build that has files but none contain unrecognized content.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST enumerate every `.wtf` file a build actually contains — loose on disk or packed
  inside the build's own data archives — not scoped to any assumed subfolder location and not scoped to
  the filesystem alone.
- **FR-002**: System MUST classify every non-blank line of every found file as a recognized
  `SET <name> "<value>"` statement, a recognized `bind <KEY> <ACTION>` statement, or an unrecognized line.
- **FR-003**: System MUST report unrecognized lines with their exact original text, not a summary,
  truncation, or paraphrase.
- **FR-004**: System MUST report, per file and in aggregate per build, the count of recognized versus
  unrecognized lines.
- **FR-005**: System MUST sweep every staged build identically — no build is skipped, scoped down, or
  given different treatment because a particular build is expected to be the interesting one.
- **FR-006**: System MUST distinguish a build/file that could not be read from a build/file that was read
  and found to contain only recognized lines — these are different facts and must be reported differently.
- **FR-007**: The sweep MUST be read-only — it MUST NOT modify, move, or delete any file it reads.
- **FR-008**: When the same unrecognized line shape recurs many times, System MUST make each distinct
  shape visible without requiring a reader to scroll through every repetition, while never suppressing a
  shape that appears only once.
- **FR-009**: System MUST let a user test one or more candidate file names directly against a build's
  archives, independent of whether that name is named in any catalogued listfile, and report whether each
  candidate resolves to real content.
- **FR-010**: A candidate-name test that resolves MUST be read and classified through the same line
  classification as FR-002, not reported as a bare yes/no.

### Key Entities

- **WTF Line**: one non-blank line from a WTF file — its exact original text, and its classification
  (recognized SET statement with parsed name/value, or unrecognized).
- **WTF File Survey**: one file's results — its real path (loose disk path or archive-internal virtual
  path), which build it came from, whether it was found via the normal sweep or a candidate-name test, and
  its lines' classifications.
- **Build WTF Survey**: one build's aggregated results — every WTF file found across both loose and
  archive-packed sources, combined recognized/unrecognized counts, and the distinct unrecognized line
  shapes seen, deduplicated for readability without losing any shape that appears only once.
- **Candidate Name Probe**: one user-supplied guess at a file's real name, the build it was tested
  against, and whether it resolved to real content.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A sweep of 0.5.3.3368 finds `WTF\DefaultBindings.wtf` and correctly classifies its real
  `bind`-statement content, even though that file exists only packed inside the build's archives and not
  as a loose file anywhere in the staged directory tree.
- **SC-002**: Every non-blank line in every found file is classified; none are silently dropped from the
  report.
- **SC-003**: Unrecognized lines appear in the report with their real, exact text — a reader can see
  actual candidate command syntax, not just a count.
- **SC-004**: The sweep runs against every staged build in one pass, including 2.0.0.5610 and the
  already-checked 0.5.3.3368/1.0.0.3980, with directly comparable reports across all of them.
- **SC-005**: 2.0.0.5610's real findings are recorded and explicitly compared against Spec 158's earlier,
  now-superseded conclusion — the correction is evidenced in the record, not just stated.
- **SC-006**: A user can test a specific candidate file name against a build's archives and get a real
  resolved-or-not answer, without that name needing to already appear in any catalogued listfile.

## Assumptions

- This spec covers reading and classifying WTF content only. Executing any recognized command (worldport,
  teleport, or anything else discovered) remains Spec 158's scope, informed by this spec's real findings
  rather than assumed syntax.
- The recognized-statement grammar covers exactly two shapes confirmed from real data: `SET name "value"`
  and `bind KEY ACTION` (confirmed via `WTF\DefaultBindings.wtf`'s real content, including the `bind ALT-P
  TOGGLEPERFORMANCEDISPLAY` line). It is deliberately not broader than that, so this tool's job stays
  "reliably tell recognized from not," not "guess at every possible command's grammar up front." A wider
  grammar can be added once more real unrecognized-line shapes are actually seen.
- No file is skipped because its build is outside this project's supported pre-4.0.0 era — the sweep
  itself is inspection, not the command-execution work Spec 158 gates at 4.0.0; seeing what a later build
  contains is informative even where executing it would not be in scope.
- This spec does not require or assume that 2.0.0's demonstration-point content is stored as literal
  `.wtf` files with recognizable command syntax — it is entirely possible the sweep finds nothing new
  there either, in which case that is itself the real, reportable finding, not a failure of the tool.
