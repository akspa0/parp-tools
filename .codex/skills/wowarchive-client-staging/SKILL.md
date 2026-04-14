# WoWArchive Client Staging

## When To Use

- The task depends on game clients that live in `G:\WoW\WoWArchive-0.X-3.X`.
- You need to mount the deduplicated archive with `MountAll.bat` before accessing client files.
- You are planning or running multi-build export or audit or inspect or training-prep work against archive-backed clients.
- You need to choose whether to read directly from the mounted archive or stage a local working copy first.
- You need to prune old staged client copies so the temp area does not grow without bound.

## Read First

1. `gillijimproject_refactor/memory-bank/data-paths.md`
2. `gillijimproject_refactor/memory-bank/activeContext.md`
3. `gillijimproject_refactor/memory-bank/progress.md`
4. `G:\WoW\WoWArchive-0.X-3.X\Readme.txt`
5. `G:\WoW\WoWArchive-0.X-3.X\MountAll.bat`
6. `AGENTS.md`

## Canonical Paths

- Archive root: `G:\WoW\WoWArchive-0.X-3.X`
- Archive readme: `G:\WoW\WoWArchive-0.X-3.X\Readme.txt`
- Mount script: `G:\WoW\WoWArchive-0.X-3.X\MountAll.bat`
- Current mount target from the batch file: `G:\WoW\WoWArchive-0.X-3.X\Mount`
- Default workspace staging root: `i:/parp/parp-tools/output/tmp/wowarchive-clients`

## Procedure

1. Confirm whether a fixed local client root already exists.
2. Mount the archive only when needed.
3. Stage the required client locally before heavy reads.
4. Keep the staging root small.
5. Keep validation notes precise.

## Guardrails

- Do not point heavy multi-map or multi-client processing directly at the mounted archive when a local staged copy is practical.
- Do not copy the whole archive just because a single build is needed; stage only the required client roots.
- Do not leave old staged client trees behind once they are no longer needed.
- Do not describe the mounted archive as fast storage; the point of the workflow is that staged local copies are materially faster.