# PROGRESS — V11

## POSITION
- V10 pipeline: dead. Two-stage nonsense with broken archive path.
- V11: v9 proven pipeline + ConvNeXt backbone + MCAL/MCLY multi-task + proper training infra.

## VALIDATED
| What | Status |
|------|--------|
| MpqArchiveCatalog probe fix | DONE — break→continue, 256 limit |
| MCAL/MCLY in v9 BuildDirectCacheEntry | DONE |
| V11 model forward pass | DONE — 35.5M, 6 output heads |
| V11 training loop (GPU) | DONE — AMP, EMA, cosine, uncertainty loss |
| V11 inference + OBJ export | DONE |
| Channel layout audit | DONE — 26ch, shadow removed, MCCV at 3x dropout |
| Cache memory (LRU 2GB cap) | DONE |
| Zero-samples / empty vocab guards | DONE |

## NOT YET
- Full extraction run on 6 game clients (800-1500 shards)
- Production training run (300 epochs)
- Model evaluation on held-out tiles
- OBJ reconstruction quality check

## KNOWN BOUNDARIES
- v9 `dataset-scan` expects `Data/World/Maps/<map>` directory structure for filesystem mode
- Alpha clients (0.5.3) need monolithic WDT handling in scan
- No MCAL/MCLY on alpha-era ADTs (pre-3.x format lacks split texture files)
