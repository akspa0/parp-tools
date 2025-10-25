# Product Context

## Users
- Technical artists and engineers maintaining Alpha-to-Lich King world data conversions.

## Problem
- RoundTrip CLI should validate byte-for-byte parity when converting Alpha ADTs through LK build pipelines, but current runs zero out `MCLY`/`MCAL` data, blocking verification.

## Desired Experience
- Engineers run automated tests that confirm extracts retain original Alpha chunk bytes before rebuilding LK ADTs.
- RoundTrip CLI reports clean parity, enabling confident refactors without manual diffing.
