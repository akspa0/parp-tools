# GV-13 Cross-Root Clipboard Package

## Intent

Design a transport format for moving assets or world selections between roots and sessions.

## Scope

- package manifest
- payload references vs embedded payloads
- source profile metadata
- target compatibility notes

## Outputs

- `InteropClipboardPackage`
- copy/paste package rules
- safe failure rules for unsupported targets

## Dependencies

- GV-10, GV-11, GV-12

## Proof

- one bounded source-to-target transfer uses the package contract

## Non-Goals

- no live collaboration protocol
