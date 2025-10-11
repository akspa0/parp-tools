# Arcane Recall: Commands

## Session Commands
- `/arcane begin` - Start a new session with context loading and initial questions
- `/arcane status` - Display current project context and session history summary
- `/arcane save` - Update session context with latest progress
- `/arcane end` - Complete session with context update and closing questions
- `/arcane tome` - Show this help information

## Context Management
- `/arcane quest <topic>` - Add a new task/topic to the context
- `/arcane lore <info>` - Add important information to project context

## ChunkVault Commands
- `/arcane parse <docfile> --chunks` - Extract chunk definitions from documentation
- `/arcane chunk <ID>` - Load specific chunk details into active context
- `/arcane implement <ID> --status <status>` - Update implementation status
- `/arcane deps <ID>` - Show all dependencies for a given chunk 