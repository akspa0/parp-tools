# Gillijim Project Refactor - 0.5.3 Integrated Server Fork

> **Note**: This is a specialized fork of the Gillijim Project / Alpha Core. The original documentation is available in [README_PARP_TOOLS.md](README_PARP_TOOLS.md).

## 🌍 The Mission: A True 0.5.3 Integrated Server

This fork aims to create a **complete, standalone server implementation** for the World of Warcraft 0.5.3 Alpha client.

Unlike the mainline project, which focuses on strict preservation and relies on external tools/workflows to patch data (often locking down features behind custom backends), this fork prioritizes **native server functionality**. We believe the core itself should handle debug, GM, and interaction commands natively, just as a fully integrated server should.

## 🚀 Key Features & Divergence

### 1. Native GM & Debug Ops
We have restored and implemented critical debug opcodes directly in the `alpha-core` Python server, removing the need for client-side hacks or external cheat engines.

*   **`CMSG_DBLOOKUP` (0x02)**: Native item/quest/creature search.
*   **`CMSG_GM_INVIS` (0x1D7)**: True server-side GM invisibility (flags update).
*   **`CMSG_GHOST` (0x1D6)**: Toggle ghost mode.
*   **`CMSG_GM_NUKE` (0x1EB)**: "Deathtouch" functionality.
*   **`MSG_GM_BIND_OTHER` (0x1D9)**: Bind other players to locations.
*   **Quest Debug**: Force accept/complete/clear quests via opcode handlers.

### 2. Integrated Security
All restored commands are protected by robust server-side GM permission checks (`is_gm()`), ensuring a secure environment without relying on obscure external auth layers.

### 3. Mod Preservation (`alpha-core-mods`)
To preserve the specific enhancements made in this fork and prevent them from being obscured by gitignore rules or submodules, all modified and new core files are mirrored in the `alpha-core-mods/` directory.

## 📂 Repository Structure

*   **`lib/alpha-core/`**: The main server codebase.
*   **`alpha-core-mods/`**: Contains the specific modifications and additions made by this fork (Handlers, Managers, Tools).
*   **`tools/`**: Utility scripts (see [README_PARP_TOOLS.md](README_PARP_TOOLS.md)).

## 🛠️ Getting Started

Follow the setup instructions in [lib/alpha-core/README.md](lib/alpha-core/README.md) to configure the server, but enjoy the enhanced functionality out of the box.

### Quick Command Reference
| Command | Opcode | Function |
| :--- | :--- | :--- |
| **.lookup [item/quest/creature] [name]** | 0x02 | Search database |
| **.invis** | 0x1D7 | Toggle GM Invisibility |
| **.ghost** | 0x1D6 | Toggle Ghost Mode |
| **.nuke** | 0x1EB | Kill target instantly |
| **.bind [target]** | 0x1D9 | Bind target to their current location |

---
*For the original preservation/mapping tool documentation, please see [README_PARP_TOOLS.md](README_PARP_TOOLS.md).*
