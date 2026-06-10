# 0.5.3 Opcode Behavior Reference

This reference catalogs every inbound opcode handled by the current 0.5.3 server dispatch map and describes expected gameplay/engine effects for tooling-network implementation.

## Method
- Source-of-truth inbound map: `alpha-core-mods/game/world/opcode_handling/Definitions.py`
- Enumerated opcode count: **222**
- Outbound server-emitted `SMSG_*` inventory artifact: `network_server_smsg_inventory.json`
- Confidence model: `High`=verified in mirrored handler code; `Medium-High`=strong handler-name+flow evidence; `Medium`=dispatch-only inference

## Critical Flow Anchors (Verified)
- Auth bootstrap: `CMSG_AUTH_SRP6_BEGIN` -> `CMSG_AUTH_SRP6_PROOF` -> `CMSG_AUTH_SESSION`
- Character entry: `CMSG_CHAR_ENUM` -> `CMSG_PLAYER_LOGIN`
- Transfer: server emits `SMSG_TRANSFER_PENDING` then `SMSG_NEW_WORLD`; client acknowledges with `MSG_MOVE_WORLDPORT_ACK`
- Movement steady state: `MSG_MOVE_HEARTBEAT` and other `MSG_MOVE_*` status packets
- Quest interaction core: status/hello/query/accept/complete/reward opcodes

## AuthSession
| Opcode | Handler | What It Does | Engine Integration | Confidence |
|---|---|---|---|---|
| `CMSG_AUTH_SESSION` | `AuthSessionHandler.handle` | Binds authenticated account to world session and enables gameplay opcode handling. | Session/auth manager, account state, character bootstrap | High |
| `CMSG_AUTH_SRP6_BEGIN` | `AuthSessionHandler.handle_srp6_begin` | Starts SRP6 auth exchange (step 1); server prepares challenge state. | Session/auth manager, account state, character bootstrap | High |
| `CMSG_AUTH_SRP6_PROOF` | `AuthSessionHandler.handle_srp6_proof` | Submits SRP6 proof (step 2); server validates account proof and session eligibility. | Session/auth manager, account state, character bootstrap | High |
| `CMSG_CHAR_CREATE` | `CharCreateHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Session/auth manager, account state, character bootstrap | Medium |
| `CMSG_CHAR_DELETE` | `CharDeleteHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Session/auth manager, account state, character bootstrap | Medium |
| `CMSG_CHAR_ENUM` | `CharEnumHandler.handle` | Requests character list for account; feeds character select UI. | Session/auth manager, account state, character bootstrap | High |
| `CMSG_LOGOUT_CANCEL` | `LogoutCancelHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Session/auth manager, account state, character bootstrap | Medium |
| `CMSG_LOGOUT_REQUEST` | `LogoutRequestHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Session/auth manager, account state, character bootstrap | Medium |
| `CMSG_PLAYER_LOGIN` | `PlayerLoginHandler.handle` | Selects one character and starts world bootstrap/create flow. | Session/auth manager, account state, character bootstrap | High |
| `CMSG_PLAYER_LOGOUT` | `PlayerLogoutHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Session/auth manager, account state, character bootstrap | Medium |
## Channel
| Opcode | Handler | What It Does | Engine Integration | Confidence |
|---|---|---|---|---|
| `CMSG_CHANNEL_ANNOUNCEMENTS` | `ChannelAnnounceHandler.handle` | Channel chat administration request (join/leave/moderation/ownership/password). | Channel/chat manager and moderation state | Medium-High |
| `CMSG_CHANNEL_BAN` | `ChannelBanHandler.handle_ban` | Channel chat administration request (join/leave/moderation/ownership/password). | Channel/chat manager and moderation state | Medium-High |
| `CMSG_CHANNEL_INVITE` | `ChannelInviteHandler.handle` | Channel chat administration request (join/leave/moderation/ownership/password). | Channel/chat manager and moderation state | Medium-High |
| `CMSG_CHANNEL_KICK` | `ChannelKickHandler.handle` | Channel chat administration request (join/leave/moderation/ownership/password). | Channel/chat manager and moderation state | Medium-High |
| `CMSG_CHANNEL_LIST` | `ChannelListHandler.handle` | Channel chat administration request (join/leave/moderation/ownership/password). | Channel/chat manager and moderation state | Medium-High |
| `CMSG_CHANNEL_MODERATE` | `ChannelModeratorHandler.handle_moderate` | Channel chat administration request (join/leave/moderation/ownership/password). | Channel/chat manager and moderation state | Medium-High |
| `CMSG_CHANNEL_MODERATOR` | `ChannelModeratorHandler.handle_add_mod` | Channel chat administration request (join/leave/moderation/ownership/password). | Channel/chat manager and moderation state | Medium-High |
| `CMSG_CHANNEL_MUTE` | `ChannelMuteHandler.handle_mute` | Channel chat administration request (join/leave/moderation/ownership/password). | Channel/chat manager and moderation state | Medium-High |
| `CMSG_CHANNEL_OWNER` | `ChannelOwnerHandler.handle` | Channel chat administration request (join/leave/moderation/ownership/password). | Channel/chat manager and moderation state | Medium-High |
| `CMSG_CHANNEL_PASSWORD` | `ChannelPasswordHandler.handle` | Channel chat administration request (join/leave/moderation/ownership/password). | Channel/chat manager and moderation state | Medium-High |
| `CMSG_CHANNEL_SET_OWNER` | `ChannelOwnerHandler.handle_set_owner` | Channel chat administration request (join/leave/moderation/ownership/password). | Channel/chat manager and moderation state | Medium-High |
| `CMSG_CHANNEL_UNBAN` | `ChannelBanHandler.handle_unban` | Channel chat administration request (join/leave/moderation/ownership/password). | Channel/chat manager and moderation state | Medium-High |
| `CMSG_CHANNEL_UNMUTE` | `ChannelMuteHandler.handle_unmute` | Channel chat administration request (join/leave/moderation/ownership/password). | Channel/chat manager and moderation state | Medium-High |
| `CMSG_JOIN_CHANNEL` | `ChannelJoinHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Channel/chat manager and moderation state | Medium |
| `CMSG_LEAVE_CHANNEL` | `ChannelLeaveHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Channel/chat manager and moderation state | Medium |
## Combat_PvP
| Opcode | Handler | What It Does | Engine Integration | Confidence |
|---|---|---|---|---|
| `CMSG_ATTACKSTOP` | `AttackSwingHandler.handle_stop` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Combat resolution, duel/pvp/resurrection state | Medium |
| `CMSG_ATTACKSWING` | `AttackSwingHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Combat resolution, duel/pvp/resurrection state | Medium |
| `CMSG_DUEL_ACCEPTED` | `DuelAcceptHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Combat resolution, duel/pvp/resurrection state | Medium |
| `CMSG_DUEL_CANCELLED` | `DuelCanceledHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Combat resolution, duel/pvp/resurrection state | Medium |
| `CMSG_MAKEMONSTERATTACKME` | `MakeMonsterAttackMeHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Combat resolution, duel/pvp/resurrection state | Medium |
| `CMSG_PVP_PORT` | `PvPPortHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Combat resolution, duel/pvp/resurrection state | Medium |
| `CMSG_RECLAIM_CORPSE` | `ResurrectResponseHandler.handle_reclaim_corpse` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Combat resolution, duel/pvp/resurrection state | Medium |
| `CMSG_RESURRECT_RESPONSE` | `ResurrectResponseHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Combat resolution, duel/pvp/resurrection state | Medium |
| `MSG_RANDOM_ROLL` | `RandomRollHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Combat resolution, duel/pvp/resurrection state | Medium |
## Friends
| Opcode | Handler | What It Does | Engine Integration | Confidence |
|---|---|---|---|---|
| `CMSG_ADD_FRIEND` | `FriendAddHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Social manager and friend/ignore persistence | Medium |
| `CMSG_ADD_IGNORE` | `FriendIgnoreHandler.handle` | Social roster mutation/query for friends/ignore lists. | Social manager and friend/ignore persistence | Medium-High |
| `CMSG_DEL_FRIEND` | `FriendDeleteHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Social manager and friend/ignore persistence | Medium |
| `CMSG_DEL_IGNORE` | `FriendDeleteIgnoreHandler.handle` | Social roster mutation/query for friends/ignore lists. | Social manager and friend/ignore persistence | Medium-High |
| `CMSG_FRIEND_LIST` | `FriendsListHandler.handle` | Social roster mutation/query for friends/ignore lists. | Social manager and friend/ignore persistence | Medium-High |
## GM_Debug
| Opcode | Handler | What It Does | Engine Integration | Confidence |
|---|---|---|---|---|
| `CMSG_BEASTMASTER` | `CheatBeastMasterHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | GM permission checks, debug tooling, anticheat logs | Medium |
| `CMSG_CHEAT_SETMONEY` | `CheatSetMoneyHandler.handle` | GM/debug control opcode guarded by permission checks and anticheat logging. | GM permission checks, debug tooling, anticheat logs | Medium-High |
| `CMSG_COOLDOWN_CHEAT` | `CooldownCheatHandler.handle` | GM/debug control opcode guarded by permission checks and anticheat logging. | GM permission checks, debug tooling, anticheat logs | Medium-High |
| `CMSG_DBLOOKUP` | `DBLookupHandler.handle` | GM-only database search by text (item/quest/creature) returning SMSG_DBLOOKUP entries. | GM permission checks, debug tooling, anticheat logs | High |
| `CMSG_DEBUG_AISTATE` | `DebugAIStateHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | GM permission checks, debug tooling, anticheat logs | Medium |
| `CMSG_GM_INVIS` | `InvisHandler.handle` | GM/debug control opcode guarded by permission checks and anticheat logging. | GM permission checks, debug tooling, anticheat logs | Medium-High |
| `CMSG_GM_NUKE` | `NukeHandler.handle` | GM/debug control opcode guarded by permission checks and anticheat logging. | GM permission checks, debug tooling, anticheat logs | Medium-High |
| `CMSG_GODMODE` | `GodModeHandler.handle` | GM/debug control opcode guarded by permission checks and anticheat logging. | GM permission checks, debug tooling, anticheat logs | Medium-High |
| `CMSG_LEVELUP_CHEAT` | `LevelUpCheatHandler.handle` | GM/debug control opcode guarded by permission checks and anticheat logging. | GM permission checks, debug tooling, anticheat logs | Medium-High |
| `CMSG_LEVEL_CHEAT` | `LevelCheatHandler.handle` | GM/debug control opcode guarded by permission checks and anticheat logging. | GM permission checks, debug tooling, anticheat logs | Medium-High |
| `CMSG_TRIGGER_CINEMATIC_CHEAT` | `TriggerCinematicCheatHandler.handle` | GM/debug control opcode guarded by permission checks and anticheat logging. | GM permission checks, debug tooling, anticheat logs | Medium-High |
| `MSG_GM_BIND_OTHER` | `BindOtherHandler.handle` | GM/debug control opcode guarded by permission checks and anticheat logging. | GM permission checks, debug tooling, anticheat logs | Medium-High |
| `MSG_GM_SHOWLABEL` | `ShowLabelHandler.handle` | GM/debug control opcode guarded by permission checks and anticheat logging. | GM permission checks, debug tooling, anticheat logs | Medium-High |
| `MSG_GM_SUMMON` | `GMSummonHandler.handle` | GM/debug control opcode guarded by permission checks and anticheat logging. | GM permission checks, debug tooling, anticheat logs | Medium-High |
## Group
| Opcode | Handler | What It Does | Engine Integration | Confidence |
|---|---|---|---|---|
| `CMSG_GROUP_ACCEPT` | `GroupInviteAcceptHandler.handle` | Party/group lifecycle request (invite, accept, leadership, loot method). | Group manager and party state propagation | Medium-High |
| `CMSG_GROUP_DECLINE` | `GroupInviteDeclineHandler.handle` | Party/group lifecycle request (invite, accept, leadership, loot method). | Group manager and party state propagation | Medium-High |
| `CMSG_GROUP_DISBAND` | `GroupDisbandHandler.handle` | Party/group lifecycle request (invite, accept, leadership, loot method). | Group manager and party state propagation | Medium-High |
| `CMSG_GROUP_INVITE` | `GroupInviteHandler.handle` | Party/group lifecycle request (invite, accept, leadership, loot method). | Group manager and party state propagation | Medium-High |
| `CMSG_GROUP_SET_LEADER` | `GroupSetLeaderHandler.handle` | Party/group lifecycle request (invite, accept, leadership, loot method). | Group manager and party state propagation | Medium-High |
| `CMSG_GROUP_UNINVITE` | `GroupUnInviteHandler.handle` | Party/group lifecycle request (invite, accept, leadership, loot method). | Group manager and party state propagation | Medium-High |
| `CMSG_GROUP_UNINVITE_GUID` | `GroupUnInviteGuidHandler.handle` | Party/group lifecycle request (invite, accept, leadership, loot method). | Group manager and party state propagation | Medium-High |
| `CMSG_LOOT_METHOD` | `GroupLootMethodHandler.handle` | Loot interaction request controlling loot open/take/release lifecycle. | Group manager and party state propagation | Medium-High |
| `CMSG_SET_LOOKING_FOR_GROUP` | `LookingForGroupHandler.handle_set` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Group manager and party state propagation | Medium |
| `MSG_LOOKING_FOR_GROUP` | `LookingForGroupHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Group manager and party state propagation | Medium |
## Guild
| Opcode | Handler | What It Does | Engine Integration | Confidence |
|---|---|---|---|---|
| `CMSG_GUILD_ACCEPT` | `GuildInviteAcceptHandler.handle` | Guild management request (membership/roles/motd/roster lifecycle). | Guild manager and membership persistence | Medium-High |
| `CMSG_GUILD_CREATE` | `GuildCreateHandler.handle` | Guild management request (membership/roles/motd/roster lifecycle). | Guild manager and membership persistence | Medium-High |
| `CMSG_GUILD_DECLINE` | `GuildInviteDeclineHandler.handle` | Guild management request (membership/roles/motd/roster lifecycle). | Guild manager and membership persistence | Medium-High |
| `CMSG_GUILD_DEMOTE` | `GuildDemoteHandler.handle` | Guild management request (membership/roles/motd/roster lifecycle). | Guild manager and membership persistence | Medium-High |
| `CMSG_GUILD_DISBAND` | `GuildDisbandHandler.handle` | Guild management request (membership/roles/motd/roster lifecycle). | Guild manager and membership persistence | Medium-High |
| `CMSG_GUILD_INFO` | `GuildInfoHandler.handle` | Guild management request (membership/roles/motd/roster lifecycle). | Guild manager and membership persistence | Medium-High |
| `CMSG_GUILD_INVITE` | `GuildInviteHandler.handle` | Guild management request (membership/roles/motd/roster lifecycle). | Guild manager and membership persistence | Medium-High |
| `CMSG_GUILD_LEADER` | `GuildLeaderHandler.handle` | Guild management request (membership/roles/motd/roster lifecycle). | Guild manager and membership persistence | Medium-High |
| `CMSG_GUILD_LEAVE` | `GuildLeaveHandler.handle` | Guild management request (membership/roles/motd/roster lifecycle). | Guild manager and membership persistence | Medium-High |
| `CMSG_GUILD_MOTD` | `GuildMOTDHandler.handle` | Guild management request (membership/roles/motd/roster lifecycle). | Guild manager and membership persistence | Medium-High |
| `CMSG_GUILD_PROMOTE` | `GuildPromoteHandler.handle` | Guild management request (membership/roles/motd/roster lifecycle). | Guild manager and membership persistence | Medium-High |
| `CMSG_GUILD_QUERY` | `GuildQueryHandler.handle` | Guild management request (membership/roles/motd/roster lifecycle). | Guild manager and membership persistence | Medium-High |
| `CMSG_GUILD_ROSTER` | `GuildRosterHandler.handle` | Guild management request (membership/roles/motd/roster lifecycle). | Guild manager and membership persistence | Medium-High |
| `MSG_SAVE_GUILD_EMBLEM` | `GuildSaveEmblemHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Guild manager and membership persistence | Medium |
## Ignored
| Opcode | Handler | What It Does | Engine Integration | Confidence |
|---|---|---|---|---|
| `CMSG_SCREENSHOT` | `NullHandler.handle` | Client notification/event intentionally ignored by server dispatcher (NullHandler). | No gameplay mutation (explicitly dropped) | Medium-High |
| `CMSG_TUTORIAL_CLEAR` | `NullHandler.handle` | Client notification/event intentionally ignored by server dispatcher (NullHandler). | No gameplay mutation (explicitly dropped) | Medium-High |
## Inventory
| Opcode | Handler | What It Does | Engine Integration | Confidence |
|---|---|---|---|---|
| `CMSG_AUTOEQUIP_ITEM` | `AutoequipItemHandler.handle` | Inventory/bag operation affecting item locations, stacks, and client inventory sync packets. | Inventory manager, item templates, update packet generation | Medium-High |
| `CMSG_AUTOSTORE_BAG_ITEM` | `AutostoreBagItemHandler.handle` | Inventory/bag operation affecting item locations, stacks, and client inventory sync packets. | Inventory manager, item templates, update packet generation | Medium-High |
| `CMSG_BUY_ITEM` | `BuyItemHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Inventory manager, item templates, update packet generation | Medium |
| `CMSG_BUY_ITEM_IN_SLOT` | `BuyItemInSlotHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Inventory manager, item templates, update packet generation | Medium |
| `CMSG_CREATEITEM` | `CreateItemHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Inventory manager, item templates, update packet generation | Medium |
| `CMSG_DESTROYITEM` | `DestroyItemHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Inventory manager, item templates, update packet generation | Medium |
| `CMSG_ITEM_QUERY_MULTIPLE` | `ItemQueryMultipleHandler.handle` | Inventory/bag operation affecting item locations, stacks, and client inventory sync packets. | Inventory manager, item templates, update packet generation | Medium-High |
| `CMSG_ITEM_QUERY_SINGLE` | `ItemQuerySingleHandler.handle` | Inventory/bag operation affecting item locations, stacks, and client inventory sync packets. | Inventory manager, item templates, update packet generation | Medium-High |
| `CMSG_LIST_INVENTORY` | `ListInventoryHandler.handle` | NPC/object interaction request executing range/faction checks and service-specific responses. | Inventory manager, item templates, update packet generation | Medium-High |
| `CMSG_OPEN_ITEM` | `OpenItemHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Inventory manager, item templates, update packet generation | Medium |
| `CMSG_READ_ITEM` | `ReadItemHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Inventory manager, item templates, update packet generation | Medium |
| `CMSG_SELL_ITEM` | `SellItemHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Inventory manager, item templates, update packet generation | Medium |
| `CMSG_SPLIT_ITEM` | `SplitItemHandler.handle` | Inventory/bag operation affecting item locations, stacks, and client inventory sync packets. | Inventory manager, item templates, update packet generation | Medium-High |
| `CMSG_SWAP_INV_ITEM` | `SwapInvItemHandler.handle` | Inventory/bag operation affecting item locations, stacks, and client inventory sync packets. | Inventory manager, item templates, update packet generation | Medium-High |
| `CMSG_SWAP_ITEM` | `SwapItemHandler.handle` | Inventory/bag operation affecting item locations, stacks, and client inventory sync packets. | Inventory manager, item templates, update packet generation | Medium-High |
| `CMSG_USE_ITEM` | `UseItemHandler.handle` | Spellcast/control request entering spell system validation and cast execution. | Inventory manager, item templates, update packet generation | Medium-High |
| `CMSG_WRAP_ITEM` | `WrapItemHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Inventory manager, item templates, update packet generation | Medium |
## Loot
| Opcode | Handler | What It Does | Engine Integration | Confidence |
|---|---|---|---|---|
| `CMSG_AUTOSTORE_LOOT_ITEM` | `AutostoreLootItemHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Loot manager and loot rights/content state | Medium |
| `CMSG_LOOT_MONEY` | `LootMoneyHandler.handle` | Loot interaction request controlling loot open/take/release lifecycle. | Loot manager and loot rights/content state | Medium-High |
| `CMSG_LOOT_RELEASE` | `LootReleaseHandler.handle` | Loot interaction request controlling loot open/take/release lifecycle. | Loot manager and loot rights/content state | Medium-High |
## Movement
| Opcode | Handler | What It Does | Engine Integration | Confidence |
|---|---|---|---|---|
| `CMSG_CHANNEL_UNMODERATOR` | `ChannelModeratorHandler.handle_remove_mod` | Channel chat administration request (join/leave/moderation/ownership/password). | Movement manager, map cell updates, teleport/worldport pipeline | Medium-High |
| `CMSG_FORCE_MOVE_ROOT_ACK` | `MovementHandler.handle_movement_status` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Movement manager, map cell updates, teleport/worldport pipeline | Medium |
| `CMSG_FORCE_MOVE_UNROOT_ACK` | `MovementHandler.handle_movement_status` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Movement manager, map cell updates, teleport/worldport pipeline | Medium |
| `CMSG_FORCE_SPEED_CHANGE_ACK` | `MovementHandler.handle_movement_status` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Movement manager, map cell updates, teleport/worldport pipeline | Medium |
| `CMSG_FORCE_SWIM_SPEED_CHANGE_ACK` | `MovementHandler.handle_movement_status` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Movement manager, map cell updates, teleport/worldport pipeline | Medium |
| `CMSG_GUILD_REMOVE` | `GuildRemoveMemberHandler.handle` | Guild management request (membership/roles/motd/roster lifecycle). | Movement manager, map cell updates, teleport/worldport pipeline | Medium-High |
| `CMSG_MOUNTSPECIAL_ANIM` | `MountSpecialAnimHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Movement manager, map cell updates, teleport/worldport pipeline | Medium |
| `CMSG_QUESTLOG_REMOVE_QUEST` | `QuestGiverRemoveQuestHandler.handle` | Quest-system request that reads/modifies quest log, objective, or reward state. | Movement manager, map cell updates, teleport/worldport pipeline | Medium-High |
| `CMSG_TELEPORT_TO_PLAYER` | `TeleportToPlayerHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Movement manager, map cell updates, teleport/worldport pipeline | Medium |
| `CMSG_WORLD_TELEPORT` | `WorldTeleportHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Movement manager, map cell updates, teleport/worldport pipeline | Medium |
| `MSG_MOVE_COLLIDE_REDIRECT` | `MovementHandler.handle_movement_status` | Movement state transition/ack consumed by movement handler for server-side position state. | Movement manager, map cell updates, teleport/worldport pipeline | Medium-High |
| `MSG_MOVE_COLLIDE_STUCK` | `MovementHandler.handle_movement_status` | Movement state transition/ack consumed by movement handler for server-side position state. | Movement manager, map cell updates, teleport/worldport pipeline | Medium-High |
| `MSG_MOVE_HEARTBEAT` | `MovementHandler.handle_movement_status` | Periodic authoritative position/orientation update used by movement reconciliation. | Movement manager, map cell updates, teleport/worldport pipeline | High |
| `MSG_MOVE_JUMP` | `MovementHandler.handle_movement_status` | Movement state transition/ack consumed by movement handler for server-side position state. | Movement manager, map cell updates, teleport/worldport pipeline | Medium-High |
| `MSG_MOVE_ROOT` | `MovementHandler.handle_movement_status` | Movement state transition/ack consumed by movement handler for server-side position state. | Movement manager, map cell updates, teleport/worldport pipeline | Medium-High |
| `MSG_MOVE_SET_ALL_SPEED_CHEAT` | `SpeedCheatHandler.handle` | Movement state transition/ack consumed by movement handler for server-side position state. | Movement manager, map cell updates, teleport/worldport pipeline | Medium-High |
| `MSG_MOVE_SET_FACING` | `MovementHandler.handle_movement_status` | Movement state transition/ack consumed by movement handler for server-side position state. | Movement manager, map cell updates, teleport/worldport pipeline | Medium-High |
| `MSG_MOVE_SET_PITCH` | `MovementHandler.handle_movement_status` | Movement state transition/ack consumed by movement handler for server-side position state. | Movement manager, map cell updates, teleport/worldport pipeline | Medium-High |
| `MSG_MOVE_SET_RUN_MODE` | `MovementHandler.handle_movement_status` | Movement state transition/ack consumed by movement handler for server-side position state. | Movement manager, map cell updates, teleport/worldport pipeline | Medium-High |
| `MSG_MOVE_SET_RUN_SPEED_CHEAT` | `SpeedCheatHandler.handle` | Movement state transition/ack consumed by movement handler for server-side position state. | Movement manager, map cell updates, teleport/worldport pipeline | Medium-High |
| `MSG_MOVE_SET_SWIM_SPEED_CHEAT` | `SpeedCheatHandler.handle` | Movement state transition/ack consumed by movement handler for server-side position state. | Movement manager, map cell updates, teleport/worldport pipeline | Medium-High |
| `MSG_MOVE_SET_TURN_RATE_CHEAT` | `SpeedCheatHandler.handle` | Movement state transition/ack consumed by movement handler for server-side position state. | Movement manager, map cell updates, teleport/worldport pipeline | Medium-High |
| `MSG_MOVE_SET_WALK_MODE` | `MovementHandler.handle_movement_status` | Movement state transition/ack consumed by movement handler for server-side position state. | Movement manager, map cell updates, teleport/worldport pipeline | Medium-High |
| `MSG_MOVE_SET_WALK_SPEED_CHEAT` | `SpeedCheatHandler.handle` | Movement state transition/ack consumed by movement handler for server-side position state. | Movement manager, map cell updates, teleport/worldport pipeline | Medium-High |
| `MSG_MOVE_START_BACKWARD` | `MovementHandler.handle_movement_status` | Movement state transition/ack consumed by movement handler for server-side position state. | Movement manager, map cell updates, teleport/worldport pipeline | Medium-High |
| `MSG_MOVE_START_FORWARD` | `MovementHandler.handle_movement_status` | Movement state transition/ack consumed by movement handler for server-side position state. | Movement manager, map cell updates, teleport/worldport pipeline | Medium-High |
| `MSG_MOVE_START_PITCH_DOWN` | `MovementHandler.handle_movement_status` | Movement state transition/ack consumed by movement handler for server-side position state. | Movement manager, map cell updates, teleport/worldport pipeline | Medium-High |
| `MSG_MOVE_START_PITCH_UP` | `MovementHandler.handle_movement_status` | Movement state transition/ack consumed by movement handler for server-side position state. | Movement manager, map cell updates, teleport/worldport pipeline | Medium-High |
| `MSG_MOVE_START_STRAFE_LEFT` | `MovementHandler.handle_movement_status` | Movement state transition/ack consumed by movement handler for server-side position state. | Movement manager, map cell updates, teleport/worldport pipeline | Medium-High |
| `MSG_MOVE_START_STRAFE_RIGHT` | `MovementHandler.handle_movement_status` | Movement state transition/ack consumed by movement handler for server-side position state. | Movement manager, map cell updates, teleport/worldport pipeline | Medium-High |
| `MSG_MOVE_START_SWIM` | `MovementHandler.handle_movement_status` | Movement state transition/ack consumed by movement handler for server-side position state. | Movement manager, map cell updates, teleport/worldport pipeline | Medium-High |
| `MSG_MOVE_START_TURN_LEFT` | `MovementHandler.handle_movement_status` | Movement state transition/ack consumed by movement handler for server-side position state. | Movement manager, map cell updates, teleport/worldport pipeline | Medium-High |
| `MSG_MOVE_START_TURN_RIGHT` | `MovementHandler.handle_movement_status` | Movement state transition/ack consumed by movement handler for server-side position state. | Movement manager, map cell updates, teleport/worldport pipeline | Medium-High |
| `MSG_MOVE_STOP` | `MovementHandler.handle_movement_status` | Movement state transition/ack consumed by movement handler for server-side position state. | Movement manager, map cell updates, teleport/worldport pipeline | Medium-High |
| `MSG_MOVE_STOP_PITCH` | `MovementHandler.handle_movement_status` | Movement state transition/ack consumed by movement handler for server-side position state. | Movement manager, map cell updates, teleport/worldport pipeline | Medium-High |
| `MSG_MOVE_STOP_STRAFE` | `MovementHandler.handle_movement_status` | Movement state transition/ack consumed by movement handler for server-side position state. | Movement manager, map cell updates, teleport/worldport pipeline | Medium-High |
| `MSG_MOVE_STOP_SWIM` | `MovementHandler.handle_movement_status` | Movement state transition/ack consumed by movement handler for server-side position state. | Movement manager, map cell updates, teleport/worldport pipeline | Medium-High |
| `MSG_MOVE_STOP_TURN` | `MovementHandler.handle_movement_status` | Movement state transition/ack consumed by movement handler for server-side position state. | Movement manager, map cell updates, teleport/worldport pipeline | Medium-High |
| `MSG_MOVE_TELEPORT_ACK` | `WorldTeleportHandler.handle_ack` | Acknowledges teleport placement update and clears movement sync gate. | Movement manager, map cell updates, teleport/worldport pipeline | High |
| `MSG_MOVE_TELEPORT_CHEAT` | `WorldTeleportHandler.handle` | Movement state transition/ack consumed by movement handler for server-side position state. | Movement manager, map cell updates, teleport/worldport pipeline | Medium-High |
| `MSG_MOVE_UNROOT` | `MovementHandler.handle_movement_status` | Movement state transition/ack consumed by movement handler for server-side position state. | Movement manager, map cell updates, teleport/worldport pipeline | Medium-High |
| `MSG_MOVE_WORLDPORT_ACK` | `WorldTeleportHandler.handle_ack` | Acknowledges worldport/new-world transfer; server continues spawn pipeline. | Movement manager, map cell updates, teleport/worldport pipeline | High |
## NPC_Object
| Opcode | Handler | What It Does | Engine Integration | Confidence |
|---|---|---|---|---|
| `CMSG_BANKER_ACTIVATE` | `BankerActivateHandler.handle` | NPC/object interaction request executing range/faction checks and service-specific responses. | Map object manager, creature/gameobject template lookup | Medium-High |
| `CMSG_BINDER_ACTIVATE` | `BinderActivateHandler.handle` | NPC/object interaction request executing range/faction checks and service-specific responses. | Map object manager, creature/gameobject template lookup | Medium-High |
| `CMSG_BUY_BANK_SLOT` | `BuyBankSlotHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Map object manager, creature/gameobject template lookup | Medium |
| `CMSG_CREATURE_QUERY` | `CreatureQueryHandler.handle` | Resolves creature template metadata (name/subname/model ids/faction fields). | Map object manager, creature/gameobject template lookup | Medium-High |
| `CMSG_GAMEOBJECT_QUERY` | `GameObjectQueryHandler.handle` | Resolves gameobject template metadata for interactable world objects. | Map object manager, creature/gameobject template lookup | Medium-High |
| `CMSG_GAMEOBJ_USE` | `GameobjUseHandler.handle` | NPC/object interaction request executing range/faction checks and service-specific responses. | Map object manager, creature/gameobject template lookup | Medium-High |
| `CMSG_INSPECT` | `InspectHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Map object manager, creature/gameobject template lookup | Medium |
| `CMSG_TRAINER_LIST` | `TrainerListHandler.handle` | NPC/object interaction request executing range/faction checks and service-specific responses. | Map object manager, creature/gameobject template lookup | Medium-High |
| `MSG_TABARDVENDOR_ACTIVATE` | `TabardVendorActivateHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Map object manager, creature/gameobject template lookup | Medium |
## Other
| Opcode | Handler | What It Does | Engine Integration | Confidence |
|---|---|---|---|---|
| `CMSG_BOOTME` | `BootMeHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | General gameplay service path based on handler implementation | Medium |
| `CMSG_BUG` | `BugHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | General gameplay service path based on handler implementation | Medium |
| `CMSG_CREATEMONSTER` | `CreateMonsterHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | General gameplay service path based on handler implementation | Medium |
| `CMSG_DESTROYMONSTER` | `DestroyMonsterHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | General gameplay service path based on handler implementation | Medium |
| `CMSG_GHOST` | `GhostHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | General gameplay service path based on handler implementation | Medium |
| `CMSG_NAME_QUERY` | `NameQueryHandler.handle` | Resolves GUID to display name/race/class metadata for UI/tooling identity. | General gameplay service path based on handler implementation | Medium-High |
| `CMSG_PAGE_TEXT_QUERY` | `PageTextQueryHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | General gameplay service path based on handler implementation | Medium |
| `CMSG_PING` | `PingHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | General gameplay service path based on handler implementation | Medium |
| `CMSG_RECHARGE` | `RechargeHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | General gameplay service path based on handler implementation | Medium |
| `CMSG_SETDEATHBINDPOINT` | `SetDeathBindPointHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | General gameplay service path based on handler implementation | Medium |
| `CMSG_SETWEAPONMODE` | `SetWeaponModeHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | General gameplay service path based on handler implementation | Medium |
| `CMSG_SET_ACTION_BUTTON` | `SetActionButtonHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | General gameplay service path based on handler implementation | Medium |
| `CMSG_SET_SELECTION` | `SetSelectionHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | General gameplay service path based on handler implementation | Medium |
| `CMSG_SET_TARGET` | `SetTargetHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | General gameplay service path based on handler implementation | Medium |
| `CMSG_STANDSTATECHANGE` | `StandStateChangeHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | General gameplay service path based on handler implementation | Medium |
| `MSG_MINIMAP_PING` | `MinimapPingHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | General gameplay service path based on handler implementation | Medium |
## Pet
| Opcode | Handler | What It Does | Engine Integration | Confidence |
|---|---|---|---|---|
| `CMSG_OFFER_PETITION` | `PetitionOfferHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Pet manager/AI and owner linkage state | Medium |
| `CMSG_PETITION_BUY` | `PetitionBuyHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Pet manager/AI and owner linkage state | Medium |
| `CMSG_PETITION_QUERY` | `PetitionQueryHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Pet manager/AI and owner linkage state | Medium |
| `CMSG_PETITION_SHOWLIST` | `PetitionShowlistHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Pet manager/AI and owner linkage state | Medium |
| `CMSG_PETITION_SHOW_SIGNATURES` | `PetitionShowSignaturesHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Pet manager/AI and owner linkage state | Medium |
| `CMSG_PETITION_SIGN` | `PetitionSignHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Pet manager/AI and owner linkage state | Medium |
| `CMSG_PET_ABANDON` | `PetAbandonHandler.handle` | Pet command/state request routed to pet manager/action system. | Pet manager/AI and owner linkage state | Medium-High |
| `CMSG_PET_ACTION` | `PetActionHandler.handle` | Pet command/state request routed to pet manager/action system. | Pet manager/AI and owner linkage state | Medium-High |
| `CMSG_PET_LEVEL_CHEAT` | `PetLevelCheatHandler.handle` | Pet command/state request routed to pet manager/action system. | Pet manager/AI and owner linkage state | Medium-High |
| `CMSG_PET_NAME_QUERY` | `PetNameQueryHandler.handle` | Pet command/state request routed to pet manager/action system. | Pet manager/AI and owner linkage state | Medium-High |
| `CMSG_PET_RENAME` | `PetRenameHandler.handle` | Pet command/state request routed to pet manager/action system. | Pet manager/AI and owner linkage state | Medium-High |
| `CMSG_PET_SET_ACTION` | `PetSetActionHandler.handle` | Pet command/state request routed to pet manager/action system. | Pet manager/AI and owner linkage state | Medium-High |
| `CMSG_TURN_IN_PETITION` | `PetitionTurnInHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Pet manager/AI and owner linkage state | Medium |
## Quest
| Opcode | Handler | What It Does | Engine Integration | Confidence |
|---|---|---|---|---|
| `CMSG_CLEAR_QUEST` | `ClearQuestHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Quest manager, object lookup, dialog/reward state | Medium |
| `CMSG_FLAG_QUEST` | `FlagQuestHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Quest manager, object lookup, dialog/reward state | Medium |
| `CMSG_FLAG_QUEST_FINISH` | `FlagQuestFinishHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Quest manager, object lookup, dialog/reward state | Medium |
| `CMSG_LOOT` | `LootRequestHandler.handle` | Loot interaction request controlling loot open/take/release lifecycle. | Quest manager, object lookup, dialog/reward state | Medium-High |
| `CMSG_QUESTGIVER_ACCEPT_QUEST` | `QuestGiverAcceptQuestHandler.handle` | Accepts selected quest; server validates hostility/range/log capacity then mutates quest state. | Quest manager, object lookup, dialog/reward state | High |
| `CMSG_QUESTGIVER_CHOOSE_REWARD` | `QuestGiverChooseRewardHandler.handle` | Quest-system request that reads/modifies quest log, objective, or reward state. | Quest manager, object lookup, dialog/reward state | Medium-High |
| `CMSG_QUESTGIVER_COMPLETE_QUEST` | `QuestGiverCompleteQuestHandler.handle` | Requests completion turn-in path; server verifies objective state and reward flow. | Quest manager, object lookup, dialog/reward state | Medium-High |
| `CMSG_QUESTGIVER_HELLO` | `QuestGiverHelloHandler.handle` | Opens quest dialog/gossip path for interactable questgiver/item. | Quest manager, object lookup, dialog/reward state | High |
| `CMSG_QUESTGIVER_QUERY_QUEST` | `QuestGiverQueryQuestHandler.handle` | Quest-system request that reads/modifies quest log, objective, or reward state. | Quest manager, object lookup, dialog/reward state | Medium-High |
| `CMSG_QUESTGIVER_REQUEST_REWARD` | `QuestGiverRequestReward.handle` | Quest-system request that reads/modifies quest log, objective, or reward state. | Quest manager, object lookup, dialog/reward state | Medium-High |
| `CMSG_QUESTGIVER_STATUS_QUERY` | `QuestGiverStatusHandler.handle` | Queries quest marker/status for target GUID; server resolves known object and sends dialog status. | Quest manager, object lookup, dialog/reward state | High |
| `CMSG_QUEST_CONFIRM_ACCEPT` | `QuestConfirmAcceptHandler.handle` | Quest-system request that reads/modifies quest log, objective, or reward state. | Quest manager, object lookup, dialog/reward state | Medium-High |
| `CMSG_QUEST_QUERY` | `QuestQueryHandler.handle` | Requests quest template/details payload for quest id. | Quest manager, object lookup, dialog/reward state | Medium-High |
| `CMSG_REPOP_REQUEST` | `RepopRequestHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Quest manager, object lookup, dialog/reward state | Medium |
## Social
| Opcode | Handler | What It Does | Engine Integration | Confidence |
|---|---|---|---|---|
| `CMSG_MESSAGECHAT` | `ChatHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Chat/LFG/macro social systems | Medium |
| `CMSG_PLAYER_MACRO` | `PlayerMacroHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Chat/LFG/macro social systems | Medium |
| `CMSG_TEXT_EMOTE` | `TextEmoteHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Chat/LFG/macro social systems | Medium |
| `CMSG_WHO` | `WhoHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Chat/LFG/macro social systems | Medium |
## Spells
| Opcode | Handler | What It Does | Engine Integration | Confidence |
|---|---|---|---|---|
| `CMSG_CANCEL_AURA` | `CancelAuraHandler.handle` | Spellcast/control request entering spell system validation and cast execution. | Spell/aura/cast systems and cooldown/resource validation | Medium-High |
| `CMSG_CANCEL_CAST` | `CancelCastHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Spell/aura/cast systems and cooldown/resource validation | Medium |
| `CMSG_CANCEL_CHANNELLING` | `CancelChannellingHandler.handle` | Spellcast/control request entering spell system validation and cast execution. | Spell/aura/cast systems and cooldown/resource validation | Medium-High |
| `CMSG_CAST_SPELL` | `CastSpellHandler.handle` | Spellcast/control request entering spell system validation and cast execution. | Spell/aura/cast systems and cooldown/resource validation | Medium-High |
| `CMSG_LEARN_SPELL` | `LearnSpellCheatHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Spell/aura/cast systems and cooldown/resource validation | Medium |
| `CMSG_NEW_SPELL_SLOT` | `NewSpellSlotHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Spell/aura/cast systems and cooldown/resource validation | Medium |
| `CMSG_TRAINER_BUY_SPELL` | `TrainerBuySpellHandler.handle` | NPC/object interaction request executing range/faction checks and service-specific responses. | Spell/aura/cast systems and cooldown/resource validation | Medium-High |
## Taxi
| Opcode | Handler | What It Does | Engine Integration | Confidence |
|---|---|---|---|---|
| `CMSG_ACTIVATETAXI` | `ActivateTaxiHandler.handle` | Flight-path query/activation that hands control to taxi manager and movement state. | Taxi manager and travel movement control | Medium-High |
| `CMSG_TAXICLEARALLNODES` | `TaxiClearAllNodesHandler.handle` | Flight-path query/activation that hands control to taxi manager and movement state. | Taxi manager and travel movement control | Medium-High |
| `CMSG_TAXIENABLEALLNODES` | `TaxiEnableAllNodesHandler.handle` | Flight-path query/activation that hands control to taxi manager and movement state. | Taxi manager and travel movement control | Medium-High |
| `CMSG_TAXINODE_STATUS_QUERY` | `TaxiNodeStatusQueryHandler.handle` | Flight-path query/activation that hands control to taxi manager and movement state. | Taxi manager and travel movement control | Medium-High |
| `CMSG_TAXIQUERYAVAILABLENODES` | `TaxiQueryNodesHandler.handle` | Flight-path query/activation that hands control to taxi manager and movement state. | Taxi manager and travel movement control | Medium-High |
## Trade
| Opcode | Handler | What It Does | Engine Integration | Confidence |
|---|---|---|---|---|
| `CMSG_ACCEPT_TRADE` | `AcceptTradeHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Trade manager and transactional item/gold exchange | Medium |
| `CMSG_BEGIN_TRADE` | `BeginTradeHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Trade manager and transactional item/gold exchange | Medium |
| `CMSG_CANCEL_TRADE` | `CancelTradeHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Trade manager and transactional item/gold exchange | Medium |
| `CMSG_CLEAR_TRADE_ITEM` | `ClearTradeItemHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Trade manager and transactional item/gold exchange | Medium |
| `CMSG_INITIATE_TRADE` | `InitiateTradeHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Trade manager and transactional item/gold exchange | Medium |
| `CMSG_SET_TRADE_GOLD` | `SetTradeGoldHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Trade manager and transactional item/gold exchange | Medium |
| `CMSG_SET_TRADE_ITEM` | `SetTradeItemHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Trade manager and transactional item/gold exchange | Medium |
| `CMSG_UNACCEPT_TRADE` | `UnacceptTradeHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | Trade manager and transactional item/gold exchange | Medium |
## WorldState
| Opcode | Handler | What It Does | Engine Integration | Confidence |
|---|---|---|---|---|
| `CMSG_AREATRIGGER` | `AreaTriggerHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | World/map time/zone/area trigger subsystems | Medium |
| `CMSG_GETDEATHBINDZONE` | `GetDeathBindPointHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | World/map time/zone/area trigger subsystems | Medium |
| `CMSG_PLAYED_TIME` | `PlayedTimeHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | World/map time/zone/area trigger subsystems | Medium |
| `CMSG_QUERY_TIME` | `TimeQueryHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | World/map time/zone/area trigger subsystems | Medium |
| `CMSG_ZONEUPDATE` | `ZoneUpdateHandler.handle` | Routed to handler named in dispatch map; behavior follows corresponding subsystem manager logic. | World/map time/zone/area trigger subsystems | Medium |

## Outbound `SMSG_*` Notes
- The inbound dispatch table does not include outbound packet handlers; those packets are emitted from subsystem managers.
- The mirrored-code outbound inventory is preserved in `network_server_smsg_inventory.json` and should be used to define receive/decode support in tooling.

## Implementation Guidance for Tooling Network Layer
- Implement protocol as stateful phases: `Auth` -> `Character` -> `WorldTransfer` -> `InWorld`.
- Gate sendable opcodes by session phase to avoid invalid-sequence disconnects.
- Build decode handlers first for outbound world-critical packets: transfer/new-world/object updates/movement corrections.
- Keep opcode table versioned per build; do not mix 0.5.3 and later-era constants.