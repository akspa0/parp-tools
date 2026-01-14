from game.world.opcode_handling.HandlerValidator import HandlerValidator
from utils.Logger import Logger
from network.packet.PacketWriter import PacketWriter
from utils.constants.OpCodes import OpCode


class BindOtherHandler:

    @staticmethod
    def handle(world_session, reader):
        # Validate session
        player_mgr, res = HandlerValidator.validate_session(world_session, reader.opcode)
        if not player_mgr:
            return res

        # GM Permission Check
        if not world_session.account_mgr.is_gm():
            Logger.anticheat(f'Player {player_mgr.get_name()} ({player_mgr.guid}) tried to use BindOther.')
            return 0

        # Read Mode (0=Locked Target, 1=Name)
        try:
            mode = reader.read_uint8()
        except IndexError:
            # Fallback if no params (unlikely based on client code, but safe parsing)
            return 0

        target = None

        if mode == 0:
            # Locked Target (GUID)
            # Client sends 0 + LockedTarget (8 bytes)
            # Actually client sends 8 bytes? Decomp said: CDataStore::Put((CDataStore *)&msg, CONCAT44(CGGameUI::m_lockedTarget._4_4_,(undefined4)CGGameUI::m_lockedTarget));
            # That is a 64-bit GUID.
            try:
                target_guid = reader.read_uint64()
                from game.world.WorldSessionStateHandler import WorldSessionStateHandler
                target_player_mgr = WorldSessionStateHandler.find_player_by_guid(target_guid)
                if target_player_mgr:
                    target = target_player_mgr
            except IndexError:
                pass
        else:
            # Name (String)
            try:
                target_name = reader.read_string()
                from game.world.WorldSessionStateHandler import WorldSessionStateHandler
                target_player_mgr = WorldSessionStateHandler.find_player_by_name(target_name)
                if target_player_mgr:
                     target = target_player_mgr
            except IndexError:
                pass
        
        # Fallback to current selection if still no target found? 
        # But client specific logic implies explicit target.
        if not target:
            # Try player's current selection as last resort if mode 0 failed or was ambiguous
             t = player_mgr.get_target()
             if t and t.is_player():
                 target = t

        if not target:
             Logger.info(f"GM BindOther: Player {player_mgr.get_name()} tried to bind invalid target.")
             return 0

        # Perform the Bind
        # Bind target to THEIR CURRENT LOCATION.
        # Arguments: map_id, position (Vector)
        target.set_death_bind(target.map_id, target.location)
        
        Logger.info(f"GM BindOther: {player_mgr.get_name()} bound {target.get_name()} to Map {target.map_id} {target.location.x} {target.location.y} {target.location.z}")
        return 0
