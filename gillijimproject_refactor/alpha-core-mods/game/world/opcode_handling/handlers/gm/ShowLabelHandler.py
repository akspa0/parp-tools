from game.world.opcode_handling.HandlerValidator import HandlerValidator
from utils.Logger import Logger


class ShowLabelHandler:

    @staticmethod
    def handle(world_session, reader):
        player_mgr, res = HandlerValidator.validate_session(world_session, reader.opcode)
        if not player_mgr:
            return res

        if not world_session.account_mgr.is_gm():
            Logger.anticheat(f'Player {player_mgr.get_name()} ({player_mgr.guid}) tried to use ShowLabel.')
            return 0

        # ShowLabel seems to be a client-side debug toggle that the server might need to acknowledge,
        # or maybe it's just a server-side flag for "Show Debug info on Units".
        # For now, we'll log it. 
        # In modern emus, this might toggle GMRank visibility or similar.
        
        Logger.info(f"GM ShowLabel: Toggle request from {player_mgr.name}")
        return 0
