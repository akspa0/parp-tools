from game.world.opcode_handling.HandlerValidator import HandlerValidator
from utils.Logger import Logger
from utils.constants.UnitCodes import UnitFlags


class InvisHandler:

    @staticmethod
    def handle(world_session, reader):
        player_mgr, res = HandlerValidator.validate_session(world_session, reader.opcode)
        if not player_mgr:
            return res

        if not world_session.account_mgr.is_gm():
            Logger.anticheat(f'Player {player_mgr.get_name()} ({player_mgr.guid}) tried to use Invis.')
            return 0

        # Toggle Invisibility
        # We use UNIT_MASK_NON_ATTACKABLE to prevent AI aggro/targeting
        if player_mgr.has_flag(UnitFlags.UNIT_MASK_NON_ATTACKABLE):
            player_mgr.remove_flag(UnitFlags.UNIT_MASK_NON_ATTACKABLE)
            state = "OFF"
        else:
            player_mgr.add_flag(UnitFlags.UNIT_MASK_NON_ATTACKABLE)
            state = "ON"

        Logger.info(f"GM Invis: {player_mgr.name} set to {state}")
        return 0
