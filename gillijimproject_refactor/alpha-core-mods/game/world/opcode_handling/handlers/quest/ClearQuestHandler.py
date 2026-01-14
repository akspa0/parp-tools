from struct import unpack
from game.world.opcode_handling.HandlerValidator import HandlerValidator
from utils.Logger import Logger


class ClearQuestHandler:

    @staticmethod
    def handle(world_session, reader):
        # Validate world session.
        player_mgr, res = HandlerValidator.validate_session(world_session, reader.opcode)
        if not player_mgr:
            return res

        if not world_session.account_mgr.is_gm():
            Logger.anticheat(f'Player {player_mgr.get_name()} ({player_mgr.guid}) tried to use ClearQuest.')
            return 0

        if len(reader.data) >= 4:
            quest_id = unpack('<I', reader.data[:4])[0]
            player_mgr.quest_manager.remove_quest(quest_id)
            Logger.info(f"ClearQuest: Player {player_mgr.name} forced removal of quest {quest_id}")

        return 0
