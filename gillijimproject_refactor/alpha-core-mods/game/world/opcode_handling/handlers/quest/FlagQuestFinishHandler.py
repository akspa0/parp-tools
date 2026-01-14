from struct import unpack
from game.world.opcode_handling.HandlerValidator import HandlerValidator
from utils.Logger import Logger


class FlagQuestFinishHandler:

    @staticmethod
    def handle(world_session, reader):
        # Validate world session.
        player_mgr, res = HandlerValidator.validate_session(world_session, reader.opcode)
        if not player_mgr:
            return res

        if not world_session.account_mgr.is_gm():
            Logger.anticheat(f'Player {player_mgr.get_name()} ({player_mgr.guid}) tried to use FlagQuestFinish.')
            return 0

        if len(reader.data) >= 4:
            quest_id = unpack('<I', reader.data[:4])[0]
            
            if quest_id in player_mgr.quest_manager.active_quests:
                active_quest = player_mgr.quest_manager.active_quests[quest_id]
                player_mgr.quest_manager.complete_quest(active_quest, update_surrounding=True, notify=True)
                player_mgr.quest_manager.update_single_quest(quest_id)
                Logger.info(f"FlagQuestFinish: Player {player_mgr.name} forced completion of quest {quest_id}")
            else:
                Logger.warning(f"FlagQuestFinish: Player {player_mgr.name} tried to finish quest {quest_id} but it is not active.")

        return 0
