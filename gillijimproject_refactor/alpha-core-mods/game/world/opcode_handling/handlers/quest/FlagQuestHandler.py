from struct import unpack
from game.world.opcode_handling.HandlerValidator import HandlerValidator
from utils.Logger import Logger


class FlagQuestHandler:

    @staticmethod
    def handle(world_session, reader):
        # Validate world session.
        player_mgr, res = HandlerValidator.validate_session(world_session, reader.opcode)
        if not player_mgr:
            return res

        if not world_session.account_mgr.is_gm():
            Logger.anticheat(f'Player {player_mgr.get_name()} ({player_mgr.guid}) tried to use FlagQuest.')
            return 0

        if len(reader.data) >= 4:
            quest_id = unpack('<I', reader.data[:4])[0]
            # FlagQuest is a "Force Accept". We pass 0 as guid to bypass proximity checks.
            # handle_accept_quest supports 0/None as valid input for "Console Source".
            player_mgr.quest_manager.handle_accept_quest(quest_id, 0)
            Logger.info(f"FlagQuest: Player {player_mgr.name} forced accept of quest {quest_id}")

        return 0
