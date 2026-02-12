
import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import struct

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from game.world.opcode_handling.handlers.quest.FlagQuestHandler import FlagQuestHandler
from game.world.opcode_handling.handlers.quest.FlagQuestFinishHandler import FlagQuestFinishHandler
from game.world.opcode_handling.handlers.quest.ClearQuestHandler import ClearQuestHandler
from utils.constants.OpCodes import OpCode

class MockPacketReader:
    def __init__(self, opcode, data):
        self.opcode = opcode
        self.data = data

class TestQuestDebugHandlers(unittest.TestCase):

    def setUp(self):
        self.mock_session = MagicMock()
        # Default to GM for positive tests
        self.mock_session.account_mgr.is_gm.return_value = True
        
        self.mock_player = MagicMock()
        self.mock_player.name = "TestPlayer"
        self.mock_player.quest_manager = MagicMock()
        
        # Mock Session validation to return the mock player
        self.validator_patch = patch('game.world.opcode_handling.HandlerValidator.HandlerValidator.validate_session', 
                                    return_value=(self.mock_player, 0))
        self.validator_patch.start()

    def tearDown(self):
        self.validator_patch.stop()

    def test_flag_quest_gm(self):
        """Test that FlagQuest works for GM"""
        quest_id = 123
        data = struct.pack('<I', quest_id)
        reader = MockPacketReader(OpCode.CMSG_FLAG_QUEST, data)
        
        FlagQuestHandler.handle(self.mock_session, reader)
        self.mock_player.quest_manager.handle_accept_quest.assert_called_with(quest_id, 0)
        print("FlagQuest (GM): Success.")

    def test_flag_quest_nongm(self):
        """Test that FlagQuest is blocked for Non-GM"""
        self.mock_session.account_mgr.is_gm.return_value = False
        quest_id = 123
        data = struct.pack('<I', quest_id)
        reader = MockPacketReader(OpCode.CMSG_FLAG_QUEST, data)
        
        FlagQuestHandler.handle(self.mock_session, reader)
        self.mock_player.quest_manager.handle_accept_quest.assert_not_called()
        print("FlagQuest (Non-GM): Blocked.")

    def test_flag_quest_finish_gm(self):
        """Test that FlagQuestFinish works for GM"""
        quest_id = 456
        data = struct.pack('<I', quest_id)
        reader = MockPacketReader(OpCode.CMSG_FLAG_QUEST_FINISH, data)
        
        mock_active_quest = MagicMock()
        self.mock_player.quest_manager.active_quests = {quest_id: mock_active_quest}
        
        FlagQuestFinishHandler.handle(self.mock_session, reader)
        self.mock_player.quest_manager.complete_quest.assert_called_with(mock_active_quest, update_surrounding=True, notify=True)
        print("FlagQuestFinish (GM): Success.")

    def test_flag_quest_finish_nongm(self):
        """Test that FlagQuestFinish is blocked for Non-GM"""
        self.mock_session.account_mgr.is_gm.return_value = False
        quest_id = 456
        data = struct.pack('<I', quest_id)
        reader = MockPacketReader(OpCode.CMSG_FLAG_QUEST_FINISH, data)
        
        mock_active_quest = MagicMock()
        self.mock_player.quest_manager.active_quests = {quest_id: mock_active_quest}
        
        FlagQuestFinishHandler.handle(self.mock_session, reader)
        self.mock_player.quest_manager.complete_quest.assert_not_called()
        print("FlagQuestFinish (Non-GM): Blocked.")

    def test_clear_quest_gm(self):
        """Test that ClearQuest works for GM"""
        quest_id = 101
        data = struct.pack('<I', quest_id)
        reader = MockPacketReader(OpCode.CMSG_CLEAR_QUEST, data)
        
        ClearQuestHandler.handle(self.mock_session, reader)
        self.mock_player.quest_manager.remove_quest.assert_called_with(quest_id)
        print("ClearQuest (GM): Success.")

    def test_clear_quest_nongm(self):
        """Test that ClearQuest is blocked for Non-GM"""
        self.mock_session.account_mgr.is_gm.return_value = False
        quest_id = 101
        data = struct.pack('<I', quest_id)
        reader = MockPacketReader(OpCode.CMSG_CLEAR_QUEST, data)
        
        ClearQuestHandler.handle(self.mock_session, reader)
        self.mock_player.quest_manager.remove_quest.assert_not_called()
        print("ClearQuest (Non-GM): Blocked.")

if __name__ == '__main__':
    unittest.main()
