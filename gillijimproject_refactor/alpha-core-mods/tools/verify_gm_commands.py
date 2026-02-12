import sys
import unittest
from unittest.mock import MagicMock, patch

# Adjust path to include lib/alpha-core
sys.path.append('j:/wowDev/parp-tools/gillijimproject_refactor/lib/alpha-core')

# --- MOCKING CONFIG AND DB BEFORE IMPORTS ---
config_mock = MagicMock()
config_mock.config.Unit.Defaults.combat_reach = 1.0
config_mock.config.Unit.Defaults.base_attack_time = 2000
config_mock.config.Unit.Defaults.offhand_attack_time = 2000
sys.modules['utils.ConfigManager'] = config_mock

# Mock DbcDatabaseManager completely to avoid DB connections
mock_dbc = MagicMock()
sys.modules['database.dbc.DbcDatabaseManager'] = mock_dbc

# Mock WorldDatabaseManager to avoid DB connections
mock_world_db = MagicMock()
sys.modules['database.world.WorldDatabaseManager'] = mock_world_db

# Mock WorldSessionStateHandler to avoid import errors
mock_wss_handler = MagicMock()
sys.modules['game.world.WorldSessionStateHandler'] = mock_wss_handler


from utils.constants.OpCodes import OpCode
from utils.constants.UnitCodes import UnitFlags
# Import handlers AFTER mocking
from game.world.opcode_handling.handlers.gm.InvisHandler import InvisHandler
from game.world.opcode_handling.handlers.gm.GhostHandler import GhostHandler
from game.world.opcode_handling.handlers.gm.NukeHandler import NukeHandler
from game.world.opcode_handling.handlers.gm.BindOtherHandler import BindOtherHandler
from game.world.opcode_handling.handlers.gm.ShowLabelHandler import ShowLabelHandler
from game.world.opcode_handling.handlers.gm.DBLookupHandler import DBLookupHandler

class TestGMCommands(unittest.TestCase):

    def setUp(self):
        self.session = MagicMock()
        self.packet_reader = MagicMock()
        self.player_mgr = MagicMock()
        self.player_mgr.guid = 1
        self.player_mgr.name = "TestGM"
        self.player_mgr.get_name.return_value = "TestGM"
        self.player_mgr.get_target.return_value = None

        # Setup Session
        self.session.player_mgr = self.player_mgr
        self.session.account_mgr.is_gm.return_value = True # Default to GM

        # Locations
        self.player_mgr.location.map_id = 0
        self.player_mgr.location.x = 100
        self.player_mgr.location.y = 200
        self.player_mgr.location.z = 300
        self.player_mgr.location.o = 1.5

    def test_invis_gm(self):
        self.packet_reader.opcode = OpCode.CMSG_GM_INVIS
        
        # Test Toggle ON
        self.player_mgr.has_flag.return_value = False
        InvisHandler.handle(self.session, self.packet_reader)
        self.player_mgr.add_flag.assert_called_with(UnitFlags.UNIT_MASK_NON_ATTACKABLE)

        # Test Toggle OFF
        self.player_mgr.has_flag.return_value = True
        InvisHandler.handle(self.session, self.packet_reader)
        self.player_mgr.remove_flag.assert_called_with(UnitFlags.UNIT_MASK_NON_ATTACKABLE)

    def test_invis_non_gm(self):
        self.packet_reader.opcode = OpCode.CMSG_GM_INVIS
        self.session.account_mgr.is_gm.return_value = False
        InvisHandler.handle(self.session, self.packet_reader)
        self.player_mgr.add_flag.assert_not_called()
        self.player_mgr.remove_flag.assert_not_called()

    def test_ghost_gm(self):
        self.packet_reader.opcode = OpCode.CMSG_GHOST
        
        # Test Toggle ON
        self.player_mgr.has_flag.return_value = False
        GhostHandler.handle(self.session, self.packet_reader)
        self.player_mgr.add_flag.assert_called_with(UnitFlags.UNIT_FLAG_GHOST)

        # Test Toggle OFF
        self.player_mgr.has_flag.return_value = True
        GhostHandler.handle(self.session, self.packet_reader)
        self.player_mgr.remove_flag.assert_called_with(UnitFlags.UNIT_FLAG_GHOST)

    def test_nuke_gm(self):
        self.packet_reader.opcode = OpCode.CMSG_GM_NUKE
        target = MagicMock()
        target.name = "Victim"
        self.player_mgr.get_target.return_value = target

        NukeHandler.handle(self.session, self.packet_reader)
        target.deal_damage.assert_called()
        # Verify damage info passed has massive damage
        args, _ = target.deal_damage.call_args
        damage_info = args[1] # 2nd arg is damage_info
        self.assertEqual(damage_info.base_damage, 999999)

    def test_bind_other_gm(self):
        self.packet_reader.opcode = OpCode.MSG_GM_BIND_OTHER
        # Mock parameter reading: Mode 0 (Locked Target) -> GUID
        self.packet_reader.read_uint8.side_effect = [0]
        self.packet_reader.read_uint64.side_effect = [12345] # Mock target GUID

        # Setup WorldSessionStateHandler to return a target
        mock_target = MagicMock()
        mock_target.is_player.return_value = True
        mock_target.get_name.return_value = "Victim"
        mock_target.map_id = 1
        mock_target.location.x = 50
        mock_target.location.y = 60
        mock_target.location.z = 70
        mock_target.location.o = 0.5
        
        # We need to access the mock we injected into sys.modules
        from game.world.WorldSessionStateHandler import WorldSessionStateHandler
        WorldSessionStateHandler.find_player_by_guid.return_value = mock_target

        BindOtherHandler.handle(self.session, self.packet_reader)
        
        # Verify set_death_bind called
        mock_target.set_death_bind.assert_called()
        args = mock_target.set_death_bind.call_args[0]
        self.assertEqual(args[0], 1) # Map
        self.assertEqual(args[1].x, 50) # Location from target
        
        print(f"[INFO] GM BindOther: TestGM bound Victim to {args[0]} {args[1].x}...")

    @patch('game.world.opcode_handling.handlers.gm.DBLookupHandler.WorldDatabaseManager')
    def test_db_lookup_gm_item(self, mock_db):
        self.packet_reader.opcode = OpCode.CMSG_DBLOOKUP
        self.packet_reader.read_string.return_value = "item Thunderfury"
        
        # Mock DB response
        item_mock = MagicMock()
        item_mock.entry = 12345
        item_mock.name = "Thunderfury, Blessed Blade of the Windseeker"
        mock_db.ItemTemplateHolder.item_template_get_by_name.return_value = [item_mock]

        DBLookupHandler.handle(self.session, self.packet_reader)
        
        self.session.send_packet.assert_called()
        # Decode packet data to verify
        # We can just verify it called send_packet for now.

if __name__ == '__main__':
    unittest.main()
