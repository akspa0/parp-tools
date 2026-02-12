from game.world.opcode_handling.HandlerValidator import HandlerValidator
from utils.Logger import Logger
from database.world.WorldDatabaseManager import WorldDatabaseManager
from network.packet.PacketWriter import PacketWriter
from utils.constants.OpCodes import OpCode
from struct import pack


class DBLookupHandler:

    @staticmethod
    def handle(world_session, reader):
        player_mgr, res = HandlerValidator.validate_session(world_session, reader.opcode)
        if not player_mgr:
            return res

        if not world_session.account_mgr.is_gm():
            Logger.anticheat(f'Player {player_mgr.get_name()} ({player_mgr.guid}) tried to use DBLookup.')
            return 0

        # Read CString query
        # Format usually: "item name" or "quest name"
        try:
            full_query = reader.read_string()
        except:
            return 0

        parts = full_query.split(' ', 1)
        if len(parts) < 2:
            return 0
        
        search_type = parts[0].lower() # item, creature, quest
        search_value = parts[1]

        results = [] # List of (entry, name)

        if search_type == "item":
            # Item Search
            items = WorldDatabaseManager.ItemTemplateHolder.item_template_get_by_name(search_value, return_all=True)
            if items:
                # Filter/Limit
                 for item in items:
                     results.append((item.entry, item.name))

        elif search_type == "quest":
            # Quest Search
            # We iterate manually
            search_value_lower = search_value.lower()
            for q_id, q_template in WorldDatabaseManager.QuestTemplateHolder.QUEST_TEMPLATES.items():
                if search_value_lower in q_template.Title.lower():
                    results.append((q_id, q_template.Title))
        
        elif search_type in ["creature", "mob", "npc"]:
            # Creature Search
            creatures = WorldDatabaseManager.CreatureTemplateHolder.creature_get_by_name(search_value, return_all=True)
            if creatures:
                for c in creatures:
                    results.append((c.entry, c.name))

        # Cap results to avoid packet overflow
        results = results[:50]

        # Build SMSG_DBLOOKUP
        # Structure assumption: Count (uint32), [Entry (uint32), Name (String)]
        
        response_data = pack('<I', len(results))
        for entry, name in results:
            response_data += pack('<I', entry)
            response_data += (name.encode('utf-8') + b'\x00')

        world_session.send_packet(PacketWriter.get_packet(OpCode.SMSG_DBLOOKUP, response_data))
        
        Logger.info(f"GM DBLookup: {player_mgr.name} searched {search_type} '{search_value}', found {len(results)} results.")
        return 0
