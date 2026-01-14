from game.world.opcode_handling.HandlerValidator import HandlerValidator
from utils.Logger import Logger
from game.world.managers.objects.units.DamageInfoHolder import DamageInfoHolder
from utils.constants.SpellCodes import SpellSchoolMask, SpellSchools
from utils.constants.MiscCodes import AttackTypes


class NukeHandler:

    @staticmethod
    def handle(world_session, reader):
        player_mgr, res = HandlerValidator.validate_session(world_session, reader.opcode)
        if not player_mgr:
            return res

        if not world_session.account_mgr.is_gm():
            Logger.anticheat(f'Player {player_mgr.get_name()} ({player_mgr.guid}) tried to use Nuke.')
            return 0

        target = player_mgr.get_target()
        if not target:
             # If no target, nuke 10 yards around? Or just warn?
             # Client usually sends nuke on selected target. 
             # Let's fallback to current target guid in player_mgr if get_target() is complex,
             # but usually get_target() returns the Unit object or None.
             # If None, maybe user meant to nuke their self? Safer to do nothing.
             Logger.info(f"GM Nuke: Player {player_mgr.name} has no target.")
             return 0

        # NUKE EM
        damage = 999999
        damage_info = DamageInfoHolder(attacker=player_mgr, target=target,
                                       attack_type=AttackTypes.BASE_ATTACK,
                                       damage_school_mask=SpellSchoolMask.SPELL_SCHOOL_MASK_NORMAL)
        damage_info.base_damage = damage
        damage_info.total_damage = damage
        
        target.deal_damage(target, damage_info)
        Logger.info(f"GM Nuke: {player_mgr.name} nuked {target.name} for {damage} damage.")

        return 0
