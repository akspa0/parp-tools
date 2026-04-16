namespace WowViewer.Core.M2;

public static class M2AnimationNameResolver
{
    public static string GetSequenceDisplayName(ushort animationId, ushort variationIndex)
    {
        string baseName = animationId switch
        {
            0 => "Stand",
            1 => "Death",
            2 => "Spell",
            3 => "Stop",
            4 => "Walk",
            5 => "Run",
            6 => "Dead",
            7 => "Rise",
            8 => "StandWound",
            9 => "CombatWound",
            10 => "CombatCritical",
            11 => "ShuffleLeft",
            12 => "ShuffleRight",
            13 => "WalkBackwards",
            14 => "Stun",
            15 => "HandsClosed",
            16 => "AttackUnarmed",
            17 => "Attack1H",
            18 => "Attack2H",
            19 => "Attack2HL",
            20 => "ParryUnarmed",
            21 => "Parry1H",
            22 => "Parry2H",
            23 => "Parry2HL",
            24 => "ShieldBlock",
            25 => "ReadyUnarmed",
            26 => "Ready1H",
            27 => "Ready2H",
            28 => "Ready2HL",
            29 => "ReadyBow",
            30 => "Dodge",
            31 => "SpellPrecast",
            32 => "SpellCast",
            33 => "SpellCastArea",
            34 => "NPCWelcome",
            35 => "NPCGoodbye",
            36 => "Block",
            37 => "JumpStart",
            38 => "Jump",
            39 => "JumpEnd",
            40 => "Fall",
            41 => "SwimIdle",
            42 => "Swim",
            43 => "SwimLeft",
            44 => "SwimRight",
            45 => "SwimBackwards",
            _ => $"Anim{animationId}",
        };

        return variationIndex == 0
            ? baseName
            : $"{baseName}_{variationIndex:D2}";
    }
}