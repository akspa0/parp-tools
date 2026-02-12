# M2Sequence Structure

## Overview
The M2Sequence structure defines an animation sequence in an M2 model. Animation sequences control the movement of bones, color changes, texture coordinate shifts, and other animated properties. Each sequence corresponds to a specific animation like walking, running, casting spells, etc.

## Structure

```cpp
struct M2Sequence {
  uint16_t animation_id;         // Animation ID (see table below)
  uint16_t sub_animation_id;     // Sub-animation ID
  uint32_t length;               // Length of the animation in milliseconds
  float moving_speed;            // Speed when moving with this animation
  uint32_t flags;                // Flags controlling animation behavior
  int16_t probability;           // Probability for this animation to be selected (used for random animations)
  uint16_t padding;              // Unused padding
  uint32_t replay_count;         // Number of times to repeat the animation (0 = infinite)
  uint32_t replay_timestamp;     // Timestamp of the end of the animation replay cycle
  uint32_t blend_time;           // Time to blend between animations in milliseconds
  C3Vector bounds_min;           // Minimum bounds of animation
  C3Vector bounds_max;           // Maximum bounds of animation
  float bound_radius;            // Bounding sphere radius
  int16_t next_animation;        // Animation ID to play next (-1 = none)
  uint16_t alias_next;           // Related to next_animation
};
```

## Fields

- **animation_id**: Identifies the animation type (e.g., 0 = stand, 1 = death, etc.)
- **sub_animation_id**: Identifies a variant of the animation
- **length**: Duration of the animation in milliseconds
- **moving_speed**: Movement speed for the model when playing this animation
- **flags**: Bit flags controlling animation behavior
- **probability**: Likelihood this animation will be chosen (for random selection)
- **padding**: Unused field for alignment
- **replay_count**: Number of times to repeat the animation (0 = infinite)
- **replay_timestamp**: End timestamp for animation replay cycle
- **blend_time**: Time to blend between this animation and the next in milliseconds
- **bounds_min**: Minimum coordinates of the animation's bounding box
- **bounds_max**: Maximum coordinates of the animation's bounding box
- **bound_radius**: Radius of the animation's bounding sphere
- **next_animation**: ID of the animation to play after this one completes (-1 = none)
- **alias_next**: Additional information related to next_animation

## Animation Flags

The flags field contains bit flags controlling animation behavior:
- **0x01**: Looping animation
- **0x02**: Sync with next animation. When next_animation is valid (>= 0), this animation will speed up or slow down to align with the start of the next animation.
- **0x04**: Unknown, often set for death animations
- **0x08**: Unknown, often set for stand/idle animations
- **0x10**: Unknown
- **0x20**: Sequence stored as half-precision quaternions (added in later versions)
- **0x40**: Blended animation (smooth transition between animations)
- **0x80**: Has root tracks - special animation type used with character mounts
- **0x100**: Unknown, often set for movement animations

## Common Animation IDs

| ID | Name | Description |
|----|------|-------------|
| 0 | Stand | Default standing/idle animation |
| 1 | Death | Death animation |
| 2 | Spell | Spell casting animation |
| 3 | Stop | Stopping movement animation |
| 4 | Walk | Walking animation |
| 5 | Run | Running animation |
| 6 | Dead | Dead (on ground) animation |
| 7 | Rise | Rising from the ground animation |
| 8 | StandWound | Standing while wounded animation |
| 9 | CombatWound | Combat wounded animation |
| 10 | CombatCritical | Critically wounded in combat animation |
| 11 | ShuffleLeft | Shuffle left animation |
| 12 | ShuffleRight | Shuffle right animation |
| 13 | Walkbackwards | Walking backwards animation |
| 14 | Stun | Stunned animation |
| 15 | HandsClosed | Hands closed animation |
| 16 | AttackUnarmed | Unarmed attack animation |
| 17 | Attack1H | One-handed weapon attack animation |
| 18 | Attack2H | Two-handed weapon attack animation |
| 19 | Attack2HL | Two-handed large weapon attack animation |
| 20 | ParryUnarmed | Unarmed parry animation |
| 21 | Parry1H | One-handed weapon parry animation |
| 22 | Parry2H | Two-handed weapon parry animation |
| 23 | Parry2HL | Two-handed large weapon parry animation |
| 24 | ShieldBlock | Shield block animation |
| 25 | ReadyUnarmed | Unarmed ready/combat stance animation |
| 26 | Ready1H | One-handed weapon ready animation |
| 27 | Ready2H | Two-handed weapon ready animation |
| 28 | Ready2HL | Two-handed large weapon ready animation |
| 29 | ReadyBow | Bow ready animation |
| 30 | Dodge | Dodge animation |
| 31 | SpellPrecast | Spell pre-casting animation |
| 32 | SpellCast | Spell casting animation |
| 33 | SpellCastArea | Area spell casting animation |
| 34 | NPCWelcome | NPC welcome animation |
| 35 | NPCGoodbye | NPC goodbye animation |
| 36 | Block | Blocking animation |
| 37 | JumpStart | Jump start animation |
| 38 | Jump | Jump loop animation |
| 39 | JumpEnd | Jump end animation |
| 40 | Fall | Falling animation |
| 41 | SwimIdle | Swimming idle animation |
| 42 | Swim | Swimming animation |
| 43 | SwimLeft | Swimming left animation |
| 44 | SwimRight | Swimming right animation |
| 45 | SwimBackwards | Swimming backwards animation |
| 46 | AttackBow | Bow attack animation |
| 47 | FireBow | Firing bow animation |
| 48 | ReadyRifle | Rifle ready animation |
| 49 | AttackRifle | Rifle attack animation |
| 50 | Loot | Looting animation |
| 51 | ReadySpellDirected | Directed spell ready animation |
| 52 | ReadySpellOmni | Omni-directional spell ready animation |
| 53 | SpellCastDirected | Directed spell cast animation |
| 54 | SpellCastOmni | Omni-directional spell cast animation |
| 55 | BattleRoar | Battle roar animation |
| 56 | ReadyAbility | Special ability ready animation |
| 57 | Special1H | Special one-handed attack animation |
| 58 | Special2H | Special two-handed attack animation |
| 59 | ShieldBash | Shield bash animation |
| 60 | EmoteTalk | Talk emote animation |
| 61 | EmoteEat | Eat emote animation |
| 62 | EmoteWork | Work emote animation |
| 63 | EmoteUseStanding | Use standing emote animation |

Note: This is a partial list of common animation IDs. The game has many more animations, especially for newer models.

## Implementation Notes

- Animation keyframes are stored in the animation tracks of the various animated properties (bones, colors, etc.)
- The sequence table in the M2 header maps animation IDs to sequence indices
- Animation IDs and sub-animation IDs together uniquely identify each animation
- Sub-animation IDs are often used for different variants of the same basic animation (e.g., different weapon types)
- When a sequence completes, the next_animation field determines what animation plays next
- The blend_time field controls smooth transitions between animations
- Looping animations repeat based on the replay_count field (0 = infinite)
- The bounds fields are used for culling and collision detection during animation
- Some animations can be randomly selected based on the probability field
- Walking/running animations use the moving_speed field to synchronize movement with animation
- Not all animations are available for all model types
- Newer versions of the format may use half-precision quaternions to save space

## Usage in Animation System

To use the animation system:
1. Select an appropriate sequence based on the desired animation ID
2. Use the sequence to determine which frames to play from the various animation tracks
3. Calculate bone positions, colors, and other animated properties at the current timestamp
4. Apply resulting transformations to the model
5. When the animation completes, transition to the next animation based on next_animation field
6. Use blend_time to smoothly transition between animations 