// hud_frame_bracket.scad
// 3D HUD corner framing bracket for in-scene camera UI cards.
// Authored for WoWViewer camera-rigged 3D HUD.

$fn = 16;

module hud_frame_bracket() {
    difference() {
        union() {
            // Horizontal arm
            translate([0.5, 0.05, 0])
            cube([1.0, 0.1, 0.06], center = true);

            // Vertical arm
            translate([0.05, 0.5, 0])
            cube([0.1, 1.0, 0.06], center = true);

            // Corner beveled boss
            translate([0.08, 0.08, 0])
            cylinder(r = 0.14, h = 0.08, center = true);
        }

        // Inner bevel cut
        translate([0.25, 0.25, 0])
        rotate([0, 0, 45])
        cube([0.15, 0.15, 0.2], center = true);
    }
}

hud_frame_bracket();
