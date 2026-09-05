// camera_hud_compass_tape.scad
// Cylindrical curved tactical heading tape / compass bar for camera-anchored 3D HUD.
// Features graduated bearing notches, cardinal ticks, and central indicator pin.

$fn = 32;

module camera_hud_compass_tape() {
    radius = 1.8;
    tape_h = 0.08;
    thickness = 0.025;
    span_deg = 60;

    // Curved backing arc
    rotate([0, 0, 90 - span_deg/2])
    rotate_extrude(angle = span_deg, $fn = 48)
    translate([radius, 0, 0])
    square([thickness, tape_h], center = true);

    // Graduated tick marks radiating along arc
    for (deg = [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25]) {
        tick_h = (deg % 10 == 0) ? 0.06 : 0.035;
        tick_w = (deg == 0) ? 0.015 : 0.008;

        rotate([0, 0, 90 + deg])
        translate([radius + thickness/2 + 0.01, 0, 0])
        cube([0.02, tick_w, tick_h], center = true);
    }

    // Top heading indicator triangle (lubber line)
    translate([0, radius + 0.04, 0])
    rotate([0, 0, 180])
    cylinder(r1 = 0.03, r2 = 0.0, h = 0.05, center = true, $fn = 3);
}

camera_hud_compass_tape();
