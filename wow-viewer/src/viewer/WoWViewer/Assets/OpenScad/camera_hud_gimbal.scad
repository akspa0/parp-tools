// camera_hud_gimbal.scad
// 3D Camera HUD Attitude & Heading Gimbal.
// Mounted in camera-local space as a 3D in-scene instrument.
// Authored for WoWViewer camera-rigged 3D HUD.

$fn = 32;

module camera_hud_gimbal() {
    // Outer yaw ring (equator)
    difference() {
        cylinder(r = 1.0, h = 0.08, center = true);
        cylinder(r = 0.90, h = 0.15, center = true);
    }

    // Outer pitch ring (vertical meridian)
    difference() {
        rotate([90, 0, 0])
        cylinder(r = 1.0, h = 0.06, center = true);
        rotate([90, 0, 0])
        cylinder(r = 0.92, h = 0.15, center = true);
    }

    // North arrow indicator (prominent arrow pointing +Y)
    translate([0, 0.75, 0])
    cylinder(r1 = 0.12, r2 = 0.0, h = 0.35, center = true, $fn = 4);

    // South, East, West pips
    for (a = [90, 180, 270]) {
        rotate([0, 0, a])
        translate([0, 0.85, 0])
        sphere(r = 0.06, $fn = 12);
    }

    // Center horizon attitude bar
    cube([1.4, 0.04, 0.04], center = true);
    sphere(r = 0.08, $fn = 16);
}

camera_hud_gimbal();
