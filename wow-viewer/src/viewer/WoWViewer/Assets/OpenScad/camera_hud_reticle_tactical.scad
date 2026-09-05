// camera_hud_reticle_tactical.scad
// Multi-ring segmented tactical camera reticle with elevation notches and central diamond.
// Designed for in-scene camera-projected 3D HUD targeting and inspection.

$fn = 32;

module camera_hud_reticle_tactical() {
    h = 0.02;

    // Outer segmented quadrant arcs
    for (a = [0, 90, 180, 270]) {
        rotate([0, 0, a + 15])
        difference() {
            cylinder(r = 0.60, h = h, center = true);
            cylinder(r = 0.56, h = h + 0.02, center = true);
            
            // Mask out 30 degrees of each quadrant to leave a 60 deg arc
            rotate([0, 0, 60])
            translate([1, 1, 0])
            cube([2, 2, h + 0.04], center = true);
            rotate([0, 0, 150])
            translate([1, 1, 0])
            cube([2, 2, h + 0.04], center = true);
        }
    }

    // Inner fine ring
    difference() {
        cylinder(r = 0.32, h = h, center = true);
        cylinder(r = 0.30, h = h + 0.02, center = true);
    }

    // Cardinal tick marks
    for (a = [0, 90, 180, 270]) {
        rotate([0, 0, a])
        translate([0.42, 0, 0])
        cube([0.16, 0.02, h], center = true);
    }

    // Elevation ladder notches (vertical ticks)
    for (y = [-0.22, -0.15, -0.08, 0.08, 0.15, 0.22]) {
        translate([0, y, 0])
        cube([0.06, 0.012, h], center = true);
    }

    // Center diamond aperture
    rotate([0, 0, 45])
    difference() {
        cube([0.08, 0.08, h], center = true);
        cube([0.05, 0.05, h + 0.02], center = true);
    }
}

camera_hud_reticle_tactical();
