// cursor_reticle.scad
// 3D targeting reticle with cardinal tick pointers and center pip.
// Authored for WoWViewer in-scene 3D spatial UI.

$fn = 36;

module cursor_reticle() {
    // Center pip
    sphere(r = 0.1, $fn = 16);

    // Outer ring with cardinal notches
    difference() {
        cylinder(r = 1.0, h = 0.15, center = true);
        cylinder(r = 0.85, h = 0.25, center = true);

        // Cardinal notch cutouts
        for (a = [0, 9], a = [0, 90, 180, 270]) {
            rotate([0, 0, a])
            translate([0.92, 0, 0])
            cube([0.25, 0.12, 0.3], center = true);
        }
    }

    // Inner ring
    difference() {
        cylinder(r = 0.65, h = 0.1, center = true);
        cylinder(r = 0.58, h = 0.2, center = true);
    }

    // 4 Cardinal tick pointers pointing inward
    for (a = [0, 90, 180, 270]) {
        rotate([0, 0, a])
        translate([0.45, 0, 0])
        cylinder(r1 = 0.06, r2 = 0.0, h = 0.25, center = true);
    }
}

cursor_reticle();
