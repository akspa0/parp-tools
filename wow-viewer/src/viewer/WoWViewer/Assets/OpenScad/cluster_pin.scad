// cluster_pin.scad
// In-scene 3D tactical candidate pin.
// Ground contact point is at (0, 0, 0), stem extends up along +Z.
// Authored for WoWViewer in-scene 3D spatial UI.

$fn = 20;

module cluster_pin() {
    // Ground contact tip at origin
    cylinder(r1 = 0.02, r2 = 0.08, h = 0.3, center = false);

    // Main slender stem
    translate([0, 0, 0.3])
    cylinder(r = 0.06, h = 1.0, center = false);

    // Upper ring collar
    translate([0, 0, 1.3])
    cylinder(r = 0.12, h = 0.1, center = false);

    // Top beacon crown (faceted diamond / double cone)
    translate([0, 0, 1.4])
    cylinder(r1 = 0.08, r2 = 0.22, h = 0.25, center = false, $fn = 8);

    translate([0, 0, 1.65])
    cylinder(r1 = 0.22, r2 = 0.0, h = 0.3, center = false, $fn = 8);
}

cluster_pin();
