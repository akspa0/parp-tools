// camera_hud_curved_bezel.scad
// Sleek curved floating HUD visor frame with chamfered corner chevrons and sensor notches.
// Designed to anchor into camera-local space for in-scene 3D HUD surfaces.

$fn = 32;

module curved_arc(radius, angle_span, thickness, height) {
    rotate_extrude(angle = angle_span, $fn = 48)
    translate([radius, 0, 0])
    polygon(points = [
        [0, -height/2],
        [thickness, -height/2 + thickness*0.2],
        [thickness, height/2 - thickness*0.2],
        [0, height/2]
    ]);
}

module chevron_wing() {
    difference() {
        union() {
            // Outer wing slab
            cube([0.35, 0.08, 0.04], center = true);
            // Angled tab
            translate([0.15, 0.06, 0])
            rotate([0, 0, 35])
            cube([0.22, 0.06, 0.04], center = true);
        }
        // Telemetry slot cutouts
        translate([-0.05, 0, 0])
        cube([0.08, 0.03, 0.08], center = true);
        translate([0.06, 0, 0])
        cube([0.08, 0.03, 0.08], center = true);
    }
}

module camera_hud_curved_bezel() {
    radius = 2.0;
    span = 50;

    // Top visor arc
    translate([0, 0.6, 0])
    rotate([0, 0, 90 - span/2])
    curved_arc(radius, span, 0.05, 0.04);

    // Bottom chin arc
    translate([0, -0.6, 0])
    rotate([0, 0, 270 - span/2])
    curved_arc(radius, span, 0.05, 0.04);

    // Left chevron wing bracket
    translate([-1.2, 0, 0])
    rotate([0, 0, 90])
    chevron_wing();

    // Right chevron wing bracket
    translate([1.2, 0, 0])
    rotate([0, 0, -90])
    chevron_wing();

    // Sensor boss nodes on corners
    for (sx = [-1, 1]) {
        for (sy = [-1, 1]) {
            translate([sx * 1.05, sy * 0.55, 0])
            cylinder(r = 0.04, h = 0.06, center = true, $fn = 16);
        }
    }
}

camera_hud_curved_bezel();
