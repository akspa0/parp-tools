// cursor_pointer.scad
// Sleek, beveled 3D cursor pointer with origin at the active tip (0, 0, 0).
// Authored for WoWViewer in-scene 3D spatial UI.

$fn = 24;

module cursor_pointer() {
    // Tip is at (0, 0, 0), arrow extends along +X (right) and -Y (down)
    difference() {
        // Main faceted arrow body
        linear_extrude(height = 0.25, center = true, convexity = 10)
        polygon(points = [
            [0.0, 0.0],       // Tip
            [0.7, -0.9],      // Bottom right edge
            [0.43, -0.9],     // Inner notch right
            [0.67, -1.5],     // Tail bottom right
            [0.43, -1.6],     // Tail bottom left
            [0.13, -1.0],     // Tail inner left
            [-0.1, -1.0]      // Far left wing
        ]);

        // Bevel top face slightly along Z
        translate([0, -0.9, 0.2])
        rotate([10, 0, 0])
        cube([2.5, 2.5, 0.2], center = true);
    }
}

cursor_pointer();
