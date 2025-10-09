import { CoordinateSystem } from '../core/CoordinateSystem.js';

/**
 * Test suite for CoordinateSystem
 * Run this in the browser console or via a test runner
 */

function runTests() {
    console.log('=== CoordinateSystem Tests ===\n');
    
    const coords = new CoordinateSystem({ coordMode: 'wowtools' });
    let passed = 0;
    let failed = 0;
    
    // Test 1: World to Tile conversion
    console.log('Test 1: World to Tile conversion');
    const worldPos = { x: 0, y: 0 }; // Center of map
    const tile = coords.worldToTile(worldPos.x, worldPos.y);
    if (tile.row === 32 && tile.col === 32) {
        console.log('✓ PASS: Center (0,0) maps to tile (32,32)');
        passed++;
    } else {
        console.log(`✗ FAIL: Expected (32,32), got (${tile.row},${tile.col})`);
        failed++;
    }
    
    // Test 2: Tile to World conversion
    console.log('\nTest 2: Tile to World conversion');
    const world = coords.tileToWorld(32, 32);
    const tolerance = 1.0; // Allow small floating point differences
    if (Math.abs(world.worldX) < tolerance && Math.abs(world.worldY) < tolerance) {
        console.log('✓ PASS: Tile (32,32) maps to world center (~0,~0)');
        passed++;
    } else {
        console.log(`✗ FAIL: Expected (~0,~0), got (${world.worldX.toFixed(2)},${world.worldY.toFixed(2)})`);
        failed++;
    }
    
    // Test 3: Round-trip conversion
    console.log('\nTest 3: Round-trip World → Tile → World');
    const originalWorld = { x: 5000, y: -3000 };
    const convertedTile = coords.worldToTile(originalWorld.x, originalWorld.y);
    const backToWorld = coords.tileToWorld(convertedTile.row, convertedTile.col);
    const xDiff = Math.abs(originalWorld.x - backToWorld.worldX);
    const yDiff = Math.abs(originalWorld.y - backToWorld.worldY);
    if (xDiff < coords.TILE_SIZE && yDiff < coords.TILE_SIZE) {
        console.log('✓ PASS: Round-trip conversion within tile size tolerance');
        passed++;
    } else {
        console.log(`✗ FAIL: Round-trip error too large: X=${xDiff.toFixed(2)}, Y=${yDiff.toFixed(2)}`);
        failed++;
    }
    
    // Test 4: Tile bounds calculation
    console.log('\nTest 4: Tile bounds calculation');
    const bounds = coords.tileBounds(30, 35);
    if (Array.isArray(bounds) && bounds.length === 2 && 
        Array.isArray(bounds[0]) && Array.isArray(bounds[1])) {
        console.log('✓ PASS: Tile bounds returns valid format');
        console.log(`  Bounds: [${bounds[0]}, ${bounds[1]}]`);
        passed++;
    } else {
        console.log('✗ FAIL: Tile bounds format invalid');
        failed++;
    }
    
    // Test 5: Elevation normalization
    console.log('\nTest 5: Elevation normalization');
    const minElev = coords.normalizeElevation(coords.MIN_ELEVATION);
    const maxElev = coords.normalizeElevation(coords.MAX_ELEVATION);
    const midElev = coords.normalizeElevation(750); // Mid-range
    if (minElev === 0 && maxElev === 1 && midElev > 0 && midElev < 1) {
        console.log('✓ PASS: Elevation normalization works correctly');
        console.log(`  Min: ${minElev}, Mid: ${midElev.toFixed(2)}, Max: ${maxElev}`);
        passed++;
    } else {
        console.log(`✗ FAIL: Elevation normalization incorrect: ${minElev}, ${midElev}, ${maxElev}`);
        failed++;
    }
    
    // Test 6: Color brightness adjustment
    console.log('\nTest 6: Color brightness adjustment');
    const baseColor = '#FF0000'; // Red
    const dimmed = coords.adjustColorBrightness(baseColor, 0.5);
    console.log(`  Input: ${baseColor}, Brightness: 0.5`);
    console.log(`  Output: ${dimmed}`);
    console.log(`  Type: ${typeof dimmed}`);
    console.log(`  Starts with #: ${dimmed.startsWith('#')}`);
    console.log(`  Length: ${dimmed.length}`);
    if (dimmed.startsWith('#') && dimmed.length === 7) {
        console.log('✓ PASS: Color brightness adjustment returns valid hex color');
        console.log(`  Original: ${baseColor}, Dimmed (50%): ${dimmed}`);
        passed++;
    } else {
        console.log(`✗ FAIL: Invalid color format: ${dimmed}`);
        console.log(`  Expected: 7-character hex string starting with #`);
        console.log(`  Got: ${dimmed} (length: ${dimmed.length})`);
        failed++;
    }
    
    // Test 7: Lat/Lng to Tile conversion (wowtools mode)
    console.log('\nTest 7: Lat/Lng to Tile conversion (wowtools mode)');
    const latLng = { lat: 31, lng: 33 };
    const tileFromLatLng = coords.latLngToTile(latLng.lat, latLng.lng);
    if (tileFromLatLng.row === 32 && tileFromLatLng.col === 33) {
        console.log('✓ PASS: Lat/Lng (31,33) maps to tile (32,33) in wowtools mode');
        passed++;
    } else {
        console.log(`✗ FAIL: Expected (32,33), got (${tileFromLatLng.row},${tileFromLatLng.col})`);
        failed++;
    }
    
    // Test 8: Tile to Lat/Lng conversion (wowtools mode)
    console.log('\nTest 8: Tile to Lat/Lng conversion (wowtools mode)');
    const latLngFromTile = coords.tileToLatLng(32, 32);
    if (latLngFromTile.lat === 31 && latLngFromTile.lng === 32) {
        console.log('✓ PASS: Tile (32,32) maps to Lat/Lng (31,32) in wowtools mode');
        passed++;
    } else {
        console.log(`✗ FAIL: Expected (31,32), got (${latLngFromTile.lat},${latLngFromTile.lng})`);
        failed++;
    }
    
    // Summary
    console.log('\n=== Test Summary ===');
    console.log(`Passed: ${passed}`);
    console.log(`Failed: ${failed}`);
    console.log(`Total: ${passed + failed}`);
    
    if (failed === 0) {
        console.log('\n✓ All tests passed!');
    } else {
        console.log(`\n✗ ${failed} test(s) failed`);
    }
    
    return { passed, failed };
}

// Auto-run tests if loaded as module
if (typeof window !== 'undefined') {
    window.runCoordinateSystemTests = runTests;
    console.log('CoordinateSystem tests loaded. Run window.runCoordinateSystemTests() to execute.');
}

export { runTests };
