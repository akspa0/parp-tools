// Fit/projection utilities for tile viewer

const TILE_SPAN = 533.33333;
const HALF_TILES = 32;

function frac(v) { return v - Math.floor(v); }

export function mapWorldToPlane(world, axis) {
    const x = world.x, y = world.y, z = world.z;
    switch (axis) {
        case 'xz': return { px: x, py: z };
        case 'zy': return { px: z, py: y };
        case 'xy':
        default:   return { px: x, py: y };
    }
}

export function computeTileFrac(px, py) {
    const tx = HALF_TILES - (px / TILE_SPAN);
    const ty = HALF_TILES - (py / TILE_SPAN);
    return { tx, ty };
}

export function computeLocalFromTileFrac(tx, ty, flipX, flipY) {
    let lx = 1 - frac(tx); // WoW map naming is map_col_row; X increases to the WEST on map textures
    let ly = frac(ty);
    if (flipX) lx = 1 - lx;
    if (flipY) ly = 1 - ly;
    return { lx, ly };
}

export function applyRotate(lx, ly, rotateDeg) {
    // rotate around center (0.5,0.5)
    const cx = lx - 0.5;
    const cy = ly - 0.5;
    switch ((rotateDeg % 360 + 360) % 360) {
        case 90:  return { lx: 0.5 - cy, ly: 0.5 + cx };
        case 180: return { lx: 0.5 - cx, ly: 0.5 - cy };
        case 270: return { lx: 0.5 + cy, ly: 0.5 - cx };
        default:  return { lx, ly };
    }
}

export function toPixels(lx, ly, width, height, invertY = true) {
    const w1 = Math.max(1, width - 1);
    const h1 = Math.max(1, height - 1);
    const x = lx * w1;
    const y = (invertY ? (1 - ly) : ly) * h1;
    return { x, y };
}

export function computeLocalForTile(world, axis, flipX, flipY, rotate, tileRow, tileCol) {
    const { px, py } = mapWorldToPlane(world, axis);
    const { tx, ty } = computeTileFrac(px, py);
    const row = Math.floor(ty);
    const col = Math.floor(tx);

    // If computed tile doesn't match active tile, mark out-of-range
    if (row !== tileRow || col !== tileCol) {
        return { inRange: false, lx: 0, ly: 0, row, col };
    }

    let { lx, ly } = computeLocalFromTileFrac(tx, ty, flipX, flipY);
    ({ lx, ly } = applyRotate(lx, ly, rotate));

    // Use a small tolerance window beyond [0,1]
    const inRange = lx >= -0.02 && lx <= 1.02 && ly >= -0.02 && ly <= 1.02;
    return { inRange, lx, ly, row, col };
}

export function scoreTransform(points, tileRow, tileCol, width, height, cfg) {
    let inRange = 0, edgeSum = 0;
    for (const p of points) {
        if (!p.world) continue;
        const { inRange: ok, lx, ly } = computeLocalForTile(
            p.world, cfg.axis, cfg.flipX, cfg.flipY, cfg.rotate, tileRow, tileCol
        );
        if (ok) inRange++;
        const ex = Math.min(Math.abs(lx), Math.abs(1 - lx));
        const ey = Math.min(Math.abs(ly), Math.abs(1 - ly));
        edgeSum += 0.5 * (ex + ey);
    }
    const meanEdge = points.length > 0 ? edgeSum / points.length : 0;
    return inRange + 0.1 * meanEdge;
}

export function autoFit(points, tileRow, tileCol, width, height) {
    const axes = ['xy', 'xz', 'zy'];
    const flips = [ [false,false], [true,false], [false,true], [true,true] ];
    const rotations = [0,90,180,270];

    let best = { score: -Infinity, cfg: { axis: 'xy', flipX: false, flipY: false, rotate: 0, invertY: true } };
    for (const axis of axes) {
        for (const [flipX, flipY] of flips) {
            for (const rotate of rotations) {
                const cfg = { axis, flipX, flipY, rotate, invertY: true };
                const score = scoreTransform(points, tileRow, tileCol, width, height, cfg);
                if (score > best.score) best = { score, cfg };
            }
        }
    }
    return best.cfg;
}
