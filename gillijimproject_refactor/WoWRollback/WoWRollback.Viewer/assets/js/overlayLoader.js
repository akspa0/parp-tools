// Loader for overlay and diff JSON data (smart caching - cleared on version/map change)
const cache = new Map();

export async function loadOverlay(path) {
    // Use path as-is for caching - cache busting handled at state level
    if (cache.has(path)) {
        const cached = cache.get(path);
        if (cached === null) {
            throw new Error(`Overlay not found (cached 404): ${path}`);
        }
        return cached;
    }

    const response = await fetch(path, { cache: 'no-store' });
    if (!response.ok) {
        // Cache the 404 to prevent repeated requests
        cache.set(path, null);
        throw new Error(`Failed to load overlay: ${path}`);
    }

    const data = await response.json();
    cache.set(path, data);
    return data;
}

export async function loadDiff(path) {
    if (cache.has(path)) {
        return cache.get(path);
    }

    const response = await fetch(path, { cache: 'no-store' });
    if (!response.ok) {
        throw new Error(`Failed to load diff: ${path}`);
    }

    const data = await response.json();
    cache.set(path, data);
    return data;
}

export function clearCache() {
    cache.clear();
}
