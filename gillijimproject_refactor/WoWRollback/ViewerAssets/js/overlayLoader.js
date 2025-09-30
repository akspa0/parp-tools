// Loader for overlay and diff JSON data
const cache = new Map();

export async function loadOverlay(path) {
    if (cache.has(path)) {
        return cache.get(path);
    }

    const response = await fetch(path);
    if (!response.ok) {
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

    const response = await fetch(path);
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
