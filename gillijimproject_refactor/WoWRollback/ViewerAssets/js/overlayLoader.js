// Loader for overlay and diff JSON data (with cache-busting)
const cache = new Map();

function withBust(url) {
    const sep = url.includes('?') ? '&' : '?';
    return `${url}${sep}t=${Date.now()}`;
}

export async function loadOverlay(path) {
    const busted = withBust(path);
    if (cache.has(busted)) {
        return cache.get(busted);
    }

    const response = await fetch(busted, { cache: 'no-store' });
    if (!response.ok) {
        throw new Error(`Failed to load overlay: ${path}`);
    }

    const data = await response.json();
    cache.set(busted, data);
    return data;
}

export async function loadDiff(path) {
    const busted = withBust(path);
    if (cache.has(busted)) {
        return cache.get(busted);
    }

    const response = await fetch(busted, { cache: 'no-store' });
    if (!response.ok) {
        throw new Error(`Failed to load diff: ${path}`);
    }

    const data = await response.json();
    cache.set(busted, data);
    return data;
}

export function clearCache() {
    cache.clear();
}
