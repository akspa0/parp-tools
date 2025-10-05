{{ ... }}

### Step 5: Incrementally Refactor All Remaining Layers

- **Action**: One by one, migrate each of the remaining overlays (`areaId`, `holes`, `liquids`, `shadowMap`) into its own file under the `js/plugins/` directory, following the pattern established in Step 4.
- **Purpose**: To methodically and safely dismantle the old `overlayManager.js` until it is empty and can be deleted.

### Step 6: A/B Refactor for M2/WMO Object Overlays

- **Action**: The complex M2/WMO object overlays will be refactored in parallel to mitigate risk and performance concerns.
- **Purpose**: To build a new, high-performance implementation from scratch without breaking the existing, functional system, allowing for direct A/B comparison.
- **Details**:
    1.  The existing object overlay logic in `main.js` and `overlayManager.js` will be **left untouched** for now.
    2.  A new, separate plugin will be created: `js/plugins/objects-next.js`. This plugin will be built from the ground up with a focus on performance (e.g., using WebGL or point clustering techniques).
    3.  A new CLI flag (e.g., `--viewer-next-objects`) will be added to `WoWRollback.Cli`. When used, it will instruct the manifest generator to include the `objects-next` plugin in the `overlay_manifest.json`.
    4.  This allows us to toggle between the old and new implementations for direct comparison and testing.
    5.  Once the `objects-next` plugin is deemed superior and stable, we will remove the old implementation and the CLI flag, making the new system the default.

## 3. Guiding Principles for Implementation
{{ ... }}
