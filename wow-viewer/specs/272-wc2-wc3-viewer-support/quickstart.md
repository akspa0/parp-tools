# Try the WC3 File Browser

The file-browser slice is implemented. Real beta pixel fidelity and existing-world isolation still need the operator's check.

1. Run the rebuilt viewer and open the existing **File Browser** section.
2. Enable **WC3 files (loose / MPQ)**.
3. Paste/select the supplied extracted beta directory in **Folder**, then click **Open WC3 data**. No WoW build selection or world loading occurs.
4. Use **Find path** and **Show → Textures / tilesets**, **Text / tables**, or **MDX models**. Select a stored full path to show its preview. Images have fit/zoom and an alpha checkerboard. BLP0 resolves its numbered companions from the same source; individual numbered mip entries can also be selected. MDX uses the existing renderer in an isolated preview with orbit, distance, elevation and available animation controls.
5. For raw unpacking/copying, choose an **Output folder** outside the source and click **Extract / copy selected**. Existing files are never overwritten; originals and stored relative names are retained.
6. Use **Close WC3 source** to release the session. The original WoW browser remains available by clearing the WC3 checkbox.

For MPQs, select a local WC3 supplemental listfile before opening if internal filenames are insufficient. Loose extracted files do not need any listfile. The Zezula package was not imported because its HTTPS endpoint failed certificate-name validation; no TLS bypass or mandatory download was added.

**Current limits**: The selected real beta BLP0 and adjacent MDX now decode/parse; [follow-up evidence](evidence/receipt-beta-blp0-and-mdx-2026-10-15.md). Other undocumented variants may still fail with diagnostics. BLP0 requires its header and selected mip companion together. MDX visual rendering/texture fidelity still require an operator check; map/world support and automatic historical filename reconstruction remain out of scope. Unsupported entries remain extractable.

Operator check: open a real tileset and build text, compare colors/alpha with a trusted reference, copy a selected file and compare hashes, switch selections/sources, and confirm any already-open WoW scene/camera is unchanged. [Receipt and validation](evidence/receipt-wc3-implementation.md).
