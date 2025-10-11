# PM4 Core Library Improvement Roadmap

## Immediate Recommendations
1. **Centralize Mesh Extraction Logic:** Move all mesh extraction and boundary export code from analysis/tests into Core.
2. **Standardize Index Types:** Audit all index handling for type safety (int vs uint).
3. **Expose Chunk Relationships:** Provide clear APIs for traversing and mapping chunk relationships (MSUR→MSVI→MSVT, MSLK→MSVI→MSVT, etc.).
4. **Robust Error Handling:** Implement and document error handling for missing/malformed chunks.
5. **Resource Management:** Ensure all file/stream resources are properly disposed.
6. **Documentation:** Add XML docs and update the memory bank with all chunk relationships and data flows.

## Roadmap
- [ ] Refactor mesh extraction logic into Core as a reusable API.
- [ ] Implement robust error handling and validation for all chunk parsing.
- [ ] Audit and refactor index types for consistency and safety.
- [ ] Expose clear, documented APIs for chunk relationship traversal.
- [ ] Add resource management patterns (IDisposable, using statements).
- [ ] Expand and update documentation in both code and memory bank.
- [ ] Use test-driven development to codify and address edge cases.
- [ ] Regularly review and update the roadmap based on new findings. 