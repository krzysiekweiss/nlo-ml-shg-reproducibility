# Source provenance

The scientific implementation derives from the development repository, revision `3f9bd0a647706080d9fb5a363e648db0f25cac93`. Its successor `1b003bb14e6e83a30a918b4e09dd4c3caba826fd` changes notebook execution state only; every cell source is unchanged.

The publication workflow derives from revision `04d421393f998c5690bbc41cbba7177fe828bf7e` of the publication repository. It adds portable notebook paths, checked run selection and the publication exporter. The final cross-check invocation is restored from the development workflow.

This source set contains the dependency closure of the seven study notebooks and the publication exporter, including package initializers. Comments and docstrings describe operations and interfaces in English. Notebook outputs and execution counts are cleared. Parameter values and statistical algorithms are retained. Unused diagnostics, duplicate imports and unreachable reporting code are excluded from this source release.

The validation record and file hashes are stored in `../SOURCE_MANIFEST.json`. Full training and inference require the external artifacts and software environment described in `REPRODUCIBILITY.md`.
