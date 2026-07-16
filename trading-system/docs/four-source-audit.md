# Four-Source Trading System Audit

Audit date: 2026-07-16 UTC

## Sources checked

| Logical source | Files | Result |
|---|---:|---|
| WSL canonical `/home/lmy/hermes_v6/sandbox_trading` | 6475 | Confirmed trading code source of truth |
| Windows C: Hermes v6 | 3482 | No independent `sandbox_trading` tree; 12 exact-hash overlaps with canonical source |
| Windows D: Hermes v6 link | 1 | Incomplete mirror, not a source of truth |
| TRAE Work CN `.trae-cn` | 14711 | Memory, skills, runtime and session artifacts; not a second trading source tree |

The audit covered 24669 files and recorded path, size, SHA-256 and classification in `manifests/four_sources_audit.json`. The manifest uses logical root labels and relative paths so host usernames and absolute Windows paths are not published.

## Findings

- TRAE project memory explicitly identifies `/home/lmy/hermes_v6/sandbox_trading` as the unique trading-code source.
- The relevant TRAE project memory contains workflow rules, CodeGraph findings, test/gate expectations and known system boundaries. It does not contain a separate source tree that supersedes the WSL tree.
- Cross-root hash overlap with the canonical trading tree is 364 files: C: 12, TRAE: 352, D: 0. Most TRAE overlaps are runtime or built-in artifacts, not missing trading modules.
- Raw TRAE sessions, caches, built-in skills and possible sensitive candidates are accounted for but are deliberately not uploaded as active project content.

## Upload boundary

`content/` remains the unique maintainable trading snapshot. The audit manifest is the complete accounting layer. Raw local sessions and runtime caches remain in their original locations and are not treated as source. Future refreshes must rerun the four-source audit before changing the snapshot.
