# Pylint Report — `develop`

Current branch snapshot after the release hardening work, retrieval/indexing cleanup, the latest translation file switching fix, and the latest pylint pass.

Configured in:

- `.pylintrc`

Ran:

```bash
UV_CACHE_DIR=/private/tmp/uvcache uv run pylint --persistent=n \
  scripts/install/install_bootstrap_models.py \
  src/informity/api/env_vars_metadata.py \
  src/informity/api/operation_state.py \
  src/informity/api/routes_chat.py \
  src/informity/api/routes_settings.py \
  src/informity/api/routes_system.py \
  src/informity/api/schemas.py \
  src/informity/config.py \
  src/informity/db/vectors.py \
  src/informity/indexer/chunker.py \
  src/informity/indexer/pipeline.py \
  src/informity/llm/engine.py \
  src/informity/llm/five_q_classifier.py \
  src/informity/llm/model_bootstrap.py \
  src/informity/llm/prompt_builder.py \
  src/informity/llm/rag_runtime/generation_closeout.py \
  src/informity/llm/retrieval.py \
  src/informity/llm/tokenization.py \
  src/informity/llm/types.py \
  src/informity/main.py \
  --output-format=json
```

Result:

- Exit code: `28`
- Pylint score: `9.66/10`
- Total findings: `243`

## Findings by type

| Count | ID | Symbol | Notes |
| ---: | --- | --- | --- |
| 46 | `C0413` | `wrong-import-position` | Imports are not at the top of the file. |
| 46 | `C0415` | `import-outside-toplevel` | Imports intentionally deferred inside functions. |
| 27 | `R0914` | `too-many-locals` | Functions with too many local variables. |
| 24 | `R0913` | `too-many-arguments` | Functions with too many parameters. |
| 20 | `R0915` | `too-many-statements` | Functions with too many statements. |
| 18 | `R0912` | `too-many-branches` | Functions with too many branches. |
| 17 | `R0917` | `too-many-positional-arguments` | Functions with too many positional args. |
| 8 | `C0302` | `too-many-lines` | Modules exceed the configured line count. |
| 9 | `R0801` | `duplicate-code` | Repeated code blocks across files. |
| 7 | `R1702` | `too-many-nested-blocks` | Nested control flow is too deep. |
| 1 | `W0718` | `broad-exception-caught` | Generic `except` clauses catch too much. |
| 5 | `R0911` | `too-many-return-statements` | Functions with too many returns. |
| 5 | `W0603` | `global-statement` | Global variables are being reassigned. |
| 5 | `R0903` | `too-few-public-methods` | Classes with very small public APIs. |
| 5 | `R0902` | `too-many-instance-attributes` | Classes with too many fields. |
## Most affected files

| Count | File |
| ---: | --- |
| 73 | `src/informity/main.py` |
| 43 | `src/informity/llm/engine.py` |
| 29 | `src/informity/api/routes_chat.py` |
| 20 | `src/informity/indexer/pipeline.py` |
| 15 | `src/informity/api/routes_system.py` |
| 15 | `scripts/install/install_bootstrap_models.py` |
| 12 | `src/informity/llm/retrieval.py` |
| 10 | `src/informity/db/vectors.py` |
| 9 | `src/informity/config.py` |
| 6 | `src/informity/api/routes_settings.py` |
| 4 | `src/informity/indexer/chunker.py` |
| 4 | `src/informity/llm/model_bootstrap.py` |
| 1 | `src/informity/llm/prompt_builder.py` |
| 1 | `src/informity/api/env_vars_metadata.py` |
| 1 | `src/informity/llm/five_q_classifier.py` |

## Notes

- The latest run reflects the current develop snapshot after the release hardening, translation fixes, and pylint cleanup pass.
- Categories cleared in this pass: `line-too-long`, `protected-access`, `unused-argument`, `unused-variable`, `invalid-name`, and `redefined-outer-name`.
- Remaining findings are dominated by import placement, complexity, and a few style and duplication issues.
