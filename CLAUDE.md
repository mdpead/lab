# CLAUDE.md

Guidance for Claude working in this repo.

## The repo

- `lab/`: the shared package (layers, Triton kernels, models, typed configs, training loop,
  generation). [lab/README.md](lab/README.md) explains how it is organised.
- `projects/en_cy_translation`, `projects/python_reasoning`: the projects built on it. Each has
  its own configs, scripts, tests and `requirements.txt`, and runs from its own folder.
- `tests/`: the `lab` package's tests.

## Git and pull requests

- **Commit as Matthew Pead.** Cloud sessions start with a different git identity and rewrite
  the global git config, so set it in each clone before the first commit, from inside the
  repo: `git config user.name "Matthew Pead"` and `git config user.email "mdpead@gmail.com"`.
  Check with `git var GIT_AUTHOR_IDENT`. Keep the `Co-Authored-By: Claude` line.
- **Ask before committing or pushing**, unless the request already says to (for example "open
  a PR").
- **Never commit to `main`.** Work on a branch and open a PR. Matthew merges PRs.
- **Keep one PR open at a time.** Only stack PRs when asked to.
- **Leave the old `en-cy-translation` and `python-reasoning-lm` repos alone.** Both projects now
  live here; don't open PRs or push branches there.

## Making changes

- **Keep it light.** Prefer the simplest design that does the job, and don't write the same
  setting or logic in two places. Don't add a framework or a large dependency (Lightning,
  Hydra, pydantic and the like) without asking: the point of this repo is that the parts are
  written from scratch.
- **Move code into `lab` when a second project needs it**, not before.
- **Refactors must not change behaviour, and must show it.** Before changing model or
  training code, record what the current code produces (seeded weights, parameter layouts,
  outputs or `results.json`), then compare after. Commit the records as test fixtures so CI
  keeps checking, as in `tests/fixtures/` and `projects/*/tests/fixtures/`.
- **Don't build backward compatibility into the code.** When a format changes, migrate the
  existing files once rather than translating old formats forever.

## Configs

- YAML lives in `projects/<project>/configs/`, checked against the project's `src/config.py`
  schema when a script loads it. A misspelt key or wrong type should fail there, so extend
  the schema along with the code.
- Use block style for settings that carry a comment. Exponent floats need a dot: `3.0e-4`,
  not `3e-4`, which YAML reads as a string.
- Run outputs go to `projects/<project>/models/`, which is not in git. They exist only on
  Matthew's GPU machine.

## Checks before pushing

```bash
ruff check lab tests tools && ruff format --check lab tests tools   # repo root
pytest                                                              # repo root
cd projects/<project> && ruff check . && ruff format --check . && pytest
```

- Tests that build the decoder-only model or call a Triton kernel are marked `cuda` and skip
  without a GPU, so CI and cloud sessions cannot run them. When a change touches that code,
  say so and ask Matthew to run `pytest` on the GPU machine before merging.
- In cloud sessions the PyTorch download server is blocked; install the pinned `torch` from
  PyPI instead.
