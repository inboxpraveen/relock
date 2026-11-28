# Contributing to relock

Thanks for helping. Bug reports, integrations, benchmark runs and docs fixes are all welcome.

## Setup

```
git clone https://github.com/inboxpraveen/relock && cd relock
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                                            # SQLite backends
ruff check src tests examples
```

To include the Postgres backend in the test run:

```
docker run -d --rm --name relock-pg -p 127.0.0.1:55432:5432 \
  -e POSTGRES_USER=relock -e POSTGRES_PASSWORD=relock -e POSTGRES_DB=relock postgres:17
RELOCK_TEST_POSTGRES_URL=postgresql://relock:relock@127.0.0.1:55432/relock pytest
```

CI runs the same checks on Python 3.10–3.14, plus the MCP tests on the `mcp` 1.x line.

## Invariants a change must keep

relock is a write discipline. Most of its value is in what it refuses to do, so a pull request
that weakens one of these needs a very good reason and a test:

- Only items recalled in the current episode can be rewritten, and a closed episode accepts no writes.
- Only trusted signals (tool, verifier, authenticated user) count as evidence. Transient errors never count.
- Nothing an untrusted author writes reaches the rewriter or steers the edit. That covers the
  agent's prediction text, its suggested corrections, the clause ids it cites, and
  `detail_trusted=False` signal text.
- There is exactly one active version per item. Old versions are superseded, never deleted.
- Every gate decision is written to the event log.

The [design doc](docs/design.md) has the full lifecycle and threat model.

## Documentation

The root `README.md` is the front page, on GitHub and on PyPI: what relock is, the results,
install, a quickstart and links. Everything else lives in [`docs/`](docs/README.md), one topic per
page. Each fact should live on one page, with other pages linking to it.

- PyPI renders the same README, with relative links rewritten to GitHub (see `pyproject.toml`).
  Avoid GitHub-only syntax there, such as Mermaid or `> [!NOTE]` alerts.
- Keep existing file paths and headings under `docs/` stable. Released READMEs on PyPI link to
  them.
- Diagrams in `docs/assets/` are hand-written SVG with light and dark palettes. Check both themes
  before committing.

## Benchmark changes

The structural RelapseBench mode is deterministic. If you change retrieval, the bench systems or
the scenarios, regenerate the published tables with `relock bench --out docs/results` and
commit them. CI fails when `docs/results/structural.md` no longer matches a fresh run.

Behavioural results on real models are the most valuable contribution right now. Run
`relock bench --mode behavioral --base-url ... --model ...` and open an issue or pull request with
the generated `behavioral.json`, `behavioral.md` and the exact model, quantisation and serving
stack you used.

## Releases (maintainers)

1. Bump `version` in `pyproject.toml` and `__version__` in `src/relock_memory/__init__.py`, and
   add a dated entry to `CHANGELOG.md`. Merge to `main` and wait for CI to pass.
2. Create a GitHub release with the tag `vX.Y.Z`, for example
   `gh release create vX.Y.Z --title vX.Y.Z --notes-file notes.md`.
3. The `release` workflow then:
   - builds the sdist and wheel and checks the version against the tag;
   - attaches both files to the release;
   - publishes them to PyPI.

PyPI publishing uses [Trusted Publishing](https://docs.pypi.org/trusted-publishers/), so no API
token is stored anywhere. PyPI trusts the `release.yml` workflow of `inboxpraveen/relock`
running in the `pypi` environment. That environment only accepts `v*` tags. To pause
publishing, set the Actions variable `PUBLISH_TO_PYPI` to anything but `true`. PyPI never
accepts the same version twice, so a mistake needs a new version, not a re-upload.
