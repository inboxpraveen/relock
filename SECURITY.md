# Security policy

relock decides when an agent's memory may be rewritten, so a way around its gates is a security
bug even if nothing crashes.

## Reporting

Please report vulnerabilities privately through
[GitHub private vulnerability reporting](https://github.com/inboxpraveen/relock/security/advisories/new),
not in a public issue. Include a minimal reproduction. A failing test is ideal.

## In scope

- Text from an untrusted author reaching a rewrite or choosing what gets rewritten. Untrusted
  authors include the environment, the agent's own reports, predictions and citations,
  unauthenticated users, and `detail_trusted=False` signals.
- Rewriting an item that was not recalled in the current episode, or writing through a closed
  episode.
- Transient errors counting as evidence, or the evidence threshold being bypassed without an
  authoritative signal.
- Two active versions of one item, lost updates under concurrency, or history being destroyed.
- The MCP server running anything other than the verifier commands its operator configured.

The [threat model](docs/design.md#threat-model) lists what v0.1 does and does not defend against.
Some cases are out of scope by design: an attacker who controls your tool code or verifier
commands, and an operator who opts in to `--trust-agent-reports` or `--allow-teach`.

## Supported versions

Only the latest release receives fixes while relock is in 0.x.
