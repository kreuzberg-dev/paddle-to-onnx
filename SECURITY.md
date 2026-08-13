# Security Policy

## Reporting a Vulnerability

**Do not open a public GitHub issue for a security vulnerability.** A public issue
tells everyone about the problem before there is a fix.

Email both maintainers directly:

- **Na'aman Hirschfeld** — <naaman@xberg.io>
- **Tobias Silva** — <tobias@xberg.io>

You may also use GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability) on this repository, which opens a private channel with the maintainers.

### What to include

1. A description of the vulnerability and the versions affected.
2. A minimal reproducer — ideally a file, request, or short script that triggers it.
3. Your assessment of impact and severity.
4. Whether you want public credit when the advisory is published.

A reproducer is the single most useful thing you can send. It turns triage from
guesswork into verification.

## Response targets

| Stage | Target |
|-------|--------|
| Acknowledgement of your report | 2 business days |
| Initial assessment and severity | 5 business days |
| Fix released — Critical / High | 14 calendar days |
| Fix released — Medium / Low | 30 calendar days |

If a fix will take longer than the target, we will tell you why and give a revised
date rather than let the report go quiet.

## Scope

This tool reads **model files that a user may not have produced themselves**. In scope: unsafe deserialization of model formats, path traversal when writing converted output, resource exhaustion from crafted model graphs, and any path that executes code embedded in a model artifact.

## Out of scope

- Vulnerabilities in third-party dependencies: report those to the dependency's
  own maintainers. Open an advisory here as well if this project's pinned version
  is affected, so we can upgrade.
- Findings from automated scanners with no demonstrated impact. Show us the
  consequence, not the signature.
- Attacks requiring physical access to a machine, or an already-compromised host
  or account.
- Social engineering of maintainers or users.

## Supported versions

Security fixes are applied to the latest release from the default branch. Fixes for
Critical and High severity issues are back-ported to the current minor series;
older minor series receive no back-ports.

## Disclosure

We coordinate disclosure with you. Our default is to publish a GitHub Security
Advisory once a fix is released, crediting you by name unless you ask to remain
anonymous. If a report is disputed or we conclude it is not a vulnerability, we
will explain our reasoning rather than simply closing it.
