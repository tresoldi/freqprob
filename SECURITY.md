# Security Policy

## Supported versions

FreqProb is under active development. Security fixes are applied to the latest
released version on PyPI. Please make sure you are running the most recent
release before reporting an issue.

| Version | Supported          |
| ------- | ------------------ |
| 0.4.x   | :white_check_mark: |
| < 0.4   | :x:                |

## Reporting a vulnerability

If you believe you have found a security vulnerability, please report it
privately rather than opening a public issue.

- Use GitHub's [private vulnerability reporting](https://github.com/tresoldi/freqprob/security/advisories/new)
  (Security → Report a vulnerability), or
- Email the maintainer at tiago.tresoldi@lingfil.uu.se.

Please include a description of the issue, the affected version(s), and, if
possible, a minimal reproduction. You can expect an initial acknowledgement
within a reasonable timeframe, and we will keep you informed as the report is
investigated and resolved.

Because FreqProb is a numerical/NLP library with no network or authentication
surface, the most likely concerns are around untrusted input (for example,
deserializing data or processing adversarial frequency distributions). Reports
in those areas are especially welcome.
