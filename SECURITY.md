# Security policy

## Supported versions

Only the latest release of Quoth gets security fixes; older releases and
development snapshots do not. Upgrade before reporting a problem when you can.

## Report a vulnerability privately

Use [GitHub private vulnerability reporting](https://github.com/abullard1/Quoth/security/advisories/new)
for security or privacy vulnerabilities, not a public issue. Without a GitHub
account, use the [contact form](https://quoth.dev/contact) and say only that you
have a security report. The maintainer replies to the email address you give it;
send the details only in the way that reply arranges.

Include the Quoth version and package channel, your desktop and session, steps
to reproduce the issue and its likely impact. Do not send recordings, dictated text,
credentials or unrelated logs. Reproduce a finding yourself before reporting it:
scanner or AI output that nobody has confirmed against Quoth is closed without a
fix.

Two limits are known: the Flatpak can run commands on the host, and
`/dev/uinput` access belongs to the seat rather than one account. See
[Flatpak](SUPPORT.md#flatpak) and
[Permissions and known limitations](SUPPORT.md#permissions-and-known-limitations)
before reporting either.

The maintainer responds within 14 days and coordinates the fix and disclosure in
the private advisory, or by email for a report sent without a GitHub account.
Fixed vulnerabilities are published as GitHub Security Advisories. Please do not
disclose the issue publicly until a fix or mitigation is released.

For ordinary crashes, compatibility problems, and feature requests, use the
[public issue tracker](https://github.com/abullard1/Quoth/issues).
