# Desktop test lab

`lab/` checks that a desktop is ready for testing, without Quoth installed: that
a VNC console can drive the logged-in Wayland session, and that a known virtual
microphone reaches the guest. It runs disposable libvirt guests on
`qemu:///session`, one at a time, with no shared folders, clipboard channels,
guest agents or device passthrough. `profiles.json` names the guest images;
the images, their credentials and the test fixtures stay outside Git.

Each run boots a clone of a powered-off source guest, checks that the source
disk and firmware variables are unchanged afterwards, and removes only the
clone it made. It refuses an evidence directory that already exists.

The host needs `virsh`, `virt-clone`, `qemu-img`, PipeWire's `pw-dump`,
`pw-loopback` and `pw-play`, and `ssh` and `scp`.

- Without a VM, as CI does: `uv run --locked --project lab pytest lab/test_lifecycle.py`.
- With an image prepared by `lab/prepare-ubuntu.sh`:
  `make lab-check PREPARED=<image>/prepared.json EVIDENCE=<path that does not exist yet>`.

`prepared.json` describes the prepared image:

| Key | Value |
|---|---|
| `profile` | A profile `id` from `profiles.json` with a `readiness` block |
| `disk`, `nvram`, `xml` | The guest's disk, firmware variables and libvirt XML, each with a `<key>_sha256` |
| `speech`, `silence` | 16 kHz mono 16-bit WAV fixtures played into the guest's microphone, each with a `<key>_sha256` |
| `user`, `password_file` | The guest account and a file holding its login password |
| `identity`, `known_hosts`, `host_alias` | The SSH key, known-hosts file and host-key alias for the guest |
| `evidence_root` | Where a `pytest` run without `--evidence` creates its evidence directory |
