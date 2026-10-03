# Security policy

The inference service is intended for a trusted robot network or SSH tunnel,
not direct Internet exposure. Keep it on loopback by default. Non-loopback
binds require a bearer secret; use TLS or SSH to protect that secret and camera
data in transit. The stdlib HTTP server is not a hardened multi-tenant gateway.

NPZ decoding disables pickle, bounds compressed/expanded sizes and checks exact
members. Model loading uses safetensors with hashes and strict state shapes.
External Python checkouts, native SDKs and upstream VAE loaders are trusted
dependencies: inspect provenance; do not point configs at untrusted content.

Do not paste sensitive vulnerability details or robot-network credentials into
public issues. Use the repository's private security reporting channel if it is
enabled. Otherwise, ask for a private contact channel without including exploit
details or sensitive data in the public request.

Use physical emergency procedures for motion hazards. Software pause and
process termination are not equivalent to a safety-rated stop.

The client rejects HTTP redirects, including same-host redirects. Configure the
final endpoint directly. Asset integrity checks cover required VGGT files,
the Wan VAE and the external Python package trees; checksums do not replace
source review or provider authentication.

Only the latest release receives security fixes. The dependency audit and any
time-limited exceptions are described in [dependency maintenance](deployment/docs/maintainers/DEPENDENCIES.md).
