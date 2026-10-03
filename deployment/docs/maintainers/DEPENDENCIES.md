# Dependency maintenance

The inference package pins PyTorch 2.10.0 and torchvision 0.25.0 for checkpoint
compatibility and the patched weights-only loader. Pillow has a 12.3.0 security floor. Keep the GPU and robot SDK
Python environments separate.

## Audit

Run against the actual installed inference environment:

```bash
python -m pip install --upgrade pip
python -m pip install pip-audit==2.10.0
python scripts/audit_dependencies.py
```

The audit enumerates installed Python packages without resolving or installing
anything. Official PyTorch CPU/CUDA wheel suffixes are mapped to the matching
upstream release for advisory lookup; other unknown packages remain failures.
Native operating-system and GPU-driver packages require separate platform checks.

The maintainer chose to leave dependency vulnerability review to their own
subsequent validation for this source release. Source CI runs CPU tests and
package checks; it does not run this audit, and a manual audit is not a condition
of source publication. This tool remains available for future review against
the resolved inference environment. When used, it rejects new findings,
unaudited dependencies and expired exceptions. The exception list is in
`dependency_exceptions.json`; each entry names one package version and advisory.
Do not add broad package exclusions or automatically extend an expiry date.

## Temporary framework exceptions

Two known advisories affect the pinned PyTorch version:

| Advisory | Affected API | Inference exposure |
|---|---|---|
| [CVE-2026-4538](https://github.com/pypa/advisory-database/blob/main/vulns/torch/PYSEC-2026-139.yaml) | PT2 loading | No PT2 archives are accepted or loaded |
| [CVE-2025-3000](https://github.com/advisories/GHSA-rrmf-rvhw-rf47) | `torch.jit.script` | No compilation of user-supplied Python in the service |

These exceptions expire on **2026-11-01**. They are a compatibility constraint,
not a claim that PyTorch 2.10.0 is vulnerability-free. External Wan/VGGT source
must be trusted and reviewed; extensions that use these APIs need a new risk
assessment. The HTTP service is limited to trusted networks with protected
transport and bounded array inputs.

Before changing the framework, test strict loading, model outputs with matching
inputs/seeds, image preprocessing, VAE temporal caches, VGGT features, GPU memory
and latency on the actual deployment. Then update the package pins, Docker base,
installation examples and exception list together. CPU tests alone do not prove
full-weight GPU compatibility.

See [Pillow 12.3.0 security fixes](https://pillow.readthedocs.io/en/stable/releasenotes/12.3.0.html)
for the image-library minimum. Audit the resolved dependency graph for each
release; version ranges do not replace vulnerability checks.

The PyTorch floor includes the upstream fix for
[CVE-2026-24747](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p),
which affects loading malicious checkpoints even with `weights_only=True`.
Only provider-verified VAE assets are supported; do not treat checksums computed
after receiving an untrusted file as proof of trust.
