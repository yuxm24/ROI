# Contributing to Zerith VA

Thanks for contributing. Keep changes focused and describe the behavior,
affected component and verification in your pull request. Start with the
[project overview](README.md#overview), [simulation guide](docs/simulation.md)
and [H1 deployment guide](docs/deployment.md).

## Before submitting

- Use synthetic examples in issues and tests. Do not commit credentials,
  model weights, recordings, customer data, local configuration, vendor SDK
  binaries or machine-specific paths.
- Preserve existing copyright and license notices. New Zerith integration
  material is Apache-2.0; third-party source keeps its own terms. See the
  [license map](LICENSE_POLICY.md).
- Add focused offline checks for behavior changes. Hardware operation and
  full-model parity require separate validation and are not prerequisites for
  a documentation change.

Useful checks from the repository root:

```bash
python scripts/check_release.py
python -m pytest -q deployment/tests
PYTHONPATH=simulation python -m pytest -q simulation/tests
```

Use the environment described in the corresponding guide and install the
deployment packages before running deployment tests. For local HTTP tests,
set `NO_PROXY=127.0.0.1,localhost` so requests bypass proxy services.
Include the commands you ran and their results in your pull request.
