# Contributing to Zerith-VA

Thanks for contributing. Keep changes focused and describe the behavior,
affected component and verification in your pull request. Start with the
[architecture](README.md#architecture), [simulation workflow](README.md#simulation-post-training) and [H1 workflow](README.md#h1-deployment).

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
python -m pytest -q simulation/tests/test_public_posttrain.py
```

Install the dependencies for the component you are testing first. The
[main README](README.md#validation-and-license) lists the environment and validation commands. Include the commands you ran and their results
in your pull request.
