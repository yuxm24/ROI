# Releasing the combined Zerith source tree

Run commands from the combined repository root. Publish only the reviewed source
snapshot; do not push the original simulation repository's history or private
deployment archives into the public remote.

1. Complete the root [publication gates](../../../RELEASE_CHECKLIST.md), especially
   ownership and third-party redistribution rights.
2. Run `python scripts/check_release.py` on the exact tree. Scan the new public
   Git history and built archives with an independent secret scanner. Gitignore
   does not remove files that were already committed.
3. Build the three deploy wheels separately and run `python scripts/check_dist.py
   deployment/dist`. The robot wheel must have no inference model or PyTorch
   dependency; the inference wheel must have no H1 SDK controller code.
4. Install each wheel in a clean role-specific environment. Run synthetic tests,
   documentation checks, and model bundle/visual asset hash checks.
5. Before any hardware claim, run matching GPU replay and authorized H1
   commissioning. Record checkpoint, asset, dependency, GPU and SDK versions.

Weights, recordings, text banks, URDFs, native SDK binaries and upstream Wan/VGGT
assets are outside this source distribution. A source hash documents identity,
not permission to publish or relicense.
