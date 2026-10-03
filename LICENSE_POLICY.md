# License and rights map

Zerith Robotics' new source code, integration work and documentation in this
repository are released under [Apache License 2.0](LICENSE), except where a
file or component carries a different license notice. Apache-2.0 permits
commercial use of the licensed code. No commercial-use restriction on code is
added by this policy or by the separate model-weight policy.

Existing source keeps the rights and notices with which it was supplied:

| Path | Existing notice / treatment |
|---|---|
| `simulation/` | Project simulation source without a separate license notice is released under Apache-2.0 by the project maintainer's decision. This covers project training, data preparation and benchmark integration code; importing or calling a simulator does not change this policy. Source copied or adapted from an official simulator retains its upstream terms and attribution. Existing Limx Dynamics Apache-2.0 source shared with FluxVLA Engine, the OpenMMLab Apache-2.0 adaptation and separately licensed FlashAttention/CUTLASS source keep their notices. See `simulation/NOTICE` and `simulation/LICENSES/`. |
| `deployment/shared/`, `deployment/robot/`, `deployment/inference/src/zwam/` | The project maintainer has confirmed Apache-2.0 publication of deployment integration and contribution code. The package-specific `LICENSE` and `NOTICE` files retain attribution. The H1 SDK is an external library; only the project's SDK adapter is included. |
| `deployment/inference/src/zwam_core/` | Inference adaptation attributed to Limx Dynamics under Apache-2.0; preserve the original NOTICE and Apache-2.0 text. |
| `simulation/fa2_custom_mask/` | Project bindings, launchers and mask integration use Apache-2.0. Modified FlashAttention Hopper headers and CUTLASS/C++ CuTe headers in `vendor/` retain BSD-3-Clause and upstream copyright. Original licenses, AUTHORS and modification notices are included in the wheel and source distribution; see `simulation/fa2_custom_mask/NOTICE`. The separately licensed CUTLASS Python CuTeDSL subtree is not bundled. |
| `simulation/metaworld_benchmark/prompts.py` | Official MetaWorld task descriptions adapted for the project, retaining Meta-World Team attribution and MIT licensing. The full terms are in `simulation/LICENSES/MetaWorld-MIT.txt` and are included in the simulation distribution. |
| `simulation/robodojo/xpolicylab_policy/ZVLA/setup_eval_env_client.sh`, `setup_eval_policy_server.sh` | XPolicyLab launch-template adaptations under upstream Apache-2.0; source revision and attribution are recorded in `simulation/NOTICE`. |
| `media/zerith-logo.png` | Zerith Robotics brand artwork displayed for project identification with publication and display authorization confirmed by the project maintainer. Zerith Robotics retains its rights in the logo. The artwork is excluded from the Apache-2.0 source-code license; this repository grants no additional trademark or branding rights. |
| [Public H1 SDK](https://github.com/inFpZero/zerith_public_sdk) | External motion and camera libraries installed separately by the user under the provider's terms. SDK source, binaries and vendor examples are not included in the repository, Python distributions or inference image. Configure `sdk_root` and `camera_sdk_root` to the local installation; no SDK rights are granted by the project's Apache-2.0 license. |
| Wan, VGGT, model assets, data | External dependencies; no license is granted by this repository. |

The root Apache-2.0 license does not remove third-party notices or grant
rights to external assets. In
particular, **model checkpoints/weights are not included in this source
release**. Zerith-owned weights, if released, need their own release-specific
terms; commercial use of those weights and their derivatives will require a
separate written license. See [MODEL_LICENSE_POLICY.md](MODEL_LICENSE_POLICY.md).

LIBERO/Plus, MetaWorld, RoboTwin and RoboDojo simulator environments are installed
separately. Their official code and any copied or adapted portions are not
relicensed by the project's Apache-2.0 decision.

The outstanding rights and redistribution checks are recorded in
[RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md).
