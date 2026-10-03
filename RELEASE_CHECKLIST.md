# Publication gates

## Agreed release scope

- The code release contains source, documentation and Python packages.
- Model weights will be published separately on Hugging Face. The READMEs and
  model-weight policy contain a clearly labeled placeholder link, to be replaced
  when that model repository is published.
- Data, text banks, Wan/VGGT assets, SDK binaries, robot descriptions and native
  IK resources are not bundled in the code release.
- Full-model training, simulator rollouts and H1 commissioning will be validated
  separately. The source release must retain the limits of its current evidence.
- The maintainer selected Apache-2.0 for project simulation files that are not
  official simulator source. Existing third-party licenses and attribution
  remain in force; official source or adaptations retain their upstream terms.
- The maintainer confirmed Apache-2.0 publication of deployment project and
  contribution code. H1 motion and camera SDKs remain separately installed
  external libraries under the provider's terms; only project adapters ship.
- The maintainer confirmed publication and display of the Zerith Robotics
  logo for project identification. The company retains its rights in the logo;
  the artwork is excluded from the Apache-2.0 source-code license.
- The maintainer selected the deployment `body_yaw_link` frame for H1 training
  observations and absolute hand targets. Data and hardware agreement with this
  definition remain part of deferred validation; old statistics are not
  relabeled or numerically transformed by the deployment tools.
- The maintainer confirmed H1 raw waist values as lift in metres, pitch in
  radians and yaw in radians, in that order. Position/waist units and joint
  order are explicit deployment requirements; SDK ABI, signs, zero positions
  and physical validation remain deferred.
- The maintainer selected the current three-camera Wan2.2 VAE, UMT5 and VGGT
  interfaces for H1 deployment: text width 4096/prefix 64, 48 latent channels
  with a canonical 42-by-14 grid, geometry width 2048/49 tokens per camera,
  three camera slots and patch size 2. Logical text lengths of at least 64
  retain the zero suffix; this interface check adds no restrictions to internal
  network widths, layer counts or RoPE settings.
- The maintainer selected protocol v3 with paired client/server upgrades and
  explicit rejection of older health metadata. The client retains its secret
  episode token before the first reset request, allowing explicit reset or
  release after a lost response. Existing ownership remains exclusive even for
  an inactive session, with no TTL takeover or automatic inference retry.
- The maintainer selected Conda-managed environments and direct source execution
  for the public simulation workflow. Clone the repository, activate the core
  or selected official simulator environment, and run module commands from
  `simulation/`. The project source needs no pip installation; the lightweight
  HTTP client needs NumPy and requests. A separate Python 3.11 client Conda
  specification is available for the standalone bridge/launcher role; official
  simulator environments retain their own Python requirements. Source launchers,
  the benchmark recipe generator and evaluation bridge resources must be complete.
  Official simulator environments, native dependencies, models and assets are installed separately;
  only the selected platform is loaded, and harness subprocesses receive the
  simulation source root on `PYTHONPATH`.
- The maintainer declined additional simulation encoder-identity enforcement
  across training checkpoints, exports and serving. Developers are responsible
  for keeping Wan VAE, UMT5/tokenizer, VGGT and preprocessing consistent between
  data preparation and inference. Preparation resume retains its existing
  content checks; exported/served assets are not automatically matched to the
  preparation record, and this additional enforcement is not a publication gate.
- The maintainer selected PyTorch 2.10.0 and torchvision 0.25.0 for the
  simulation core environment and CPU CI, matching deployment's library
  versions. Simulation retains its CUDA 13.0 installation example and Conda
  source workflow; selected official simulator environments remain separate.
- The maintainer selected `Pillow>=12.3,<13` for simulation, matching deployment's
  security floor and excluding the reviewed vulnerable 12.0–12.2 releases.
- The maintainer selected documentation of the simulation checkpoint trust
  boundary while retaining the current full-state training recovery format.
  `train --resume` is only for training directories created by the developer
  or whose source they trust. Public policy initialization and model exports
  retain safetensors; internal recovery may deserialize Python/NumPy state.
- The maintainer declined automatic dependency vulnerability auditing in CI and
  a manual vulnerability audit as part of this source-release preparation.
  Dependency review and later testing remain the maintainer's responsibility.
  Existing audit tooling and documented temporary framework exceptions remain
  available; this decision does not establish that all advisories are resolved.

## Candidate checks completed on 2026-09-30

- `python scripts/check_release.py`: passed on 1,083 publication files after
  the Apache-2.0/code, model-weight policy and bilingual homepage updates.
- `detect-secrets` 1.5.0 scanned the source tree before the license-only
  update, locally with network verification disabled. Its 39 high-entropy
  findings were reviewed as source fingerprints and upstream revision hashes
  in four files; no credential was identified. Scan the final public Git
  history and release archives separately.
- Deployment synthetic tests: 109 passed. Simulation post-training tests:
  21 passed. These runs did not use a full model bundle or robot hardware.
- Three rebuilt deployment wheels previously passed `scripts/check_dist.py`.
  Rebuild after the Apache-only license update. A prior robot-only
  installation passed its offline mapping check with no PyTorch or server
  module; the inference-only installation imported its server with no H1
  module. Rebuild the simulation wheel after replacing its logger and metric
  modules; it must include Apache-2.0 and NOTICE.
- A minimal container build verified that the inference Docker context admits
  the package source, example config and license files while excluding the
  robot package and local resource directories. The full CUDA image was not
  built in this environment.

## Candidate checks completed on 2026-10-01

- Replaced the training logger and metric modules with Apache-2.0
  implementations and removed the former license files and package
  metadata. The remaining `openvla/modified_libero_rlds` reference identifies
  an external LIBERO dataset and its target convention, not bundled source.
- `python scripts/check_release.py`: passed on 1,079 publication files.
- Simulation post-training and metric tests: 23 passed. Deployment synthetic
  tests: 109 passed with the three package source roots on `PYTHONPATH`.
- Built the simulation and three deployment wheels with Apache-2.0 licensing
  files. `scripts/check_dist.py` passed on the deployment wheels.
- Compared simulation engine source against FluxVLA. The registry file is
  identical and the builder/root utilities remain substantially similar;
  preserve their attribution. The logger and metric replacements are separate
  implementations. This comparison does not establish rights over other
  unmarked files.
- Consolidated user-facing technical guides into the English and Chinese root
  READMEs. Updated the H1 adapter and offline diagnostics for the separately
  located motion/camera Python libraries in the public Zerith SDK. Synthetic
  tests pass, but the SDK binaries and hardware have not been exercised here.
- After consolidation, the release-tree audit passed on 1,063 files;
  simulation tests passed (23), deployment tests passed (112), all four wheels
  built, and the three deployment wheels passed `scripts/check_dist.py`.

## Candidate checks completed on 2026-10-03

- Added the separate Hugging Face model-release placeholder and recorded the
  maintainer's Apache-2.0 decision for project simulation code. Preserved
  MetaWorld MIT task-description attribution and XPolicyLab Apache-2.0 template
  provenance, including the license text in the simulation distribution.
- Retained the optional FA3 extension, restored package-local CUTLASS discovery
  and complete build resources, and preserved upstream BSD-3-Clause notices and
  AUTHORS. The original kernel source fingerprint remains unchanged.
- Built the simulation wheel and source archive; all 850 extension resources
  matched the source bytes, including the 805 pinned CUTLASS headers. All seven
  declared license/notice files were present without path collisions. An
  independently located wheel import used its own bundled resources and passed
  the original source fingerprint check.
- Simulation CPU tests: 34 passed, including 11 focused FA3 resource and ABI
  checks. Documentation links and the release-tree audit passed (1,068 files).
  The test environment used Python 3.12 and PyTorch 2.10.0; this did not compile
  CUDA, execute the custom kernel or validate real training and hardware.
- Recorded the maintainer's deployment Apache-2.0 decision and external H1 SDK
  boundary. Rebuilt all three deployment wheels and verified their package
  boundaries without SDK binaries or vendor examples. Recorded authorized logo
  display while retaining the company's rights and excluding brand artwork
  from the source-code license.
- Added explicit preparation `--resume`, raw-file and RGB/episode content
  identities, encoder/code/software identities, immutable completed-shard
  validation, exclusive output locking and atomic shard/summary publication.
  Legacy caches without receipts and changed or damaged completed caches are
  rejected before encoder loading. Dataset fingerprints now include the actual
  episode, text, task and statistics outputs.
- Simulation CPU tests: 75 passed, including 41 preparation identity, recovery
  and source-reader checks. New implementation/test Ruff checks passed. Built
  the simulation wheel and source archive; independently located wheel imports
  and `prepare --help` exposed the new `--resume` option outside the checkout.
  All 41 preparation checks also passed against the extracted wheel from an
  independent working directory. Documentation links and the release-tree
  audit passed on the updated 1,073-file source candidate.
- Added one shared H1 `body_yaw_link` pose-frame check to codec statistics,
  bundle preparation, policy/session loading and offline diagnostics. Missing,
  unresolved or other frames fail before model imports or weight loading.
  Unverified physical status and source evidence are retained; editing a
  manifest alone cannot bypass the checkpoint statistics fingerprint.
- Deployment CPU tests: 196 passed, including 84 new frame-contract checks.
  Rebuilt all three deployment wheels; package-boundary checks passed and 101
  frame/doctor tests passed from independently located extracted wheels with
  no skipped cases. The installed contract check initialized neither SDK nor
  CUDA. Ruff checks, changed-file formatting and the 1,074-file release-tree
  audit passed. These checks do not validate physical data or hardware.
- Enforced hand-position metres and waist joint order/units as lift in metres,
  pitch in radians and yaw in radians, following the maintainer's confirmation
  of the raw H1 data. Bundle preparation, codec statistics, policy/session
  loading and diagnostics reject incompatible declarations before loading
  weights. Correct legacy statistics retain their exact contents and hash;
  optional explicit waist semantics must match the declared order.
- Deployment CPU tests: 218 passed, including 22 new unit/order checks. All
  three rebuilt deployment wheels passed package-boundary checks; 123 focused
  contract/doctor tests passed from independently located extracted wheels
  without skipped cases. Ruff checks and changed-file formatting passed. The
  release-tree audit passed on 1,075 publication files. SDK unit conventions,
  signs and physical zero positions still require hardware validation.
- Added a dependency-free check for the fixed UMT5, Wan2.2 VAE and VGGT
  three-camera interfaces to bundle preparation, policy/session loading and
  offline diagnostics. Incompatible dimensions fail before model imports,
  weight reads or CUDA checks. Omitted interface fields use production
  defaults; explicit sizes must be integers. Longer logical text contexts
  retain their zero-padded suffix, and internal widths, layers and camera RoPE
  values retain their existing structural configuration rules.
- Deployment CPU tests: 317 passed, including 99 new interface and early
  rejection checks. Rebuilt all three deployment wheels; package-boundary
  checks passed and 222 focused contract/doctor tests passed outside the
  checkout against extracted wheels, with no skipped cases. Ruff checks and
  changed-file formatting passed; the release-tree audit passed on 1,078
  publication files. Full GPU inference and hardware remain deferred.
- Upgraded the paired client/server wire contract to v3. The client retains
  its generated secret before the first reset request and activates only after
  a matching token and exact sequence-zero acknowledgement. Lost or malformed
  reset/release replies retain that secret and block inference until explicit
  recovery. The server keeps ownership across partial model-reset failures and
  inactive sessions; unrelated clients cannot take over. POST requests require
  the v3 header before payload parsing, and rejected requests or parser/reply
  failures before a session operation cannot interrupt an existing owner.
- Deployment CPU tests: 416 passed, with 99 additional ownership, client
  recovery and server-isolation cases. Rebuilt all three deployment wheels;
  package-boundary checks passed and 350 focused contract, recovery and doctor
  tests passed against extracted wheels outside the checkout, without skipped
  cases. Loopback HTTP tests exercised lost acknowledgements, malformed replies,
  authentication headers and legacy-version rejection. Ruff and changed-file
  formatting passed, and the 1,081-file release-tree audit passed. These tests
  used a synthetic session and did not invoke a GPU model or H1 hardware.
- Completed the public Conda/source simulation workflow: the training launcher
  uses its source module, retains distributed/resume/configuration arguments,
  and imports training dependencies only when needed. Evaluation loads only the
  chosen simulator and copies all three RoboTwin or eight RoboDojo project
  bridge files before invoking the official harness. Missing roots, tasks,
  environments or required files fail before launching a subprocess.
- Simulation CPU tests: 117 passed, including 42 new launcher, resource and
  dependency-isolation checks. Those 42 checks were rerun against the final
  source. A fresh Python 3.11 environment created from `environment-client.yml`
  with micromamba passed source CLI/import checks without installing the project,
  PyTorch, Megatron or any simulator. Mocked harness launches verified the copied
  bridge files and propagated source path; missing selected simulator libraries
  produced clear errors before creating output. These were entry-point checks,
  not training or simulator rollouts. Bilingual documentation checks and Ruff
  checks/formatting for the changed Python files passed.
- Simulation source archive after the entry-point changes includes the client Conda specification,
  requirements, compatibility training script and five configuration examples.
  All 13 checked package entry/bridge resources, eight source-run resources,
  850 FA3 files and seven license/notice files match the source byte-for-byte.
  An internal wheel resource check also passed 53 focused CPU tests; installing
  that wheel is not part of the public simulation workflow. The final release-tree
  audit passed on 1,087 publication files.
- Updated both simulation READMEs and the simulation CPU CI job to PyTorch
  2.10.0/torchvision 0.25.0, matching deployment and removing the 2.9.1 version
  affected by [GHSA-63cw-57p8-fm3p](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p).
  With PyTorch 2.10.0+cu130 and torchvision 0.25.0+cu130, all 117 simulation
  CPU tests and the documentation check passed. The parsed CI workflow and
  bilingual examples use the same library pair. Local API inspection found the
  required DCP, DTensor and AdamW interfaces available; 32 existing pin-memory
  deprecation warnings remain. This did not exercise full Megatron training,
  CUDA optimizer kernels or FA3 compilation.
- Updated simulation to `Pillow>=12.3,<13`, matching deployment and excluding
  the 12.0–12.2 versions affected by
  [GHSA-pg7v-jwj7-p798](https://github.com/python-pillow/Pillow/security/advisories/GHSA-pg7v-jwj7-p798).
  A PyPI dry-run selected Pillow 12.3.0 for Python 3.12; requirement parsing
  confirmed the installed 12.3.0 satisfies both simulation and deployment.
  All 117 simulation CPU tests passed with that version, with the same 32
  existing pin-memory deprecation warnings. The release-tree audit passed on
  1,087 files; this is not a complete clean-role dependency audit.
- Added the agreed training-directory trust statement to both public READMEs
  and both training parsers. The public CLI, direct module and compatibility
  script expose it without loading training dependencies. All 16 training
  entry-point checks passed; the full-state recovery behavior is unchanged.
- Preserved `shared/NOTICE` in the inference Docker context and corrected the
  two outstanding CI formatting failures without changing Python syntax trees.
  All 416 deployment CPU tests passed; Ruff checks and formatting passed across
  the CI's 67 Python files, and the release-tree audit passed on 1,087 files.
  Static Docker-ignore review confirms the shared license notice is included
  while retaining the existing context scope. The Docker daemon is unavailable,
  so a full CUDA image build remains unverified.

- Rebuilt the final three deployment wheels and checked their package boundaries.
  All 44 Python modules and seven license/notice files match the current source
  bytes; all wheel RECORD hashes are valid. The only runtime difference from the
  previously tested v3 artifacts is AST-identical H1 adapter formatting.
- Rebuilt the final simulation wheel and source distribution as internal
  completeness checks. All 100 Python modules, 13 entry/bridge resources, eight
  source-run resources, 850 FA3 files and seven license/notice files match the
  current source. The independently extracted wheel passed 53 focused CPU
  checks without skips. Public simulation use remains Conda plus source.
- Checked the exact 1,087-file publication snapshot and scanned it offline with
  detect-secrets 1.5.0, with network verification disabled. All 43 findings were
  reviewed as public source fingerprints, upstream revisions, resource paths
  or an explicit synthetic test credential; no credential was identified.
  Safe extraction and scans of all three deployment wheels, the simulation
  wheel and its source distribution found no additional credential. No Git
  history is included in the reviewed source archive; scan any newly created
  public history before pushing it, and do not import the original private
  repositories' history.
- Corrected the Git ignore rule for local datasets so that all six Python
  dataset implementation modules under `simulation/zvla/datasets/` enter the
  public commit. Local datasets and generated caches remain ignored. The
  reviewed publication file count remains 1,087; runtime code is unchanged.

## Source release checklist

- [x] Record the maintainer's Apache-2.0 decision for project simulation source
      without a separate license notice, excluding official simulator source
      and retaining existing third-party licenses and attribution.
- [x] Preserve identified official adaptations: MetaWorld task descriptions with
      the upstream MIT notice and terms, and XPolicyLab launch templates with
      their upstream Apache-2.0 attribution and source revision.
- [x] Confirm publication authorization for deployment contributions. Keep Limx
      Dynamics, OpenMMLab and other third-party attribution.
- [x] Keep the H1 SDK external: no vendor source, binaries, examples or automatic
      SDK installation in the repository, Python packages or inference image.
- [x] Retain the optional FlashAttention/CUTLASS extension with upstream
      BSD-3-Clause licenses, original copyright, AUTHORS and modification
      notices. Package the build resources in the wheel and source distribution;
      verify source/ABI fingerprints without substituting new kernel contents.
- [x] Record the maintainer's authorization to publish and display the Zerith
      Robotics logo for project identification; retain the company's rights
      and exclude the artwork from the Apache-2.0 source-code license.
- [x] Make preparation resume explicit and verify source, episode, encoder,
      configuration and completed-output content before reuse. Recover ordinary
      interruptions without mixing different data; reject unsupported legacy
      caches and corrupted completed artifacts.
- [x] Verify complete simulation source entry points, the benchmark recipe
      generator and evaluation bridge resources for the public Conda workflow.
      Run module commands from `simulation/` without installing the project;
      load only the selected external simulator environment and assets, and
      propagate the source root to official harness subprocesses.
- [x] Enforce the H1 `body_yaw_link` pose-frame declaration consistently in
      bundle preparation, codec statistics, model/session loading and offline
      diagnostics; reject unresolved or other frames before weight loading,
      retaining checkpoint/statistics identity and deferred physical validation.
- [x] Confirm and enforce H1 position/waist units and lift/pitch/yaw joint
      semantics consistently before preparing or loading model weights. Preserve
      correct legacy statistics without adding fields or changing fingerprints;
      reject unresolved declarations and incompatible units/order.
- [x] Reject incompatible fixed H1 encoder dimensions during bundle preparation,
      policy/session loading and offline diagnostics before reading model
      weights or initializing the GPU. Check text 4096/prefix 64, latent
      48/42-by-14, geometry 2048/49 tokens per camera, three cameras and patch
      size 2; retain logical `text_length >= 64` with its zero suffix.
- [x] Make the first session reset recoverable after a lost response using a
      retained client-generated secret episode token and explicit reset/close.
      Preserve exclusive ownership while inactive, with no TTL takeover or
      automatic inference retry. Upgrade client/server together to protocol v3
      and reject older or incompatible health metadata.
- [x] Record the maintainer's decision to leave simulation encoder-version
      matching between preparation, training, export and serving to developers,
      without adding automatic cross-stage identity enforcement.
- [x] Document the simulation checkpoint trust boundary in both READMEs and
      training CLI help, retaining full-state recovery only for developer-created
      or trusted-source run directories.
- [x] Record the maintainer's decision to defer dependency vulnerability auditing
      to their own subsequent review, without adding CI auditing or making a
      manual audit a condition of this source release.
- [x] Preserve the shared package NOTICE in the filtered Docker context and
      make the existing CI formatting check pass.
- [x] Run `python scripts/check_release.py` on the exact publication tree and
      scan source and built archives with an independent secret scanner.
      The reviewed source archive contains no Git history; scan new public history
      before pushing it.
- [x] Build all three deployment wheels and confirm package boundaries with
      `scripts/check_dist.py`; run the synthetic test suites and verify that
      final packaged source matches the tested candidate, with only
      AST-identical formatting changes.
- [x] Review the public README/license wording: Apache-2.0 code permits
      commercial use, third-party permissions remain, and only
      separately released Zerith-owned weights and derived weights are subject
      to a separate commercial model license.

## Separate model release gates

- [ ] Replace the Hugging Face placeholder link with the published model
      repository in both READMEs and MODEL_LICENSE_POLICY.md.
- [ ] Obtain and review rights for any checkpoint, data, text bank, Wan/VGGT
      asset or H1 SDK material before distributing it; none is included here.
      For each Zerith-owned checkpoint release, attach binding model-specific
      terms on Hugging Face identifying the model/version, permitted uses,
      derivative weights, redistribution and the written commercial licensing
      process. Review third-party base-model terms before asserting the right
      to grant commercial permissions.

## Deferred training and hardware validation

- [ ] Validate full-model simulation training and closed-loop benchmark
      evaluation with the actual assets and supported simulator environments.
- [ ] Compile and exercise the optional FA3 extension on a compatible Hopper
      GPU; measure numerical agreement, speed and memory use with actual inputs.
- [ ] Validate the H1 checkpoint against the strict loader, then run GPU
      replay with matching observations and perform separately authorized
      hardware commissioning. Synthetic tests do not replace these checks.
