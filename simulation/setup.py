# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

from setuptools import find_packages, setup

fa3_root = Path(__file__).resolve().parent / "fa2_custom_mask"
fa3_headers = sorted(
    path.relative_to(fa3_root).as_posix()
    for path in (fa3_root / "vendor").rglob("*")
    if path.is_file() and path.suffix in {".h", ".hpp", ".inl", ".cuh"}
)

long_description = (
    "Zerith VA benchmark post-training component. "
    "The repository README contains installation, training, export and evaluation instructions."
)

setup(
    name="zvla",
    version="0.0.1",
    description="Zerith VA benchmark post-training source",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    license="Apache-2.0 AND MIT AND BSD-3-Clause",
    license_files=[
        "LICENSES/Apache-2.0.txt",
        "LICENSES/MetaWorld-MIT.txt",
        "NOTICE",
        "fa2_custom_mask/NOTICE",
        "fa2_custom_mask/vendor/flash_attn_hopper/LICENSE",
        "fa2_custom_mask/vendor/flash_attn_hopper/AUTHORS",
        "fa2_custom_mask/vendor/flash_attn_cutlass/LICENSE.txt",
    ],
    package_data={
        "robotwin.public_policy": ["deploy_policy.yml"],
        "robodojo.xpolicylab_policy.ZVLA": ["*.sh", "*.yml"],
        "fa2_custom_mask": [
            "*.cpp",
            "*.cu",
            "*.h",
            "NOTICE",
            "vendor/flash_attn_hopper/LICENSE",
            "vendor/flash_attn_hopper/AUTHORS",
            "vendor/flash_attn_cutlass/LICENSE.txt",
            *fa3_headers,
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[],
    extras_require={"client": ["numpy>=1.26", "requests>=2.32"]},
)
