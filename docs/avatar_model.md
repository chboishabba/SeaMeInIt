# Avatar model integration

This guide explains how to install the Python tooling, fetch either the licensed
SMPL-X assets or an open SMPLer-X bundle, and generate parametric meshes using
the `BodyModel` wrapper provided in `src/avatar_model/body_model.py`.

## Environment preparation

1. Create a virtual environment with Python 3.10 or 3.11.
2. Install the package in editable mode along with the tooling extras:
   ```bash
   pip install -e .[tools]
   ```
3. Download a body-model bundle using the helper script:
   ```bash
   # Licensed SMPL-X download (requires authenticated link)
   export SMPLX_DOWNLOAD_URL="https://smpl-x.is.tue.mpg.de/..."
   python tools/download_smplx.py --model smplx --dest assets/smplx --sha256 <optional-checksum>

   # or fetch the open SMPLer-X bundle (S-Lab License 1.0, non-commercial use)
   python tools/download_smplx.py --model smplerx --dest assets/smplerx --sha256 <optional-checksum>
   ```
4. Each command writes an `assets/<model>/manifest.json` file recording the
   source URL, checksum, license, and top-level contents. Use it to confirm the
   bundle provenance before loading the model.
5. (Optional) Supply `SMPLX_ARCHIVE_SHA256` or `--sha256` with a known checksum
   to enable verification during download. The script skips the network call
   when the `--archive` option is provided.

## Loading the body model

```python
from pathlib import Path
import numpy as np

from avatar_model import BodyModel

# Load the default SMPL-X bundle
model = BodyModel(model_path=Path("assets/smplx"), batch_size=2)
print(model.parameter_shapes())
```

To use the SMPLer-X release, point `model_path` at `assets/smplerx`. The
manifest ensures the loader checks the correct subdirectories and surfaces
actionable error messages when files are missing.

The `parameter_shapes()` method describes the tensor layout managed by the
wrapper. By default the parameters contain zeros and represent the canonical
T-pose of the SMPL-X mesh.

## Editing shape and pose

`BodyModel` accepts numpy arrays or tensors when setting or adjusting any of the
supported parameters.

```python
import numpy as np

# Make the avatar taller/narrower.
model.set_shape(np.array([[ 0.5, -0.3, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [-0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]))

# Rotate both bodies slightly around the vertical axis.
model.adjust_pose(global_orient=np.array([[0.0, 0.2, 0.0],
                                          [0.0, -0.2, 0.0]]))

# Move the avatars forward in the scene.
model.adjust_pose(transl=np.array([[0.0, 0.0, 0.3],
                                   [0.0, 0.0, 0.5]]))
```

## Generating meshes

`BodyModel.vertices()` and `BodyModel.joints()` return torch tensors on the
configured device. Convert to numpy for downstream tools by calling
`tensor.detach().cpu().numpy()`.

```python
import torch

with torch.no_grad():
    vertices = model.vertices()  # Shape: (batch_size, num_vertices, 3)
    joints = model.joints()      # Shape: (batch_size, num_joints, 3)

mesh_np = vertices.detach().cpu().numpy()
print(mesh_np.shape)
```

Save the output in the format required by your renderer (OBJ, PLY, etc.) using
existing mesh utilities. The wrapper keeps all tensors on the selected device so
you can integrate it with differentiable pipelines or export the meshes for
offline rendering.
