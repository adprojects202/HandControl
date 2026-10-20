# `world/h4nd_left` — the hand the project reads

The SolidWorks export of the left hand, and the one file the code loads.

| file | read by | what it is |
| --- | --- | --- |
| `h4nd_left_genesis.urdf` | **everything** — `robot/model.py:URDF` | generated from `urdf/h4nd_left.urdf`, and only two things change: LF line endings and `filename="meshes/…"` in place of `package://`. Otherwise a straight transcription. |
| `meshes/*.STL` | `robot/model.py`, `robot/mjcf.py` | the 21 link meshes, and nothing else. The export also emitted ~455 Unity artefacts (`.meta`, `.prefab`, `.asset`) alongside them; nothing read those and they are removed. |
| `urdf/h4nd_left.urdf` | nothing | the vendored original, kept as provenance for the file above. |
| `launch/`, `config/`, `CMakeLists.txt`, `package.xml`, `export.log` | nothing | ROS scaffolding the exporter emits. |

**This one is natively LEFT**, which the previous export was not — see
`model.MIRRORED_FOR`. The old package was a right hand that had to be
mirrored to get the bench hand, so the label and the geometry disagreed on
purpose and that disagreement was written down in three places. Here they
agree: `--hand left` loads the vendored model, `--hand right` loads the
mirror, and `Retargeter.chirality()` measures each as the hand it says it is.

## Still to fix at the exporter

Neither is urgent and neither is worked around silently, but both are real:

- ~~**`ring_dstial_link`.**~~ **Fixed here.** The export named the ring's
  distal link with a transposition, consistently on the link and its mesh, so
  the URDF was internally sound and only the *name* was odd — but
  `DISTAL_LINKS` builds `ring_distal_link` like every other digit. The link is
  renamed in `urdf/h4nd_left.urdf` and the STL renamed to match, so the
  generated copy is a straight transcription again. **Fix it at the exporter
  too**, or the next export brings it back.
- **The Unity artefacts.** The export dropped ~455 `.meta`, `.prefab` and
  `.asset` files into `meshes/` beside the 21 STLs. Removed here; if the
  exporter keeps emitting them, they will come back with the next export.

And one that is not the exporter's fault but bites the same way: each export so
far has carried a **nested duplicate** of itself (`h4nd_2/h4nd_2/`,
`h4nd_2_2/h4nd_2/`) whose URDF was an OLDER revision disagreeing on the index
adduction axis and its limits. Nothing read them; anything that had would have
driven the finger backwards. Check for one after every re-export.

Everything generated from these files — the mirrored URDF and meshes, the
MJCF, the collision calibration, the pose bank — lands in `world/build/`,
which is not tracked. Delete it and it rebuilds.
