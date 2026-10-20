# HandControl

Work on the **h4nd** robot hand.

| folder | what |
| --- | --- |
| [`handlab/`](handlab/README.md) | **the project**: one MuJoCo world for the hand and its DYNAMIXEL hardware, a camera that tracks the human hand and records tasks, and the steps that turn a recording into task data (describe, segment, ...). Self-contained. |
| [`archive/`](archive/README.md) | the earlier HandControl codebase (pygame panel, Leap Motion retargeting, learning pipeline), kept as it was. Not developed further. |
| `.venv/` | the Python environment both use (not committed) |

```bash
cd handlab
../.venv/bin/python main.py
```

`handlab/README.md` has everything else.
