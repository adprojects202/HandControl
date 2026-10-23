"""The ClearML pipeline's GPU training step, run by the agent (see
src/learning/remote.py): PPO over the bundle's task sims on the GPU
(src/learning/gpu.py), the test, and the run folder uploaded as the "run"
artifact. Parameters ("handlab"): bundle (a ClearML dataset id), envs,
iterations. Not for running by hand: the pipeline sets them.
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from clearml import Dataset, Task  # noqa: E402

from src.learning import gpu, train  # noqa: E402

PARAMETERS = {"bundle": "", "envs": 8192, "iterations": 1500}


def main():
    task = Task.init(project_name="HandRL", task_name="handlab train_gpu",
                     reuse_last_task_id=False)
    params = task.connect(dict(PARAMETERS), name="handlab")
    if not torch.cuda.is_available():
        raise RuntimeError(f"PyTorch {torch.__version__} sees no CUDA GPU - the agent "
                           "installed a CPU build")
    print(f"GPU: {torch.cuda.get_device_name(0)}, PyTorch {torch.__version__}", flush=True)
    bundle = Path(Dataset.get(dataset_id=params["bundle"]).get_local_copy())
    folder = train.RUNS / datetime.now().strftime("%Y%m%d-%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    envs, iterations = int(params["envs"]), int(params["iterations"])
    (folder / "run.json").write_text(json.dumps({
        "method": "ppo", "device": torch.cuda.get_device_name(0), "envs": envs,
        "iterations": iterations, "bundle": params["bundle"],
        "library_made": json.loads((bundle / "bundle.json").read_text())["library_made"],
        "config": train.CONFIG}, indent=2))
    writer = {"class_name": "src.learning.train:ProgressWriter", "iterations": iterations}
    gpu.train(bundle, folder, envs, iterations, train.CONFIG, writer)
    results = gpu.evaluate(bundle, folder, train.latest_checkpoint(folder), train.CONFIG)
    (folder / "evaluation.json").write_text(json.dumps(results, indent=2))
    archive = shutil.make_archive(str(folder), "zip", root_dir=folder.parent, base_dir=folder.name)
    task.upload_artifact("run", artifact_object=archive, wait_on_upload=True)


if __name__ == "__main__":
    main()
