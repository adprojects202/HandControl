"""The ClearML pipeline's GPU training step, run by the agent (see
src/learning/remote.py): PPO over the bundle's sims on the GPU, the test,
and the run folder uploaded as the "run" artifact. Parameters ("handlab"):
bundle (a ClearML dataset id), learn ("object tracking": gpu.py; "hand
motion": gpu_hand.py), envs, iterations, and the contact solver's
solver_iterations and linesearch_iterations (gpu.py). Not for running by
hand: the pipeline sets them - or clone the task in ClearML and change them
there.
"""

import json
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from clearml import Dataset, Task  # noqa: E402

from src.learning import background, gpu, gpu_hand, train  # noqa: E402

PARAMETERS = {"bundle": "", "learn": "object tracking", "envs": 8192, "iterations": 1500,
              "solver_iterations": gpu.SOLVER_ITERATIONS,
              "linesearch_iterations": gpu.LINESEARCH_ITERATIONS}


def main():
    task = Task.init(project_name="HandRL", task_name="handlab train_gpu",
                     reuse_last_task_id=False)
    params = task.connect(dict(PARAMETERS), name="handlab")
    background.begin("train")
    _mirror_progress(task)
    if not torch.cuda.is_available():
        raise RuntimeError(f"PyTorch {torch.__version__} sees no CUDA GPU - the agent "
                           "installed a CPU build")
    print(f"GPU: {torch.cuda.get_device_name(0)}, PyTorch {torch.__version__}", flush=True)
    bundle = Path(Dataset.get(dataset_id=params["bundle"]).get_local_copy())
    folder = train.RUNS / datetime.now().strftime("%Y%m%d-%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    envs, iterations = int(params["envs"]), int(params["iterations"])
    physics = {"iterations": int(params["solver_iterations"]),
               "linesearch": int(params["linesearch_iterations"])}
    learn = params["learn"]
    if learn not in train.LEARN:
        raise ValueError(f"learn is {learn!r}: one of {', '.join(train.LEARN)}")
    module = gpu_hand if learn == "hand motion" else gpu
    (folder / "run.json").write_text(json.dumps({
        "learn": learn, "method": "ppo", "device": torch.cuda.get_device_name(0),
        "envs": envs, "iterations": iterations, "bundle": params["bundle"], "physics": physics,
        "library_made": json.loads((bundle / "bundle.json").read_text())["library_made"],
        "config": train.CONFIG,
        **({"env": {name: getattr(gpu_hand, name) for name in gpu_hand.SETTINGS}}
           if module is gpu_hand else {})}, indent=2))
    writer = {"class_name": "src.learning.train:ProgressWriter", "iterations": iterations}
    module.train(bundle, folder, envs, iterations, train.CONFIG, writer, physics=physics)
    results = module.evaluate(bundle, folder, train.latest_checkpoint(folder), train.CONFIG,
                              physics=physics)
    if module is gpu_hand:                     # the hand library goes with the run
        shutil.copy(bundle / "bundle.json", folder / "bundle.json")
    (folder / "evaluation.json").write_text(json.dumps(results, indent=2))
    task.upload_artifact("run", artifact_object=_pack(folder), wait_on_upload=True)
    task.set_progress(100)


def _pack(folder):
    """The run as the Mac needs it, zipped: everything but the earlier
    checkpoints and the TensorBoard log (ClearML keeps the scalars) - a few
    MB, where all of it is ~100 MB and stalls on the way down."""
    last = train.latest_checkpoint(folder)
    packed = folder.parent / "packed" / folder.name
    packed.mkdir(parents=True)
    for path in folder.iterdir():
        if path.is_file() and not path.name.startswith("events.out") and \
                (not path.name.startswith("model_") or path == last):
            shutil.copy(path, packed / path.name)
    return shutil.make_archive(str(packed), "zip", root_dir=packed.parent, base_dir=folder.name)


def _mirror_progress(task):
    """The rounds done (ProgressWriter's status file here), as the ClearML
    task's progress - which the Mac's launcher shows in the Learning panel."""
    def mirror():
        while True:
            status = background.read() or {}
            if status.get("total"):
                task.set_progress(int(100 * status.get("done", 0) / status["total"]))
            time.sleep(15)
    threading.Thread(target=mirror, daemon=True).start()


if __name__ == "__main__":
    main()
