"""A processing step run off the GUI thread - one at a time, with progress.

    job = Job()
    job.start("segment: <folder>", work, on_done)   # work(progress) -> result
    job.busy, job.label, job.done, job.total, job.text, job.error
    job.cancel()

`work` reports with progress(done, total, text); once `cancel()` is asked,
the next progress() call ends the work (raises Cancelled). `on_done(result)`
runs on the job's thread after the work succeeds.
"""

from __future__ import annotations

import threading
import time


class Cancelled(Exception):
    pass


class Job:
    def __init__(self):
        self.busy = False
        self.label = ""
        self.done = self.total = 0
        self.text = ""
        self.error = ""
        self.started = 0.0
        self._cancel = threading.Event()

    def start(self, label, work, on_done=None):
        """False if another step is still running."""
        if self.busy:
            return False
        self.busy, self.label, self.error = True, label, ""
        self.done, self.total, self.text = 0, 0, "starting"
        self.started = time.monotonic()
        self._cancel.clear()
        threading.Thread(target=self._run, args=(work, on_done),
                         name="handlab-processing", daemon=True).start()
        return True

    def cancel(self):
        self._cancel.set()

    def _progress(self, done, total, text=""):
        if self._cancel.is_set():
            raise Cancelled()
        self.done, self.total = done, total
        if text:
            self.text = text

    def _run(self, work, on_done):
        try:
            result = work(self._progress)
            if on_done is not None:
                on_done(result)
            self.text = "done"
        except Cancelled:
            self.text = "cancelled"
        except Exception as failure:          # a failed step never takes the app down
            self.error = str(failure) or type(failure).__name__
            self.text = "failed"
        finally:
            self.busy = False
