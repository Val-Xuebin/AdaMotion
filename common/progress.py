from __future__ import annotations

import time
from typing import Any, Dict

from tqdm.auto import tqdm


class ExperimentProgress:
    def __init__(
        self,
        stage: str,
        epochs: int,
        train_steps: int,
        val_steps: int,
        run=None,
        log_interval: int = 20,
    ) -> None:
        self.stage = stage
        self.epochs = epochs
        self.train_steps = train_steps
        self.val_steps = val_steps
        self.run = run
        self.log_interval = max(1, log_interval)
        self.total_steps = max(1, epochs * (train_steps + val_steps))
        self.completed_steps = 0
        self.wall_start = time.time()
        self.phase_bar = None
        self.epoch = 0
        self._last_log_step = -1
        self.overall = tqdm(
            total=self.total_steps,
            desc=f"{stage} experiment",
            position=0,
            leave=True,
            dynamic_ncols=True,
            smoothing=0.1,
        )

    def start_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def start_phase(self, phase: str, total: int) -> tqdm:
        if self.phase_bar is not None:
            self.phase_bar.close()
        self.phase_bar = tqdm(
            total=total,
            desc=f"{self.stage} {phase} epoch {self.epoch + 1}/{self.epochs}",
            position=1,
            leave=False,
            dynamic_ncols=True,
            smoothing=0.1,
        )
        return self.phase_bar

    def update(self, phase: str, metrics: Dict[str, Any] | None = None) -> None:
        self.completed_steps += 1
        self.overall.update(1)
        if self.phase_bar is not None:
            self.phase_bar.update(1)
            if metrics:
                self.phase_bar.set_postfix({k: _fmt(v) for k, v in metrics.items()})
        elapsed = max(1e-6, time.time() - self.wall_start)
        rate = self.completed_steps / elapsed
        eta_seconds = max(0.0, (self.total_steps - self.completed_steps) / rate) if rate > 0 else 0.0
        progress = self.completed_steps / self.total_steps
        overall_metrics = {
            "epoch": f"{self.epoch + 1}/{self.epochs}",
            "phase": phase,
            "done": f"{progress * 100:.1f}%",
            "eta": _fmt_seconds(eta_seconds),
        }
        if metrics:
            overall_metrics.update({k: _fmt(v) for k, v in metrics.items()})
        self.overall.set_postfix(overall_metrics)
        self._log_to_wandb(phase=phase, metrics=metrics or {}, eta_seconds=eta_seconds, progress=progress)

    def end_phase(self) -> None:
        if self.phase_bar is not None:
            self.phase_bar.close()
            self.phase_bar = None

    def close(self) -> None:
        self.end_phase()
        self.overall.close()

    def _log_to_wandb(self, phase: str, metrics: Dict[str, Any], eta_seconds: float, progress: float) -> None:
        if self.run is None:
            return
        if self.completed_steps == self._last_log_step:
            return
        should_log = (
            self.completed_steps == 1
            or self.completed_steps == self.total_steps
            or self.completed_steps % self.log_interval == 0
        )
        if not should_log:
            return
        payload = {
            "progress/stage": self.stage,
            "progress/epoch": self.epoch + 1,
            "progress/phase": phase,
            "progress/completed_steps": self.completed_steps,
            "progress/total_steps": self.total_steps,
            "progress/fraction": progress,
            "progress/eta_seconds": eta_seconds,
        }
        for key, value in metrics.items():
            payload[f"progress/{phase}/{key}"] = value
        self.run.log(payload, step=self.completed_steps)
        self._last_log_step = self.completed_steps


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _fmt_seconds(seconds: float) -> str:
    seconds = int(max(0, round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
