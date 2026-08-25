"""
src/export_dialog.py

The export dialog, and the progress window that follows it.

Replaces a chain of yes/no prompts -- clean up? which model? include the
originals? -- with one panel where every choice is visible at once and Save
commits them together. Chained prompts make the user answer questions before
they can see what else is coming, and give no way back.

The progress window exists because cleanup is slow enough to look like a
crash. A real meeting produced 1311 lines; at 25 lines a batch and 20-50s a
batch that is 18 to 45 minutes. Run on the UI thread, as it originally was,
the app freezes solid for that long: no progress, no cancel, and every
instinct tells the user to force-quit and lose the work.
"""

import threading
import time
import tkinter as tk
from dataclasses import dataclass
from typing import Callable, Optional

import customtkinter as ctk


@dataclass
class ExportChoices:
    path: str
    polish: bool = False
    backend: str = "cli"          # "cli" | "api"
    include_original: bool = False

    @property
    def is_markdown(self) -> bool:
        return self.path.lower().endswith(".md")


class ExportDialog(ctk.CTkToplevel):
    """One panel: format, cleanup, backend, originals. Then Save."""

    def __init__(self, parent, default_path: str, cli_available: bool = True,
                 line_count: int = 0):
        super().__init__(parent)
        self.title("Export conversation")
        self.geometry("520x460")
        self.resizable(False, False)
        self.transient(parent)

        self._result: Optional[ExportChoices] = None
        self._path = default_path
        self._cli_available = cli_available
        self._line_count = line_count

        self._polish = tk.BooleanVar(value=False)
        self._backend = tk.StringVar(value="cli" if cli_available else "api")
        self._originals = tk.BooleanVar(value=False)

        self._build()
        self._sync_enabled()

        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._cancel)

    # -- layout ------------------------------------------------------------

    def _build(self) -> None:
        pad = {"padx": 20, "pady": (0, 6)}

        ctk.CTkLabel(self, text="Export conversation",
                     font=("Arial", 18, "bold")).pack(anchor="w", padx=20,
                                                      pady=(18, 2))
        summary = f"{self._line_count} lines" if self._line_count else ""
        ctk.CTkLabel(self, text=summary, font=("Arial", 12),
                     text_color="#8a8a8a").pack(anchor="w", padx=20, pady=(0, 12))

        self._path_label = ctk.CTkLabel(self, text=self._short_path(),
                                        font=("Arial", 12), anchor="w")
        self._path_label.pack(fill="x", **pad)
        ctk.CTkButton(self, text="Change location...", width=150, height=26,
                      command=self._choose_path).pack(anchor="w", padx=20,
                                                      pady=(0, 16))

        ctk.CTkCheckBox(
            self, text="Clean up the transcript with a language model",
            variable=self._polish, command=self._sync_enabled,
        ).pack(anchor="w", padx=20, pady=(0, 2))
        ctk.CTkLabel(
            self,
            text="Fixes mis-heard words and punctuation. The original wording "
                 "of every line is kept either way.",
            font=("Arial", 11), text_color="#8a8a8a",
            wraplength=460, justify="left",
        ).pack(anchor="w", padx=44, pady=(0, 10))

        self._backend_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._backend_frame.pack(fill="x", padx=44, pady=(0, 4))

        self._cli_radio = ctk.CTkRadioButton(
            self._backend_frame, text="Claude CLI  —  no per-token cost, slower",
            variable=self._backend, value="cli", command=self._sync_estimate)
        self._cli_radio.pack(anchor="w", pady=2)

        self._api_radio = ctk.CTkRadioButton(
            self._backend_frame, text="API from conf.yaml  —  faster, billed per token",
            variable=self._backend, value="api", command=self._sync_estimate)
        self._api_radio.pack(anchor="w", pady=2)

        self._estimate = ctk.CTkLabel(self, text="", font=("Arial", 11),
                                      text_color="#8a8a8a")
        self._estimate.pack(anchor="w", padx=44, pady=(0, 10))

        self._originals_box = ctk.CTkCheckBox(
            self, text="Also show the original wording of corrected lines",
            variable=self._originals)
        self._originals_box.pack(anchor="w", padx=44, pady=(0, 4))
        self._originals_hint = ctk.CTkLabel(
            self,
            text="For when the transcript is evidence rather than notes. "
                 "Roughly doubles its length.",
            font=("Arial", 11), text_color="#8a8a8a",
            wraplength=440, justify="left")
        self._originals_hint.pack(anchor="w", padx=68, pady=(0, 12))

        buttons = ctk.CTkFrame(self, fg_color="transparent")
        buttons.pack(side="bottom", fill="x", padx=20, pady=16)
        ctk.CTkButton(buttons, text="Cancel", width=100, fg_color="#3a3a3a",
                      command=self._cancel).pack(side="right", padx=(8, 0))
        ctk.CTkButton(buttons, text="Save", width=120,
                      command=self._save).pack(side="right")

    # -- behaviour ---------------------------------------------------------

    def _sync_enabled(self) -> None:
        """Backend and originals only matter when cleanup is on."""
        on = self._polish.get()
        state = "normal" if on else "disabled"
        for widget in (self._cli_radio, self._api_radio, self._originals_box):
            widget.configure(state=state)
        if not self._cli_available:
            self._cli_radio.configure(state="disabled")
            self._backend.set("api")
        colour = "#8a8a8a" if on else "#4a4a4a"
        self._originals_hint.configure(text_color=colour)
        self._sync_estimate()

    def _sync_estimate(self) -> None:
        if not self._polish.get() or not self._line_count:
            self._estimate.configure(text="")
            return
        self._estimate.configure(
            text=f"Estimated {estimate_minutes(self._line_count, self._backend.get())}")

    def _short_path(self) -> str:
        import os
        return "Saving to:  " + os.path.basename(self._path)

    def _choose_path(self) -> None:
        from tkinter import filedialog
        import os
        chosen = filedialog.asksaveasfilename(
            parent=self,
            initialfile=os.path.basename(self._path),
            initialdir=os.path.dirname(self._path),
            defaultextension=".md",
            filetypes=[("Markdown (for reading)", "*.md"),
                       ("JSON (full record)", "*.json")],
            title="Export conversation")
        if chosen:
            self._path = chosen
            self._path_label.configure(text=self._short_path())

    def _save(self) -> None:
        self._result = ExportChoices(
            path=self._path,
            polish=self._polish.get(),
            backend=self._backend.get(),
            include_original=self._originals.get() and self._polish.get(),
        )
        self.grab_release()
        self.destroy()

    def _cancel(self) -> None:
        self._result = None
        self.grab_release()
        self.destroy()

    def ask(self) -> Optional[ExportChoices]:
        self.wait_window()
        return self._result


def estimate_minutes(lines: int, backend: str, batch_size: int = 25) -> str:
    """
    A rough figure, phrased as a range.

    Measured per 25-line batch: 20-50s through the Claude CLI (the spread is
    queueing upstream, not anything local) and 5-8s through the OpenAI API.
    A single number here would be a lie in one direction or the other.
    """
    batches = max(1, (lines + batch_size - 1) // batch_size)
    low, high = (20, 50) if backend == "cli" else (5, 8)
    lo_min = batches * low / 60
    hi_min = batches * high / 60
    if hi_min < 1:
        return "under a minute"
    if lo_min < 1:
        return f"up to {hi_min:.0f} minutes"
    return f"{lo_min:.0f}-{hi_min:.0f} minutes"


class ProgressWindow(ctk.CTkToplevel):
    """
    Shows cleanup running, and lets it be stopped.

    The work happens on a worker thread; this polls it. Tk is not thread-safe,
    so the worker only ever sets plain attributes and the UI reads them from
    its own `after` loop.
    """

    def __init__(self, parent, total_batches: int, backend: str):
        super().__init__(parent)
        self.title("Cleaning up transcript")
        self.geometry("460x210")
        self.resizable(False, False)
        self.transient(parent)

        self.cancelled = threading.Event()
        self._total = max(1, total_batches)
        self._done = 0
        self._started = time.time()
        self._finished = False

        ctk.CTkLabel(self, text="Cleaning up transcript",
                     font=("Arial", 16, "bold")).pack(anchor="w", padx=24,
                                                      pady=(22, 4))
        ctk.CTkLabel(self, text=f"Using {backend}", font=("Arial", 12),
                     text_color="#8a8a8a").pack(anchor="w", padx=24, pady=(0, 14))

        self._bar = ctk.CTkProgressBar(self, width=410)
        self._bar.set(0)
        self._bar.pack(padx=24, pady=(0, 8))

        self._status = ctk.CTkLabel(self, text="Starting...", font=("Arial", 12))
        self._status.pack(anchor="w", padx=24)

        self._remaining = ctk.CTkLabel(self, text="", font=("Arial", 11),
                                       text_color="#8a8a8a")
        self._remaining.pack(anchor="w", padx=24, pady=(2, 0))

        ctk.CTkButton(self, text="Stop", width=100, fg_color="#5a3a3a",
                      command=self._stop).pack(side="bottom", pady=14)

        # Closing the window stops the work rather than orphaning it.
        self.protocol("WM_DELETE_WINDOW", self._stop)
        self._tick()

    def advance(self, done: int, total: int) -> None:
        """Called from the worker thread. Attribute writes only."""
        self._done = done
        self._total = max(1, total)

    def finish(self) -> None:
        self._finished = True

    def _stop(self) -> None:
        self.cancelled.set()
        self._status.configure(text="Stopping after this batch...")

    def _tick(self) -> None:
        if self._finished:
            self.grab_release()
            self.destroy()
            return

        fraction = self._done / self._total
        self._bar.set(fraction)
        self._status.configure(text=f"Batch {self._done} of {self._total}")

        if self._done >= 1:
            elapsed = time.time() - self._started
            remaining = elapsed / self._done * (self._total - self._done)
            self._remaining.configure(
                text=f"About {_humanise(remaining)} left. "
                     "Stopping keeps whatever has been cleaned so far.")
        else:
            self._remaining.configure(text="Working out how long this will take...")

        self.after(300, self._tick)


def _humanise(seconds: float) -> str:
    if seconds < 60:
        return "under a minute"
    minutes = round(seconds / 60)
    return "a minute" if minutes <= 1 else f"{minutes} minutes"


def run_with_progress(parent, total_batches: int, backend: str,
                      work: Callable[[Callable[[int, int], None],
                                      threading.Event], object]):
    """
    Run `work` on a worker thread behind a progress window.

    `work` is handed a progress callback and a cancellation Event, and should
    check the latter between batches. Returns whatever it returned, or None if
    it raised -- the caller decides what a failure means.
    """
    window = ProgressWindow(parent, total_batches, backend)
    outcome = {}

    def worker():
        try:
            outcome["result"] = work(window.advance, window.cancelled)
        except Exception as e:                      # noqa: BLE001
            outcome["error"] = e
        finally:
            window.finish()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    window.wait_window()
    thread.join(timeout=5)

    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("result")
