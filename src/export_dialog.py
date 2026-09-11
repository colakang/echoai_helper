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
    cli_command: str = "claude"   # which CLI, when backend is "cli"
    include_original: bool = False
    merge_speakers_to: int = 0    # 0 = leave the labels alone

    @property
    def is_markdown(self) -> bool:
        return self.path.lower().endswith(".md")


class ExportDialog(ctk.CTkToplevel):
    """One panel: format, cleanup, backend, originals. Then Save."""

    # How each CLI is described. Anything installed but unnamed here still
    # gets offered, under its own command name.
    CLI_LABELS = {
        "claude": "Claude CLI",
        "codex": "Codex CLI",
        "gemini": "Gemini CLI",
    }

    def __init__(self, parent, default_path: str, cli_commands=("claude",),
                 line_count: int = 0, speakers_found: int = 0,
                 can_merge: bool = False):
        super().__init__(parent)
        self.title("Export conversation")
        self.geometry("520x560")
        self.resizable(False, False)
        self.transient(parent)

        self._result: Optional[ExportChoices] = None
        self._path = default_path
        # Every agent CLI found on this machine, not just Claude. The dialog
        # offered one hardcoded choice before, so an installed Codex was
        # unreachable and an absent Claude left the API as the only option.
        self._cli_commands = list(cli_commands or ())
        self._line_count = line_count

        self._speakers_found = speakers_found
        self._can_merge = can_merge
        self._merge = tk.BooleanVar(value=False)
        self._merge_to = tk.StringVar(value=str(max(1, min(speakers_found, 4))))
        self._polish = tk.BooleanVar(value=False)
        self._backend = tk.StringVar(
            value=f"cli:{self._cli_commands[0]}" if self._cli_commands
            else "api")
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

        self._cli_radios = []
        for command in self._cli_commands:
            label = self.CLI_LABELS.get(command, command)
            radio = ctk.CTkRadioButton(
                self._backend_frame,
                text=f"{label}  \u2014  no per-token cost, slower",
                variable=self._backend, value=f"cli:{command}",
                command=self._sync_estimate)
            radio.pack(anchor="w", pady=2)
            self._cli_radios.append(radio)

        if not self._cli_commands:
            ctk.CTkLabel(
                self._backend_frame,
                text="No agent CLI found, so only the API is available.",
                font=("Arial", 11), text_color="#8a8a8a",
            ).pack(anchor="w", pady=2)

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

        if self._can_merge and self._speakers_found > 1:
            merge_row = ctk.CTkFrame(self, fg_color="transparent")
            merge_row.pack(fill="x", padx=20, pady=(4, 0))
            ctk.CTkCheckBox(
                merge_row,
                text=f"Merge {self._speakers_found} detected voices down to",
                variable=self._merge).pack(side="left")
            ctk.CTkOptionMenu(
                merge_row, variable=self._merge_to,
                # From 1: the same omission as the live control had. Merging
                # everything into one person is the whole answer when a
                # single voice came back as three.
                values=[str(i) for i in range(1, max(2, self._speakers_found))],
                width=64).pack(side="left", padx=8)
            ctk.CTkLabel(merge_row, text="people",
                         font=("Arial", 12)).pack(side="left")
            ctk.CTkLabel(
                self,
                text="Voices drift with volume and connection quality, so one "
                     "person often ends up split across several labels. "
                     "Re-groups them using the recorded voice prints.",
                font=("Arial", 11), text_color="#8a8a8a",
                wraplength=460, justify="left").pack(anchor="w", padx=44,
                                                     pady=(2, 8))

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
        for widget in (*self._cli_radios, self._api_radio, self._originals_box):
            widget.configure(state=state)
        colour = "#8a8a8a" if on else "#4a4a4a"
        self._originals_hint.configure(text_color=colour)
        self._sync_estimate()

    def _sync_estimate(self) -> None:
        if not self._polish.get() or not self._line_count:
            self._estimate.configure(text="")
            return
        kind = "api" if self._backend.get() == "api" else "cli"
        self._estimate.configure(
            text=f"Estimated {estimate_minutes(self._line_count, kind)}")

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
        chosen = self._backend.get()
        self._result = ExportChoices(
            path=self._path,
            polish=self._polish.get(),
            backend="api" if chosen == "api" else "cli",
            cli_command=(chosen.split(":", 1)[1] if chosen.startswith("cli:")
                         else "claude"),
            include_original=self._originals.get() and self._polish.get(),
            merge_speakers_to=(int(self._merge_to.get())
                               if self._merge.get() else 0),
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

    Measured per 25-line batch: 20-90s through an agent CLI and 5-8s through
    the OpenAI API. A single number here would be a lie in one direction or
    the other.

    The CLI range was 20-50 and came from short batches. Over a real 229-line
    meeting it ran 16-90s with a mean of 65, so the old figure under-promised
    by roughly half -- and an estimate that reads "4 minutes" for an 11-minute
    job is how a progress bar starts looking like a hang.
    """
    batches = max(1, (lines + batch_size - 1) // batch_size)
    low, high = (20, 90) if backend == "cli" else (5, 8)
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
