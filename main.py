#main.py

from datetime import datetime
import threading
from tkinter import filedialog, messagebox
import customtkinter as ctk
import queue
import time

import sys
import subprocess
import os
import glob
import json
import tkinter as tk  # 添加这一行

from src.audio import get_audio_backend
from src.AudioTranscriber import AudioTranscriber
from src.GPTResponder import GPTResponder
from src.ResponseManager import ResponseManager
from src.SettingsManager import SettingsManager
from src.TemplateManager import TemplateManager
import src.TranscriberModels as TranscriberModels
from src.config import EnvConfig, SystemConfig, AudioConfig, PathConfig
from src import profiles
from src.TranscriptUI import TranscriptUI
#import torch

def update_response_UI(responder, textbox, freeze_state, transcript_ui):
    if not freeze_state[0] and not transcript_ui.is_response_frozen():
        new_response = responder.response

        # 只在响应内容变化时更新
        current_text = textbox.get("1.0", "end-1c")
        if new_response != current_text:
            # 保存当前的选择范围
            try:
                selection_start = textbox.index("sel.first")
                selection_end = textbox.index("sel.last")
                has_selection = True
            except tk.TclError:  # 没有选择时会抛出异常
                has_selection = False

            # 更新文本
            textbox.configure(state="normal")
            textbox.delete("1.0", "end")
            textbox.insert("1.0", new_response)
            
            # 如果之前有选择，恢复选择
            if has_selection:
                try:
                    textbox.tag_add("sel", selection_start, selection_end)
                except Exception:
                    pass  # 如果无法恢复选择，就忽略错误，保持界面流畅
                
            textbox.configure(state="normal")  # 保持可选择状态

    # 定时调用以保持UI更新
    textbox.after(300, update_response_UI, responder, textbox, freeze_state,
                  transcript_ui)

    
def _run_first_launch_setup(root=None):
    """
    Offer to finish setup the first time this is run.

    Capture needs a virtual audio device and a Multi-Output routing system
    sound through it as well as the speakers. Asking a user to build that in
    Audio MIDI Setup before their first meeting loses most of them, and there
    is no reason to: all of it is scriptable, and only the driver install
    needs a password.

    Declining is remembered, so this asks once rather than at every launch.
    """
    if sys.platform != "darwin":
        return True

    from src.audio import setup_macos as setup

    state = setup.inspect()
    if state.blackhole is not None and state.multi_output is not None:
        return True

    settings = SettingsManager()
    if settings.get_setting("setup_declined"):
        print("[INFO] Audio routing is incomplete; run "
              "scripts/setup_audio.py to finish it.")
        return False

    needs_install = state.blackhole is None
    detail = ("EchoAI Helper needs a virtual audio device to hear the other "
              "side of a call, and a Multi-Output device so you still hear it "
              "yourself.\n\nSet this up now?")
    if needs_install:
        detail += ("\n\nmacOS will ask for your password once, to install the "
                   "audio driver.")

    if not messagebox.askyesno("Finish setting up audio?", detail):
        settings.update_setting("setup_declined", True)
        messagebox.showinfo(
            "Recording is limited",
            "Without it, only your microphone is recorded — not the other "
            "side of the call.\n\nRun scripts/setup_audio.py whenever you "
            "want to finish.")
        return False

    state = setup.run(auto_activate=True)
    if state.ready:
        messagebox.showinfo("Ready",
                            "Meeting audio will now be both heard and recorded.")
        return True

    messagebox.showwarning(
        "Setup did not finish",
        "Some of it could not be completed:\n\n" + state.describe())
    return False


def _offer_recovery(root, transcriber, response_manager):
    """
    Offer to reopen a session that ended without being exported.

    Only when the last one crashed, had content, and was recent. A meeting
    from last week belongs in the history list, not spliced onto the front of
    today's -- resuming it would merge two unrelated conversations into one
    transcript.
    """
    if not messagebox:
        return
    try:
        from src.session import find_recoverable, to_conversation

        recoverable = find_recoverable()
        if recoverable is None:
            return

        if not messagebox.askyesno(
                "Recover the last meeting?",
                f"{recoverable.name} ended without being exported, and has "
                f"{recoverable.line_count} lines.\n\n"
                "Load it so you can export it now?\n\n"
                "It stays on disk either way — you can also reach it from "
                "Past meetings."):
            return

        conversation = to_conversation(recoverable.path)
        _load_conversation(transcriber, conversation)
        messagebox.showinfo(
            "Loaded",
            f"{recoverable.line_count} lines restored. Export them from the "
            "button below.\n\nAnything said from now on is recorded to a new "
            "session.")
    except Exception as e:
        print(f"Could not recover the last session: {e}")


def _load_conversation(transcriber, conversation):
    """Put a saved conversation back into the in-memory transcript."""
    from datetime import datetime as _dt

    messages = conversation.get("conversation", {}).get("messages", [])
    for message in messages:
        text = message.get("text") or ""
        if not text.strip():
            continue
        role = (message.get("role") or "speaker").lower()
        try:
            when = _dt.fromisoformat(message["timestamp"]).replace(tzinfo=None)
        except (KeyError, TypeError, ValueError):
            when = _dt.now()

        entry = (text, when, message.get("response_id"))
        transcriber.structured_transcript[role].insert(0, entry)
        transcriber.structured_transcript["combined"].insert(
            0, (text, when, message.get("response_id"), role))

        embedding = message.get("embedding")
        if embedding and message.get("response_id"):
            transcriber.speaker_embeddings[message["response_id"]] = embedding


def _restore_on_exit(root):
    """
    Put the system output back on the way out, without asking.

    This used to be a prompt. It should not be: the user has no way to judge
    the answer, and the cost of the wrong one is delayed and baffling --
    macOS cannot control the volume of a Multi-Output device at all, so the
    volume keys and the menu-bar slider silently stop working, and that is
    discovered days later with no connection to this app.

    Restores the device that was in use before we took over, rather than
    guessing: someone listening through an external monitor should not end up
    on the built-in speakers.
    """
    try:
        session = getattr(_restore_on_exit, "session", None)
        if session is not None:
            session.close()
    except Exception as e:
        print(f"Could not close the session: {e}")

    if sys.platform == "darwin":
        try:
            from src.audio import setup_macos as setup
            state = setup.inspect()
            if (state.default_output is not None
                    and state.default_output.uid == setup.MULTI_OUTPUT_UID):
                setup.restore()
        except Exception as e:
            print(f"Could not restore the audio output: {e}")

    root.destroy()


def _polish_with_progress(root, conversation_data, backend):
    """
    Clean the transcript on a worker thread, behind a progress window.

    It used to run inline on the UI thread. On a real meeting -- 1311 lines,
    53 batches -- that is 18 to 45 minutes of a frozen window with no
    progress and no way out, which is indistinguishable from a hang and
    invites a force-quit that loses the work.
    """
    try:
        from src.polish import polish_transcript, DEFAULT_BATCH_SIZE
        from src.export_dialog import run_with_progress

        provider = _build_polish_provider(backend)
        if provider is None:
            return "\n\nCleanup skipped: no language model available."

        messages = conversation_data["conversation"]["messages"]
        countable = len([m for m in messages if (m.get("text") or "").strip()])
        batches = max(1, (countable + DEFAULT_BATCH_SIZE - 1) // DEFAULT_BATCH_SIZE)

        def work(report, cancelled):
            return polish_transcript(
                messages, provider,
                progress=lambda done, total: report(done, total),
                cancelled=cancelled)

        result = run_with_progress(root, batches, provider.get_model_name(), work)
        if result is None:
            return "\n\nCleanup did not run."

        conversation_data["conversation"]["messages"] = result.segments
        conversation_data.setdefault("metadata", {})["cleanup"] = {
            "model": provider.get_model_name(),
            "lines_cleaned": result.polished_count,
            "batches_failed": result.batches_failed,
            "stopped_early": result.cancelled,
        }
        note = f"\n\nCleanup: {result.summary()}"
        if result.cancelled:
            note += "\nStopped early; the rest was left as recorded."
        return note

    except Exception as e:
        print(f"Transcript cleanup failed: {e}")
        import traceback
        traceback.print_exc()
        return f"\n\nCleanup failed ({e}); the transcript was saved unchanged."


def _build_polish_provider(backend):
    """Build the provider for one cleanup run. 'cli' overrides conf.yaml."""
    import yaml
    from src.llm import create_llm_provider

    with open(f"{PathConfig.get_project_root()}/conf.yaml", "rb") as f:
        llm_config = (yaml.safe_load(f) or {}).get("LLM", {})

    if backend == "cli":
        return create_llm_provider("cli", dict(llm_config.get("cli", {})))

    provider_type = llm_config.get("provider", "openai").lower()
    if provider_type == "cli":
        provider_type = "openai"
    if provider_type == "openai":
        config = {"api_key": EnvConfig.get_openai_key(),
                  "model": llm_config.get("openai", {}).get("model", "gpt-4o-mini")}
    else:
        config = dict(llm_config.get(provider_type, {}))
    return create_llm_provider(provider_type, config)


def _ask_polish_backend():
    """
    Ask whether to clean the transcript, and on what.

    The two backends differ in ways the user is the only one who can weigh:
    an API key is metered but fast, a subscription CLI is already paid for but
    roughly three times slower and subject to its own rate limits. Measured on
    an hour-long meeting: about 3 minutes against about 9.
    """
    if not messagebox.askyesno(
            "Clean up transcript?",
            "Run the transcript through a language model to fix "
            "speech-recognition errors before saving?\n\n"
            "The original wording of every line is kept either way; "
            "corrections are stored alongside it."):
        return None

    from src.llm.cli_provider import CLIProvider
    cli_available = CLIProvider(command="claude").validate_config()

    if not cli_available:
        return "config"

    use_cli = messagebox.askyesno(
        "Which model?",
        "Use the Claude CLI on your existing subscription?\n\n"
        "Yes  —  Claude CLI. No per-token cost, but slower: roughly "
        "9 minutes for an hour-long meeting, and subject to your "
        "subscription's rate limits.\n\n"
        "No  —  the API configured in conf.yaml. Around 3 minutes for the "
        "same meeting, billed per token.")
    return "cli" if use_cli else "config"


def _save_markdown(filepath, conversation_data, include_original):
    """Render and write the human-readable export."""
    try:
        from src.export_markdown import render
        text = render(conversation_data, include_original=include_original)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)
        return True
    except Exception as e:
        print(f"Markdown export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def _polish_before_export(conversation_data, backend="config"):
    """
    Clean the transcript in place and describe what happened.

    Runs here rather than during the meeting: one request per batch instead of
    one per utterance, the model gets to see the surrounding conversation, and
    nobody is waiting on it -- which is also what makes the CLI provider, at
    roughly 4s a call, a sensible backend for this and not for live prompting.

    Never fatal. A transcript that failed to be cleaned is still a transcript,
    and refusing to export it because the model was unavailable would be the
    worse outcome.
    """
    try:
        from src.polish import polish_transcript
        from src.llm import create_llm_provider
        import yaml

        with open(f"{PathConfig.get_project_root()}/conf.yaml", "rb") as f:
            llm_config = (yaml.safe_load(f) or {}).get("LLM", {})

        # "cli" overrides conf.yaml for this export only; the live responder
        # keeps whatever is configured, since 4s per answer is unusable there.
        provider_type = ("cli" if backend == "cli"
                         else llm_config.get("provider", "openai").lower())
        if provider_type == "openai":
            provider_config = {
                "api_key": EnvConfig.get_openai_key(),
                "model": llm_config.get("openai", {}).get("model", "gpt-4o-mini"),
            }
        else:
            provider_config = dict(llm_config.get(provider_type, {}))

        provider = create_llm_provider(provider_type, provider_config)
        if provider is None:
            return "\n\nCleanup skipped: no language model configured."

        messages = conversation_data["conversation"]["messages"]
        result = polish_transcript(messages, provider)
        conversation_data["conversation"]["messages"] = result.segments
        conversation_data.setdefault("metadata", {})["cleanup"] = {
            "model": provider.get_model_name(),
            "lines_cleaned": result.polished_count,
            "batches_failed": result.batches_failed,
        }
        return f"\n\nCleanup: {result.summary()}"

    except Exception as e:
        print(f"Transcript cleanup failed: {e}")
        import traceback
        traceback.print_exc()
        return f"\n\nCleanup failed ({e}); original transcript saved."


def clear_context(transcriber, mic_queue, speaker_queue, transcript_ui):
    """
    Phase 2: 清除所有上下文（双队列版本）
    """
    print("Clearing context...")
    # 清除transcriber数据
    transcriber.clear_transcript_data()
    # Phase 2: 清除两个音频队列
    with mic_queue.mutex:
        mic_queue.queue.clear()
    with speaker_queue.mutex:
        speaker_queue.queue.clear()
    # 清除UI显示
    transcript_ui.clear()
    print("Context cleared")

def create_ui_components(root, response_manager, transcriber, mic_queue, speaker_queue):
    """Phase 2: 创建并配置所有UI组件（双队列版本）"""
    # 基础设置
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
    root.title("EchoAI 365 (Helper Mode)")
    root.configure(bg='#252422')
    root.geometry("1200x800")

    # 创建设置管理器
    settings_manager = SettingsManager()
    # 设置更小的组件高度

    button_height = 25  # 减小按钮高度
    dropdown_height = 25  # 减小下拉菜单高度
    font_size = 20
    
    # 主要内容区域
    transcript_textbox = ctk.CTkTextbox(
        root, 
        width=400, 
        font=("Arial", font_size), 
        text_color='#FFFCF2', 
        wrap="word",
        state="normal"  # 确保可以选择文本
    )
    transcript_textbox.grid(row=0, column=0, padx=10, pady=(20,10), sticky="nsew")

    response_textbox = ctk.CTkTextbox(
        root, 
        width=600, 
        font=("Arial", font_size), 
        text_color='#639cdc', 
        wrap="word",
        state="normal"  # 确保可以选择文本
    )
    response_textbox.grid(row=0, column=1, padx=10, pady=(20,10), sticky="nsew")

    # 创建main_control_frame时设置较小的padding
    main_control_frame = ctk.CTkFrame(root, fg_color="#252422")
    main_control_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=0)

    # 配置main_control_frame的行高和列宽
    main_control_frame.grid_rowconfigure(0, minsize=25)  # 原来是weight=1
    main_control_frame.grid_rowconfigure(1, minsize=25)
    main_control_frame.grid_rowconfigure(2, minsize=25)

    for i in range(4):
        main_control_frame.grid_columnconfigure(i, weight=1)
    # === Column 1: Prompt Templates ===
    system_role_files = TemplateManager.get_template_files('system_role')
    case_detail_files = TemplateManager.get_template_files('case_detail')
    knowledge_files = TemplateManager.get_template_files('knowledge')
    
    templates = {
        "System Role": (system_role_files, "system_role"),
        "Case Detail": (case_detail_files, "case_detail"),
        "Knowledge Base": (knowledge_files, "knowledge")
    }

    template_vars = {}
    row = 0
    for label, (options, setting_key) in templates.items():
        label_widget = ctk.CTkLabel(
            main_control_frame,
            text=label,
            font=("Arial", 12),
            text_color="#FFFCF2"
        )
        label_widget.grid(row=row, column=0, padx=5, pady=2, sticky="w")

        saved_value = settings_manager.get_setting(setting_key)
        var = ctk.StringVar(value=saved_value if saved_value in (options or ['default']) else (options or ['default'])[0])
        menu = ctk.CTkOptionMenu(
            main_control_frame,
            variable=var,
            values=options or ['default'],
            width=160,
            height=dropdown_height,  # 新增这行
        )
        menu.grid(row=row, column=0, padx=(80, 5), pady=1, sticky="e")
        template_vars[setting_key] = var
        row += 1

    def on_selection_change(*args):
        """处理模板选择变化"""
        try:
            for key, var in template_vars.items():
                settings_manager.update_setting(key, var.get())
            
            new_role = TemplateManager.update_system_role(
                template_vars["system_role"].get(),
                template_vars["case_detail"].get(),
                template_vars["knowledge"].get()
            )
            if new_role is None:
                print("Warning: Failed to update system role")
        except Exception as e:
            print(f"Error updating system role: {e}")

    for var in template_vars.values():
        # trace_add('write'), not the legacy trace('w'): Tk 9.0 — which
        # Homebrew's python-tk@3.12 ships — removed the old form, and it
        # raises TclError('bad option "variable"') at startup.
        var.trace_add('write', on_selection_change)

    # === Column 2: Action Buttons ===
    def export_responses():
        """Export the conversation, optionally cleaning it up first."""
        try:
            conversation_data = response_manager.export_structured_conversation(
                transcriber.structured_transcript, reverse_chronological=False,
                speaker_embeddings=getattr(transcriber, "speaker_embeddings", None))

            messages = (conversation_data.get("conversation", {}) or {}).get(
                "messages", []) if conversation_data else []
            if not messages:
                messagebox.showwarning(
                    "Nothing to export",
                    "No conversation has been recorded yet.")
                return

            from src.export_dialog import ExportDialog, run_with_progress
            from src.llm.cli_provider import CLIProvider

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_path = os.path.join(
                os.path.expanduser("~/Desktop"), f"conversation_{timestamp}.md")

            import re as _re
            found = {_re.match(r"S(\d+)", m.get("speaker") or "")
                     for m in messages if m.get("speaker")}
            found = {m.group(1) for m in found if m}

            choices = ExportDialog(
                root, default_path,
                cli_available=CLIProvider(command="claude").validate_config(),
                line_count=len([m for m in messages if (m.get("text") or "").strip()]),
                speakers_found=len(found),
                can_merge=any(m.get("embedding") for m in messages),
            ).ask()
            if choices is None:
                return

            cleanup_note = ""
            if choices.merge_speakers_to:
                from src.polish import merge_speakers
                changed = merge_speakers(messages, choices.merge_speakers_to)
                cleanup_note += (f"\n\nRe-grouped voices into "
                                 f"{choices.merge_speakers_to} speakers "
                                 f"({changed} lines relabelled).")
            if choices.polish:
                cleanup_note = _polish_with_progress(
                    root, conversation_data, choices.backend)

            if choices.is_markdown:
                saved = _save_markdown(choices.path, conversation_data,
                                       choices.include_original)
            else:
                saved = response_manager.save_structured_conversation(
                    choices.path, conversation_data)

            if not saved:
                messagebox.showerror(
                    "Export failed",
                    f"Could not write to:\n{choices.path}\n\n"
                    "Check the folder is writable and has space.")
                return

            lines = len(conversation_data["conversation"]["messages"])
            messagebox.showinfo(
                "Exported",
                f"{lines} lines saved to:\n{choices.path}" + cleanup_note)

        except Exception as e:
            messagebox.showerror("Export error", f"{e}")
            print(f"Export error: {e}")
            import traceback
            traceback.print_exc()

    # === Column 2: Action Buttons ===
    buttons_data = [
        ("Clear Transcript", lambda: clear_context(transcriber, mic_queue, speaker_queue, transcript_ui), "#1f538d"),
        ("Export Conversation", export_responses, "#1B4332"),
        ("Pop Up", None, "#1B4332")
    ]

    # 创建按钮并保存引用
    clear_transcript_button = None
    export_button = None
    freeze_button = None
    
    for i, (text, command, color) in enumerate(buttons_data):
        btn = ctk.CTkButton(
            main_control_frame,
            text=text,
            command=command,
            width=160,
            height=button_height,  # 新增这行
            fg_color=color,
            hover_color="#2B7A0B",
        )
        btn.grid(row=i, column=1, padx=5, pady=1)
        
        # 保存按钮引用
        if text == "Clear Transcript":
            clear_transcript_button = btn
        elif text == "Export Conversation":
            export_button = btn
        elif text == "Pop Up":
            freeze_button = btn

    # 创建TranscriptUI实例
    transcript_ui = TranscriptUI(transcript_textbox, response_manager)
    transcript_ui.add_click_handler(response_textbox)

    # === Column 3: Transcription Controls ===
    #
    # These three used to be "Phrase Timeout", "Buffer Chunks" and "Update
    # Interval". All three had stopped doing anything: segmentation moved to
    # the VAD, chunks_buffer was deleted, and the responder stored the
    # interval without ever reading it. A control that silently does nothing
    # is worse than no control.

    profile_label = ctk.CTkLabel(
        main_control_frame,
        text="Mode:",
        font=("Arial", 12)
    )
    profile_label.grid(row=0, column=2, padx=5, pady=2, sticky="w")

    saved_profile = profiles.by_key(settings_manager.get_setting("profile"))
    profile_var = ctk.StringVar(value=saved_profile.label)

    profile_hint = ctk.CTkLabel(
        main_control_frame,
        text=saved_profile.description,
        font=("Arial", 10),
        text_color="#8a8a8a",
        wraplength=260,
        justify="left",
    )
    profile_hint.grid(row=1, column=2, columnspan=1, padx=5, pady=(0, 2), sticky="w")

    def on_profile_change(label):
        profile = profiles.by_label(label)
        profiles.apply(profile, transcriber)
        settings_manager.update_setting("profile", profile.key)
        settings_manager.update_setting("min_silence_ms", profile.min_silence_ms)
        profile_hint.configure(text=profile.description)
        pause_var.set(str(profile.min_silence_ms))
        profile_label.configure(text_color="#639cdc")
        root.after(500, lambda: profile_label.configure(text_color="#FFFCF2"))

    profile_dropdown = ctk.CTkOptionMenu(
        main_control_frame,
        variable=profile_var,
        values=profiles.labels(),
        width=150,
        height=dropdown_height,
        command=on_profile_change,
    )
    profile_dropdown.grid(row=0, column=2, padx=(60, 5), pady=2, sticky="w")

    # Pause length -- the one segmentation knob worth exposing. Shorter reacts
    # sooner but splits sentences; longer merges separate turns.
    pause_label = ctk.CTkLabel(
        main_control_frame,
        text="Pause:",
        font=("Arial", 12)
    )
    pause_label.grid(row=2, column=2, padx=5, pady=2, sticky="w")

    pause_values = ["300", "450", "600", "700", "900", "1200", "1500"]
    pause_var = ctk.StringVar(value=str(settings_manager.get_setting("min_silence_ms")))

    def on_pause_change(value):
        settings_manager.update_setting("min_silence_ms", int(value))
        current = profiles.by_label(profile_var.get())
        config = current.segmenter_config()
        config.min_silence_ms = int(value)
        transcriber.apply_segmenter_config(config)
        pause_label.configure(text_color="#639cdc")
        root.after(500, lambda: pause_label.configure(text_color="#FFFCF2"))

    pause_dropdown = ctk.CTkOptionMenu(
        main_control_frame,
        variable=pause_var,
        values=pause_values,
        width=70,
        command=on_pause_change,
    )
    pause_dropdown.grid(row=2, column=2, padx=(60, 5), pady=2, sticky="w")

    pause_unit = ctk.CTkLabel(
        main_control_frame,
        text="ms silence ends a sentence",
        font=("Arial", 10),
        text_color="#8a8a8a",
    )
    pause_unit.grid(row=2, column=2, padx=(135, 5), pady=2, sticky="w")

    # === Column 4: Window Controls ===
    # Create a frame for the first row controls
    controls_frame = ctk.CTkFrame(main_control_frame, fg_color="transparent")
    controls_frame.grid(row=0, column=3, padx=5, pady=(2, 0), sticky="w")  # 减少下方padding

    # Record Only Checkbox
    record_only_var = tk.BooleanVar(value=settings_manager.get_setting("record_only_mode"))

    def toggle_record_only():
        is_record_only = record_only_var.get()
        SystemConfig.set_record_only_mode(is_record_only)
        settings_manager.update_setting("record_only_mode", is_record_only)

    record_only_checkbox = ctk.CTkCheckBox(
        controls_frame,
        text="Record Only",
        variable=record_only_var,
        command=toggle_record_only,
        width=100,
        height=button_height,  # 新增这行
        checkbox_width=16,
        checkbox_height=16
    )
    record_only_checkbox.pack(side="left", padx=(0, 5))  # 减少右侧padding

    # Speaker labelling. Only meaningful on the far-end track, which carries
    # everyone in the meeting; costs one voice embedding per utterance.
    diarize_var = tk.BooleanVar(value=settings_manager.get_setting("diarization"))

    def toggle_diarization():
        enabled = diarize_var.get()
        AudioConfig.set_diarization(enabled)
        settings_manager.update_setting("diarization", enabled)
        if enabled:
            # Warm it here rather than inside the next utterance, where the
            # load would stall transcription for 20s or more.
            threading.Thread(target=transcriber.preload_speaker_model,
                             daemon=True).start()

    diarize_checkbox = ctk.CTkCheckBox(
        controls_frame,
        text="Speakers",
        variable=diarize_var,
        command=toggle_diarization,
        width=90,
        height=button_height,
        checkbox_width=16,
        checkbox_height=16,
    )
    diarize_checkbox.pack(side="left", padx=(0, 5))

    # How many people are on the call. Voice embeddings drift with volume,
    # codec and network conditions, so left to itself the clustering splits
    # one person into several -- a real call produced 12 speakers, exactly the
    # cap, for a handful of people. Given the real number it stops inventing.
    people_values = ["auto"] + [str(i) for i in range(2, 13)]
    saved_people = settings_manager.get_setting("speaker_count")
    people_var = ctk.StringVar(
        value=str(saved_people) if saved_people else "auto")

    def on_people_change(value):
        count = 0 if value == "auto" else int(value)
        AudioConfig.set_speaker_count(count)
        settings_manager.update_setting("speaker_count", count)

    people_menu = ctk.CTkOptionMenu(
        controls_frame, variable=people_var, values=people_values,
        width=76, height=button_height, command=on_people_change)
    people_menu.pack(side="left", padx=(0, 5))
    ctk.CTkLabel(controls_frame, text="people", font=("Arial", 11),
                 text_color="#8a8a8a").pack(side="left", padx=(0, 5))

    # Topmost Button
    topmost_var = tk.BooleanVar(value=settings_manager.get_setting("window_topmost"))

    def toggle_topmost():
        is_topmost = topmost_var.get()
        root.attributes('-topmost', is_topmost)
        settings_manager.update_setting("window_topmost", is_topmost)
        topmost_button.configure(
            fg_color="#1B4332" if is_topmost else "#2B2B2B"
        )

    topmost_button = ctk.CTkButton(
        controls_frame,
        text="📌",
        width=30,
        command=lambda: [topmost_var.set(not topmost_var.get()), toggle_topmost()]
    )
    topmost_button.pack(side="left", padx=0)
    topmost_button.configure(fg_color="#1B4332" if topmost_var.get() else "#2B2B2B")

    # Opacity Control 
    saved_opacity = settings_manager.get_setting("window_opacity")

    # 创建一个frame来容纳标签和slider，占用剩余行
    opacity_frame = ctk.CTkFrame(main_control_frame, fg_color="transparent")
    opacity_frame.grid(row=1, column=3, rowspan=2, padx=5, pady=(0, 2), sticky="nsew")  # 减少垂直padding

    # 配置opacity_frame的行权重，让滑块区域可以伸展
    opacity_frame.grid_rowconfigure(0, weight=0)  # label行不伸展
    opacity_frame.grid_rowconfigure(1, weight=1)  # 滑块行填充剩余空间

    # 添加标签显示标题和当前值
    opacity_label = ctk.CTkLabel(
        opacity_frame,
        text=f"Opacity: {int(saved_opacity * 100)}%",
        font=("Arial", 12),
        text_color="#FFFCF2"
    )
    opacity_label.grid(row=0, pady=1)  # 减少垂直padding

    def update_opacity(value):
        opacity = float(value)
        root.attributes('-alpha', opacity)
        settings_manager.update_setting("window_opacity", opacity)
        opacity_label.configure(text=f"Opacity: {int(opacity * 100)}%")

    opacity_slider = ctk.CTkSlider(
        opacity_frame,
        from_=0.3,
        to=1.0,
        orientation="vertical",
        height=80,  # 设置一个合理的固定高度        
        command=update_opacity
    )
    opacity_slider.grid(row=1, pady=1, sticky="n")  # 减少底部padding
    opacity_slider.set(saved_opacity)

    # Window Drag Support
    drag_data = {"x": 0, "y": 0, "dragging": False}

    def start_drag(event):
        drag_data["dragging"] = True
        drag_data["x"] = event.x_root - root.winfo_x()
        drag_data["y"] = event.y_root - root.winfo_y()

    def stop_drag(event):
        drag_data["dragging"] = False

    # 移除所有子组件的内部padding
    for child in root.winfo_children():
        child.grid_configure(pady=0)

    # A dict rather than a positional tuple: the tuple had grown to ten
    # entries and still carried three widgets whose controls no longer did
    # anything, which is exactly how that happens.
    return {
        "transcript_ui": transcript_ui,
        "response_textbox": response_textbox,
        "freeze_button": freeze_button,
        "clear_transcript_button": clear_transcript_button,
        "export_button": export_button,
        "profile_dropdown": profile_dropdown,
        "pause_dropdown": pause_dropdown,
    }

def main():
    try:
        # 初始化环境配置
        EnvConfig.initialize()
        if not EnvConfig.ensure_api_key():
            print("Please set up your OpenAI API key and restart the application.")
            input("Press Enter to exit...")
            return
                
        # 检查ffmpeg
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("ERROR: The ffmpeg library is not installed. Please install ffmpeg and try again.")
        return

    TemplateManager.ensure_template_directories()

    # Phase 2: Dual-queue architecture for full-duplex processing
    mic_queue = queue.Queue()
    speaker_queue = queue.Queue()

    # Platform-neutral capture: WASAPI loopback on Windows, sounddevice +
    # virtual audio device on macOS. Either recorder may be None (no mic, or
    # no loopback route), so start only the tracks that actually exist.
    backend = get_audio_backend()
    print(f"[INFO] Audio backend: {backend.name}")

    user_audio_recorder = backend.create_mic_recorder()
    speaker_audio_recorder = backend.create_speaker_recorder()

    if user_audio_recorder is not None:
        user_audio_recorder.record_into_queue(mic_queue)

    if speaker_audio_recorder is not None:
        speaker_audio_recorder.record_into_queue(speaker_queue)

    if user_audio_recorder is None and speaker_audio_recorder is None:
        print("ERROR: No audio input available. Run "
              "`python scripts/check_audio.py` to diagnose.")
        return

    model = TranscriberModels.get_model('--api' in sys.argv)

    # 创建ResponseManager实例
    response_manager = ResponseManager()

    transcriber = AudioTranscriber(
        user_audio_recorder.source if user_audio_recorder else None,
        speaker_audio_recorder.source if speaker_audio_recorder else None,
        model,
        response_manager,
    )

    # One session per launch, created without asking. Requiring a step before
    # recording hands the user an implementation detail, and forgetting it
    # costs the whole meeting.
    from src.session import SessionWriter
    transcriber.session = SessionWriter()
    print(f"[INFO] Recording to {transcriber.session.path}")

    # Phase 2: Dual-thread transcription (shared model, independent queues)
    if user_audio_recorder is not None:
        mic_transcribe = threading.Thread(target=transcriber.transcribe_audio_queue, args=(mic_queue,), name="MicTranscriber")
        mic_transcribe.daemon = True
        mic_transcribe.start()

    if speaker_audio_recorder is not None:
        speaker_transcribe = threading.Thread(target=transcriber.transcribe_audio_queue, args=(speaker_queue,), name="SpeakerTranscriber")
        speaker_transcribe.daemon = True
        speaker_transcribe.start()

    responder = GPTResponder(response_manager)
    respond = threading.Thread(target=responder.respond_to_transcriber, args=(transcriber,))
    respond.daemon = True
    respond.start()

    #monitor = threading.Thread(target=transcriber.self_check)
    #monitor.daemon = True
    #monitor.start()

    root = ctk.CTk()
    widgets = create_ui_components(root, response_manager, transcriber,
                                   mic_queue, speaker_queue)
    transcript_ui = widgets["transcript_ui"]
    response_textbox = widgets["response_textbox"]
    freeze_button = widgets["freeze_button"]
    clear_transcript_button = widgets["clear_transcript_button"]


    # 创建设置管理器实例
    settings_manager = SettingsManager()
    
    # 加载窗口设置
    saved_opacity = settings_manager.get_setting("window_opacity")
    saved_topmost = settings_manager.get_setting("window_topmost")
    
    root.attributes('-alpha', saved_opacity)  # 设置透明度
    root.attributes('-topmost', saved_topmost)  # 设置置顶状态   
    
    SystemConfig.set_record_only_mode(settings_manager.get_setting("record_only_mode"))
    AudioConfig.set_diarization(settings_manager.get_setting("diarization"))
    AudioConfig.set_speaker_count(settings_manager.get_setting("speaker_count"))
    if AudioConfig.get_diarization():
        # Off the main thread: loading it costs ~20s the first time, and the
        # UI should come up regardless. Segments transcribe unlabelled until
        # it is ready rather than waiting for it.
        threading.Thread(target=transcriber.preload_speaker_model,
                         daemon=True).start()

    # Apply the saved transcription profile before the first chunk arrives.
    saved_profile = profiles.by_key(settings_manager.get_setting("profile"))
    config = saved_profile.segmenter_config()
    config.min_silence_ms = int(settings_manager.get_setting("min_silence_ms"))
    profiles.apply(saved_profile, transcriber)
    transcriber.apply_segmenter_config(config)
    print(f"[INFO] Profile: {saved_profile.label} "
          f"(live partials {'on' if saved_profile.live_partials else 'off'}, "
          f"pause {config.min_silence_ms}ms)")
 

    # 允许窗口在任务栏显示 (Windows-only Tk attribute; macOS raises TclError)
    if sys.platform == "win32":
        root.wm_attributes('-toolwindow', False)

    _run_first_launch_setup(root)

    # Make sure system audio is actually going through our Multi-Output.
    # The user may have other output configurations, or macOS may have moved
    # the output when a device connected -- either way recording the far end
    # fails silently if we do not check.
    if sys.platform == "darwin":
        try:
            from src.audio import setup_macos as setup
            # A previous run that crashed or was force-quit never reached its
            # restore, leaving the machine on a device whose volume macOS
            # cannot control. Clear that first, then take the output again.
            setup.restore(progress=lambda *_: None)
            setup.ensure_active()
        except Exception as e:
            print(f"Could not select the recording output: {e}")

    _restore_on_exit.session = transcriber.session
    root.protocol("WM_DELETE_WINDOW", lambda: _restore_on_exit(root))
    _offer_recovery(root, transcriber, response_manager)

    print("READY")
    root.grid_rowconfigure(0, weight=85)  # 主内容区域占70%
    root.grid_rowconfigure(1, weight=15)  # 控制区域占30%
    root.grid_columnconfigure(0, weight=2)
    root.grid_columnconfigure(1, weight=3)

    clear_transcript_button.configure(
        command=lambda: clear_context(transcriber, mic_queue, speaker_queue, transcript_ui)
    )
    def show_popup():
        try:
            # 获取最新的话语内容
            if transcriber.structured_transcript["speaker"]:
                latest_text = transcriber.structured_transcript["speaker"][0][0]
                messagebox.showinfo(
                    "Pop Up Successful",
                    f"Last sentence: {latest_text}"
                )
            else:
                messagebox.showinfo(
                    "Pop Up Information",
                    "No sentence detected yet."
                )
        except Exception as e:
            print(f"Error in show_popup: {e}")
            messagebox.showerror(
                "Error",
                "Failed to get last sentence."
            )    
    freeze_state = [False]  # Using list to be able to change its content inside inner functions

    freeze_button.configure(command=show_popup)

    # 更新transcript UI调用
    transcript_ui.update_transcript(transcriber)
    update_response_UI(responder, response_textbox, freeze_state, transcript_ui)

    TemplateManager.initialize_default_role()

    root.mainloop()

if __name__ == "__main__":
    main()