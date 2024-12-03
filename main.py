#main.py

from datetime import datetime
import threading
from tkinter import filedialog, messagebox
import customtkinter as ctk
import queue
import time
import torch
import sys
import subprocess
import os
import glob
import json
import tkinter as tk  # 添加这一行

import src.AudioRecorder as AudioRecorder
from src.AudioTranscriber import AudioTranscriber
from src.GPTResponder import GPTResponder
from src.ResponseManager import ResponseManager
from src.SettingsManager import SettingsManager
from src.TemplateManager import TemplateManager
import src.TranscriberModels as TranscriberModels
from src.config import EnvConfig, SystemConfig, AudioConfig
from src.TranscriptUI import TranscriptUI


def validate_phrase_timeout(value):
    try:
        if value == "": return True
        val = float(value)
        return 0.01 <= val <= 50
    except ValueError:
        return False

def validate_buffer_chunks(value):
    try:
        if value == "": return True
        val = int(value)
        return 0 <= val <= 10
    except ValueError:
        return False

def create_dropdown(root, options, row, column):
    var = ctk.StringVar(value=options[0])
    menu = ctk.CTkOptionMenu(root, variable=var, values=options)
    menu.grid(row=row, column=column, padx=10, pady=3)
    return menu, var

def create_buffer_config(root, transcriber):
    buffer_frame = ctk.CTkFrame(root)
    buffer_frame.grid(row=2, column=1, padx=10, pady=3, sticky="ew")
    
    buffer_label = ctk.CTkLabel(
        buffer_frame, 
        text="Buffer Chunks (0-10):", 
        font=("Arial", 12),
        text_color="#FFFCF2"
    )
    buffer_label.pack(side="left", padx=5)
    
    def validate_chunks(value):
        try:
            if value == "": return True
            val = int(value)
            return 0 <= val <= 10
        except ValueError:
            return False
    
    validate_cmd = root.register(validate_chunks)
    
    buffer_entry = ctk.CTkEntry(
        buffer_frame,
        width=100,
        validate="key",
        validatecommand=(validate_cmd, '%P')
    )
    buffer_entry.pack(side="left", padx=5)
    buffer_entry.insert(0, str(AudioConfig.get_buffer_chunks()))
    
    def update_chunks():
        value = buffer_entry.get()
        if AudioConfig.set_buffer_chunks(value):
            buffer_label.configure(text_color="#FFFCF2")
            # 不需要特别的重新计算，因为新的chunk数量会在下次update时自动应用
        else:
            buffer_label.configure(text_color="#FF6B6B")
    
    update_button = ctk.CTkButton(
        buffer_frame,
        text="Update Chunks",
        width=100,
        command=update_chunks
    )
    update_button.pack(side="left", padx=5)
    
    return buffer_frame

def create_timeout_config(root):
    # 创建配置框架
    config_frame = ctk.CTkFrame(root)
    config_frame.grid(row=2, column=0, padx=10, pady=3, sticky="ew")
    
    # 标签
    timeout_label = ctk.CTkLabel(
        config_frame, 
        text="Phrase Timeout (0.01-50s):", 
        font=("Arial", 12),
        text_color="#FFFCF2"
    )
    timeout_label.pack(side="left", padx=5)
    
    # 验证函数
    def validate_timeout(value):
        try:
            if value == "": return True
            val = float(value)
            return 0.01 <= val <= 50
        except ValueError:
            return False
    
    validate_cmd = root.register(validate_timeout)
    
    # 输入框
    timeout_entry = ctk.CTkEntry(
        config_frame,
        width=100,
        validate="key",
        validatecommand=(validate_cmd, '%P')
    )
    timeout_entry.pack(side="left", padx=5)
    timeout_entry.insert(0, str(AudioConfig.get_phrase_timeout()))
    
    # 更新按钮
    def update_timeout():
        value = timeout_entry.get()
        if AudioConfig.set_phrase_timeout(value):
            timeout_label.configure(text_color="#FFFCF2")  # 正常颜色
        else:
            timeout_label.configure(text_color="#FF6B6B")  # 错误颜色

    update_button = ctk.CTkButton(
        config_frame,
        text="Update",
        width=80,
        command=update_timeout
    )
    update_button.pack(side="left", padx=5)
    
    # 实时更新
    def on_timeout_change(*args):
        value = timeout_entry.get()
        if validate_timeout(value):
            AudioConfig.set_phrase_timeout(value)
            timeout_label.configure(text_color="#FFFCF2")
        else:
            timeout_label.configure(text_color="#FF6B6B")
    
    timeout_entry.bind('<KeyRelease>', on_timeout_change)
    
    return config_frame

def write_in_textbox(textbox, text):
    textbox.delete("0.0", "end")
    textbox.insert("0.0", text)

def update_response_UI(responder, textbox, update_interval_slider_label, update_interval_slider, freeze_state, transcript_ui):
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

        # 更新响应间隔
        update_interval = int(update_interval_slider.get())
        responder.update_response_interval(update_interval)
        update_interval_slider_label.configure(text=f"Update interval: {update_interval} seconds")

    # 定时调用以保持UI更新
    textbox.after(300, update_response_UI, responder, textbox, update_interval_slider_label, 
                 update_interval_slider, freeze_state, transcript_ui)

    
def clear_context_(transcriber, audio_queue):
    transcriber.clear_transcript_data()
    with audio_queue.mutex:
        audio_queue.queue.clear()

def clear_context(transcriber, audio_queue, transcript_ui):
    """
    清除所有上下文
    """
    print("Clearing context...")
    # 清除transcriber数据
    transcriber.clear_transcript_data()
    # 清除音频队列
    with audio_queue.mutex:
        audio_queue.queue.clear()
    # 清除UI显示
    transcript_ui.clear()
    print("Context cleared")

def create_ui_components(root, response_manager, transcriber, audio_queue):
    """创建并配置所有UI组件"""
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
        var.trace('w', on_selection_change)

    # === Column 2: Action Buttons ===
    def export_responses():
        """处理导出对话记录的函数"""
        try:
            conversation_data = response_manager.export_structured_conversation(
                transcriber.structured_transcript,
                reverse_chronological=False
            )
            
            if not conversation_data or not conversation_data["conversation"]["messages"]:
                messagebox.showwarning(
                    "Export Notice",
                    "No conversation data available for export."
                )
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            filepath = filedialog.asksaveasfilename(
                defaultextension=".json",
                initialfile=f"conversation_export_{timestamp}.json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Export Conversation Data"
            )
            
            if filepath:
                success = response_manager.save_structured_conversation(
                    filepath, 
                    conversation_data
                )
                
                if success:
                    total_messages = len(conversation_data["conversation"]["messages"])
                    messages_with_responses = sum(
                        1 for msg in conversation_data["conversation"]["messages"] 
                        if "response" in msg
                    )
                    
                    messagebox.showinfo(
                        "Export Successful", 
                        f"Conversation data has been saved to:\n{filepath}\n\n"
                        f"Total messages: {total_messages}\n"
                        f"Messages with responses: {messages_with_responses}"
                    )
                else:
                    messagebox.showerror(
                        "Export Failed", 
                        f"Error occurred while saving the file.\n"
                        f"Please check file permissions and disk space.\n"
                        f"Target path: {filepath}"
                    )
        except Exception as e:
            messagebox.showerror(
                "Export Error", 
                f"An error occurred during export:\n{str(e)}\n\n"
                "Please contact technical support."
            )
            print(f"Export error: {e}")
            import traceback
            traceback.print_exc()

    # === Column 2: Action Buttons ===
    buttons_data = [
        ("Clear Transcript", lambda: clear_context(transcriber, audio_queue, transcript_ui), "#1f538d"),
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

    # === Column 3: Configuration Controls ===
    # Phrase Timeout
    phrase_time_label = ctk.CTkLabel(
        main_control_frame,
        text="Phrase Timeout:",
        font=("Arial", 12)
    )
    phrase_time_label.grid(row=0, column=2, padx=5, pady=2, sticky="w")
    
    validate_cmd = root.register(validate_phrase_timeout)
    phrase_time_entry = ctk.CTkEntry(
        main_control_frame,
        width=70,
        placeholder_text="0.01-50s",
        validate="key",
        validatecommand=(validate_cmd, '%P')
    )
    phrase_time_entry.grid(row=0, column=2, padx=(100, 5), pady=2, sticky="w")
    phrase_time_entry.insert(0, str(settings_manager.get_setting("phrase_timeout")))
    
    def update_settings():
        phrase_timeout = phrase_time_entry.get()
        if AudioConfig.set_phrase_timeout(phrase_timeout):
            settings_manager.update_setting("phrase_timeout", float(phrase_timeout))
            phrase_time_label.configure(text_color="#FFFCF2")
        else:
            phrase_time_label.configure(text_color="#FF6B6B")
            
    update_button = ctk.CTkButton(
        main_control_frame,
        text="Update",
        width=60,
        command=update_settings
    )
    update_button.grid(row=0, column=2, padx=(180, 5), pady=2, sticky="w")

    # Buffer Chunks
    buffer_label = ctk.CTkLabel(
        main_control_frame, 
        text="Buffer Chunks:",
        font=("Arial", 12)
    )
    buffer_label.grid(row=1, column=2, padx=5, pady=2, sticky="w")
    
    buffer_options = [str(i) for i in range(11)]
    saved_buffer = str(settings_manager.get_setting("buffer_chunks"))
    buffer_var = ctk.StringVar(value=saved_buffer)
    
    def on_buffer_change(value):
        settings_manager.update_setting("buffer_chunks", int(value))
        AudioConfig.set_buffer_chunks(value)
        buffer_label.configure(text_color="#639cdc")
        root.after(500, lambda: buffer_label.configure(text_color="#FFFCF2"))
    
    buffer_dropdown = ctk.CTkOptionMenu(
        main_control_frame,
        variable=buffer_var,
        values=buffer_options,
        width=70,
        command=on_buffer_change
    )
    buffer_dropdown.grid(row=1, column=2, padx=(100, 5), pady=2, sticky="w")

    # Update Interval
    interval_label = ctk.CTkLabel(
        main_control_frame, 
        text="Update Interval:",
        font=("Arial", 12)
    )
    interval_label.grid(row=2, column=2, padx=5, pady=2, sticky="w")
    
    saved_interval = str(int(settings_manager.get_setting("update_interval")))
    interval_values = [str(i) for i in range(1, 11)]
    
    def on_interval_change(value):
        settings_manager.update_setting("update_interval", float(value))
    
    interval_dropdown = ctk.CTkOptionMenu(
        main_control_frame,
        values=interval_values,
        width=70,
        command=on_interval_change
    )
    interval_dropdown.grid(row=2, column=2, padx=(100, 5), pady=2, sticky="w")
    interval_dropdown.set(saved_interval)

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

    # 在这里加入return语句
    return (
        transcript_ui,
        response_textbox,
        interval_dropdown,
        interval_label,
        freeze_button,
        clear_transcript_button,
        phrase_time_entry,
        buffer_dropdown,
        update_button,
        export_button  # 添加这个
    )

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
    audio_queue = queue.Queue()

    user_audio_recorder = AudioRecorder.DefaultMicRecorder()
    user_audio_recorder.record_into_queue(audio_queue)

    time.sleep(2)

    speaker_audio_recorder = AudioRecorder.DefaultSpeakerRecorder()
    speaker_audio_recorder.record_into_queue(audio_queue)

    model = TranscriberModels.get_model('--api' in sys.argv)

    # 创建ResponseManager实例
    response_manager = ResponseManager()

    transcriber = AudioTranscriber(user_audio_recorder.source, speaker_audio_recorder.source, model,response_manager)
    transcribe = threading.Thread(target=transcriber.transcribe_audio_queue, args=(audio_queue,))
    transcribe.daemon = True
    transcribe.start()

    responder = GPTResponder(response_manager)
    respond = threading.Thread(target=responder.respond_to_transcriber, args=(transcriber,))
    respond.daemon = True
    respond.start()

    #monitor = threading.Thread(target=transcriber.self_check)
    #monitor.daemon = True
    #monitor.start()

    root = ctk.CTk()
    (
        transcript_ui, 
        response_textbox, 
        update_interval_slider, 
        update_interval_slider_label, 
        freeze_button,
        clear_transcript_button,
        phrase_time_entry,
        buffer_dropdown,
        update_button,
        export_button
    ) = create_ui_components(root, response_manager, transcriber, audio_queue)


    # 创建设置管理器实例
    settings_manager = SettingsManager()
    
    # 加载窗口设置
    saved_opacity = settings_manager.get_setting("window_opacity")
    saved_topmost = settings_manager.get_setting("window_topmost")
    
    root.attributes('-alpha', saved_opacity)  # 设置透明度
    root.attributes('-topmost', saved_topmost)  # 设置置顶状态   
    
    SystemConfig.set_record_only_mode(settings_manager.get_setting("record_only_mode"))
 

    # 允许窗口在任务栏显示
    root.wm_attributes('-toolwindow', False)

    print("READY")
    root.grid_rowconfigure(0, weight=85)  # 主内容区域占70%
    root.grid_rowconfigure(1, weight=15)  # 控制区域占30%
    root.grid_columnconfigure(0, weight=2)
    root.grid_columnconfigure(1, weight=3)

    clear_transcript_button.configure(
        command=lambda: clear_context(transcriber, audio_queue, transcript_ui)
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

    update_interval_slider_label.configure(text=f"Update interval: {update_interval_slider.get()} seconds")

    # 更新transcript UI调用
    transcript_ui.update_transcript(transcriber)
    update_response_UI(responder, response_textbox, update_interval_slider_label, 
                      update_interval_slider, freeze_state,transcript_ui)

    TemplateManager.initialize_default_role()

    root.mainloop()

if __name__ == "__main__":
    main()