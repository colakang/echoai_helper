#main.py

from datetime import datetime
import threading
from tkinter import filedialog, messagebox
from AudioTranscriber import AudioTranscriber
from GPTResponder import GPTResponder
import customtkinter as ctk
import AudioRecorder 
import queue
import time
import torch
import sys
from ResponseManager import ResponseManager
from SettingsManager import SettingsManager
from TemplateManager import TemplateManager
import TranscriberModels
import subprocess
import os
import glob
from config import EnvConfig, SystemConfig, AudioConfig
from TranscriptUI import TranscriptUI
import json
import tkinter as tk  # 添加这一行




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

def create_ui_components(root, response_manager,transcriber):
    """创建并配置所有UI组件"""
    # 基础设置
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")
    root.title("EchoAI 365 (Helper Mode)")
    root.configure(bg='#252422')
    root.geometry("1000x600")

    # 创建设置管理器
    settings_manager = SettingsManager()

    font_size = 20
    
    # 主要内容区域
    transcript_textbox = ctk.CTkTextbox(
        root, 
        width=250, 
        font=("Arial", font_size), 
        text_color='#FFFCF2', 
        wrap="word",
        state="normal"  # 确保可以选择文本

    )
    transcript_textbox.grid(row=0, column=0, padx=10, pady=20, sticky="nsew")

    response_textbox = ctk.CTkTextbox(
        root, 
        width=400, 
        font=("Arial", font_size), 
        text_color='#639cdc', 
        wrap="word",
        state="normal"  # 确保可以选择文本

    )
    response_textbox.grid(row=0, column=1, padx=10, pady=20, sticky="nsew")

    # 控制区域框架
    control_frame = ctk.CTkFrame(root)
    control_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=3)

    # Clear Transcript按钮
    clear_transcript_button = ctk.CTkButton(
        control_frame, 
        text="Clear Transcript", 
        command=None, 
        width=120
    )
    clear_transcript_button.pack(side="left", padx=(10, 20))

    # Phrase Timeout区域
    phrase_time_frame = ctk.CTkFrame(control_frame)
    phrase_time_frame.pack(side="left", padx=5)

    phrase_time_label = ctk.CTkLabel(
        phrase_time_frame, 
        text="Phrase Timeout:", 
        font=("Arial", 12),
        text_color="#FFFCF2"
    )
    phrase_time_label.pack(side="left", padx=2)

    validate_timeout = root.register(validate_phrase_timeout)
    phrase_time_entry = ctk.CTkEntry(
        phrase_time_frame,
        width=70,
        placeholder_text="0.01-50s",
        validate="key",
        validatecommand=(validate_timeout, '%P')
    )
    phrase_time_entry.pack(side="left", padx=2)
    
    # 使用保存的设置或默认值
    saved_timeout = settings_manager.get_setting("phrase_timeout")
    phrase_time_entry.insert(0, str(saved_timeout))
    AudioConfig.set_phrase_timeout(saved_timeout)

    # Update Timeout按钮（移到phrase_time_frame中）
    update_button = ctk.CTkButton(
        phrase_time_frame, 
        text="Update", 
        width=80
    )
    update_button.pack(side="left", padx=5)

    # Buffer Chunks区域
    buffer_chunks_frame = ctk.CTkFrame(control_frame)
    buffer_chunks_frame.pack(side="left", padx=20)
    
    buffer_label = ctk.CTkLabel(
        buffer_chunks_frame, 
        text="Buffer Chunks:", 
        font=("Arial", 12),
        text_color="#FFFCF2"
    )
    buffer_label.pack(side="left", padx=2)
    
    buffer_options = [str(i) for i in range(11)]
    saved_buffer = str(settings_manager.get_setting("buffer_chunks"))
    buffer_var = ctk.StringVar(value=saved_buffer)
    
    def on_buffer_change(*args):
        value = buffer_var.get()
        settings_manager.update_setting("buffer_chunks", int(value))
        AudioConfig.set_buffer_chunks(value)
        buffer_label.configure(text_color="#639cdc")
        root.after(500, lambda: buffer_label.configure(text_color="#FFFCF2"))
    
    buffer_dropdown = ctk.CTkOptionMenu(
        buffer_chunks_frame,
        variable=buffer_var,
        values=buffer_options,
        width=70,
        command=lambda _: on_buffer_change()
    )
    buffer_dropdown.pack(side="left", padx=2)

    # Freeze按钮
    freeze_button = ctk.CTkButton(
        control_frame, 
        text="Pop Up", 
        command=None,  # 稍后设置
        width=100,
        hover_color="#2B7A0B",  # 添加鼠标悬停效果
        fg_color="#1B4332"  # 使用不同的颜色以区分其他按钮
    )
    freeze_button.pack(side="right", padx=10)

    # Update interval区域
    update_interval_slider_label = ctk.CTkLabel(
        root, 
        text="", 
        font=("Arial", 12), 
        text_color="#FFFCF2"
    )
    update_interval_slider_label.grid(row=2, column=1, padx=10, pady=3, sticky="nsew")

    saved_interval = settings_manager.get_setting("update_interval")
    update_interval_slider = ctk.CTkSlider(
        root, 
        from_=1, 
        to=10, 
        width=300, 
        height=20, 
        number_of_steps=9
    )
    update_interval_slider.set(saved_interval)
    update_interval_slider.grid(row=3, column=1, padx=10, pady=10, sticky="nsew")

    # 模板选择区域
    system_role_files = TemplateManager.get_template_files('system_role')
    case_detail_files = TemplateManager.get_template_files('case_detail')
    knowledge_files = TemplateManager.get_template_files('knowledge')
    
    if not all([system_role_files, case_detail_files, knowledge_files]):
        print("Warning: Some template directories are empty")

    template_frame = ctk.CTkFrame(root)
    template_frame.grid(row=4, column=0, columnspan=2, sticky="ew", padx=10, pady=10)

    # 创建带标签的模板选择下拉菜单
    templates = {
        "System Role": (system_role_files, "system_role"),
        "Case Detail": (case_detail_files, "case_detail"),
        "Knowledge Base": (knowledge_files, "knowledge")
    }

    template_vars = {}
    for i, (label, (options, setting_key)) in enumerate(templates.items()):
        frame = ctk.CTkFrame(template_frame)
        frame.grid(row=0, column=i, padx=10, pady=3)

        label_widget = ctk.CTkLabel(
            frame,
            text=label,
            font=("Arial", 12),
            text_color="#FFFCF2"
        )
        label_widget.pack(pady=2)

        saved_value = settings_manager.get_setting(setting_key)
        var = ctk.StringVar(value=saved_value if saved_value in (options or ['default']) else (options or ['default'])[0])
        menu = ctk.CTkOptionMenu(
            frame,
            variable=var,
            values=options or ['default'],
            width=160
        )
        menu.pack(pady=2)
        template_vars[setting_key] = var

    def on_selection_change(*args):
        """处理模板选择变化"""
        try:
            # 保存选择
            for key, var in template_vars.items():
                settings_manager.update_setting(key, var.get())
            
            # 更新系统角色
            new_role = TemplateManager.update_system_role(
                template_vars["system_role"].get(),
                template_vars["case_detail"].get(),
                template_vars["knowledge"].get()
            )
            if new_role is None:
                print("Warning: Failed to update system role")
        except Exception as e:
            print(f"Error updating system role: {e}")

    # 绑定变化事件
    for var in template_vars.values():
        var.trace('w', on_selection_change)

    # 更新按钮的回调函数
    def update_settings():
        phrase_timeout = phrase_time_entry.get()
        if AudioConfig.set_phrase_timeout(phrase_timeout):
            settings_manager.update_setting("phrase_timeout", float(phrase_timeout))
            phrase_time_label.configure(text_color="#FFFCF2")
        else:
            phrase_time_label.configure(text_color="#FF6B6B")

    update_button.configure(command=update_settings)

    # 配置网格权重
    root.grid_rowconfigure(0, weight=100)
    root.grid_rowconfigure(1, weight=1)
    root.grid_rowconfigure(2, weight=1)
    root.grid_rowconfigure(3, weight=1)
    root.grid_rowconfigure(4, weight=1)
    root.grid_columnconfigure(0, weight=2)
    root.grid_columnconfigure(1, weight=3)

    # 创建TranscriptUI实例
    transcript_ui = TranscriptUI(transcript_textbox, response_manager)
    transcript_ui.add_click_handler(response_textbox)

    # 保存update_interval的回调
    def on_interval_change(value):
        settings_manager.update_setting("update_interval", float(value))
        update_interval_slider_label.configure(
            text=f"Update interval: {int(float(value))} seconds"
        )

    update_interval_slider.configure(command=on_interval_change)
    # 添加导出按钮
    def export_responses():
        """处理导出对话记录的函数"""
        try:
            # 检查是否有对话数据可导出
            #transcript_data = transcriber.get_transcript()
            conversation_data = response_manager.export_structured_conversation(
                transcriber.structured_transcript,
                #transcript_data,
                reverse_chronological=False
            )
            
            if not conversation_data or not conversation_data["conversation"]["messages"]:
                messagebox.showwarning(
                    "Export Notice",
                    "No conversation data available for export."
                )
                return

            # 获取当前时间戳用于文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 使用文件对话框获取保存位置
            filepath = filedialog.asksaveasfilename(
                defaultextension=".json",
                initialfile=f"conversation_export_{timestamp}.json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Export Conversation Data"
            )
            
            if filepath:  # 如果用户没有取消对话框
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


    # 在control_frame中添加导出按钮，使用新的样式
    export_button = ctk.CTkButton(
        control_frame, 
        text="Export Conversation", 
        command=export_responses,
        width=120,
        hover_color="#2B7A0B",  # 添加鼠标悬停效果
        fg_color="#1B4332"  # 使用不同的颜色以区分其他按钮
    )
    export_button.pack(side="left", padx=(10, 20))

    # 返回需要的组件
    return (
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
        buffer_dropdown,  # 接收dropdown
        update_button,
        export_button
    ) = create_ui_components(root,response_manager,transcriber)


    print("READY")

    root.grid_rowconfigure(0, weight=100)
    root.grid_rowconfigure(1, weight=1)
    root.grid_rowconfigure(2, weight=1)
    root.grid_rowconfigure(3, weight=1)
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