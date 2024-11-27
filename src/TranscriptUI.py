# src/TranscriptUI.py

import customtkinter as ctk
from typing import Optional, Dict, List, Any
import traceback

class TranscriptUI:
    """处理对话记录的UI显示和交互"""
    
    def __init__(self, textbox: ctk.CTkTextbox, response_manager: Any):
        """
        初始化TranscriptUI
            
        Args:
            textbox: 用于显示对话记录的CTkTextbox
            response_manager: 管理对话响应的ResponseManager实例
        """
        self.textbox = textbox
        self.text_widget = textbox._textbox
        self.response_manager = response_manager
        self.last_speaker_count = 0
        self.last_you_count = 0
        self.debug_mode = False
        self.is_response_locked = False
        self.response_textbox = None
        self.selected_response_id = None
        self.last_response_id = None
        self.current_streaming_id = None  # 添加当前正在流式更新的ID
        self.latest_speaker_response = None  # 新增：跟踪最新的Speaker响应
        self.last_speaker_content = {}  # 添加用于跟踪每个记录最新内容的字典
        self.last_you_content = {}
        self._initialize_default_lines()

        # 配置文本框
        self._configure_textbox()
        
        # 注册响应更新回调
        if hasattr(response_manager, 'register_update_callback'):
            response_manager.register_update_callback(self._on_response_update)
        
        if self.debug_mode:
            print("TranscriptUI initialized")
    def _initialize_default_lines(self) -> None:
        """初始化默认的Speaker和You行"""
        try:
            self.text_widget.configure(state="normal")
            
            # 插入Speaker行
            speaker_text = "Speaker: [Ready...]\n\n"
            self.text_widget.insert("1.0", speaker_text)
            
            # 插入You行
            you_text = "You: [Ready...]\n\n"
            self.text_widget.insert("1.0", you_text)
            
            # 添加默认行标签
            self.text_widget.tag_add("default_line", "1.0", "3.0")
            self.text_widget.tag_configure("default_line", foreground='#666666')
            
            self.text_widget.configure(state="normal")
            
        except Exception as e:
            print(f"Error initializing default lines: {e}")
            traceback.print_exc()

    def _configure_textbox(self) -> None:
        """配置文本框的基本设置"""
        self.textbox.configure(cursor="hand2")
        self.text_widget.configure(state="normal")  # 确保文本可以选择
        
    def toggle_debug(self, enabled: bool = None) -> None:
        """
        切换调试模式
        
        Args:
            enabled: 如果提供，直接设置调试模式状态；如果未提供，切换当前状态
        """
        if enabled is None:
            self.debug_mode = not self.debug_mode
        else:
            self.debug_mode = enabled
       
    def update_transcript(self, transcriber: Any) -> None:
        """
        更新对话记录显示（流式更新）
        
        Args:
            transcriber: AudioTranscriber实例，包含对话记录数据
        """
        try:
            current_speaker_count = len(transcriber.structured_transcript['speaker'])
            current_you_count = len(transcriber.structured_transcript['you'])
            
            # 只在有变化时输出调试信息
            if (current_speaker_count != self.last_speaker_count or 
                current_you_count != self.last_you_count):
                if self.debug_mode:
                    print("\nDebug TranscriptUI update:")
                    print(f"Current speaker records: {current_speaker_count}")
                    print(f"Current 'you' records: {current_you_count}")
                    print(f"Last speaker count: {self.last_speaker_count}")
                    print(f"Last you count: {self.last_you_count}")
            
            # 获取新记录
            new_records = self._get_new_records(transcriber)
            
            if new_records and self.debug_mode:
                print(f"New records found: {len(new_records)}")
                print("New records content:")
                for record in new_records:
                    print(f"- {record['type']}: {record['text']}")
            
            # 如果有新记录，追加到显示
            if new_records:
                # 保存当前的选择状态和滚动位置
                try:
                    selection_start = self.text_widget.index("sel.first")
                    selection_end = self.text_widget.index("sel.last")
                    has_selection = True
                except:
                    has_selection = False
                
                current_pos = self.textbox.yview()[1]  # 使用底部位置
                was_at_bottom = current_pos >= 0.9  # 如果接近底部就认为是在底部
                
                # 临时启用文本框，以便更新新内容
                self.text_widget.configure(state="normal")
                
                # 添加新记录
                for record in new_records:
                    record_type = record["type"]
                    # 查找第一个匹配的记录
                    line_start = self.text_widget.search(
                        f"{record_type}:", 
                        "1.0", 
                        stopindex="end"
                    )
                    
                    if line_start:
                        # 找到下一条记录的开始位置或文本结束
                        next_record_start = self.text_widget.search(
                            f"(Speaker:|You:)", 
                            f"{line_start} + 1c", 
                            stopindex="end",
                            regexp=True
                        )
                        # 确定删除范围
                        delete_end = next_record_start if next_record_start else "end"

                    if record["is_update"]:
                        # 更新最新记录的内容
                            existing_tags = self.text_widget.tag_names(line_start)
                            response_id = None
                            for tag in existing_tags:
                                if tag.startswith("response_"):
                                    response_id = tag.replace("response_", "")
                                    break                            
                            # 删除当前记录的全部内容（包括空行）
                            self.text_widget.delete(line_start, delete_end)
                            
                            # 插入更新后的内容
                            text = f"{record_type}: [{record['text']}]\n\n"
                            self.text_widget.insert(line_start, text)
                            if response_id:
                                record['response_id'] = response_id
                                self._add_record_tags(line_start, record)
                    else:
                        # 删除当前记录的全部内容（包括空行）
                        self.text_widget.delete(line_start, delete_end)
                        # 计算插入位置
                        insert_position = "1.0"
                        text = f"{record['type']}: [{record['text']}]\n\n"
                        self.text_widget.insert(insert_position, text)
                        
                        # 如果记录有response_id，添加tag和交互效果
                        if record['response_id']:
                            self._add_record_tags(insert_position, record)
                            # 如果是Speaker记录，更新最新响应
                            if record['type'] == 'Speaker':
                                self.latest_speaker_response = record['response_id']
                                response = self.response_manager.get_response(record['response_id'])
                                if response and response.response_text and not self.is_response_locked:
                                    self._update_response_text(response.response_text,response.question_text)
                        # 重复插入，用于增量更新记录
                        insert_position = "1.0"
                        text = f"{record['type']}: [Catching...]\n\n"
                        self.text_widget.insert(insert_position, text)                                                   
                self.text_widget.see("1.0")

                # 恢复选择状态
                if has_selection:
                    try:
                        self.text_widget.tag_add("sel", selection_start, selection_end)
                    except:
                        pass

                # 更新完成后设置为可交互状态
                self.text_widget.configure(state="normal")
        
        except Exception as e:
            print(f"Error in update_transcript: {str(e)}")
            traceback.print_exc()
            
        finally:
            # 设置下一次更新，每隔300ms检查一次是否有新内容
            self.textbox.after(300, self.update_transcript, transcriber)

    def _get_new_records(self, transcriber: Any) -> List[Dict]:
        """
        获取新的对话记录，包括新增记录和现有记录的更新
        
        Args:
            transcriber: AudioTranscriber实例
            
        Returns:
            List[Dict]: 新记录和更新的记录列表
        """
        new_records = []
        
        try:
            current_speaker_records = transcriber.structured_transcript["speaker"]
            current_you_records = transcriber.structured_transcript["you"]
            
            # 处理speaker记录
            if current_speaker_records:
                # 检查是否有新增记录
                if len(current_speaker_records) > self.last_speaker_count:
                    # 有新记录时，处理新记录
                    new_count = len(current_speaker_records) - self.last_speaker_count
                    for record in reversed(current_speaker_records[:new_count]):
                        text, timestamp, response_id = record
                        new_records.append({
                            "type": "Speaker",
                            "text": text,
                            "timestamp": timestamp,
                            "response_id": response_id,
                            "is_update": False
                        })
                        self.current_speaker_text = text
                    self.last_speaker_count = len(current_speaker_records)
                else:
                    # 没有新记录时，检查最新记录是否有更新
                    latest_record = current_speaker_records[0]
                    text, timestamp, response_id = latest_record
                    if text != self.current_speaker_text:
                        new_records.append({
                            "type": "Speaker",
                            "text": text,
                            "timestamp": timestamp,
                            "response_id": response_id,
                            "is_update": True
                        })
                        self.current_speaker_text = text
            
            # 处理you记录（类似逻辑）
            if current_you_records:
                if len(current_you_records) > self.last_you_count:
                    new_count = len(current_you_records) - self.last_you_count
                    for record in reversed(current_you_records[:new_count]):
                        text, timestamp, response_id = record
                        new_records.append({
                            "type": "You",
                            "text": text,
                            "timestamp": timestamp,
                            "response_id": response_id,
                            "is_update": False
                        })
                        self.current_you_text = text
                    self.last_you_count = len(current_you_records)
                else:
                    latest_record = current_you_records[0]
                    text, timestamp, response_id = latest_record
                    if text != self.current_you_text:
                        new_records.append({
                            "type": "You",
                            "text": text,
                            "timestamp": timestamp,
                            "response_id": response_id,
                            "is_update": True
                        })
                        self.current_you_text = text
            
        except Exception as e:
            print(f"Error in _get_new_records: {str(e)}")
            traceback.print_exc()
            
        # 按时间戳排序，最新的在前
        new_records.sort(key=lambda x: x["timestamp"], reverse=True)
        return new_records
      
    def _append_new_records(self, records: List[Dict]) -> None:
        """追加新记录到文本框"""
        try:
            self.text_widget.configure(state="normal")
            insert_position = "1.0"
            
            latest_speaker_record = None
            
            for record in records:
                text = f"{record['type']}: [{record['text']}]\n\n"
                self.text_widget.insert(insert_position, text)
                
                if record['response_id']:
                    self._add_record_tags(insert_position, record)
                    # 跟踪最新的Speaker记录
                    if record['type'] == 'Speaker':
                        latest_speaker_record = record
            
            # 在所有记录添加完成后，更新最新响应
            if latest_speaker_record and not self.is_response_locked:
                self.latest_speaker_response = latest_speaker_record['response_id']
                response = self.response_manager.get_response(latest_speaker_record['response_id'])
                if response and response.response_text:
                    self.update_latest_response(latest_speaker_record['response_id'], response.response_text)
                    
            self.text_widget.configure(state="normal")
            
        except Exception as e:
            print(f"Error in _append_new_records: {e}")
            traceback.print_exc()

    def _add_record_tags(self, position: str, record: Dict) -> None:
        """
        为记录添加tag和交互效果
        """
        try:
            # 修改：精确计算行结束位置
            line_end = self.text_widget.index(f"{position} lineend")
            tag_name = f"response_{record['response_id']}"
            
            # 修改：确保tag覆盖整行文本，包括换行符
            self.text_widget.tag_add(tag_name, position, f"{line_end}+1c")
            
            # 其余标签和交互效果设置保持不变
            hover_bg = '#2f3746' if record['type'] == 'Speaker' else '#1f2736'
            self.text_widget.tag_configure(tag_name, background='')
            
            def on_enter(e, tag=tag_name, bg=hover_bg):
                try:
                    self.text_widget.tag_configure(tag, background=bg)
                except Exception as e:
                    print(f"Error in on_enter: {e}")
                    
            def on_leave(e, tag=tag_name):
                try:
                    self.text_widget.tag_configure(tag, background='')
                except Exception as e:
                    print(f"Error in on_leave: {e}")
            
            self.text_widget.tag_bind(tag_name, '<Enter>', on_enter)
            self.text_widget.tag_bind(tag_name, '<Leave>', on_leave)
            
            if record['type'] == 'You':
                self.text_widget.tag_configure(tag_name, foreground='#A0A0A0')
                
        except Exception as e:
            print(f"Error in _add_record_tags: {str(e)}")
            traceback.print_exc()

    def _add_record_tags_(self, position: str, record: Dict) -> None:
        """
        为记录添加tag和交互效果
        
        Args:
            position: 插入位置
            record: 记录数据
        """
        try:
            # 获取插入的文本行的结束位置
            line_end = self.text_widget.index(f"{position} lineend")
            tag_name = f"response_{record['response_id']}"
            
            # 添加tag到整行文本
            self.text_widget.tag_add(tag_name, position, f"{line_end}+1c")
            
            # 设置tag样式
            hover_bg = '#2f3746' if record['type'] == 'Speaker' else '#1f2736'
            self.text_widget.tag_configure(tag_name, background='')
            
            # 添加鼠标悬停效果
            def on_enter(e, tag=tag_name, bg=hover_bg):
                try:
                    self.text_widget.tag_configure(tag, background=bg)
                except Exception as e:
                    print(f"Error in on_enter: {e}")
                    
            def on_leave(e, tag=tag_name):
                try:
                    self.text_widget.tag_configure(tag, background='')
                except Exception as e:
                    print(f"Error in on_leave: {e}")
            
            # 绑定鼠标事件
            self.text_widget.tag_bind(tag_name, '<Enter>', on_enter)
            self.text_widget.tag_bind(tag_name, '<Leave>', on_leave)
            
            # 为You类型设置特殊颜色
            if record['type'] == 'You':
                self.text_widget.tag_configure(tag_name, foreground='#A0A0A0')
            
            if self.debug_mode:
                print(f"Added tag {tag_name} to text at position {position}")
                
        except Exception as e:
            print(f"Error in _add_record_tags: {str(e)}")
            traceback.print_exc()

    def _update_response_text_(self, response_text: str) -> None:
        """更新响应文本框内容"""
        try:
            if not self.response_textbox:
                print("Warning: response_textbox is not initialized.")
                return

            current_text = self.response_textbox.get("1.0", "end-1c")
            if response_text != current_text:  # 只在内容变化时更新
                self.response_textbox.configure(state="normal")
                self.response_textbox.delete("1.0", "end")
                self.response_textbox.insert("1.0", response_text)
                self.response_textbox.configure(state="normal")
        except Exception as e:
            print(f"Error updating response text: {e}")
            traceback.print_exc()

    def _update_response_text(self, response_text: str, question_text: str = None) -> None:
        """
        更新响应文本框内容，包括问题和答案
        
        Args:
            response_text: 响应文本
            question_text: 关联的问题文本
        """
        try:
            if not self.response_textbox:
                print("Warning: response_textbox is not initialized.")
                return

            # 格式化显示内容
            display_text = self._format_response_display(question_text, response_text)
            #print (f'display text: {display_text}')
            current_text = self.response_textbox.get("1.0", "end-1c")
            if display_text != current_text:  # 只在内容变化时更新
                self.response_textbox.configure(state="normal")
                self.response_textbox.delete("1.0", "end")
                self.response_textbox.insert("1.0", display_text)
                self.response_textbox.configure(state="normal")
        except Exception as e:
            print(f"Error updating response text: {e}")
            traceback.print_exc()

    def _format_response_display(self, question_text: str, response_text: str) -> str:
        """
        格式化问题和响应的显示内容
        
        Args:
            question_text: 问题文本
            response_text: 响应文本
            
        Returns:
            str: 格式化后的显示文本
        """
        if question_text:
            return f"Q: {question_text}\n\n---\n\nA: {response_text}"
        return response_text
    
    def clear(self) -> None:
        """清除所有内容和计数器"""
        try:
            self.text_widget.configure(state="normal")
            self.text_widget.delete("1.0", "end")
            self.text_widget.configure(state="normal")  # 保持可选择状态
            self.last_speaker_count = 0
            self.last_you_count = 0
            self._initialize_default_lines()
            if self.debug_mode:
                print("TranscriptUI cleared")
                
        except Exception as e:
            print(f"Error in clear: {str(e)}")
            traceback.print_exc()

    def _on_response_update(self, response_id: str, response_text: str, is_complete: bool) -> None:
        """
        响应更新回调函数
        
        Args:
            response_id: 响应ID
            response_text: 更新的响应文本
            is_complete: 响应是否完成
        """
        try:
            # 强制更新最新的响应，即使在锁定状态下
            if self.is_response_locked and response_id != self.selected_response_id:
                return

            if not self.response_textbox:
                print("Warning: response_textbox is not initialized.")
                return
            # 获取关联的response对象以获取问题文本
            response = self.response_manager.get_response(response_id)
            question_text = response.question_text if response else None
            self.response_textbox.configure(state="normal")
            
            if not is_complete:
                # 如果响应尚未完成，追加更新文本而不是替换整个内容
                current_text = self.response_textbox.get("1.0", "end-1c")
                updated_text = current_text + response_text
                self.response_textbox.delete("1.0", "end")
                self.response_textbox.insert("1.0", updated_text)
            else:
                # 如果响应完成，则替换整个内容
                display_text = self._format_response_display(question_text, response_text)
                self.response_textbox.delete("1.0", "end")
                self.response_textbox.insert("1.0", display_text)
                #self.response_textbox.insert("1.0", response_text)
            
            self.response_textbox.configure(state="normal")  # 保持可选择状态

            # 更新最新的响应ID
            self.last_response_id = response_id
            
            if self.debug_mode:
                print(f"Response updated: {response_id}, complete: {is_complete}")
        except Exception as e:
            print(f"Error in _on_response_update: {e}")
            traceback.print_exc()


    def update_latest_response(self, response_id: str, response_text: str, question_text: str = None) -> None:
        """强制更新最新的响应文本，无论锁定状态"""
        try:
            if not self.response_textbox:
                print("Warning: response_textbox is not initialized.")
                return
            display_text = self._format_response_display(question_text, response_text)
            self.response_textbox.configure(state="normal")
            self.response_textbox.delete("1.0", "end")
            #self.response_textbox.insert("1.0", response_text)
            self.response_textbox.insert("1.0", display_text)
            self.response_textbox.configure(state="normal")  # 保持可选择状态

            if self.debug_mode:
                print(f"Latest response forcibly updated: {response_id}")
        except Exception as e:
            print(f"Error in update_latest_response: {e}")
            traceback.print_exc()

    def add_click_handler(self, response_textbox: ctk.CTkTextbox) -> None:
        """添加点击事件处理"""
        self.response_textbox = response_textbox

        def on_click(event):
            try:
                tags = self.text_widget.tag_names(f"@{event.x},{event.y}")
                for tag in tags:
                    if tag.startswith("response_"):
                        response_id = tag.replace("response_", "")
                        response = self.response_manager.get_response(response_id)
                        if response and response.response_text:
                            if self.selected_response_id == response_id:
                                # 解锁并恢复自动更新
                                self.is_response_locked = False
                                self.selected_response_id = None
                                # 立即更新到最新响应
                                if self.last_response_id:
                                    latest_response = self.response_manager.get_response(self.last_response_id)
                                    if latest_response and latest_response.response_text:
                                        self.update_latest_response(self.last_response_id, latest_response.response_text,latest_response.question_text)
                            else:
                                # 锁定并显示选中的response
                                self.is_response_locked = True
                                self.selected_response_id = response_id
                                display_text = self._format_response_display(response.question_text, response.response_text)
                                self.response_textbox.configure(state="normal")
                                self.response_textbox.delete("1.0", "end")
                                #self.response_textbox.insert("1.0", response.response_text)
                                self.response_textbox.insert("1.0", display_text)
                                self.response_textbox.configure(state="normal")
                            break
            except Exception as e:
                print(f"Error in click handler: {e}")
                traceback.print_exc()

        self.text_widget.bind('<Button-1>', on_click)
        
        def unlock_response(event):
            # 检查点击是否在transcript或response textbox之外
            if (str(event.widget) != str(self.text_widget) and 
                str(event.widget) != str(self.response_textbox._textbox)):
                self.is_response_locked = False
                self.selected_response_id = None
                
                # 恢复显示最新响应
                if self.last_response_id:
                    response = self.response_manager.get_response(self.last_response_id)
                    if response and response.response_text:
                        self.update_latest_response(self.last_response_id, response.response_text, response.question_text )

        root = self.textbox.winfo_toplevel()
        root.bind('<Button-1>', unlock_response, add="+")

    def is_response_frozen(self) -> bool:
        """
        检查响应是否被锁定
        
        Returns:
            bool: 如果响应被锁定返回True，否则返回False
        """
        return self.is_response_locked