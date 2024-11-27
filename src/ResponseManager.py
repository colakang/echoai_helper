#src/ResponseManager.py

import uuid
from dataclasses import dataclass
from typing import Optional, Dict, List
import threading
import json
import os
import traceback
from datetime import datetime, timezone
import pytz


@dataclass
class Response:
    response_id: str
    question_time: datetime
    question_text: str
    response_time: Optional[datetime] = None
    response_text: Optional[str] = None
    is_complete: bool = False

    def to_dict(self):
        """转换为可序列化的字典"""
        return {
            'response_id': self.response_id,
            'question_time': self.question_time.isoformat() if self.question_time else None,
            'question_text': self.question_text,
            'response_time': self.response_time.isoformat() if self.response_time else None,
            'response_text': self.response_text,
            'is_complete': self.is_complete
        }
    
class ResponseManager:
    def __init__(self):
        self._responses: Dict[str, Response] = {}
        self._lock = threading.Lock()
        self._latest_response_id: Optional[str] = None
        self._new_response_event = threading.Event()
        # 获取本地时区
        self._local_tz = datetime.now().astimezone().tzinfo

    def _convert_to_local_time(self, dt: datetime) -> datetime:
        """将时间转换为本地时区"""
        if dt.tzinfo is None:
            # 如果时间没有时区信息，假定为UTC
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(self._local_tz)

    def _format_datetime(self, dt: Optional[datetime]) -> Optional[str]:
        """格式化日期时间为本地时间字符串"""
        if dt is None:
            return None
        local_dt = self._convert_to_local_time(dt)
        return local_dt.isoformat()

    def export_responses(self) -> list:
        """
        导出所有响应数据为可序列化的格式
        
        Returns:
            list: 包含所有响应数据的列表
        """
        with self._lock:
            try:
                # 按时间顺序排序
                sorted_responses = sorted(
                    self._responses.values(),
                    key=lambda x: x.question_time,
                    reverse=True  # 最新的在前
                )
                
                # 转换为可序列化的格式
                responses_data = [response.to_dict() for response in sorted_responses]
                
                print(f"Exporting {len(responses_data)} responses")  # 调试信息
                return responses_data
                
            except Exception as e:
                print(f"Error in export_responses: {e}")
                return []

    def save_responses_to_file(self, filepath: str) -> bool:
        """
        将响应数据保存到JSON文件
        
        Args:
            filepath (str): 文件保存路径
            
        Returns:
            bool: 保存成功返回True，否则返回False
        """
        try:
            # 获取数据
            data = self.export_responses()
            
            if not data:
                print("No responses to export")
                return False
                
            print(f"Saving {len(data)} responses to {filepath}")  # 调试信息
            
            # 确保文件以.json结尾
            if not filepath.endswith('.json'):
                filepath += '.json'
            
            # 保存文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            # 验证文件是否正确保存
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                print(f"Successfully saved to {filepath}")
                return True
            else:
                print(f"File was created but may be empty: {filepath}")
                return False
                
        except Exception as e:
            print(f"Error saving responses: {e}")
            import traceback
            traceback.print_exc()  # 打印详细错误信息
            return False

    def export_structured_conversation(self, structured_transcript: dict, reverse_chronological: bool = False) -> dict:
        """
        基于structured_transcript导出完整的对话数据，使用本地时区
        """
        with self._lock:
            try:
                # 获取combined messages
                combined_messages = list(structured_transcript.get("combined", []))
                
                # 提取speaker类型的消息
                speaker_messages = []
                other_messages = []
                print (f'Combined: {combined_messages}')
                print (f'---------------')
                # 分离speaker和其他类型的消息
                for msg in combined_messages:
                    text, timestamp, response_id, speaker_type = msg
                    if speaker_type == "speaker":
                        speaker_messages.append(msg)
                    else:
                        other_messages.append(msg)
                
                # 如果有speaker消息，进行response_id前移处理
                if speaker_messages:
                    # 获取所有response_ids
                    response_ids = [msg[2] for msg in speaker_messages]  # [id1, id2, id3, ...]
                    
                    # 创建一个新的空response id
                    #new_first_id = response_ids[0]  # 保存第一个id用于复制
                    
                    # 后移response_ids
                    shifted_response_ids = [None] + response_ids[0:]   # [None,id1,id2, id3, ..., ]
                    
                    # 更新speaker_messages的response_ids
                    new_speaker_messages = []
                    for i, msg in enumerate(speaker_messages):
                        text, timestamp, _, speaker_type = msg
                        new_response_id = shifted_response_ids[i]
                        new_speaker_messages.append((text, timestamp, new_response_id, speaker_type))
                    
                # 根据时间戳合并消息
                all_messages = []
                all_messages.extend(new_speaker_messages)
                #all_messages.extend(speaker_messages)
                all_messages.extend(other_messages)
                # 按时间戳排序
                all_messages.sort(key=lambda x: x[1])
                
                # 设置排序
                #if not reverse_chronological:
                #    all_messages = all_messages[::-1]
                # 如果需要倒序（从新到旧），则反转列表
                if reverse_chronological:
                    all_messages.reverse()                

                # 构建response字典
                responses_dict = {}
                for response_id, response in self._responses.items():
                    if response_id:
                        responses_dict[response_id] = {
                            "id": response_id,
                            "question_time": self._format_datetime(response.question_time),
                            "question_text": response.question_text,
                            "response_time": self._format_datetime(response.response_time),
                            "response_text": response.response_text,
                            "is_complete": response.is_complete
                        }
                
                # 创建导出数据结构
                export_data = {
                    "metadata": {
                        "export_time": self._format_datetime(datetime.now().astimezone(self._local_tz)),
                        "version": "2.0",
                        "total_messages": len(all_messages),
                        "order": "newest_first" if reverse_chronological else "oldest_first",
                        "timezone": str(self._local_tz)
                    },
                    "conversation": {
                        "messages": []
                    }
                }
                
                # 构建最终的消息列表
                for idx, (text, timestamp, response_id, speaker_type) in enumerate(all_messages):
                    message = {
                        "role": speaker_type,
                        "text": text,
                        "timestamp": self._format_datetime(timestamp),
                        "response_id": response_id,
                        "index": idx
                    }
                    
                    # 只为有效的response_id添加响应
                    if response_id and response_id in responses_dict:
                        message["response"] = responses_dict[response_id]
                    
                    export_data["conversation"]["messages"].append(message)
                
                # Debug输出
                if hasattr(self, 'debug_mode') and self.debug_mode:
                    print("\nDebug - Message Processing:")
                    print("\nOriginal Speaker Messages:")
                    for msg in speaker_messages:
                        print(f"Text: {msg[0]}, Response ID: {msg[2]}")
                        
                    print("\nProcessed Messages:")
                    for msg in export_data["conversation"]["messages"]:
                        if msg["role"] == "speaker":
                            print(f"Text: {msg['text']}")
                            print(f"Response ID: {msg['response_id']}")
                            if "response" in msg:
                                print(f"Response Text: {msg['response']['response_text']}")
                            print("---")
                
                return export_data
                
            except Exception as e:
                print(f"Error in export_structured_conversation: {e}")
                traceback.print_exc()
                return {}
                    
    def save_structured_conversation(self, filepath: str, structured_transcript: dict) -> bool:
        """
        将结构化对话数据保存到JSON文件
        
        Args:
            filepath: 文件保存路径
            structured_transcript: 结构化的对话记录
                
        Returns:
            bool: 保存成功返回True，否则返回False
        """
        try:
            # 获取结构化对话数据
            data = structured_transcript
            
            if not data or not data["conversation"]["messages"]:
                print("No conversation data to export")
                return False
                
            print(f"Saving conversation with {len(data['conversation']['messages'])} messages")
            
            # 确保文件扩展名正确
            if not filepath.endswith('.json'):
                filepath += '.json'
            
            # 保存文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            # 验证文件保存成功
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                print(f"Successfully saved conversation to {filepath}")
                return True
            else:
                print(f"File was created but may be empty: {filepath}")
                return False
                
        except Exception as e:
            print(f"Error saving conversation: {e}")
            traceback.print_exc()
            return False

    def create_response(self, question_time: datetime, question_text: str) -> str:
        """为新的问题创建response记录，返回response_id"""
        response_id = str(uuid.uuid4())
        with self._lock:
            # 处理输入时间
            if question_time.tzinfo is None:
                question_time = question_time.replace(tzinfo=timezone.utc)
            question_time = question_time.astimezone(self._local_tz)   

            self._responses[response_id] = Response(
                response_id=response_id,
                question_time=question_time,
                question_text=question_text
            )
            self._latest_response_id = response_id
        return response_id

    def update_response(self, response_id: str, response_text: str, 
                    is_complete: bool = False, is_incremental: bool = False):
        """更新response内容，支持增量更新"""
        with self._lock:
            if response_id not in self._responses:
                return False
            
            response = self._responses[response_id]
            if response.response_time is None:
                response.response_time = datetime.now().astimezone(self._local_tz)
            
            if is_incremental:
                response.response_text = (response.response_text or "") + response_text
            else:
                response.response_text = response_text
                
            response.is_complete = is_complete
            
            if is_complete:
                self._new_response_event.set()
            return True
            
    def get_response(self, response_id: str) -> Optional[Response]:
        """获取指定response"""
        #print(f"Get Response ID: {response_id}")
        return self._responses.get(response_id)
    
    def get_latest_response(self) -> Optional[Response]:
        """获取最新的response"""
        print(f"Get latest response:\n\n")
        print(f"Latest response_id: {self._latest_response_id} \n\n")

        if self._latest_response_id:
            return self._responses.get(self._latest_response_id)
        return None
    
    def wait_for_new_response(self, timeout: Optional[float] = None) -> bool:
        """等待新的完整response"""
        result = self._new_response_event.wait(timeout)
        self._new_response_event.clear()
        return result