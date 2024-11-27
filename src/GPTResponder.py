##src/GPTResponder.py

import threading
import openai
from .prompts import create_prompt, INITIAL_RESPONSE
import time
import sys
from .config import SystemConfig,EnvConfig

class GPTResponder:
    def __init__(self, response_manager):
        self.response_manager = response_manager
        self.response = ""
        self._response_update_interval = 2
        self._lock = threading.Lock()
        self._processing = False
        self._last_processed_id = None
        # 初始化OpenAI配置
        if not self._initialize_openai():
            raise ValueError("Failed to initialize OpenAI configuration. Please check your API key.")

    def _initialize_openai(self) -> bool:
        """
        初始化OpenAI配置
        
        Returns:
            bool: 初始化成功返回True，否则返回False
        """
        if not EnvConfig.ensure_api_key():
            return False
            
        openai.api_key = EnvConfig.get_openai_key()
        return True

    def _generate_response_from_transcript(self, lastContent, latest_response_text="", latest_response_q_text="", current_response_id=None):
        """
        从转录内容生成流式回复
        
        Args:
            lastContent (str): 最新的转录内容
            latest_response_text (str): 上一次的回复内容
            latest_response_q_text (str): 上一次的问题内容
            current_response_id (str): 当前响应的ID
            
        Yields:
            str: 生成的部分回复内容
        """
        # 添加对短内容的过滤
        if lastContent.strip() == "" or len(lastContent.strip()) < 4:
            print(f"Skipping due to too short content (length: {len(lastContent.strip())})")
            return

        conversation_history = []
        recent_speakers = [f"Speaker: [{latest_response_q_text}]\n\n"]
        conversation_history.extend(recent_speakers)

        # 添加调试信息
        #print(f"\nDebug generate_response_from_transcript:")
        #print(f"Latest response: {latest_response_text}")
        
        # 将记录组合成字符串
        recent_transcript = "".join(conversation_history)
        #print(f"Recent transcript: {recent_transcript}")
        #print(f"Last content: {lastContent}")
        
        try:
            content = create_prompt(recent_speakers, lastContent, latest_response_text)
            #print(f"Created prompt: {content}")

            # 使用流式API
            stream = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SystemConfig.get_system_role()},
                    {"role": "user", "content": content},
                ],
                temperature=0.6,
                stream=True  # 启用流式响应
            )

            accumulated_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    chunk_content = chunk.choices[0].delta.content
                    accumulated_response += chunk_content
                    
                    # 尝试解析方括号中的内容
                    try:
                        if '[' in accumulated_response and ']' in accumulated_response:
                            response_text = accumulated_response.split("[")[1].split("]")[0]
                        else:
                            response_text = accumulated_response
                            
                        # 更新响应
                        if current_response_id:
                            self.response = response_text
                            self.response_manager.update_response(
                                current_response_id,
                                response_text,
                                is_complete=False
                            )
                        
                        yield response_text
                        
                    except Exception as e:
                        print(f"Error parsing chunk: {e}")
                        yield chunk_content

            # 完成后标记为完整响应
            if current_response_id:
                try:
                    # 尝试获取方括号中的内容，如果失败则使用完整响应
                    if '[' in accumulated_response and ']' in accumulated_response:
                        final_response = accumulated_response.split("[")[1].split("]")[0]
                    else:
                        print("No brackets found in response, using full response")
                        final_response = accumulated_response
                    
                    self.response_manager.update_response(
                        current_response_id,
                        final_response,
                        is_complete=True
                    )
                except Exception as e:
                    print(f"Error processing final response: {e}")
                    # 如果解析失败，使用累积的完整响应
                    self.response_manager.update_response(
                        current_response_id,
                        accumulated_response,
                        is_complete=True
                    )
                
        except Exception as e:
            print(f"Error in generate_response: {e}")
            error_message = str(e)
            if current_response_id:
                self.response_manager.update_response(
                    current_response_id,
                    error_message,
                    is_complete=True
                )
            yield error_message

    def respond_to_transcriber(self, transcriber):
        """
        持续监听并响应转录器的输出
        
        Args:
            transcriber: 转录器实例
        """
        while True:
            try:
                # 先等待 transcript_changed_event
                if transcriber.transcript_changed_event.wait(0.1):
                    transcriber.transcript_changed_event.clear()
                    
                    if transcriber.structured_transcript["speaker"]:
                        latest_record = transcriber.structured_transcript["speaker"][0]
                        current_response_id = latest_record[2]
                        
                        if (current_response_id and 
                            current_response_id != self._last_processed_id and 
                            not self._processing):
                            
                            with self._lock:
                                self._processing = True
                            
                            try:
                                question_text = latest_record[0]
                                self.response = "Thinking..."
                                self.response_manager.update_response(current_response_id, self.response)
                                
                                latest_response = self.response_manager.get_response(self._last_processed_id)
                                latest_response_text = ""
                                latest_response_q_text = ""
                                if latest_response and latest_response.is_complete:
                                    latest_response_text = latest_response.response_text
                                    latest_response_q_text = latest_response.question_text
                                
                                response_text = ''
                                # 使用生成器处理流式响应
                                for response_text in self._generate_response_from_transcript(
                                    question_text,
                                    latest_response_text,
                                    latest_response_q_text,
                                    current_response_id
                                ):
                                    if response_text.strip():
                                        #print(f"Generated partial response: {response_text}")
                                        self.response = response_text
                                
                                print(f"Generated response: {response_text}")
                                self._last_processed_id = current_response_id
                                
                            finally:
                                with self._lock:
                                    self._processing = False
            
            except Exception as e:
                print(f"Error in respond_to_transcriber: {e}")
                time.sleep(0.1)

    def update_response_interval(self, interval):
        self._response_update_interval = interval