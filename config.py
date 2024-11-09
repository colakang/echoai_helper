# config.py

import os
import sys
from dotenv import load_dotenv
from typing import Optional

class EnvConfig:
    """环境配置管理类"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def initialize(cls) -> None:
        """初始化环境配置"""
        if cls._initialized:
            return
            
        # 获取程序根目录
        if getattr(sys, 'frozen', False):
            # 如果是打包后的可执行文件
            root_dir = os.path.dirname(sys.executable)
        else:
            # 如果是源码运行
            root_dir = os.path.dirname(os.path.abspath(__file__))
            
        env_path = os.path.join(root_dir, '.env')
        
        # 如果.env文件不存在，创建它
        if not os.path.exists(env_path):
            cls.create_env_template(env_path)
            print(f"Please set your OpenAI API key in {env_path}")
            return
            
        # 加载.env文件
        load_dotenv(env_path)
        
        # 验证API密钥
        if not os.getenv('OPENAI_API_KEY'):
            print(f"OPENAI_API_KEY not found in {env_path}")
            print("Please add your OpenAI API key to the .env file")
            return
            
        cls._initialized = True
    
    @classmethod
    def create_env_template(cls, env_path: str) -> None:
        """创建.env模板文件"""
        template = (
            "# OpenAI API Configuration\n"
            "OPENAI_API_KEY=your_api_key_here\n"
            "\n"
            "# Add other configuration variables below\n"
        )
        try:
            with open(env_path, 'w', encoding='utf-8') as f:
                f.write(template)
            print(f"Created template .env file at {env_path}")
        except Exception as e:
            print(f"Error creating .env template: {e}")
    
    @classmethod
    def get_openai_key(cls) -> Optional[str]:
        """获取OpenAI API密钥"""
        # 确保已初始化
        if not cls._initialized:
            cls.initialize()
        return os.getenv('OPENAI_API_KEY')
    
    @classmethod
    def ensure_api_key(cls) -> bool:
        """确保API密钥已设置且有效"""
        api_key = cls.get_openai_key()
        if not api_key or api_key == 'your_api_key_here':
            env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
            print(f"Please set your OpenAI API key in {env_path}")
            return False
        return True
    
class SystemConfig:
    _instance = None
    _system_role = ""

    @classmethod
    def get_system_role(cls):
        return cls._system_role

    @classmethod
    def set_system_role(cls, role):
        cls._system_role = role

class AudioConfig:
    _instance = None
    _phrase_timeout = 5.2  # 默认值
    _buffer_chunks = 1  # 默认保存5个chunks

    @classmethod
    def get_buffer_chunks(cls):
        return cls._buffer_chunks

    @classmethod
    def set_buffer_chunks(cls, value):
        try:
            value = int(value)
            if 0 <= value <= 10:  # 限制chunks数量在0-10之间
                cls._buffer_chunks = value
                return True
            return False
        except ValueError:
            return False

    @classmethod
    def get_phrase_timeout(cls):
        return cls._phrase_timeout

    @classmethod
    def set_phrase_timeout(cls, value):
        try:
            value = float(value)
            if 0.01 <= value <= 50:
                cls._phrase_timeout = value
                return True
            return False
        except ValueError:
            return False