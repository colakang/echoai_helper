# src/config.py

import os
import sys
from dotenv import load_dotenv
from typing import Optional

class PathConfig:
    """路径配置管理"""
    
    @staticmethod
    def get_project_root():
        """
        The directory the app was launched from.

        Only meaningful for a source checkout. Anything the app *ships* --
        conf.yaml, the prompt templates, the default settings -- lives inside
        the package instead, because once this is installed from a wheel there
        is no project root: the code lands in site-packages and everything
        beside it belongs to other packages.
        """
        if getattr(sys, 'frozen', False):
            return os.path.dirname(sys.executable)
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @staticmethod
    def get_package_root():
        """Where the installed package lives -- the one path that is always real."""
        return os.path.dirname(os.path.abspath(__file__))

    @staticmethod
    def get_resource_path():
        """
        Shipped, read-only data: prompt templates and default settings.

        Resolved against the package rather than the project, so it is the same
        directory whether this is a git checkout or a wheel in site-packages.
        """
        if getattr(sys, 'frozen', False):
            return os.path.join(sys._MEIPASS, 'resources')
        return os.path.join(PathConfig.get_package_root(), 'resources')

    @staticmethod
    def get_config_path():
        """获取配置文件目录"""
        return os.path.join(PathConfig.get_resource_path(), 'config')

    @staticmethod
    def get_prompt_path():
        """获取prompt目录"""
        return os.path.join(PathConfig.get_resource_path(), 'prompt')

    @staticmethod
    def get_user_prompt_path():
        """
        Where a user's own imported templates live.

        Outside the package, for the same reason conf.yaml has a copy there:
        installed from a wheel the shipped templates sit in site-packages,
        which is not somewhere anyone should be writing and which a reinstall
        overwrites. Imports go here and survive.
        """
        return os.path.join(PathConfig.get_user_config_path(), "prompt")

    @staticmethod
    def get_conf_file():
        """
        conf.yaml, preferring the user's own copy.

        The shipped default is inside the package, which is not somewhere
        anyone can reasonably edit once this is installed from a wheel -- it is
        in site-packages, and a reinstall overwrites it. So a copy in the user
        config directory wins if it exists; `echoai-helper config` puts one
        there. With neither, the shipped defaults are all "auto" and work
        unedited, which is the common case.
        """
        user_copy = os.path.join(PathConfig.get_user_config_path(), "conf.yaml")
        if os.path.exists(user_copy):
            return user_copy
        return os.path.join(PathConfig.get_package_root(), "conf.yaml")

    @staticmethod
    def get_user_config_path():
        """
        Where this user's own settings live.

        Deliberately outside the repo. resources/config/settings.json is
        tracked in git but was being rewritten on every change, so one
        person's window opacity and mode showed up as a diff, and committing
        it pushed their preferences to everyone. That file is now the shipped
        default, read once and never written.
        """
        if sys.platform == "darwin":
            base = os.path.expanduser("~/Library/Application Support")
        elif sys.platform == "win32":
            base = os.environ.get("APPDATA") or os.path.expanduser("~")
        else:
            base = os.environ.get(
                "XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
        return os.path.join(base, "EchoAI Helper")

class EnvConfig:
    """环境配置管理类"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    # Files holding nothing but the API key itself, checked before .env.
    # Keeping the secret in one place beats copying it into a second file;
    # every one of these must stay in .gitignore.
    KEY_FILES = ('.llm', '.openai_key')

    @classmethod
    def _load_bare_key_file(cls, root: str) -> bool:
        """Read a file whose entire contents are the API key. Returns True on success."""
        for name in cls.KEY_FILES:
            path = os.path.join(root, name)
            if not os.path.exists(path):
                continue
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    key = f.read().strip()
            except Exception as e:
                print(f"Error reading {path}: {e}")
                continue
            if not key:
                continue
            # Tolerate `OPENAI_API_KEY=sk-...` as well as a bare `sk-...`.
            if '=' in key.split('\n')[0]:
                key = key.split('\n')[0].split('=', 1)[1].strip().strip('"\'')
            os.environ['OPENAI_API_KEY'] = key
            print(f"[INFO] Loaded OpenAI key from {name}")
            return True
        return False

    @classmethod
    def initialize(cls) -> None:
        """初始化环境配置"""
        if cls._initialized:
            return

        root = PathConfig.get_project_root()

        # 优先读裸密钥文件（.llm 等），其次 .env
        if cls._load_bare_key_file(root):
            cls._initialized = True
            return

        # 获取.env文件路径
        env_path = os.path.join(root, '.env')

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
        if not cls._initialized:
            cls.initialize()
        return os.getenv('OPENAI_API_KEY')
    
    @classmethod
    def ensure_api_key(cls) -> bool:
        api_key = cls.get_openai_key()
        return bool(api_key and api_key != 'your_api_key_here')

class SystemConfig:
    _instance = None
    _system_role = ""
    _record_only_mode = False  # Add new class variable for record-only mode

    @classmethod
    def get_system_role(cls):
        return cls._system_role

    @classmethod
    def set_system_role(cls, role):
        cls._system_role = role
    @classmethod
    def get_record_only_mode(cls):
        """Get the current state of record-only mode"""
        return cls._record_only_mode

    @classmethod
    def set_record_only_mode(cls, value: bool):
        """Set the record-only mode state
        
        Args:
            value (bool): True to enable record-only mode, False to disable
        """
        cls._record_only_mode = bool(value)
        
class AudioConfig:
    _instance = None
    _phrase_timeout = 5.2  # 默认值
    _buffer_chunks = 1     # 默认值

    # Whether to run the fast lane: re-transcribe the utterance in progress on
    # every chunk so text appears while someone is still speaking.
    #
    # Off by default, which suits meeting notes -- the primary use case. There
    # the only thing that matters is the accurate pass at each pause, and
    # skipping partials removes roughly two thirds of all model calls.
    #
    # On for live interview, where seeing words ~0.6s after they are spoken is
    # the entire point and paying for it is the trade.
    _live_partials = False

    # Whether the microphone track is paused.
    #
    # Muting yourself in the meeting app does not reach us -- we hold our own
    # input stream, and Zoom or WeChat silencing your outgoing audio has no
    # effect on what CoreAudio hands this process. So a meeting spent muted
    # still fills the transcript with your side of the room.
    #
    # It is also the cheapest real-time headroom available: the measured
    # real-time factor is a *dual-track* figure, so dropping one track roughly
    # halves the model's work. On a machine that is only just keeping up, this
    # is the difference.
    _mic_paused = False

    @classmethod
    def get_mic_paused(cls):
        return cls._mic_paused

    @classmethod
    def set_mic_paused(cls, value: bool):
        cls._mic_paused = bool(value)

    @classmethod
    def get_live_partials(cls):
        return cls._live_partials

    @classmethod
    def set_live_partials(cls, value: bool):
        cls._live_partials = bool(value)

    # Which named profile is current. See src/profiles.py.
    _profile = "meeting"

    @classmethod
    def replies_enabled(cls):
        """
        Whether to ask a language model for a suggested reply.

        Two conditions, and the mode one was missing. Reply suggestions are an
        interview feature: in meeting notes the pause is long and partials are
        off, so an answer arrives well after the moment it was for, and nobody
        is looking at that pane anyway.

        Without this the only gate was the Record Only checkbox, which meant a
        user in meeting mode who unticked it started paying for an API call on
        every sentence the far end spoke, with nothing on screen to say so.
        """
        return (cls._profile == "interview"
                and not SystemConfig.get_record_only_mode())

    @classmethod
    def get_profile(cls):
        return cls._profile

    @classmethod
    def set_profile(cls, key: str):
        cls._profile = key

    # Label who is speaking on a track that carries several people. Costs one
    # CAM++ embedding per segment (~293ms on an M4) on top of transcription,
    # so it is opt-in: pointless for a one-to-one call, close to essential for
    # a meeting transcript.
    _diarization = False

    @classmethod
    def get_diarization(cls):
        return cls._diarization

    @classmethod
    def set_diarization(cls, value: bool):
        cls._diarization = bool(value)

    # How many people are actually in the meeting. 0 means "work it out".
    #
    # Worth telling it: voice embeddings drift with volume, codec and network
    # conditions, so online clustering splits one person into several. A real
    # WeChat call produced 12 speakers -- exactly the cap -- for a handful of
    # people. Given the real number, the registry stops inventing new ones and
    # assigns each utterance to the nearest voice it already knows.
    _speaker_count = 0

    @classmethod
    def get_speaker_count(cls):
        return cls._speaker_count

    @classmethod
    def set_speaker_count(cls, value):
        try:
            cls._speaker_count = max(0, int(value))
        except (TypeError, ValueError):
            cls._speaker_count = 0

    @classmethod
    def get_buffer_chunks(cls):
        return cls._buffer_chunks

    @classmethod
    def set_buffer_chunks(cls, value):
        try:
            value = int(value)
            if 0 <= value <= 10:
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