# src/settings_manager.py

import json
import os
from typing import Dict, Any, Optional
from .config import PathConfig

class SettingsManager:
    """管理应用程序设置的保存和加载"""
    
    DEFAULT_SETTINGS = {
        # Segmentation is driven by the VAD now. "profile" selects a bundle of
        # settings (see src/profiles.py); min_silence_ms is the one knob from
        # that bundle the UI exposes on its own.
        "profile": "meeting",
        "min_silence_ms": 700,
        # Retained so older settings.json files still load without warnings.
        # Nothing reads them: phrase_timeout and buffer_chunks belonged to the
        # accumulate loop that the segmenter replaced, and update_interval was
        # stored by the responder but never used.
        "phrase_timeout": 5.2,
        "buffer_chunks": 1,
        "update_interval": 2,
        "system_role": "inbound_cs",
        "case_detail": "inbound_cs",
        "knowledge": "none",
        "window_opacity": 1.0,
        "window_topmost": False,
        # True means "do not ask a model for replies". The default suppresses
        # them, because turning them on spends money on every sentence the far
        # end speaks and nobody should discover that by accident. It also has
        # to agree with the shipped settings.json, which it did not: a user
        # whose file predated this key fell through to the code default and
        # got the opposite of what the shipped one says.
        "record_only_mode": True,
        "diarization": False,
        # 0 = work it out; otherwise the number of people on the call.
        "speaker_count": 0,
        # Remembered so the first-run setup asks once, not every launch.
        "setup_declined": False
    }
    
    def __init__(self):
        """初始化设置管理器"""
        # User settings live outside the repo; see PathConfig.get_user_config_path.
        self.config_dir = PathConfig.get_user_config_path()
        os.makedirs(self.config_dir, exist_ok=True)
        self.settings_file = os.path.join(self.config_dir, "settings.json")

        # Shipped defaults, read-only.
        self.defaults_file = os.path.join(
            PathConfig.get_config_path(), "settings.json")

        self._migrate_old_settings()
        self.settings = self.load_settings()

        if self.debug_mode:
            print(f"Settings file location: {self.settings_file}")

    def _migrate_old_settings(self):
        """Carry a pre-existing in-repo settings file over to the user dir."""
        if os.path.exists(self.settings_file):
            return
        if not os.path.exists(self.defaults_file):
            return
        try:
            with open(self.defaults_file, 'r', encoding='utf-8') as f:
                previous = json.load(f)
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(previous, f, indent=4)
            print(f"Settings migrated to {self.settings_file}")
        except Exception as e:
            print(f"Error migrating settings: {e}")
            
    def load_settings(self) -> Dict[str, Any]:
        """
        从文件加载设置
        
        Returns:
            Dict[str, Any]: 设置字典
        """
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    saved_settings = json.load(f)
                    # 合并保存的设置和默认设置，确保所有必要的键都存在
                    merged_settings = self.DEFAULT_SETTINGS.copy()
                    merged_settings.update(saved_settings)
                    return merged_settings
            return self.DEFAULT_SETTINGS.copy()
        except Exception as e:
            print(f"Error loading settings from {self.settings_file}: {e}")
            return self.DEFAULT_SETTINGS.copy()
            
    def save_settings(self, settings: Dict[str, Any]) -> bool:
        """
        保存设置到文件
        
        Args:
            settings: 要保存的设置字典
            
        Returns:
            bool: 保存成功返回True，否则返回False
        """
        try:
            # 确保配置目录存在
            os.makedirs(self.config_dir, exist_ok=True)
            
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=4)
            self.settings = settings
            return True
        except Exception as e:
            print(f"Error saving settings to {self.settings_file}: {e}")
            return False
            
    def get_setting(self, key: str) -> Any:
        """
        获取指定设置的值
        
        Args:
            key: 设置键名
            
        Returns:
            Any: 设置值，如果键不存在则返回默认值
        """
        return self.settings.get(key, self.DEFAULT_SETTINGS.get(key))
        
    def update_setting(self, key: str, value: Any) -> bool:
        """
        更新指定设置的值
        
        Args:
            key: 设置键名
            value: 新的设置值
            
        Returns:
            bool: 更新成功返回True，否则返回False
        """
        try:
            # 类型转换确保与默认值类型一致
            if key in self.DEFAULT_SETTINGS:
                value = type(self.DEFAULT_SETTINGS[key])(value)
            self.settings[key] = value
            return self.save_settings(self.settings)
        except Exception as e:
            print(f"Error updating setting {key}: {e}")
            return False
    
    @property
    def debug_mode(self) -> bool:
        """是否启用调试模式"""
        return False  # 可以根据需要修改