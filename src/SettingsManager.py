# src/settings_manager.py

import json
import os
from typing import Dict, Any, Optional
from .config import PathConfig

class SettingsManager:
    """管理应用程序设置的保存和加载"""
    
    DEFAULT_SETTINGS = {
        "phrase_timeout": 5.2,
        "buffer_chunks": 1,
        "update_interval": 2,
        "system_role": "inbound_cs",
        "case_detail": "inbound_cs",
        "knowledge": "none",
        "window_opacity": 1.0,
        "window_topmost": False,
        "record_only_mode": False  # 添加新设置项
    }
    
    def __init__(self):
        """初始化设置管理器"""
        # 使用PathConfig获取配置目录
        self.config_dir = PathConfig.get_config_path()
        os.makedirs(self.config_dir, exist_ok=True)
        
        # 设置文件的完整路径
        self.settings_file = os.path.join(self.config_dir, "settings.json")
        
        # 尝试从旧位置迁移设置
        self._migrate_old_settings()
        
        # 加载设置
        self.settings = self.load_settings()
        
        if self.debug_mode:
            print(f"Settings file location: {self.settings_file}")
    
    def _migrate_old_settings(self):
        """迁移旧版本的设置文件"""
        old_config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
        old_settings_file = os.path.join(old_config_dir, "settings.json")
        
        if os.path.exists(old_settings_file) and not os.path.exists(self.settings_file):
            try:
                # 读取旧设置
                with open(old_settings_file, 'r', encoding='utf-8') as f:
                    old_settings = json.load(f)
                
                # 保存到新位置
                os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
                with open(self.settings_file, 'w', encoding='utf-8') as f:
                    json.dump(old_settings, f, indent=4)
                    
                print(f"Settings migrated from {old_settings_file} to {self.settings_file}")
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