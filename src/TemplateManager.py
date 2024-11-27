"""
src/template_manager.py
处理系统角色和模板的管理类，负责模板的加载、更新和维护。
"""

import os
import glob
import traceback
from typing import List, Optional, Tuple, Dict
from .SettingsManager import SettingsManager
from .config import SystemConfig, PathConfig

class TemplateManager:
    """模板管理器类，处理系统角色相关的模板文件"""
    
    @classmethod
    def _get_template_paths(cls) -> Dict[str, Tuple[str, str]]:
        """获取模板路径配置"""
        prompt_path = PathConfig.get_prompt_path()
        return {
            'system_role': (os.path.join(prompt_path, 'system_role'), '.py'),
            'case_detail': (os.path.join(prompt_path, 'case_detail'), '.txt'),
            'knowledge': (os.path.join(prompt_path, 'knowledge'), '.txt')
        }
    
    @classmethod
    def initialize_default_role(cls) -> bool:
        """初始化默认系统角色"""
        try:
            settings_manager = SettingsManager()
            
            # 获取所有可用的模板文件
            system_role_files = cls.get_template_files('system_role')
            case_detail_files = cls.get_template_files('case_detail')
            knowledge_files = cls.get_template_files('knowledge')
            
            # 从设置中获取保存的值
            saved_role = settings_manager.get_setting("system_role")
            saved_detail = settings_manager.get_setting("case_detail")
            saved_knowledge = settings_manager.get_setting("knowledge")
            
            # 验证保存的值是否有效
            saved_role_valid = saved_role in system_role_files
            saved_detail_valid = saved_detail in case_detail_files
            saved_knowledge_valid = saved_knowledge in knowledge_files
            
            if saved_role_valid and saved_detail_valid and saved_knowledge_valid:
                print(f"Using saved template settings: {saved_role}, {saved_detail}, {saved_knowledge}")
                success = cls.update_system_role(saved_role, saved_detail, saved_knowledge)
                if success:
                    print(f"Initialized role from settings: {saved_role}")
                    return True
            
            # 使用默认值
            print("Using default templates")
            default_role = system_role_files[0] if system_role_files else 'inbound_cs'
            default_detail = case_detail_files[0] if case_detail_files else 'inbound_cs'
            default_knowledge = knowledge_files[0] if knowledge_files else 'none'
            
            success = cls.update_system_role(default_role, default_detail, default_knowledge)
            if success:
                settings_manager.update_setting("system_role", default_role)
                settings_manager.update_setting("case_detail", default_detail)
                settings_manager.update_setting("knowledge", default_knowledge)
                print(f"Initialized default role: {default_role}")
                return True
                
            print("Failed to initialize default role")
            return False
            
        except Exception as e:
            print(f"Error initializing default role: {e}")
            traceback.print_exc()
            return False

    @classmethod
    def load_template(cls, filepath: str) -> str:
        """加载模板文件内容"""
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                if not content.strip():
                    print(f"Warning: Template file is empty: {filepath}")
                return content
        except Exception as e:
            print(f"Error loading template {filepath}: {e}")
            traceback.print_exc()
            return ""

    @classmethod
    def get_template_files(cls, category: str) -> List[str]:
        """获取指定类别的所有模板文件"""
        template_paths = cls._get_template_paths()
        if category not in template_paths:
            print(f"Invalid template category: {category}")
            return []
            
        path, ext = template_paths[category]
        pattern = os.path.join(path, f"*{ext}")
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(pattern), exist_ok=True)
            
            files = glob.glob(pattern)
            return [os.path.basename(f).replace(ext, '') for f in files]
        except Exception as e:
            print(f"Error getting template files for {category}: {e}")
            traceback.print_exc()
            return []

    @classmethod
    def update_system_role(cls, system_role_file: str, case_detail_file: str, 
                          knowledge_file: str) -> Optional[str]:
        """更新系统角色配置"""
        try:
            template_paths = cls._get_template_paths()
            
            # 构建完整路径
            system_role_path = os.path.join(template_paths['system_role'][0], 
                                          f"{system_role_file}{template_paths['system_role'][1]}")
            case_detail_path = os.path.join(template_paths['case_detail'][0], 
                                          f"{case_detail_file}{template_paths['case_detail'][1]}")
            knowledge_path = os.path.join(template_paths['knowledge'][0], 
                                        f"{knowledge_file}{template_paths['knowledge'][1]}")
            
            # 加载模板内容
            system_role = cls.load_template(system_role_path)
            case_detail = cls.load_template(case_detail_path)
            knowledge = cls.load_template(knowledge_path)
            
            if not all([system_role, case_detail, knowledge]):
                print("Error: One or more templates could not be loaded")
                return None
            
            try:
                new_role = system_role.format(case_detail=case_detail, knowledge=knowledge)
                if new_role.strip():  # 确保不是空字符串
                    SystemConfig.set_system_role(new_role)
                    return new_role
                print("Error: Formatted role is empty")
                return None
            except KeyError as e:
                print(f"Template format error: Missing key {e}")
                traceback.print_exc()
                return None
                
        except Exception as e:
            print(f"Error updating system role: {e}")
            traceback.print_exc()
            return None

    @classmethod
    def ensure_template_directories(cls) -> None:
        """确保所有模板目录都存在"""
        for path, _ in cls._get_template_paths().values():
            try:
                os.makedirs(path, exist_ok=True)
            except Exception as e:
                print(f"Error creating directory {path}: {e}")
                traceback.print_exc()

    @classmethod
    def get_current_role(cls) -> Optional[str]:
        """获取当前的系统角色配置"""
        return SystemConfig.get_system_role()