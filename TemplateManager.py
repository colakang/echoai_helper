"""
TemplateManager.py
处理系统角色和模板的管理类，负责模板的加载、更新和维护。
"""

import os
import glob
import traceback
from typing import List, Optional, Tuple, Dict
from SettingsManager import SettingsManager
from config import SystemConfig

class TemplateManager:
    """
    模板管理器类，处理系统角色相关的模板文件
    
    负责:
    - 管理不同类型的模板（系统角色、案例细节、知识库）
    - 加载和更新模板
    - 维护模板目录结构
    - 初始化默认系统角色
    
    Attributes:
        TEMPLATE_PATHS (Dict[str, Tuple[str, str]]): 模板类型及其对应的路径和扩展名配置
    """
    
    TEMPLATE_PATHS = {
        'system_role': ('prompt/system_role', '.py'),
        'case_detail': ('prompt/case_detail', '.txt'),
        'knowledge': ('prompt/knowledge', '.txt')
    }
    
    @classmethod
    def initialize_default_role(cls) -> bool:
        """
        初始化默认系统角色
        
        首先尝试从设置中加载，如果设置不存在或无效，
        则从可用的模板中选择默认值并初始化系统角色。
        
        Returns:
            bool: 初始化成功返回True，失败返回False
        """
        try:
            # 创建设置管理器实例
            settings_manager = SettingsManager()
            
            # 获取所有可用的模板文件
            system_role_files = cls.get_template_files('system_role')
            case_detail_files = cls.get_template_files('case_detail')
            knowledge_files = cls.get_template_files('knowledge')
            
            # 从设置中获取保存的值
            saved_role = settings_manager.get_setting("system_role")
            saved_detail = settings_manager.get_setting("case_detail")
            saved_knowledge = settings_manager.get_setting("knowledge")
            
            # 验证保存的值是否有效（检查文件是否存在）
            saved_role_valid = saved_role in system_role_files
            saved_detail_valid = saved_detail in case_detail_files
            saved_knowledge_valid = saved_knowledge in knowledge_files
            
            if saved_role_valid and saved_detail_valid and saved_knowledge_valid:
                # 使用保存的设置
                print(f"Using saved template settings: {saved_role}, {saved_detail}, {saved_knowledge}")
                success = cls.update_system_role(saved_role, saved_detail, saved_knowledge)
                if success:
                    print(f"Initialized role from settings: {saved_role}")
                    return True
                
            # 如果保存的设置无效或更新失败，使用可用模板中的第一个
            print("Saved settings invalid or update failed, using first available templates")
            
            default_role = system_role_files[0] if system_role_files else 'inbound_cs'
            default_detail = case_detail_files[0] if case_detail_files else 'inbound_cs'
            default_knowledge = knowledge_files[0] if knowledge_files else 'none'
            
            # 更新系统角色
            success = cls.update_system_role(default_role, default_detail, default_knowledge)
            if success:
                # 更新设置
                settings_manager.update_setting("system_role", default_role)
                settings_manager.update_setting("case_detail", default_detail)
                settings_manager.update_setting("knowledge", default_knowledge)
                print(f"Initialized default role and updated settings: {default_role} / {default_detail} / {default_knowledge}")
                return True
                
            print("Failed to initialize default role")
            return False
            
        except Exception as e:
            print(f"Error initializing default role: {e}")
            traceback.print_exc()
            return False

    @classmethod
    def load_template(cls, filepath: str) -> str:
        """
        加载模板文件内容
        
        Args:
            filepath: 模板文件的完整路径
            
        Returns:
            str: 模板内容，如果加载失败返回空字符串
        """
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
        """
        获取指定类别的所有模板文件
        
        Args:
            category: 模板类别('system_role', 'case_detail', 或 'knowledge')
            
        Returns:
            List[str]: 模板文件名列表（不含扩展名）
        """
        if category not in cls.TEMPLATE_PATHS:
            print(f"Invalid template category: {category}")
            return []
            
        path, ext = cls.TEMPLATE_PATHS[category]
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
        """
        更新系统角色配置
        
        组合加载的模板内容，更新系统角色配置。
        
        Args:
            system_role_file: 系统角色模板文件名
            case_detail_file: 案例细节模板文件名
            knowledge_file: 知识模板文件名
            
        Returns:
            Optional[str]: 更新后的角色配置，失败时返回None
        """
        try:
            # 构建完整路径
            path_role = cls.TEMPLATE_PATHS['system_role']
            path_detail = cls.TEMPLATE_PATHS['case_detail']
            path_knowledge = cls.TEMPLATE_PATHS['knowledge']
            
            system_role_path = os.path.join(path_role[0], f"{system_role_file}{path_role[1]}")
            case_detail_path = os.path.join(path_detail[0], f"{case_detail_file}{path_detail[1]}")
            knowledge_path = os.path.join(path_knowledge[0], f"{knowledge_file}{path_knowledge[1]}")
            
            # 加载模板内容
            system_role = cls.load_template(system_role_path)
            case_detail = cls.load_template(case_detail_path)
            knowledge = cls.load_template(knowledge_path)
            
            if not all([system_role, case_detail, knowledge]):
                print("Error: One or more templates could not be loaded")
                return None
            
            # 格式化角色配置
            try:
                new_role = system_role.format(case_detail=case_detail, knowledge=knowledge)
                if new_role.strip():  # 确保不是空字符串
                    SystemConfig.set_system_role(new_role)
                    print(f"Updated system role successfully")
                    return new_role
                else:
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
        """
        确保所有模板目录都存在
        
        创建所有必要的模板目录，如果它们不存在的话。
        """
        for path, _ in cls.TEMPLATE_PATHS.values():
            try:
                os.makedirs(path, exist_ok=True)
            except Exception as e:
                print(f"Error creating directory {path}: {e}")
                traceback.print_exc()

    @classmethod
    def get_current_role(cls) -> Optional[str]:
        """
        获取当前的系统角色配置
        
        Returns:
            Optional[str]: 当前的系统角色配置，如果未设置返回None
        """
        return SystemConfig.get_system_role()