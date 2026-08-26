"""
src/template_manager.py
处理系统角色和模板的管理类，负责模板的加载、更新和维护。
"""

import os
import re
import glob
import traceback
from typing import List, Optional, Tuple, Dict
from .SettingsManager import SettingsManager
from .config import SystemConfig, PathConfig

class TemplateManager:
    """模板管理器类，处理系统角色相关的模板文件"""
    
    # Extension per category. system_role is .py by history -- the files hold a
    # triple-quoted assignment -- but they are read as text, never imported.
    EXTENSIONS = {'system_role': '.py', 'case_detail': '.txt', 'knowledge': '.txt'}

    @classmethod
    def _get_template_paths(cls) -> Dict[str, Tuple[str, str]]:
        """The shipped templates, inside the package."""
        prompt_path = PathConfig.get_prompt_path()
        return {category: (os.path.join(prompt_path, category), ext)
                for category, ext in cls.EXTENSIONS.items()}

    @classmethod
    def _get_user_paths(cls) -> Dict[str, Tuple[str, str]]:
        """The user's own imported templates, outside the package."""
        prompt_path = PathConfig.get_user_prompt_path()
        return {category: (os.path.join(prompt_path, category), ext)
                for category, ext in cls.EXTENSIONS.items()}

    @classmethod
    def import_template(cls, category: str, source_path: str) -> Optional[str]:
        """
        Copy a file in as a template of `category`, and return its name.

        Imports land in the user config directory rather than beside the
        shipped ones. Installed from a wheel those sit in site-packages, where
        writing is wrong and a reinstall would silently delete whatever the
        user had added.

        A name that already exists is not overwritten silently; a numeric
        suffix is added instead. Losing a template someone spent an afternoon
        writing because they picked a familiar filename is not a reasonable
        cost for the convenience of skipping a prompt.
        """
        import shutil

        if category not in cls.EXTENSIONS:
            print(f"Invalid template category: {category}")
            return None
        if not os.path.isfile(source_path):
            print(f"No such file: {source_path}")
            return None

        directory, ext = cls._get_user_paths()[category]
        os.makedirs(directory, exist_ok=True)

        stem = os.path.splitext(os.path.basename(source_path))[0]
        name, suffix = stem, 1
        while os.path.exists(os.path.join(directory, f"{name}{ext}")):
            name = f"{stem}_{suffix}"
            suffix += 1

        destination = os.path.join(directory, f"{name}{ext}")
        try:
            shutil.copy(source_path, destination)
        except OSError as e:
            print(f"Could not import {source_path}: {e}")
            return None
        print(f"[INFO] Imported {category} template {name!r} -> {destination}")
        return name

    @classmethod
    def resolve_template(cls, category: str, name: str) -> Optional[str]:
        """
        The file backing a template name, preferring the user's own.

        A user who imports a template named like a shipped one means theirs.
        """
        for paths in (cls._get_user_paths(), cls._get_template_paths()):
            directory, ext = paths[category]
            candidate = os.path.join(directory, f"{name}{ext}")
            if os.path.exists(candidate):
                return candidate
        return None
    
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
        """
        加载模板文件内容

        system_role templates live in .py files that assign a triple-quoted
        string to a name, but they are read as plain text and never imported.
        Read naively, the assignment syntax ends up inside the system prompt
        sent to the model: prompts were starting with the literal text
        `SYSTEM_ROLE =` followed by an opening triple quote. Unwrap it here.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                if not content.strip():
                    print(f"Warning: Template file is empty: {filepath}")
                return cls._unwrap_python_string(content)
        except Exception as e:
            print(f"Error loading template {filepath}: {e}")
            traceback.print_exc()
            return ""

    @staticmethod
    def _fill_placeholders(system_role: str, case_detail: str,
                           knowledge: str) -> str:
        """
        Substitute the two placeholders this supports, and nothing else.

        This used to call str.format(), which interprets *every* brace in the
        template. That was survivable while the only templates were the two
        shipped ones; it stopped being survivable the moment templates could be
        imported, because a prompt is exactly the kind of text that contains
        braces on purpose:

            {"role": "user"}   -- a JSON example    -> KeyError
            {customer_name}    -- a literal marker  -> KeyError
            use { for grouping -- an unmatched one  -> ValueError

        Every one of those left the role unchanged and the dropdown showing the
        new selection, so the app went on using the previous persona while
        appearing to have switched.

        Replacing the two names directly cannot fail, and leaves everything
        else in the prompt exactly as written.
        """
        return (system_role
                .replace("{case_detail}", case_detail)
                .replace("{knowledge}", knowledge))

    @staticmethod
    def _unwrap_python_string(content: str) -> str:
        """If the whole file is one `NAME = <triple-quoted string>` assignment,
        return just the string body. Anything else is returned untouched."""
        match = re.match(
            r'''\s*[A-Za-z_]\w*\s*=\s*(?P<q>"""|\'\'\')(?P<body>.*)(?P=q)\s*\Z''',
            content,
            re.DOTALL,
        )
        return match.group('body').strip() if match else content

    @classmethod
    def get_template_files(cls, category: str) -> List[str]:
        """
        Every template of this category: the shipped ones and the user's own.

        Names are unique across both, so a user template shadowing a shipped
        one appears once -- and resolve_template() picks theirs.
        """
        if category not in cls.EXTENSIONS:
            print(f"Invalid template category: {category}")
            return []

        names = []
        for paths in (cls._get_template_paths(), cls._get_user_paths()):
            path, ext = paths[category]
            try:
                for found in sorted(glob.glob(os.path.join(path, f"*{ext}"))):
                    name = os.path.basename(found)[:-len(ext)]
                    if name not in names:
                        names.append(name)
            except OSError as e:
                print(f"Error listing templates in {path}: {e}")
        return names

    @classmethod
    def update_system_role(cls, system_role_file: str, case_detail_file: str, 
                          knowledge_file: str) -> Optional[str]:
        """更新系统角色配置"""
        try:
            system_role_path = cls.resolve_template('system_role', system_role_file)
            case_detail_path = cls.resolve_template('case_detail', case_detail_file)
            knowledge_path = cls.resolve_template('knowledge', knowledge_file)

            missing = [name for name, path in
                       (('system_role', system_role_path),
                        ('case_detail', case_detail_path),
                        ('knowledge', knowledge_path)) if path is None]
            if missing:
                print(f"Error: template not found for {', '.join(missing)}")
                return None

            system_role = cls.load_template(system_role_path)
            case_detail = cls.load_template(case_detail_path)
            knowledge = cls.load_template(knowledge_path)
            
            if not all([system_role, case_detail, knowledge]):
                print("Error: One or more templates could not be loaded")
                return None
            
            new_role = cls._fill_placeholders(system_role, case_detail, knowledge)
            if not new_role.strip():
                print("Error: the assembled role is empty")
                return None
            SystemConfig.set_system_role(new_role)
            return new_role
                
        except Exception as e:
            print(f"Error updating system role: {e}")
            traceback.print_exc()
            return None

    @classmethod
    def ensure_template_directories(cls) -> None:
        """Only the user's own directories; the shipped ones come with the package."""
        for path, _ in cls._get_user_paths().values():
            try:
                os.makedirs(path, exist_ok=True)
            except Exception as e:
                print(f"Error creating directory {path}: {e}")
                traceback.print_exc()

    @classmethod
    def get_current_role(cls) -> Optional[str]:
        """获取当前的系统角色配置"""
        return SystemConfig.get_system_role()