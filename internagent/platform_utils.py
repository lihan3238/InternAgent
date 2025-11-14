"""
跨平台兼容性工具模块

提供跨平台的文件路径处理、命令执行等功能，确保在Windows、MacOS和Linux上都能正常工作。
"""

import os
import sys
import platform
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PlatformUtils:
    """跨平台工具类"""
    
    @staticmethod
    def get_platform() -> str:
        """
        获取当前操作系统平台
        
        Returns:
            str: 'windows', 'macos', 或 'linux'
        """
        system = platform.system().lower()
        if system == 'darwin':
            return 'macos'
        elif system == 'windows':
            return 'windows'
        else:
            return 'linux'
    
    @staticmethod
    def is_windows() -> bool:
        """判断是否为Windows系统"""
        return PlatformUtils.get_platform() == 'windows'
    
    @staticmethod
    def is_macos() -> bool:
        """判断是否为MacOS系统"""
        return PlatformUtils.get_platform() == 'macos'
    
    @staticmethod
    def is_linux() -> bool:
        """判断是否为Linux系统"""
        return PlatformUtils.get_platform() == 'linux'
    
    @staticmethod
    def normalize_path(path: str) -> str:
        """
        规范化路径，使其在当前平台上正确
        
        Args:
            path: 输入路径
            
        Returns:
            str: 规范化后的路径
        """
        if not path:
            return path
        
        # 转换为Path对象处理
        p = Path(path)
        
        # 在Windows上，确保使用反斜杠
        # 在Unix系统上，确保使用正斜杠
        return str(p.resolve())
    
    @staticmethod
    def join_paths(*paths: str) -> str:
        """
        跨平台的路径拼接
        
        Args:
            *paths: 要拼接的路径部分
            
        Returns:
            str: 拼接后的路径
        """
        return str(Path(*paths))
    
    @staticmethod
    def ensure_dir(directory: str) -> None:
        """
        确保目录存在，不存在则创建
        
        Args:
            directory: 目录路径
        """
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def get_shell_executable() -> str:
        """
        获取当前平台的shell可执行文件
        
        Returns:
            str: shell路径
        """
        if PlatformUtils.is_windows():
            return 'cmd.exe'
        else:
            return '/bin/bash'
    
    @staticmethod
    def run_command(command: str,
                   cwd: Optional[str] = None,
                   env: Optional[Dict[str, str]] = None,
                   shell: bool = True,
                   timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        跨平台执行命令
        
        Args:
            command: 要执行的命令
            cwd: 工作目录
            env: 环境变量
            shell: 是否使用shell执行
            timeout: 超时时间（秒）
            
        Returns:
            Dict: 包含stdout, stderr, returncode的字典
        """
        try:
            # Windows上的特殊处理
            if PlatformUtils.is_windows():
                # 在Windows上，某些命令需要特殊处理
                if shell and not command.startswith('cmd /c'):
                    command = f'cmd /c {command}'
            
            # 执行命令
            result = subprocess.run(
                command,
                shell=shell,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return {
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode,
                'success': result.returncode == 0
            }
            
        except subprocess.TimeoutExpired as e:
            logger.error(f"Command timeout: {command}")
            return {
                'stdout': '',
                'stderr': f'Command timeout after {timeout} seconds',
                'returncode': -1,
                'success': False
            }
        except Exception as e:
            logger.error(f"Error running command: {str(e)}")
            return {
                'stdout': '',
                'stderr': str(e),
                'returncode': -1,
                'success': False
            }
    
    @staticmethod
    def run_script(script_path: str,
                  args: Optional[List[str]] = None,
                  cwd: Optional[str] = None) -> Dict[str, Any]:
        """
        跨平台执行脚本文件
        
        Args:
            script_path: 脚本文件路径
            args: 脚本参数
            cwd: 工作目录
            
        Returns:
            Dict: 包含stdout, stderr, returncode的字典
        """
        script_path = PlatformUtils.normalize_path(script_path)
        
        if not os.path.exists(script_path):
            return {
                'stdout': '',
                'stderr': f'Script not found: {script_path}',
                'returncode': -1,
                'success': False
            }
        
        # 根据文件扩展名确定执行方式
        ext = os.path.splitext(script_path)[1].lower()
        
        if PlatformUtils.is_windows():
            # Windows系统
            if ext == '.bat' or ext == '.cmd':
                command = [script_path]
            elif ext == '.ps1':
                command = ['powershell', '-ExecutionPolicy', 'Bypass', '-File', script_path]
            elif ext == '.sh':
                # Windows上可能需要Git Bash或WSL
                if shutil.which('bash'):
                    command = ['bash', script_path]
                else:
                    return {
                        'stdout': '',
                        'stderr': 'Bash not found on Windows. Please install Git Bash or WSL.',
                        'returncode': -1,
                        'success': False
                    }
            elif ext == '.py':
                command = [sys.executable, script_path]
            else:
                command = [script_path]
        else:
            # Unix系统（MacOS和Linux）
            if ext == '.sh':
                command = ['bash', script_path]
            elif ext == '.py':
                command = [sys.executable, script_path]
            else:
                # 尝试直接执行
                command = [script_path]
        
        # 添加参数
        if args:
            command.extend(args)
        
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True
            )
            
            return {
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode,
                'success': result.returncode == 0
            }
        except Exception as e:
            logger.error(f"Error running script: {str(e)}")
            return {
                'stdout': '',
                'stderr': str(e),
                'returncode': -1,
                'success': False
            }
    
    @staticmethod
    def get_python_executable() -> str:
        """
        获取Python可执行文件路径
        
        Returns:
            str: Python解释器路径
        """
        return sys.executable
    
    @staticmethod
    def check_dependencies(dependencies: List[str]) -> Dict[str, bool]:
        """
        检查系统依赖是否可用
        
        Args:
            dependencies: 依赖项列表（命令名）
            
        Returns:
            Dict: 依赖项名称到可用性的映射
        """
        result = {}
        for dep in dependencies:
            result[dep] = shutil.which(dep) is not None
        return result
    
    @staticmethod
    def get_env_var(name: str, default: Optional[str] = None) -> Optional[str]:
        """
        跨平台获取环境变量
        
        Args:
            name: 环境变量名
            default: 默认值
            
        Returns:
            str: 环境变量值
        """
        return os.environ.get(name, default)
    
    @staticmethod
    def set_env_var(name: str, value: str) -> None:
        """
        跨平台设置环境变量
        
        Args:
            name: 环境变量名
            value: 值
        """
        os.environ[name] = value
    
    @staticmethod
    def get_home_dir() -> str:
        """
        获取用户主目录
        
        Returns:
            str: 主目录路径
        """
        return str(Path.home())
    
    @staticmethod
    def get_temp_dir() -> str:
        """
        获取临时目录
        
        Returns:
            str: 临时目录路径
        """
        import tempfile
        return tempfile.gettempdir()
    
    @staticmethod
    def copy_file(src: str, dst: str) -> bool:
        """
        跨平台复制文件
        
        Args:
            src: 源文件路径
            dst: 目标文件路径
            
        Returns:
            bool: 是否成功
        """
        try:
            shutil.copy2(src, dst)
            return True
        except Exception as e:
            logger.error(f"Error copying file: {str(e)}")
            return False
    
    @staticmethod
    def copy_dir(src: str, dst: str) -> bool:
        """
        跨平台复制目录
        
        Args:
            src: 源目录路径
            dst: 目标目录路径
            
        Returns:
            bool: 是否成功
        """
        try:
            shutil.copytree(src, dst, dirs_exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error copying directory: {str(e)}")
            return False
    
    @staticmethod
    def remove_file(path: str) -> bool:
        """
        跨平台删除文件
        
        Args:
            path: 文件路径
            
        Returns:
            bool: 是否成功
        """
        try:
            Path(path).unlink(missing_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error removing file: {str(e)}")
            return False
    
    @staticmethod
    def remove_dir(path: str) -> bool:
        """
        跨平台删除目录
        
        Args:
            path: 目录路径
            
        Returns:
            bool: 是否成功
        """
        try:
            shutil.rmtree(path, ignore_errors=True)
            return True
        except Exception as e:
            logger.error(f"Error removing directory: {str(e)}")
            return False
    
    @staticmethod
    def list_files(directory: str, pattern: str = "*") -> List[str]:
        """
        列出目录中的文件
        
        Args:
            directory: 目录路径
            pattern: 文件模式（支持通配符）
            
        Returns:
            List[str]: 文件路径列表
        """
        try:
            p = Path(directory)
            if not p.exists():
                return []
            return [str(f) for f in p.glob(pattern) if f.is_file()]
        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            return []
    
    @staticmethod
    def get_file_size(path: str) -> int:
        """
        获取文件大小
        
        Args:
            path: 文件路径
            
        Returns:
            int: 文件大小（字节）
        """
        try:
            return Path(path).stat().st_size
        except:
            return 0
    
    @staticmethod
    def file_exists(path: str) -> bool:
        """
        检查文件是否存在
        
        Args:
            path: 文件路径
            
        Returns:
            bool: 是否存在
        """
        return Path(path).is_file()
    
    @staticmethod
    def dir_exists(path: str) -> bool:
        """
        检查目录是否存在
        
        Args:
            path: 目录路径
            
        Returns:
            bool: 是否存在
        """
        return Path(path).is_dir()


# 便捷函数
def normalize_path(path: str) -> str:
    """规范化路径"""
    return PlatformUtils.normalize_path(path)


def join_paths(*paths: str) -> str:
    """拼接路径"""
    return PlatformUtils.join_paths(*paths)


def ensure_dir(directory: str) -> None:
    """确保目录存在"""
    PlatformUtils.ensure_dir(directory)


def run_command(command: str, **kwargs) -> Dict[str, Any]:
    """执行命令"""
    return PlatformUtils.run_command(command, **kwargs)


def run_script(script_path: str, **kwargs) -> Dict[str, Any]:
    """执行脚本"""
    return PlatformUtils.run_script(script_path, **kwargs)

