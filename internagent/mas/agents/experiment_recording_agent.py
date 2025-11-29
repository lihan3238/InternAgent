"""
实验记录智能体 (Experiment Recording Agent)

该智能体负责记录实验执行过程中的所有重要信息，包括：
- 实验参数配置
- 中间结果和日志
- 性能指标变化
- 遇到的问题和解决方案
- 代码变更历史
"""

import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ExperimentRecordingAgent(BaseAgent):
    """
    实验记录智能体
    
    负责结构化记录实验过程，便于后续分析和复现：
    1. 实验配置记录
    2. 执行日志收集
    3. 结果数据记录
    4. 异常情况记录
    5. 代码版本追踪
    """
    
    def __init__(self, model, config: Dict[str, Any]):
        """
        初始化实验记录智能体
        
        Args:
            model: 语言模型实例
            config: 配置字典
        """
        super().__init__(model, config)
        self.agent_type = "experiment_recording"
        logger.info("ExperimentRecordingAgent initialized")
    
    async def execute(self, context: Dict[str, Any], options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        执行实验记录任务
        
        Args:
            context: 上下文信息，包含：
                - experiment_id: 实验ID
                - action: 记录操作类型 (start, log, result, error, complete)
                - data: 要记录的数据
                - work_dir: 工作目录
            options: 可选参数
            
        Returns:
            Dict: 记录结果
                - success: 是否成功
                - record_file: 记录文件路径
                - summary: 摘要信息
        """
        options = options or {}
        
        experiment_id = context.get("experiment_id", "unknown")
        action = context.get("action", "log")
        data = context.get("data", {})
        work_dir = context.get("work_dir", ".")
        
        # 创建记录目录
        record_dir = os.path.join(work_dir, "experiment_records")
        os.makedirs(record_dir, exist_ok=True)
        
        try:
            if action == "start":
                result = self._record_experiment_start(experiment_id, data, record_dir)
            elif action == "log":
                result = self._record_log_entry(experiment_id, data, record_dir)
            elif action == "result":
                result = self._record_result(experiment_id, data, record_dir)
            elif action == "error":
                result = self._record_error(experiment_id, data, record_dir)
            elif action == "complete":
                result = self._record_completion(experiment_id, data, record_dir)
            else:
                result = self._record_generic(experiment_id, action, data, record_dir)
            
            logger.info(f"Recorded {action} for experiment {experiment_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error in experiment recording: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_record_file_path(self, experiment_id: str, record_dir: str) -> str:
        """获取记录文件路径"""
        return os.path.join(record_dir, f"experiment_{experiment_id}.jsonl")
    
    def _append_record(self, record_file: str, entry: Dict[str, Any]) -> None:
        """追加记录到文件"""
        entry["timestamp"] = datetime.now().isoformat()
        with open(record_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    def _record_experiment_start(self, 
                                experiment_id: str, 
                                data: Dict[str, Any],
                                record_dir: str) -> Dict[str, Any]:
        """记录实验开始"""
        record_file = self._get_record_file_path(experiment_id, record_dir)
        
        entry = {
            "type": "start",
            "experiment_id": experiment_id,
            "configuration": data.get("configuration", {}),
            "method_details": data.get("method_details", {}),
            "baseline_results": data.get("baseline_results", {}),
            "environment": data.get("environment", {})
        }
        
        self._append_record(record_file, entry)
        
        return {
            "success": True,
            "record_file": record_file,
            "summary": f"Experiment {experiment_id} started"
        }
    
    def _record_log_entry(self,
                         experiment_id: str,
                         data: Dict[str, Any],
                         record_dir: str) -> Dict[str, Any]:
        """记录日志条目"""
        record_file = self._get_record_file_path(experiment_id, record_dir)
        
        entry = {
            "type": "log",
            "experiment_id": experiment_id,
            "message": data.get("message", ""),
            "level": data.get("level", "INFO"),
            "context": data.get("context", {})
        }
        
        self._append_record(record_file, entry)
        
        return {
            "success": True,
            "record_file": record_file,
            "summary": "Log entry recorded"
        }
    
    def _record_result(self,
                      experiment_id: str,
                      data: Dict[str, Any],
                      record_dir: str) -> Dict[str, Any]:
        """记录实验结果"""
        record_file = self._get_record_file_path(experiment_id, record_dir)
        
        entry = {
            "type": "result",
            "experiment_id": experiment_id,
            "metrics": data.get("metrics", {}),
            "comparison_with_baseline": data.get("comparison", {}),
            "visualizations": data.get("visualizations", []),
            "artifacts": data.get("artifacts", [])
        }
        
        self._append_record(record_file, entry)
        
        return {
            "success": True,
            "record_file": record_file,
            "summary": f"Results recorded: {list(data.get('metrics', {}).keys())}"
        }
    
    def _record_error(self,
                     experiment_id: str,
                     data: Dict[str, Any],
                     record_dir: str) -> Dict[str, Any]:
        """记录错误信息"""
        record_file = self._get_record_file_path(experiment_id, record_dir)
        
        entry = {
            "type": "error",
            "experiment_id": experiment_id,
            "error_type": data.get("error_type", "Unknown"),
            "error_message": data.get("error_message", ""),
            "stack_trace": data.get("stack_trace", ""),
            "attempted_solution": data.get("solution", ""),
            "resolved": data.get("resolved", False)
        }
        
        self._append_record(record_file, entry)
        
        return {
            "success": True,
            "record_file": record_file,
            "summary": f"Error recorded: {data.get('error_type', 'Unknown')}"
        }
    
    def _record_completion(self,
                          experiment_id: str,
                          data: Dict[str, Any],
                          record_dir: str) -> Dict[str, Any]:
        """记录实验完成"""
        record_file = self._get_record_file_path(experiment_id, record_dir)
        
        entry = {
            "type": "complete",
            "experiment_id": experiment_id,
            "status": data.get("status", "completed"),
            "final_results": data.get("final_results", {}),
            "duration": data.get("duration", ""),
            "conclusions": data.get("conclusions", "")
        }
        
        self._append_record(record_file, entry)
        
        # 生成实验摘要
        summary = self._generate_experiment_summary(record_file)
        
        # 保存摘要
        summary_file = os.path.join(record_dir, f"experiment_{experiment_id}_summary.json")
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        return {
            "success": True,
            "record_file": record_file,
            "summary_file": summary_file,
            "summary": summary
        }
    
    def _record_generic(self,
                       experiment_id: str,
                       action: str,
                       data: Dict[str, Any],
                       record_dir: str) -> Dict[str, Any]:
        """记录通用信息"""
        record_file = self._get_record_file_path(experiment_id, record_dir)
        
        entry = {
            "type": action,
            "experiment_id": experiment_id,
            "data": data
        }
        
        self._append_record(record_file, entry)
        
        return {
            "success": True,
            "record_file": record_file,
            "summary": f"Generic record added: {action}"
        }
    
    def _generate_experiment_summary(self, record_file: str) -> Dict[str, Any]:
        """生成实验摘要"""
        if not os.path.exists(record_file):
            return {}
        
        summary = {
            "total_entries": 0,
            "start_time": None,
            "end_time": None,
            "errors_count": 0,
            "results_count": 0,
            "final_metrics": {},
            "key_events": []
        }
        
        try:
            with open(record_file, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    summary["total_entries"] += 1
                    
                    if entry.get("type") == "start":
                        summary["start_time"] = entry.get("timestamp")
                    elif entry.get("type") == "complete":
                        summary["end_time"] = entry.get("timestamp")
                        summary["final_metrics"] = entry.get("final_results", {})
                    elif entry.get("type") == "error":
                        summary["errors_count"] += 1
                    elif entry.get("type") == "result":
                        summary["results_count"] += 1
                        
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
        
        return summary

