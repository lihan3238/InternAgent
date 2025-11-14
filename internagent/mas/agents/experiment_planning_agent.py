"""
实验流程规划智能体 (Experiment Planning Agent)

该智能体负责根据研究方案设计详细的实验执行流程，确保实验步骤有序、完整、可执行。
包括：数据准备、环境配置、实验步骤、验证方法、资源需求等。
"""

import logging
from typing import Dict, Any, List
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ExperimentPlanningAgent(BaseAgent):
    """
    实验流程规划智能体
    
    根据研究方案生成详细的实验执行计划，包括：
    1. 实验环境配置要求
    2. 数据准备步骤
    3. 代码实现步骤（分解为可执行的子任务）
    4. 实验执行流程
    5. 结果验证方法
    6. 资源需求评估
    """
    
    def __init__(self, model, config: Dict[str, Any]):
        """
        初始化实验流程规划智能体
        
        Args:
            model: 语言模型实例
            config: 配置字典
        """
        super().__init__(model, config)
        self.agent_type = "experiment_planning"
        logger.info("ExperimentPlanningAgent initialized")
    
    async def execute(self, context: Dict[str, Any], options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        执行实验流程规划任务
        
        Args:
            context: 上下文信息，包含：
                - method_details: 研究方案详情
                - baseline_code: 基线代码路径
                - task: 任务描述
                - constraints: 资源约束
            options: 可选参数
            
        Returns:
            Dict: 包含实验执行计划的字典
                - experiment_plan: 完整的实验计划
                - steps: 具体步骤列表
                - resource_requirements: 资源需求
                - estimated_time: 预计执行时间
        """
        options = options or {}
        
        method_details = context.get("method_details", {})
        baseline_code = context.get("baseline_code", "")
        task = context.get("task", {})
        constraints = context.get("constraints", [])
        
        # 构建提示词
        prompt = self._build_planning_prompt(method_details, baseline_code, task, constraints)
        
        try:
            # 调用语言模型
            response = await self.model.generate(
                prompt=prompt,
                temperature=self.config.get("temperature", 0.3),
                max_tokens=self.config.get("max_tokens", 4096)
            )
            
            # 解析响应
            result = self._parse_planning_response(response)
            
            logger.info(f"Generated experiment plan with {len(result.get('steps', []))} steps")
            return result
            
        except Exception as e:
            logger.error(f"Error in experiment planning: {str(e)}")
            raise
    
    def _build_planning_prompt(self, 
                               method_details: Dict[str, Any],
                               baseline_code: str,
                               task: Dict[str, Any],
                               constraints: List[str]) -> str:
        """
        构建实验规划的提示词
        
        Args:
            method_details: 方法详情
            baseline_code: 基线代码路径
            task: 任务描述
            constraints: 约束条件
            
        Returns:
            str: 完整的提示词
        """
        method_text = method_details.get("method", "")
        method_name = method_details.get("name", "")
        method_description = method_details.get("description", "")
        task_description = task.get("description", "")
        
        constraints_text = "\n".join([f"- {c}" for c in constraints]) if constraints else "无特殊约束"
        
        prompt = f"""你是一位经验丰富的实验规划专家。请根据以下研究方案，制定详细的实验执行计划。

# 研究任务
{task_description}

# 研究方案
**方法名称**: {method_name}

**方法描述**: {method_description}

**详细方法**:
{method_text}

# 基线代码路径
{baseline_code if baseline_code else "无基线代码，需从头实现"}

# 资源约束
{constraints_text}

# 任务要求
请制定一个详细的实验执行计划，包括以下内容：

## 1. 环境配置
- 所需的软件包和依赖
- 环境变量配置
- 硬件要求（CPU/GPU/内存等）

## 2. 数据准备
- 数据集准备步骤
- 数据预处理要求
- 数据验证方法

## 3. 代码实现步骤
将整个实现分解为可执行的子任务，每个子任务应该：
- 明确说明要实现什么功能
- 指出需要修改或创建哪些文件
- 给出具体的实现建议
- 说明与其他部分的接口

## 4. 实验执行流程
- 训练步骤
- 验证步骤
- 测试步骤
- 结果保存方法

## 5. 结果验证
- 评估指标
- 与基线对比方法
- 结果可视化建议

## 6. 资源需求评估
- 预计所需计算资源
- 预计执行时间
- 潜在的性能瓶颈

请以结构化的JSON格式输出，格式如下：
```json
{{
    "experiment_plan": {{
        "overview": "实验计划概述",
        "environment": {{
            "dependencies": ["依赖1", "依赖2"],
            "hardware": {{"cpu": "...", "gpu": "...", "memory": "..."}},
            "environment_vars": {{"VAR1": "value1"}}
        }},
        "data_preparation": {{
            "steps": ["步骤1", "步骤2"],
            "validation": "数据验证方法"
        }},
        "implementation_steps": [
            {{
                "step_id": 1,
                "task": "任务描述",
                "files_to_modify": ["文件1", "文件2"],
                "implementation_guide": "实现指南",
                "dependencies": []
            }}
        ],
        "execution_flow": {{
            "training": ["训练步骤1", "训练步骤2"],
            "validation": ["验证步骤"],
            "testing": ["测试步骤"]
        }},
        "validation": {{
            "metrics": ["指标1", "指标2"],
            "baseline_comparison": "对比方法",
            "visualization": "可视化建议"
        }},
        "resource_requirements": {{
            "compute": "计算资源需求",
            "estimated_time": "预计时间",
            "bottlenecks": ["潜在瓶颈"]
        }}
    }}
}}
```

请确保计划详细、可执行、并考虑了资源约束。
"""
        return prompt
    
    def _parse_planning_response(self, response: str) -> Dict[str, Any]:
        """
        解析模型响应，提取实验计划
        
        Args:
            response: 模型的原始响应
            
        Returns:
            Dict: 解析后的实验计划
        """
        import json
        import re
        
        # 尝试从响应中提取JSON
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                plan_data = json.loads(json_match.group(1))
                experiment_plan = plan_data.get("experiment_plan", {})
                
                return {
                    "experiment_plan": experiment_plan,
                    "steps": experiment_plan.get("implementation_steps", []),
                    "resource_requirements": experiment_plan.get("resource_requirements", {}),
                    "estimated_time": experiment_plan.get("resource_requirements", {}).get("estimated_time", "未知"),
                    "raw_response": response
                }
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {str(e)}")
        
        # 如果JSON解析失败，返回基础结构
        return {
            "experiment_plan": {
                "overview": response[:500] + "..." if len(response) > 500 else response,
                "implementation_steps": []
            },
            "steps": [],
            "resource_requirements": {},
            "estimated_time": "未知",
            "raw_response": response
        }

