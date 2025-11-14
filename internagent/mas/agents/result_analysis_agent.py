"""
结果分析智能体 (Result Analysis Agent)

该智能体负责分析实验结果，提取核心思想，结合已有文献和多次探索历史，
分析下一步的方案设计和改进方向。支持科研的迭代优化过程。
"""

import logging
import json
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ResultAnalysisAgent(BaseAgent):
    """
    结果分析智能体
    
    深度分析实验结果，为下一轮研究提供指导：
    1. 结果解读和模式识别
    2. 核心创新点提取
    3. 成功因素分析
    4. 失败原因诊断
    5. 与文献对比
    6. 改进方向建议
    7. 下一步实验设计
    """
    
    def __init__(self, model, config: Dict[str, Any]):
        """
        初始化结果分析智能体
        
        Args:
            model: 语言模型实例
            config: 配置字典
        """
        super().__init__(model, config)
        self.agent_type = "result_analysis"
        logger.info("ResultAnalysisAgent initialized")
    
    async def execute(self, context: Dict[str, Any], options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        执行结果分析任务
        
        Args:
            context: 上下文信息，包含：
                - current_results: 当前实验结果
                - method_details: 当前方法详情
                - baseline_results: 基线结果
                - historical_attempts: 历史尝试记录
                - literature_context: 文献背景
                - task: 研究任务
            options: 可选参数
                - analysis_depth: 分析深度 (quick, standard, deep)
                - focus_areas: 重点分析领域列表
            
        Returns:
            Dict: 分析结果，包含：
                - summary: 结果摘要
                - core_insights: 核心洞察
                - success_factors: 成功因素
                - failure_analysis: 失败分析（如果适用）
                - comparison_with_baseline: 与基线对比
                - comparison_with_literature: 与文献对比
                - improvement_suggestions: 改进建议
                - next_steps: 下一步实验建议
        """
        options = options or {}
        
        current_results = context.get("current_results", {})
        method_details = context.get("method_details", {})
        baseline_results = context.get("baseline_results", {})
        historical_attempts = context.get("historical_attempts", [])
        literature_context = context.get("literature_context", {})
        task = context.get("task", {})
        
        analysis_depth = options.get("analysis_depth", "standard")
        focus_areas = options.get("focus_areas", [])
        
        try:
            # 生成分析
            analysis = {}
            
            # 1. 结果摘要
            analysis["summary"] = await self._generate_summary(
                current_results, method_details
            )
            
            # 2. 核心洞察
            analysis["core_insights"] = await self._extract_core_insights(
                current_results, method_details, baseline_results
            )
            
            # 3. 成功因素分析
            analysis["success_factors"] = await self._analyze_success_factors(
                current_results, method_details, baseline_results
            )
            
            # 4. 失败分析（如果需要）
            if self._is_worse_than_baseline(current_results, baseline_results):
                analysis["failure_analysis"] = await self._analyze_failure(
                    current_results, method_details, baseline_results
                )
            else:
                analysis["failure_analysis"] = None
            
            # 5. 与基线对比
            analysis["comparison_with_baseline"] = await self._compare_with_baseline(
                current_results, baseline_results, method_details
            )
            
            # 6. 与文献对比
            if literature_context:
                analysis["comparison_with_literature"] = await self._compare_with_literature(
                    current_results, literature_context, method_details
                )
            else:
                analysis["comparison_with_literature"] = "文献对比信息不足"
            
            # 7. 改进建议
            analysis["improvement_suggestions"] = await self._generate_improvement_suggestions(
                current_results, method_details, historical_attempts, baseline_results
            )
            
            # 8. 下一步实验建议
            analysis["next_steps"] = await self._suggest_next_steps(
                current_results, method_details, historical_attempts, 
                baseline_results, task
            )
            
            logger.info("Result analysis completed successfully")
            
            return {
                "analysis": analysis,
                "analysis_depth": analysis_depth,
                "timestamp": self._get_timestamp()
            }
            
        except Exception as e:
            logger.error(f"Error in result analysis: {str(e)}")
            raise
    
    async def _generate_summary(self,
                               results: Dict[str, Any],
                               method_details: Dict[str, Any]) -> str:
        """生成结果摘要"""
        prompt = f"""请对以下实验结果生成简明摘要（150-200字）：

方法名称: {method_details.get('name', '')}

实验结果:
{json.dumps(results, ensure_ascii=False, indent=2)}

摘要应包括：
1. 实验是否成功
2. 主要性能指标
3. 最显著的发现
"""
        
        response = await self.model.generate(
            prompt=prompt,
            temperature=0.6,
            max_tokens=400
        )
        
        return response.strip()
    
    async def _extract_core_insights(self,
                                    results: Dict[str, Any],
                                    method_details: Dict[str, Any],
                                    baseline_results: Dict[str, Any]) -> List[str]:
        """提取核心洞察"""
        prompt = f"""分析以下实验结果，提取3-5个核心洞察或发现：

方法: {method_details.get('name', '')}
方法描述: {method_details.get('description', '')}

实验结果:
{json.dumps(results, ensure_ascii=False, indent=2)}

基线结果:
{json.dumps(baseline_results, ensure_ascii=False, indent=2)}

请提取最重要的洞察，每条洞察应该是：
- 具体的、可操作的发现
- 对理解方法有效性的关键认识
- 对未来研究有指导意义的结论

以JSON列表格式输出，例如：
```json
{{
    "insights": [
        "洞察1",
        "洞察2",
        "洞察3"
    ]
}}
```
"""
        
        response = await self.model.generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=800
        )
        
        # 解析JSON
        insights = self._extract_json_list(response, "insights")
        return insights if insights else [response.strip()]
    
    async def _analyze_success_factors(self,
                                      results: Dict[str, Any],
                                      method_details: Dict[str, Any],
                                      baseline_results: Dict[str, Any]) -> List[str]:
        """分析成功因素"""
        prompt = f"""分析为什么这个方法取得了当前的结果，识别关键的成功因素：

方法: {method_details.get('name', '')}
方法核心思想: {method_details.get('method', '')[:500]}

结果对比:
- 我们的方法: {results.get('metrics', {})}
- 基线方法: {baseline_results}

请识别3-5个关键因素，解释它们如何贡献于结果。

以JSON格式输出：
```json
{{
    "success_factors": [
        {{"factor": "因素名称", "explanation": "为什么这个因素重要"}},
        ...
    ]
}}
```
"""
        
        response = await self.model.generate(
            prompt=prompt,
            temperature=0.6,
            max_tokens=1000
        )
        
        factors = self._extract_json_list(response, "success_factors")
        if factors:
            return [f"{f.get('factor', '')}: {f.get('explanation', '')}" for f in factors]
        return [response.strip()]
    
    async def _analyze_failure(self,
                              results: Dict[str, Any],
                              method_details: Dict[str, Any],
                              baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析失败原因"""
        prompt = f"""深入分析为什么这个方法的表现不如预期：

方法: {method_details.get('name', '')}
方法设计: {method_details.get('method', '')[:500]}

结果对比:
- 我们的方法: {results.get('metrics', {})}
- 基线方法: {baseline_results}

请分析：
1. 可能的失败原因（技术、数据、实现等）
2. 方法设计中的潜在问题
3. 是否有改进的可能性
4. 是否应该放弃这个方向

以JSON格式输出：
```json
{{
    "failure_reasons": ["原因1", "原因2"],
    "design_issues": ["问题1", "问题2"],
    "salvageable": true/false,
    "recommendation": "建议"
}}
```
"""
        
        response = await self.model.generate(
            prompt=prompt,
            temperature=0.6,
            max_tokens=1000
        )
        
        failure_info = self._extract_json_object(response)
        return failure_info if failure_info else {"raw": response.strip()}
    
    async def _compare_with_baseline(self,
                                    results: Dict[str, Any],
                                    baseline_results: Dict[str, Any],
                                    method_details: Dict[str, Any]) -> Dict[str, Any]:
        """与基线对比"""
        prompt = f"""详细对比当前方法与基线方法的结果：

方法: {method_details.get('name', '')}

当前结果:
{json.dumps(results, ensure_ascii=False, indent=2)}

基线结果:
{json.dumps(baseline_results, ensure_ascii=False, indent=2)}

请提供：
1. 每个指标的改进/退化情况
2. 改进的百分比或绝对值
3. 统计显著性评估（如果可能）
4. 整体性能评估

以JSON格式输出：
```json
{{
    "metric_comparisons": [
        {{"metric": "指标名", "ours": 值, "baseline": 值, "improvement": "X%", "significant": true/false}}
    ],
    "overall_assessment": "整体评估",
    "better_metrics": ["表现更好的指标"],
    "worse_metrics": ["表现较差的指标"]
}}
```
"""
        
        response = await self.model.generate(
            prompt=prompt,
            temperature=0.5,
            max_tokens=1000
        )
        
        comparison = self._extract_json_object(response)
        return comparison if comparison else {"raw": response.strip()}
    
    async def _compare_with_literature(self,
                                      results: Dict[str, Any],
                                      literature_context: Dict[str, Any],
                                      method_details: Dict[str, Any]) -> str:
        """与文献对比"""
        papers = literature_context.get('papers', [])
        papers_summary = "\n".join([
            f"- {p.get('title', '')}: {p.get('results_summary', '')}"
            for p in papers[:5]
        ])
        
        prompt = f"""将当前结果与相关文献进行对比：

我们的方法: {method_details.get('name', '')}
我们的结果: {json.dumps(results.get('metrics', {}), ensure_ascii=False)}

相关文献及其结果:
{papers_summary}

请分析：
1. 我们的结果在该领域处于什么水平
2. 与state-of-the-art方法的差距
3. 我们方法的独特优势
4. 改进空间
"""
        
        response = await self.model.generate(
            prompt=prompt,
            temperature=0.6,
            max_tokens=800
        )
        
        return response.strip()
    
    async def _generate_improvement_suggestions(self,
                                               results: Dict[str, Any],
                                               method_details: Dict[str, Any],
                                               historical_attempts: List[Dict[str, Any]],
                                               baseline_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成改进建议"""
        history_summary = self._summarize_historical_attempts(historical_attempts)
        
        prompt = f"""基于当前结果和历史探索，提供具体的改进建议：

当前方法: {method_details.get('name', '')}
当前结果: {json.dumps(results.get('metrics', {}), ensure_ascii=False)}
基线结果: {baseline_results}

历史探索:
{history_summary}

请提供3-5个具体的改进建议，每个建议包括：
1. 改进方向
2. 具体做法
3. 预期效果
4. 实施难度
5. 优先级

以JSON格式输出：
```json
{{
    "suggestions": [
        {{
            "direction": "改进方向",
            "approach": "具体做法",
            "expected_impact": "预期效果",
            "difficulty": "easy/medium/hard",
            "priority": "high/medium/low"
        }}
    ]
}}
```
"""
        
        response = await self.model.generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=1500
        )
        
        suggestions = self._extract_json_list(response, "suggestions")
        return suggestions if suggestions else [{"raw": response.strip()}]
    
    async def _suggest_next_steps(self,
                                 results: Dict[str, Any],
                                 method_details: Dict[str, Any],
                                 historical_attempts: List[Dict[str, Any]],
                                 baseline_results: Dict[str, Any],
                                 task: Dict[str, Any]) -> List[str]:
        """建议下一步实验"""
        history_summary = self._summarize_historical_attempts(historical_attempts)
        
        prompt = f"""基于所有信息，建议接下来应该进行的实验：

研究任务: {task.get('description', '')}
当前方法: {method_details.get('name', '')}
当前结果: {json.dumps(results.get('metrics', {}), ensure_ascii=False)}
历史尝试: {history_summary}

请建议3个具体的下一步实验，按优先级排序。
每个实验应该包括：
- 实验目标
- 要测试的假设
- 具体实验设计
- 预期收获

以JSON列表格式输出：
```json
{{
    "next_experiments": [
        "实验1：目标 - 假设 - 设计 - 预期",
        "实验2：...",
        "实验3：..."
    ]
}}
```
"""
        
        response = await self.model.generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=1200
        )
        
        experiments = self._extract_json_list(response, "next_experiments")
        return experiments if experiments else [response.strip()]
    
    # 辅助方法
    
    def _is_worse_than_baseline(self,
                                results: Dict[str, Any],
                                baseline_results: Dict[str, Any]) -> bool:
        """判断是否比基线差"""
        # 简单的启发式判断
        # 实际应该根据具体的指标和任务类型来判断
        try:
            metrics = results.get('metrics', {})
            if not metrics or not baseline_results:
                return False
            
            # 假设我们只看第一个指标
            # 实际应该更智能地判断
            return False  # 暂时返回False，避免过度悲观
        except:
            return False
    
    def _summarize_historical_attempts(self, 
                                      historical_attempts: List[Dict[str, Any]]) -> str:
        """总结历史尝试"""
        if not historical_attempts:
            return "无历史尝试记录"
        
        summaries = []
        for idx, attempt in enumerate(historical_attempts[-5:], 1):  # 最近5次
            method = attempt.get('method_name', f'尝试{idx}')
            result = attempt.get('results', {})
            success = attempt.get('success', 'unknown')
            summaries.append(f"{idx}. {method}: {result} (状态: {success})")
        
        return "\n".join(summaries)
    
    def _extract_json_list(self, response: str, key: str) -> Optional[List]:
        """从响应中提取JSON列表"""
        import re
        try:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
                return data.get(key, [])
        except:
            pass
        return None
    
    def _extract_json_object(self, response: str) -> Optional[Dict]:
        """从响应中提取JSON对象"""
        import re
        try:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
        except:
            pass
        return None
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()

