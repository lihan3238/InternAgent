"""
研究整合智能体 (Research Integration Agent)

该智能体负责将研究方案、实验过程、实验结果整合成完整的技术报告或论文草稿。
输出格式接近学术论文标准，包括摘要、引言、方法、实验、结果、讨论等部分。
"""

import logging
import os
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ResearchIntegrationAgent(BaseAgent):
    """
    研究整合智能体
    
    将研究的各个阶段整合成完整的技术报告或论文：
    1. 摘要 (Abstract)
    2. 引言 (Introduction)
    3. 相关工作 (Related Work)
    4. 研究动机 (Motivation)
    5. 方法 (Method)
    6. 实验设置 (Experimental Setup)
    7. 结果 (Results)
    8. 讨论 (Discussion)
    9. 结论 (Conclusion)
    10. 参考文献 (References)
    """
    
    def __init__(self, model, config: Dict[str, Any]):
        """
        初始化研究整合智能体
        
        Args:
            model: 语言模型实例
            config: 配置字典
        """
        super().__init__(model, config)
        self.agent_type = "research_integration"
        logger.info("ResearchIntegrationAgent initialized")
    
    async def execute(self, context: Dict[str, Any], options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        执行研究整合任务
        
        Args:
            context: 上下文信息，包含：
                - task: 研究任务描述
                - method_details: 方法详情
                - experiment_records: 实验记录
                - experiment_results: 实验结果
                - literature_survey: 文献综述
                - baseline_results: 基线结果
            options: 可选参数
                - output_format: 输出格式 (markdown, latex, docx)
                - include_code: 是否包含代码片段
            
        Returns:
            Dict: 包含完整报告的字典
                - report_sections: 各章节内容
                - full_report: 完整报告文本
                - report_file: 保存的文件路径
        """
        options = options or {}
        
        task = context.get("task", {})
        method_details = context.get("method_details", {})
        experiment_records = context.get("experiment_records", [])
        experiment_results = context.get("experiment_results", {})
        literature_survey = context.get("literature_survey", {})
        baseline_results = context.get("baseline_results", {})
        
        output_format = options.get("output_format", "markdown")
        include_code = options.get("include_code", False)
        
        try:
            # 生成各个章节
            sections = {}
            
            # 1. 摘要
            sections["abstract"] = await self._generate_abstract(
                task, method_details, experiment_results
            )
            
            # 2. 引言
            sections["introduction"] = await self._generate_introduction(
                task, literature_survey
            )
            
            # 3. 相关工作
            sections["related_work"] = await self._generate_related_work(
                literature_survey
            )
            
            # 4. 研究动机
            sections["motivation"] = await self._generate_motivation(
                task, method_details, baseline_results
            )
            
            # 5. 方法
            sections["method"] = await self._generate_method_section(
                method_details, include_code
            )
            
            # 6. 实验设置
            sections["experimental_setup"] = await self._generate_experimental_setup(
                experiment_records, method_details
            )
            
            # 7. 结果
            sections["results"] = await self._generate_results_section(
                experiment_results, baseline_results
            )
            
            # 8. 讨论
            sections["discussion"] = await self._generate_discussion(
                method_details, experiment_results, baseline_results
            )
            
            # 9. 结论
            sections["conclusion"] = await self._generate_conclusion(
                task, method_details, experiment_results
            )
            
            # 10. 参考文献
            sections["references"] = self._format_references(literature_survey)
            
            # 整合完整报告
            full_report = self._assemble_report(sections, output_format)
            
            # 保存报告
            work_dir = context.get("work_dir", ".")
            report_file = self._save_report(
                full_report, work_dir, output_format
            )
            
            logger.info(f"Generated research report: {report_file}")
            
            return {
                "report_sections": sections,
                "full_report": full_report,
                "report_file": report_file,
                "output_format": output_format
            }
            
        except Exception as e:
            logger.error(f"Error in research integration: {str(e)}")
            raise
    
    async def _generate_abstract(self,
                                 task: Dict[str, Any],
                                 method_details: Dict[str, Any],
                                 results: Dict[str, Any]) -> str:
        """生成摘要"""
        prompt = f"""请为以下研究生成一个学术风格的摘要（200-300字）：

研究任务: {task.get('description', '')}

方法名称: {method_details.get('name', '')}

方法描述: {method_details.get('description', '')}

主要结果: {json.dumps(results.get('metrics', {}), ensure_ascii=False)}

摘要应包括：研究背景、方法创新点、主要实验结果、结论。
"""
        
        response = await self.model.generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.strip()
    
    async def _generate_introduction(self,
                                    task: Dict[str, Any],
                                    literature_survey: Dict[str, Any]) -> str:
        """生成引言"""
        prompt = f"""请为以下研究任务撰写引言部分（500-800字）：

研究任务: {task.get('description', '')}
研究领域: {task.get('domain', '')}
背景信息: {task.get('background', '')}

文献综述摘要: {literature_survey.get('summary', '')}

引言应包括：
1. 研究背景和重要性
2. 当前领域的挑战
3. 研究动机
4. 本文的主要贡献
5. 论文结构概述
"""
        
        response = await self.model.generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.strip()
    
    async def _generate_related_work(self, literature_survey: Dict[str, Any]) -> str:
        """生成相关工作"""
        papers = literature_survey.get('papers', [])
        
        if not papers:
            return "（暂无相关文献）"
        
        papers_text = "\n\n".join([
            f"论文: {p.get('title', '')}\n摘要: {p.get('abstract', '')[:200]}..."
            for p in papers[:10]  # 最多10篇
        ])
        
        prompt = f"""基于以下文献，撰写相关工作部分（400-600字）：

{papers_text}

请组织成逻辑清晰的相关工作综述，包括：
1. 按主题分类的文献回顾
2. 各方法的优缺点
3. 与本研究的关系
"""
        
        response = await self.model.generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.strip()
    
    async def _generate_motivation(self,
                                   task: Dict[str, Any],
                                   method_details: Dict[str, Any],
                                   baseline_results: Dict[str, Any]) -> str:
        """生成研究动机"""
        prompt = f"""请撰写研究动机部分（300-500字）：

研究问题: {task.get('description', '')}

现有基线方法的局限性:
{json.dumps(baseline_results, ensure_ascii=False)}

我们的方法概述: {method_details.get('description', '')}

应包括：
1. 现有方法存在的问题
2. 为什么需要新方法
3. 我们的方法如何解决这些问题
4. 预期的改进
"""
        
        response = await self.model.generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=800
        )
        
        return response.strip()
    
    async def _generate_method_section(self,
                                      method_details: Dict[str, Any],
                                      include_code: bool) -> str:
        """生成方法部分"""
        method_text = method_details.get('method', '')
        method_name = method_details.get('name', '')
        method_desc = method_details.get('description', '')
        
        prompt = f"""请将以下方法描述整理成学术论文的方法部分（800-1200字）：

方法名称: {method_name}

方法概述: {method_desc}

详细方法:
{method_text}

要求：
1. 使用正式的学术语言
2. 清晰的逻辑结构（可分为多个子节）
3. 如有必要，说明数学公式
4. 说明算法流程
5. 指出与现有方法的关键区别
"""
        
        if include_code:
            prompt += "\n6. 包含关键代码片段的伪代码"
        
        response = await self.model.generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=2000
        )
        
        return response.strip()
    
    async def _generate_experimental_setup(self,
                                          experiment_records: List[Dict[str, Any]],
                                          method_details: Dict[str, Any]) -> str:
        """生成实验设置"""
        # 从实验记录中提取配置信息
        config_info = ""
        if experiment_records:
            for record in experiment_records:
                if record.get("type") == "start":
                    config_info = json.dumps(
                        record.get("configuration", {}),
                        ensure_ascii=False,
                        indent=2
                    )
                    break
        
        prompt = f"""请撰写实验设置部分（400-600字）：

方法配置:
{config_info}

应包括：
1. 数据集描述
2. 实验环境（硬件、软件）
3. 超参数设置
4. 评估指标
5. 基线方法
6. 实现细节
"""
        
        response = await self.model.generate(
            prompt=prompt,
            temperature=0.6,
            max_tokens=1000
        )
        
        return response.strip()
    
    async def _generate_results_section(self,
                                       results: Dict[str, Any],
                                       baseline_results: Dict[str, Any]) -> str:
        """生成结果部分"""
        prompt = f"""请撰写实验结果部分（500-800字）：

我们的结果:
{json.dumps(results, ensure_ascii=False, indent=2)}

基线结果:
{json.dumps(baseline_results, ensure_ascii=False, indent=2)}

应包括：
1. 主要结果表格或数值对比
2. 与基线方法的比较
3. 统计显著性分析
4. 可视化结果描述
5. 消融实验结果（如有）
"""
        
        response = await self.model.generate(
            prompt=prompt,
            temperature=0.6,
            max_tokens=1200
        )
        
        return response.strip()
    
    async def _generate_discussion(self,
                                  method_details: Dict[str, Any],
                                  results: Dict[str, Any],
                                  baseline_results: Dict[str, Any]) -> str:
        """生成讨论部分"""
        prompt = f"""请撰写讨论部分（400-600字）：

方法: {method_details.get('name', '')}

结果摘要:
- 我们的方法: {results.get('metrics', {})}
- 基线方法: {baseline_results}

应包括：
1. 结果分析和解释
2. 为什么我们的方法有效
3. 局限性分析
4. 失败案例（如有）
5. 未来改进方向
"""
        
        response = await self.model.generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.strip()
    
    async def _generate_conclusion(self,
                                  task: Dict[str, Any],
                                  method_details: Dict[str, Any],
                                  results: Dict[str, Any]) -> str:
        """生成结论"""
        prompt = f"""请撰写结论部分（200-300字）：

研究任务: {task.get('description', '')}

方法贡献: {method_details.get('description', '')}

主要成果: {results.get('metrics', {})}

应包括：
1. 研究总结
2. 主要贡献
3. 实验结论
4. 未来工作展望
"""
        
        response = await self.model.generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.strip()
    
    def _format_references(self, literature_survey: Dict[str, Any]) -> str:
        """格式化参考文献"""
        papers = literature_survey.get('papers', [])
        
        if not papers:
            return "（暂无参考文献）"
        
        references = []
        for idx, paper in enumerate(papers, 1):
            title = paper.get('title', '')
            authors = paper.get('authors', [])
            year = paper.get('year', '')
            venue = paper.get('venue', '')
            
            # 简单格式化
            author_str = ', '.join(authors[:3])
            if len(authors) > 3:
                author_str += ' et al.'
            
            ref = f"[{idx}] {author_str}. {title}. {venue}, {year}."
            references.append(ref)
        
        return "\n".join(references)
    
    def _assemble_report(self, sections: Dict[str, str], output_format: str) -> str:
        """组装完整报告"""
        if output_format == "markdown":
            return self._assemble_markdown(sections)
        elif output_format == "latex":
            return self._assemble_latex(sections)
        else:
            return self._assemble_markdown(sections)
    
    def _assemble_markdown(self, sections: Dict[str, str]) -> str:
        """组装Markdown格式报告"""
        report = f"""# 研究报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 摘要 (Abstract)

{sections.get('abstract', '')}

---

## 1. 引言 (Introduction)

{sections.get('introduction', '')}

---

## 2. 相关工作 (Related Work)

{sections.get('related_work', '')}

---

## 3. 研究动机 (Motivation)

{sections.get('motivation', '')}

---

## 4. 方法 (Method)

{sections.get('method', '')}

---

## 5. 实验设置 (Experimental Setup)

{sections.get('experimental_setup', '')}

---

## 6. 结果 (Results)

{sections.get('results', '')}

---

## 7. 讨论 (Discussion)

{sections.get('discussion', '')}

---

## 8. 结论 (Conclusion)

{sections.get('conclusion', '')}

---

## 参考文献 (References)

{sections.get('references', '')}

"""
        return report
    
    def _assemble_latex(self, sections: Dict[str, str]) -> str:
        """组装LaTeX格式报告"""
        # 简单的LaTeX模板
        report = r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{graphicx}

\title{研究报告}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
""" + sections.get('abstract', '') + r"""
\end{abstract}

\section{引言}
""" + sections.get('introduction', '') + r"""

\section{相关工作}
""" + sections.get('related_work', '') + r"""

\section{研究动机}
""" + sections.get('motivation', '') + r"""

\section{方法}
""" + sections.get('method', '') + r"""

\section{实验设置}
""" + sections.get('experimental_setup', '') + r"""

\section{结果}
""" + sections.get('results', '') + r"""

\section{讨论}
""" + sections.get('discussion', '') + r"""

\section{结论}
""" + sections.get('conclusion', '') + r"""

\begin{thebibliography}{99}
""" + sections.get('references', '') + r"""
\end{thebibliography}

\end{document}
"""
        return report
    
    def _save_report(self, report: str, work_dir: str, output_format: str) -> str:
        """保存报告到文件"""
        # 创建报告目录
        report_dir = os.path.join(work_dir, "research_reports")
        os.makedirs(report_dir, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = "md" if output_format == "markdown" else "tex"
        filename = f"research_report_{timestamp}.{ext}"
        filepath = os.path.join(report_dir, filename)
        
        # 保存文件
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)
        
        return filepath

