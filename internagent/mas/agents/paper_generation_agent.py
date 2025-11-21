import os
import shutil
import subprocess
import json
import time
import logging
import inspect
import re
from datetime import datetime    
from typing import Dict, Any, List, Optional, Set, Tuple

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

DEFAULT_AUTOFORECAST_ROOT = os.path.join("results", "AutoForecast")
DEFAULT_IDEA_RESULT_DIRS: List[str] = []

FIGURE_SECTION_DEFAULT = 'Results'

SECTION_MARKER_MAP = {
    'introduction': ['%%%%%%%%%INTRODUCTION%%%%%%%%%', 'INTRODUCTION'],
    'related': ['%%%%%%%%%RELATED WORK%%%%%%%%%', 'RELATED WORK', 'related work'],
    'background': ['%%%%%%%%%BACKGROUND%%%%%%%%%', 'BACKGROUND'],
    'method': ['%%%%%%%%%METHOD%%%%%%%%%', 'METHOD'],
    'experimental': ['%%%%%%%%%EXPERIMENTAL SETUP%%%%%%%%%', 'EXPERIMENTAL SETUP'],
    'experiments': ['%%%%%%%%%EXPERIMENTS%%%%%%%%%', 'EXPERIMENTS'],
    'conclusion': ['%%%%%%%%%CONCLUSION%%%%%%%%%', 'CONCLUSION'],
}

FIGURE_ALLOWED_SECTION_NORMS = {
    'method',
    'experimentalsetup',
    'experiments',
    'conclusion',
}

ALLOWED_FIGURE_EXTENSIONS = (
    '.png',
    '.jpg',
    '.jpeg',
    '.bmp',
    '.gif',
    '.svg',
    '.pdf',
)


def _default_results_root(task_name: Optional[str] = None) -> str:
    """Resolve results/<task> root based on pipeline-provided task name."""
    name = (task_name or "").strip()
    if not name:
        return DEFAULT_AUTOFORECAST_ROOT
    return os.path.join("results", name)


def _safe_makedirs(path: str) -> None:
    """确保目录存在，若已存在则忽略以避免报错。"""
    os.makedirs(path, exist_ok=True)


def _has_run_subdirs(path: str) -> bool:
    """检查目录下是否存在 run_* 子目录，用于推断实验结果文件夹。"""
    try:
        for entry in os.listdir(path):
            full = os.path.join(path, entry)
            if entry.startswith('run_') and os.path.isdir(full):
                return True
    except Exception:
        return False
    return False


def _discover_idea_result_dirs(iteration_root: str) -> List[str]:
    """在迭代根目录下自动发现包含 run_* 结果的实验子目录。"""
    if not iteration_root or not os.path.isdir(iteration_root):
        return []

    discovered = []
    seen = set()

    def _record(path: str):
        norm = os.path.normpath(path)
        if norm not in seen:
            seen.add(norm)
            discovered.append(path)

    if _has_run_subdirs(iteration_root):
        _record(iteration_root)

    try:
        for entry in os.listdir(iteration_root):
            child = os.path.join(iteration_root, entry)
            if os.path.isdir(child) and _has_run_subdirs(child):
                _record(child)
    except Exception:
        pass

    return discovered


def _infer_figure_semantics(record: Dict[str, Any]) -> Dict[str, str]:
    """根据文件名和 run 信息给出插图放置、宽度与描述提示。"""
    name = (record.get('filename') or record.get('path') or '').lower()
    category = 'general'
    section = FIGURE_SECTION_DEFAULT
    width_cmd = '0.9\\linewidth'
    placement = '[ht]'
    caption_focus = 'Explain axis meaning, highlight peaks/valleys, connect to performance.'

    def _match(tokens: List[str]) -> bool:
        return any(tok in name for tok in tokens)

    if _match(['metric', 'rmse', 'mae', 'score', 'loss', 'curve']) or 'final_info' in name:
        category = 'metrics'
        section = 'Results'
        width_cmd = '0.75\\linewidth'
        placement = '[tb]'
        caption_focus = 'Summarize MAE/RMSE/R2 trends, best run, and performance gaps.'
    elif _match(['predict', 'forecast', 'target', 'vs', 'comparison']):
        category = 'predictions'
        section = 'Results'
        width_cmd = '0.9\\linewidth'
        placement = '[ht]'
        caption_focus = 'Describe prediction vs. target alignment and where errors concentrate.'
    elif _match(['time', 'series', 'sequence']):
        category = 'timeseries'
        section = 'Results'
        width_cmd = '0.95\\linewidth'
        placement = '[hb]'
        caption_focus = 'Emphasize long-span trends, sudden changes, and model response speed.'
    elif _match(['ablation', 'sens', 'sensitivity']):
        category = 'ablation'
        section = 'Experiments'
        width_cmd = '0.78\\linewidth'
        placement = '[t]'
        caption_focus = 'State the ablated component, percent change in metrics, and key finding.'
    elif _match(['arch', 'pipeline', 'overview', 'framework', 'flow']):
        category = 'architecture'
        section = 'Method'
        width_cmd = '0.85\\linewidth'
        placement = '[ht]'
        caption_focus = 'Describe module structure, data flow, and critical components.'

    return {
        'category': category,
        'recommended_section': section,
        'width_cmd': width_cmd,
        'placement': placement,
        'caption_focus': caption_focus,
    }


def _slugify_label(name: str, prefix: str = 'fig') -> str:
    base = os.path.splitext(os.path.basename(name or ''))[0]
    slug = re.sub(r'[^a-z0-9]+', '-', base.lower()).strip('-')
    slug = slug or 'figure'
    return f"{prefix}:{slug}"


def _escape_bib_value(value: Optional[str]) -> str:
    if not value:
        return ''
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
    }
    escaped = value
    for raw, repl in replacements.items():
        escaped = escaped.replace(raw, repl)
    return escaped


class PaperGenerationAgent(BaseAgent):
    """负责生成 LaTeX 论文草稿及 PDF 的智能体。

        可通过配置项控制的内容：
            - template_dir：可选的 LaTeX 模版目录；
            - num_cite_rounds：引用检索轮数，默认 2；
            - n_writeup_reflections：正文反思迭代次数，默认 2；
            - compile_timeout：pdflatex 单次执行超时时间（秒），默认 60。
    """

    def __init__(self, model, config: Dict[str, Any] = None):
        super().__init__(model, config or {})
        self.config = config or {}
        self._default_agent_conf = None

    async def _maybe_await(self, value):
        """若传入对象可等待则进行 await，否则直接返回，用于兼容同步/异步模型封装。"""
        try:
            if inspect.isawaitable(value):
                return await value
        except Exception:
            # Fallback: if inspect can't determine, try awaiting in case it's a coroutine
            pass
        return value

    # --- Initialization / template handling ---------------------------------
    def _init_latex_workspace(self, root_output_dir: str, iteration_suffix: str) -> str:
        """根据迭代信息搭建新的 LaTeX 工作目录，优先复制模版，其次创建最小模版。"""
        latex_root = os.path.join(root_output_dir, f"latex_{iteration_suffix}")
        # Remove old latex folder to avoid interference
        if os.path.exists(latex_root):
            try:
                shutil.rmtree(latex_root)
            except Exception:
                logger.warning(f"Could not remove existing latex dir: {latex_root}")
        _safe_makedirs(latex_root)

        template_dir = self.config.get("template_dir") or os.path.join(os.getcwd(), "blank_icml_latex")
        if os.path.exists(template_dir):
            try:
                shutil.copytree(template_dir, latex_root, dirs_exist_ok=True)
                logger.info(f"Copied LaTeX template from {template_dir} to {latex_root}")
            except Exception as e:
                logger.warning(f"Failed to copy template: {e}; creating minimal template")
                self._create_minimal_template(latex_root)
        else:
            logger.info(f"Template not found at {template_dir}; creating minimal template at {latex_root}")
            self._create_minimal_template(latex_root)

        # Ensure references.bib exists
        bib_path = os.path.join(latex_root, "references.bib")
        if not os.path.exists(bib_path):
            with open(bib_path, "w", encoding="utf-8") as bf:
                bf.write("% references.bib generated by PaperGenerationAgent\n")

        # If the template uses 'template.tex' (as in provided blank_icml_latex),
        # create a 'main.tex' copy so internals expect a consistent filename.
        template_tex = os.path.join(latex_root, 'template.tex')
        main_tex = os.path.join(latex_root, 'main.tex')
        if os.path.exists(template_tex) and not os.path.exists(main_tex):
            try:
                shutil.copy2(template_tex, main_tex)
                logger.info(f"Copied template.tex to main.tex in {latex_root}")
            except Exception:
                logger.warning("Failed to copy template.tex to main.tex")

        return latex_root

    def _create_minimal_template(self, latex_root: str) -> None:
        """在缺失标准模版时生成最小可编译的 main.tex，用于兜底。"""
        main_tex = os.path.join(latex_root, "main.tex")
        tex_content = r"""\\documentclass{article}
\\usepackage[utf8]{inputenc}
\\usepackage{graphicx}
    \usepackage{float}
    \usepackage{placeins}
\\begin{document}
\\title{<TITLE>}
\\author{InternAgent}
\\maketitle
\\begin{abstract}
<ABSTRACT>
\\end{abstract}

\\section{Introduction}
\\section{Method}
\\section{Experiments}
\\section{Conclusion}

\\bibliographystyle{plain}
\\bibliography{references}
\\end{document}
"""
        with open(main_tex, "w", encoding="utf-8") as mf:
            mf.write(tex_content)

    # --- Citation search (best-effort, optional external API) ---------------
    def _search_for_papers(self, keywords: List[str], max_rounds: int = 5) -> List[Dict[str, Any]]:
        """当存在 API Key 时调用 Semantic Scholar 多轮检索文献，返回包含标题/作者/年份等信息的列表。"""
        try:
            import requests
        except Exception:
            logger.info("requests not available; skipping online paper search")
            return []

        api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or os.environ.get("S2_API_KEY")
        if not api_key:
            logger.info("No Semantic Scholar API key found in env; skipping online search")
            return []

        papers = []
        headers = {"x-api-key": api_key}
        base = "https://api.semanticscholar.org/graph/v1/paper/search"

        for rnd in range(max_rounds):
            for kw in keywords:
                params = {"query": kw, "limit": 5, "fields": "title,authors,venue,year,externalIds"}
                try:
                    resp = requests.get(base, headers=headers, params=params, timeout=10)
                    if resp.status_code == 200:
                        data = resp.json()
                        for item in data.get("data", [])[:5]:
                            papers.append({
                                "title": item.get("title"),
                                "authors": [a.get("name") for a in item.get("authors", [])],
                                "venue": item.get("venue"),
                                "year": item.get("year"),
                                "externalIds": item.get("externalIds", {}),
                            })
                except Exception as e:
                    logger.warning(f"Paper search request failed for '{kw}': {e}")

        return papers

    # --- Multi-round interactive citation collection using a small model ---
    async def _collect_citations_with_small_model(self, small_model, context: Dict[str, Any], num_rounds: int = 2) -> List[Dict[str, Any]]:
        """通过小模型决定每轮检索关键词并挑选引用，输出去重后的候选文献列表。"""
        if small_model is None:
            logger.info("No small model provided; skipping citation collection")
            return []

        selected_papers = []
        # 构造简洁上下文，便于小模型理解任务与已有思路
        task = context.get("task", {})
        ideas = context.get("ideas", []) or []
        results = context.get("results", []) or []

        base_info = f"Task: {task.get('description','')}\nTop ideas:\n"
        for i, idea in enumerate(ideas[:5], 1):
            base_info += f"{i}. {idea.get('title') or idea.get('name')} - { (idea.get('description') or '')[:200]}\n"

        # 逐轮询问小模型要检索的关键词，必要时提前终止
        for rnd in range(num_rounds):
            prompt = (
                "You are an assistant that identifies key citation queries for a paper.\n"
                "Given the task/context below, propose up to 3 short search queries (one per line) that would help find missing references.\n"
                "If no more citations are needed, reply with the single line: NO_MORE_CITATIONS\n\n"
                f"Context:\n{base_info}\nRound: {rnd+1}/{num_rounds}\n"
            )

            try:
                if hasattr(small_model, 'generate'):
                    resp = await self._maybe_await(small_model.generate(prompt))
                    text = resp.get('text') if isinstance(resp, dict) else str(resp)
                elif hasattr(small_model, 'call'):
                    resp = await self._maybe_await(small_model.call(prompt))
                    text = str(resp)
                else:
                    text = str(small_model)
            except Exception as e:
                logger.warning(f"Small model failed proposing queries: {e}")
                break

            if not text:
                break
            if "NO_MORE_CITATIONS" in text.upper():
                break

            # 提取小模型给出的检索词，最多保留三条
            queries = [ln.strip() for ln in text.splitlines() if ln.strip()][:3]
            if not queries:
                break

            # 针对每个关键词调用 API 检索候选文献
            papers = self._search_for_papers(queries, max_rounds=1)
            if not papers:
                continue

            # 让小模型在候选集中挑选最相关的条目
            paper_summaries = []
            for i, p in enumerate(papers):
                paper_summaries.append(f"{i}: {p.get('title')} ({', '.join(p.get('authors') or [])}) {p.get('year')}")
            selection_prompt = (
                "From the following search results, select the indices of the most relevant papers for inclusion in references.\n"
                "Return a JSON array such as [0,1] or [] if none. Only output the JSON.\n\n"
                + "\n\n".join(paper_summaries)
            )

            try:
                if hasattr(small_model, 'generate'):
                    sel_resp = await self._maybe_await(small_model.generate(selection_prompt))
                    sel_text = sel_resp.get('text') if isinstance(sel_resp, dict) else str(sel_resp)
                elif hasattr(small_model, 'call'):
                    sel_resp = await self._maybe_await(small_model.call(selection_prompt))
                    sel_text = str(sel_resp)
                else:
                    sel_text = "[]"
            except Exception as e:
                logger.warning(f"Small model failed selecting papers: {e}")
                sel_text = "[]"

            # 解析 JSON 形式的索引列表，解析失败则视为未选
            import re
            m = re.search(r"\[.*?\]", sel_text)
            try:
                idxs = json.loads(m.group(0)) if m else []
            except Exception:
                idxs = []

            for idx in idxs:
                try:
                    sel = papers[int(idx)]
                    selected_papers.append(sel)
                except Exception:
                    continue

        # 以标题去重，避免重复引用
        unique = {}
        for p in selected_papers:
            key = (p.get('title') or '').strip()
            if key:
                unique[key] = p
        return list(unique.values())

    def _append_bib_entries(self, bib_path: str, papers: List[Dict[str, Any]]) -> None:
        """向 references.bib 追加最简 BibTeX 条目，忽略空列表。"""
        if not papers:
            return
        with open(bib_path, "a", encoding="utf-8") as bf:
            for i, p in enumerate(papers, 1):
                # Create a minimal bib entry
                key = f"ref{i}"
                title = _escape_bib_value((p.get("title") or "Untitled").replace("{", "").replace("}", ""))
                authors = _escape_bib_value(" and ".join(p.get("authors") or []))
                year = _escape_bib_value(str(p.get("year") or ""))
                venue = _escape_bib_value(p.get("venue") or "")
                bf.write(
                    f"@article{{{key},\n"
                    f"  title = {{{title}}},\n"
                    f"  author = {{{authors}}},\n"
                    f"  journal = {{{venue}}},\n"
                    f"  year = {{{year}}},\n"
                    "}\n\n"
                )

    # --- Draft generation & refinement -------------------------------------
    def _compose_prompt_for_draft(self, context: Dict[str, Any]) -> str:
        """根据任务描述、想法和实验摘要拼接写作提示词，引导模型输出结构化正文。"""
        task = context.get("task", {})
        ideas = context.get("ideas", []) or []
        results = context.get("results", []) or []
        # Adopt a more detailed system-style prompt similar to perform_writeup.py
        page_limit = int(self.config.get("page_limit", 20))
        prompt_lines = []
        prompt_lines.append("You are an ambitious AI researcher writing a LaTeX manuscript for a top-tier ML venue.")
        prompt_lines.append(f"Task: {task.get('description','')}")
        prompt_lines.append(
            f"Target page limit (main text, excluding references/impact statement): {page_limit} pages. "
            "Aim for a manuscript that would realistically fill around 70–90% of this page budget in a double-column layout."
        )
        prompt_lines.append(
            "Avoid overly brief sections: each major section (Introduction, Method, Experimental Setup, Experiments, Conclusion) "
            "should contain multiple paragraphs with detailed reasoning, not just a short summary."
        )
        prompt_lines.append("Write clear, factual, and non-hallucinated LaTeX body content covering: Introduction, Related Work, Background (if needed), Method, Experimental Setup, Results, Conclusion.")
        prompt_lines.append(
            "Prefer figures/tables for results. Use \\cite{...} placeholders for references; do not invent citation keys—use refX placeholders, and ensure the final manuscript cites at least seven distinct references."
        )
        prompt_lines.append(
            "Keep floats local to their sections: every figure/table must use \\begin{figure}[H] from the float package, and insert \\FloatBarrier before starting a new \\section so floats cannot drift downward."
        )
        prompt_lines.append(
            "\nSection-specific writing guidance:\n"
            "- Title: informative, engaging, and under two lines so readers instantly grasp the topic.\n"
            # "- Abstract: single flowing paragraph summarizing motivation, goal relevance, approach, and findings.\n"
            "- Introduction: expand on the abstract with context, motivation, realistic discussion of positive or negative outcomes, and a clear contributions list.\n"
            "- Related Work: cite comparable efforts, highlight similarities/differences, and ground all claims with references.\n"
            "- Background: introduce only the definitions/problem setup necessary to understand the method.\n"
            "- Method: explain the proposed approach, the hypotheses it tests, and discuss limitations or improvements if results are weak.\n"
            "- Experimental Setup: describe datasets, evaluation settings, and baselines (omit hardware unless essential).\n"
            "- Experiments: report truthful metrics, compare to baselines when available, and analyze why results meet or miss expectations (merge related figures when practical).\n"
            "- Conclusion: recap core findings, note implications, and suggest future work or fixes when outcomes are inconclusive.\n"
            "\nEnsure you are always writing good compilable LaTeX code. Common mistakes that should be fixed include:\n"
            "- Unescaped special characters (e.g., %, $, #, _, &) anywhere, including figure captions/labels.\n"
            "- Missing figure labels or incorrect references.\n"
            "- Incomplete sections or lack of clarity in explanations.\n"
            "- Ensure all LaTeX commands are properly closed and environments are correctly nested.\n"
            "- Proper table/figure closure.\n"
        )

        prompt_lines.append("\nIdeas and short descriptions:")
        for i, idea in enumerate(ideas, 1):
            title = idea.get("title") or idea.get("name") or f"idea_{i}"
            desc_parts: List[str] = []
            if idea.get("description"):
                desc_parts.append(str(idea.get("description")))
            if idea.get("method"):
                desc_parts.append(f"Method: {idea.get('method')}")
            desc = " \n".join(desc_parts)[:6000]
            prompt_lines.append(f"{i}. {title}: {desc}")

        prompt_lines.append("\nConcise results summary (up to 10 items):")
        for r in results[:10]:
            s = "success" if r.get("success") else "failure"
            label = r.get('idea_label') or r.get('idea_name') or r.get('artifact_dir') or ''
            prompt_lines.append(f"- {label}: {s}; details: { (r.get('error') or r.get('notes') or '')[:800] }")

        prompt_lines.append(
            "\nDo NOT include an Abstract section; the template already supplies it. Begin the body with Introduction and proceed through the other sections."
        )
        prompt_lines.append("\nProduce LaTeX body only (no documentclass), ensure figures are referenced by filename, avoid unescaped special characters, and keep sections self-contained. Return only LaTeX body content.")

        figure_block = context.get('figure_prompt_block')
        if figure_block:
            prompt_lines.append("\n可用图片资源（idea/run/相对路径: 描述），使用时写入 figures/.. 路径：")
            prompt_lines.append(figure_block)
            prompt_lines.append(
                "插图要求：严格遵循列表中的推荐章节/浮动位置/width，例如 \\begin{figure}[ht] \\centering \\includegraphics[width=0.78\\linewidth]{figures/foo}."
            )
            prompt_lines.append(
                "caption 至少说明坐标轴含义、指标峰谷或对比结论，并引用具体数值或 run ID。"
            )
            prompt_lines.append(
                "每个 figure 内务必添加 \\label{...}，其中 {...} 使用上方列表提供的 label= 提示，以避免未定义引用。"
            )
            prompt_lines.append(
                "所有插图必须放在 Method、Experimental Setup、Experiments 或 Conclusion 章节内，且全文引用的 figure 总数不超过 6 个。"
            )
            prompt_lines.append(
                "为避免 LaTeX 浮动堆积，所有图必须使用 [H] 定位并在每个章节结束前插入 \\FloatBarrier（placeins 已启用）。"
            )
        return "\n".join(prompt_lines)

    async def _generate_abstract(self, context: Dict[str, Any], big_model) -> str:
        """调用大模型生成摘要段落，若失败或无模型则返回空字符串。"""
        if not big_model:
            return ""

        try:
            lines = [
                "You are an assistant writing a concise, factual abstract for a machine learning paper.",
                "Produce a single standalone paragraph (100-200 words) summarizing the paper: problem, method, key results, and main conclusion.",
                "Do NOT invent citations or unverifiable claims. Keep it factual and self-contained.",
                "\nContext:\n",
            ]
            task = context.get('task', {})
            lines.append(f"Task: {task.get('description','')}")
            ideas = context.get('ideas', []) or []
            if ideas:
                lines.append("Top ideas:")
                for i, idea in enumerate(ideas[:5], 1):
                    title = idea.get('title') or idea.get('name') or ''
                    desc = (idea.get('description') or '')[:400]
                    lines.append(f"{i}. {title}: {desc}")
            results = context.get('results', []) or []
            if results:
                lines.append("Concise results summary:")
                for r in results[:10]:
                    s = "success" if r.get('success') else "failure"
                    label = r.get('idea_label') or r.get('idea_name') or r.get('artifact_dir') or ''
                    lines.append(f"- {label}: {s}; { (r.get('error') or r.get('notes') or '')[:200] }")

            prompt = "\n".join(lines)

            if hasattr(big_model, 'generate'):
                resp = await self._maybe_await(big_model.generate(prompt))
                text = resp.get('text') if isinstance(resp, dict) else str(resp)
            elif hasattr(big_model, 'call'):
                resp = await self._maybe_await(big_model.call(prompt))
                text = str(resp)
            else:
                return ""

            if not text:
                return ""

            paras = [p.strip() for p in text.split('\n\n') if p.strip()]
            return paras[0] if paras else text.strip()
        except Exception as e:
            logger.warning(f"Abstract generation failed: {e}")
            return ""

    async def _ask_model_for_draft(self, prompt: str) -> str:
        """根据给定提示词让底层模型产出 LaTeX 正文，若调用失败则返回兜底模板。"""
        try:
            # Prefer a 'generate' or 'call' style API if present
            if hasattr(self.model, "generate"):
                resp = await self._maybe_await(self.model.generate(prompt))
                # handle dict-like responses
                if isinstance(resp, dict):
                    return resp.get("text") or resp.get("content") or str(resp)
                return str(resp)
            elif hasattr(self.model, "call"):
                resp = await self._maybe_await(self.model.call(prompt))
                return str(resp)
            else:
                # Last resort: include prompt into a minimal template
                logger.info("Model has no generate/call; using fallback template")
                return "\\section{Introduction}\nThis is an auto-generated draft."
        except Exception as e:
            logger.warning(f"Model failed to generate draft: {e}")
            return "\\section{Introduction}\nThis is an auto-generated draft (model error)."

    def _sanitize_latex_body(self, text: str) -> str:
        """移除 ```latex 围栏并在 \texttt 内转义下划线，避免 LaTeX 编译错误。"""
        if not text:
            return ""
        cleaned = text.strip()
        fenced = re.search(r"```(?:latex|tex)?\s*([\s\S]*?)```", cleaned, flags=re.IGNORECASE)
        if fenced:
            cleaned = fenced.group(1).strip()

        def _escape_texttt(match: re.Match) -> str:
            inner = match.group(1)
            sentinel = "__ESCAPED_UNDERSCORE__"
            inner = inner.replace(r"\_", sentinel)
            inner = inner.replace("_", r"\_")
            inner = inner.replace(sentinel, r"\_")
            return f"\\texttt{{{inner}}}"

        cleaned = re.sub(r"\\texttt\{([^{}]*)\}", _escape_texttt, cleaned)
        cleaned = cleaned.replace("\r\n", "\n").strip()
        cleaned = re.sub(r"(?<!\\)([A-Za-z0-9]+)_(\d+)", r"\1\\_\2", cleaned)
        cleaned = re.sub(r"\\bibliographystyle\s*\{[^}]+\}\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\\bibliography\s*\{[^}]+\}\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = self._normalize_graphics_commands(cleaned)
        return cleaned

    @staticmethod
    def _sanitize_caption_text(text: str) -> str:
        if not text:
            return ''
        cleaned = text.replace('{', '').replace('}', '')
        cleaned = cleaned.replace('\n', ' ')
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        try:
            ascii_text = cleaned.encode('ascii', 'ignore').decode()
        except Exception:
            ascii_text = cleaned
        replacements = {
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '~': r'\textasciitilde{}',
            '^': r'\^{}',
        }
        for raw, escaped in replacements.items():
            ascii_text = ascii_text.replace(raw, escaped)
        return ascii_text

    def _figure_asset_exists(self, latex_root: str, graphic_path: str) -> bool:
        """检查 includegraphics 引用的资源是否存在，兼容相对/绝对路径。"""
        if not graphic_path:
            return False

        candidate = graphic_path.strip().strip('"\'')
        if not candidate:
            return False

        candidate = candidate.replace('\\', '/').lstrip('./')
        search_paths: List[str] = []

        if os.path.isabs(graphic_path):
            search_paths.append(graphic_path)
        else:
            search_paths.append(os.path.join(latex_root, candidate))
            if not candidate.startswith('figures/'):
                search_paths.append(os.path.join(latex_root, 'figures', candidate))

        for spath in search_paths:
            norm = os.path.normpath(spath)
            if os.path.exists(norm):
                return True
            stem, ext = os.path.splitext(norm)
            if not ext:
                for extra in ALLOWED_FIGURE_EXTENSIONS:
                    if os.path.exists(stem + extra):
                        return True
        return False

    def _remove_missing_figures(self, latex_root: str, text: str) -> str:
        """剔除引用缺失图片资源的 figure 环境，避免编译失败。"""
        if not text or not latex_root:
            return text

        figure_pattern = re.compile(
            r"\\begin\{figure\*?\}(?:\[[^\]]*\])?[\s\S]*?\\end\{figure\*?\}",
            flags=re.IGNORECASE,
        )
        include_pattern = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", re.IGNORECASE)
        removed: List[str] = []

        def _replace(block_match: re.Match) -> str:
            block = block_match.group(0)
            includes = include_pattern.findall(block)
            missing = [path for path in includes if not self._figure_asset_exists(latex_root, path)]
            if not missing:
                return block

            removed.extend(missing)
            caption_match = re.search(r"\\caption\{([^}]*)\}", block)
            caption = caption_match.group(1).strip() if caption_match else ""
            label_match = re.search(r"\\label\{([^}]*)\}", block)
            label = label_match.group(1).strip() if label_match else ""
            descriptor = caption or (label and f"reference {label}") or os.path.basename(missing[0]) or "missing figure"
            safe_descriptor = descriptor.replace('_', r"\_")
            missing_hint = missing[0].replace('_', r"\_")
            placeholder = f"\\noindent\\textit{{[Figure omitted: {safe_descriptor} (missing asset {missing_hint})]}}\n"
            return placeholder

        cleaned = figure_pattern.sub(_replace, text)
        if removed:
            missing_set = sorted({path for path in removed})
            logger.warning(
                "[PaperGenerationAgent] Removed figure blocks referencing missing assets: %s",
                ", ".join(missing_set),
            )
        return cleaned

    @staticmethod
    def _normalize_graphics_commands(text: str) -> str:
        """在 \includegraphics 与 \label 等指令中去掉多余的 \_，避免路径解析失败。"""
        if not text:
            return text

        def _cleanup(arg: str) -> str:
            if not arg:
                return arg
            normalized = arg.replace('\\\\_', '_')
            normalized = normalized.replace('\\_', '_')
            return normalized

        graphics_pattern = re.compile(r"(\\includegraphics(?:\[[^\]]*\])?\{)([^}]*)\}", flags=re.IGNORECASE)
        text = graphics_pattern.sub(lambda m: f"{m.group(1)}{_cleanup(m.group(2))}}}", text)

        label_pattern = re.compile(r"(\\label\{)([^}]*)\}", flags=re.IGNORECASE)
        text = label_pattern.sub(lambda m: f"{m.group(1)}{_cleanup(m.group(2))}}}", text)

        return text

    def _inject_sections_into_template(self, latex_root: str, draft_body: str, params: Dict[str, Any]) -> None:
        """清洗模板内容：仅替换 <TITLE>/<ABSTRACT> 并确保正文通过 \input{body.tex} 引入。"""
        _ = draft_body  # 正文由 body.tex 统一承载，此处无需解析
        template_tex = os.path.join(latex_root, "template.tex")
        main_tex = os.path.join(latex_root, "main.tex")
        target = main_tex

        minimal_body = (
            "\\documentclass{article}\n"
            "\\usepackage[utf8]{inputenc}\n"
            "\\usepackage{graphicx}\n"
            "\\usepackage{float}\n"
            "\\usepackage{placeins}\n"
            "\\begin{document}\n"
            "\\title{<TITLE>}\n"
            "\\author{InternAgent}\n"
            "\\maketitle\n"
            "\\begin{abstract}\n"
            "<ABSTRACT>\n"
            "\\end{abstract}\n\n"
            "\\input{body.tex}\n"
            "\\bibliographystyle{plain}\n"
            "\\bibliography{references}\n"
            "\\end{document}\n"
        )

        if not os.path.exists(main_tex):
            if os.path.exists(template_tex):
                try:
                    shutil.copy2(template_tex, main_tex)
                except Exception:
                    try:
                        with open(template_tex, 'r', encoding='utf-8') as tf:
                            templ_content = tf.read()
                        with open(main_tex, 'w', encoding='utf-8') as mf:
                            mf.write(templ_content)
                    except Exception:
                        with open(main_tex, 'w', encoding='utf-8') as mf:
                            mf.write(minimal_body)
            else:
                with open(main_tex, 'w', encoding='utf-8') as mf:
                    mf.write(minimal_body)

        try:
            with open(target, 'r', encoding='utf-8') as tf:
                content = tf.read()
        except Exception:
            return

        title_text = params.get('_paper_title') or params.get('title') or "Paper"
        abstract_text = params.get('_generated_abstract') or self.config.get("default_abstract") or ""

        content = content.replace('<TITLE>', title_text)
        content = content.replace('<ABSTRACT>', abstract_text)

        for markers in SECTION_MARKER_MAP.values():
            for mk in markers:
                content = content.replace(mk, '')

        content = self._ensure_required_packages(content)
        content = self._ensure_body_input_block(content)
        content = self._ensure_bibliography_block(content)
        content = self._ensure_numeric_citations(content)

        try:
            with open(target, 'w', encoding='utf-8') as tf:
                tf.write(content)
        except Exception:
            logger.warning("Failed to write sanitized main.tex")

    def _ensure_body_input_block(self, content: str) -> str:
        """确保模板中正文区域仅通过 \input{body.tex} 引入，去除重复插入。"""
        body_block = "\n\\input{body.tex}\n\n"
        content = content.replace('\\input{body.tex}', '')

        abstract_end = content.find('\\end{abstract}')
        tail_idx = self._find_tail_index(content)

        if abstract_end != -1 and tail_idx != -1 and tail_idx > abstract_end:
            pre = content[:abstract_end + len('\\end{abstract}')]
            tail = content[tail_idx:]
            return pre + body_block + tail

        if tail_idx != -1:
            return content[:tail_idx] + body_block + content[tail_idx:]

        return content + body_block

    @staticmethod
    def _ensure_numeric_citations(content: str) -> str:
        """在 preamble 中强制 natbib 采用数字引用格式，避免 author-year 冲突。"""
        if 'setcitestyle' in content:
            return content
        insertion = "\n\\setcitestyle{numbers,square}\n"
        marker = '\\begin{document}'
        idx = content.find(marker)
        if idx != -1:
            return content[:idx] + insertion + content[idx:]
        return insertion + content

    @staticmethod
    def _find_tail_index(content: str) -> int:
        markers = ['\\bibliography', '\\bibliographystyle', '\\end{document}']
        positions = []
        for marker in markers:
            idx = content.find(marker)
            if idx != -1:
                positions.append(idx)
        if positions:
            return min(positions)
        return -1

    @staticmethod
    def _ensure_bibliography_block(content: str) -> str:
        """若模板缺少引用指令则补齐，存在时保持原状避免重复。"""
        has_bib = re.search(r"\\bibliography\s*\{", content)
        has_style = re.search(r"\\bibliographystyle\s*\{", content)

        insertion = ''
        if not has_style:
            insertion += '\n\\bibliographystyle{icml2025}\n'
        if not has_bib:
            insertion += '\n\\bibliography{references}\n'

        if not insertion:
            return content

        end_doc = content.rfind('\\end{document}')
        if end_doc == -1:
            return content + insertion

        return content[:end_doc] + insertion + content[end_doc:]

    def _ensure_required_packages(self, content: str) -> str:
        """注入 float/placeins 以支持 [H] 定位和 \\FloatBarrier。"""
        if not content:
            return content

        required = ['float', 'placeins']
        missing = [pkg for pkg in required if f"\\usepackage{{{pkg}}}" not in content]
        if not missing:
            return content

        insertion_block = "\n".join(f"\\usepackage{{{pkg}}}" for pkg in missing) + "\n"
        anchor = '\\usepackage{graphicx}'
        idx = content.find(anchor)
        if idx != -1:
            insert_pos = idx + len(anchor)
            return content[:insert_pos] + "\n" + insertion_block + content[insert_pos:]

        marker = '\\begin{document}'
        idx = content.find(marker)
        if idx != -1:
            return content[:idx] + insertion_block + content[idx:]

        return insertion_block + content

    def _ensure_placeholder_bib_entries(self, draft_body: str, latex_root: str) -> None:
        """若正文含有引用但 references.bib 缺少对应条目，则自动补全占位 BibTeX。"""
        if not draft_body:
            return

        cite_pattern = re.compile(r"\\cite[\w]*\{([^}]*)\}")
        requested: Set[str] = set()
        for match in cite_pattern.finditer(draft_body):
            keys = [k.strip() for k in match.group(1).split(',') if k.strip()]
            requested.update(keys)

        if not requested:
            return

        bib_path = os.path.join(latex_root, "references.bib")
        existing: Set[str] = set()
        if os.path.exists(bib_path):
            with open(bib_path, 'r', encoding='utf-8') as bf:
                bib_content = bf.read()
            for key in requested:
                if re.search(rf"@\w+\{{\s*{re.escape(key)}[\s,]", bib_content):
                    existing.add(key)

        missing = sorted(requested - existing)
        if not missing:
            return

        _safe_makedirs(os.path.dirname(bib_path))
        with open(bib_path, 'a', encoding='utf-8') as bf:
            for key in missing:
                title = _escape_bib_value(f"Placeholder for {key}")
                author = _escape_bib_value("AutoGenerated")
                bf.write(
                    f"@misc{{{key},\n"
                    f"  title = {{{title}}},\n"
                    f"  author = {{{author}}},\n"
                    "  year = {2025},\n"
                    "}\n\n"
                )
        logger.info(
            "[PaperGenerationAgent] Added placeholder BibTeX entries for missing citations: %s",
            ", ".join(missing),
        )

    def _resolve_iteration_result_dirs(self, root_output_dir: str, params: Dict[str, Any]) -> List[str]:
        """解析需要收集图片的实验 iteration 目录，优先来自参数或配置。"""
        raw_dirs = params.get('idea_result_dirs')
        if raw_dirs is None:
            raw_dirs = self.config.get('idea_result_dirs')
        if raw_dirs is None:
            raw_dirs = DEFAULT_IDEA_RESULT_DIRS
        if isinstance(raw_dirs, str):
            raw_dirs = [raw_dirs]

        resolved: List[str] = []
        base_root = os.path.abspath(root_output_dir or os.getcwd())

        for entry in raw_dirs or []:
            if not entry:
                continue

            candidates = []
            if os.path.isabs(entry):
                candidates.append(entry)
            else:
                candidates.append(entry)
                candidates.append(os.path.join(base_root, entry))

            match = None
            for candidate in candidates:
                if not candidate:
                    continue
                if os.path.exists(candidate):
                    match = os.path.abspath(candidate)
                    break
            if match and match not in resolved:
                resolved.append(match)
            else:
                logger.warning(f"Idea result dir not found: {entry}")
        if not resolved:
            auto_dirs = _discover_idea_result_dirs(base_root)
            if auto_dirs:
                logger.info("[PaperGenerationAgent] Auto-detected idea result dirs: %s", auto_dirs)
                resolved.extend(auto_dirs)
        return resolved

    def _copy_iteration_figures(self, idea_dirs: List[str], root_output_dir: str, latex_root: str) -> List[Dict[str, Any]]:
        """遍历 iteration 目录的 run_* 结果并汇总图片到 LaTeX 工程内的 figures 目录。"""
        latex_fig_dir = os.path.join(latex_root, 'figures')
        archive_dir = os.path.join(root_output_dir, 'collected_figures')
        _safe_makedirs(latex_fig_dir)
        _safe_makedirs(archive_dir)

        if not idea_dirs:
            logger.info("No iteration directories supplied; created empty figures folder")
            return []

        records: List[Dict[str, Any]] = []

        for idea_dir in idea_dirs:
            if not os.path.isdir(idea_dir):
                continue
            idea_name = os.path.basename(os.path.normpath(idea_dir))
            for run_entry in os.listdir(idea_dir):
                if not run_entry.startswith('run_'):
                    continue
                run_path = os.path.join(idea_dir, run_entry)
                if not os.path.isdir(run_path):
                    continue
                for root_dir, _, files in os.walk(run_path):
                    for fname in files:
                        if not fname.lower().endswith(ALLOWED_FIGURE_EXTENSIONS):
                            continue
                        src = os.path.join(root_dir, fname)
                        rel_part = os.path.relpath(root_dir, run_path)
                        rel_part = '' if rel_part == '.' else rel_part.replace(os.sep, '_')
                        base_name = "_".join(filter(None, [idea_name, run_entry, rel_part, fname]))
                        base_name = base_name.replace(' ', '_')
                        dest_latex_path = os.path.join(latex_fig_dir, base_name)
                        dest_latex_path = self._dedupe_path(dest_latex_path)
                        try:
                            shutil.copy2(src, dest_latex_path)
                            # 再额外备份一份到总 output 目录，方便后续浏览
                            archive_path = os.path.join(archive_dir, os.path.basename(dest_latex_path))
                            if os.path.abspath(archive_path) != os.path.abspath(dest_latex_path):
                                try:
                                    shutil.copy2(dest_latex_path, archive_path)
                                except Exception:
                                    pass
                            rec = {
                                'idea': idea_name,
                                'run': run_entry,
                                'path': dest_latex_path,
                                'filename': os.path.basename(dest_latex_path),
                                'latex_rel_path': os.path.join('figures', os.path.basename(dest_latex_path)).replace('\\', '/'),
                                'source': src,
                            }
                            rec['label'] = _slugify_label(rec['filename'])
                            rec['semantics'] = _infer_figure_semantics(rec)
                            records.append(rec)
                        except Exception as exc:
                            logger.warning(f"Failed to copy figure {src}: {exc}")
        return records

    @staticmethod
    def _dedupe_path(target: str) -> str:
        """若目标文件已存在则追加编号以避免覆盖。"""
        if not os.path.exists(target):
            return target
        stem, ext = os.path.splitext(target)
        idx = 1
        while True:
            candidate = f"{stem}_{idx}{ext}"
            if not os.path.exists(candidate):
                return candidate
            idx += 1

    async def _describe_figures_with_vlm(self, vlm_model, figure_records: List[Dict[str, Any]]) -> Dict[str, str]:
        """调用 VLM 模型为收集到的图片生成描述（兼容异步接口）。"""
        descriptions: Dict[str, str] = {}
        if not vlm_model or not figure_records:
            return descriptions

        async def _call_vlm(method, *args, **kwargs):
            try:
                result = method(*args, **kwargs)
            except TypeError:
                result = method(*args)
            return await self._maybe_await(result)

        for rec in figure_records:
            path = rec['path']
            semantics = rec.get('semantics') or _infer_figure_semantics(rec)
            prompt = (
                "角色：资深实验报告作者。请结合图像内容与以下上下文，输出 2-3 句结构化描述：\n"
                f"- 图类别：{semantics['category']}，推荐章节：{semantics['recommended_section']}。\n"
                f"- 期望说明：{semantics['caption_focus']}\n"
                f"- 来源：idea={rec.get('idea','unknown')}，run={rec.get('run','unknown')}，文件={rec.get('filename')}。\n"
                "描述中至少包含：1) 坐标轴或图例含义；2) 关键峰谷/对比结论；3) 与整体研究问题的联系。"
            )
            try:
                desc = None
                if hasattr(vlm_model, 'describe'):
                    try:
                        desc = await _call_vlm(vlm_model.describe, path, prompt=prompt)
                    except TypeError:
                        desc = await _call_vlm(vlm_model.describe, path)
                elif hasattr(vlm_model, 'analyze_image'):
                    try:
                        desc = await _call_vlm(vlm_model.analyze_image, path, prompt=prompt)
                    except TypeError:
                        desc = await _call_vlm(vlm_model.analyze_image, path)
                elif hasattr(vlm_model, 'generate'):
                    text_prompt = f"[FIGURE_PATH={path}] {prompt}"
                    desc = await self._maybe_await(vlm_model.generate(text_prompt))
                if isinstance(desc, dict):
                    desc = desc.get('text') or desc.get('description') or str(desc)
                if not desc:
                    desc = (
                        f"{semantics['caption_focus']}，来源 idea={rec.get('idea','unknown')} / {rec.get('run','unknown')}。"
                    )
                descriptions[rec['filename']] = str(desc).strip()
            except Exception as exc:
                logger.warning(f"VLM failed on {path}: {exc}")
                descriptions[rec['filename']] = (
                    f"{semantics['caption_focus']}，来源 idea={rec.get('idea','unknown')} / {rec.get('run','unknown')}。"
                )
        return descriptions

    async def _review_figures_with_vlm(
        self,
        vlm_model,
        figure_records: List[Dict[str, Any]],
        figure_descriptions: Dict[str, str],
        max_reviews: int = 6,
    ) -> str:
        """利用 VLM 对图表质量、标题匹配与文本引用情况给出评审意见。"""
        if not vlm_model or not figure_records:
            return "VLM 评审：未配置视觉模型或无可用图表。"

        async def _call_vlm(method, *args, **kwargs):
            try:
                result = method(*args, **kwargs)
            except TypeError:
                result = method(*args)
            return await self._maybe_await(result)

        reviews = []
        for rec in figure_records[:max_reviews]:
            path = rec['path']
            semantics = rec.get('semantics') or _infer_figure_semantics(rec)
            caption_hint = figure_descriptions.get(rec['filename']) or semantics.get('caption_focus')
            prompt = (
                "角色：顶会审稿人（图表方向）。请查看图像并结合给定 caption，逐项点评：\n"
                "1) 图像是否清晰、要素齐全；\n"
                "2) caption 与图像信息是否匹配；\n"
                "3) 与正文引用（label={label}) 是否容易对齐；\n"
                "4) 如需修改，提出具体可执行的建议。\n"
                "输出 Markdown 列表，格式：- 结论；- 问题；- 建议。"
            ).format(label=rec.get('label') or _slugify_label(rec['filename']))
            try:
                verdict = None
                if hasattr(vlm_model, 'review'):
                    verdict = await _call_vlm(vlm_model.review, path, caption=caption_hint, prompt=prompt)
                elif hasattr(vlm_model, 'analyze_image'):
                    verdict = await _call_vlm(vlm_model.analyze_image, path, prompt=f"CAPTION:{caption_hint}\n{prompt}")
                elif hasattr(vlm_model, 'describe'):
                    verdict = await _call_vlm(vlm_model.describe, path, prompt=f"CAPTION:{caption_hint}\n{prompt}")
                elif hasattr(vlm_model, 'generate'):
                    verdict = await self._maybe_await(
                        vlm_model.generate(f"[FIGURE_PATH={path}] CAPTION:{caption_hint}\n{prompt}")
                    )
                if isinstance(verdict, dict):
                    verdict = verdict.get('text') or verdict.get('review') or str(verdict)
                if not verdict:
                    verdict = "- 结论：VLM 未返回有效结果\n- 问题：无\n- 建议：检查图像引用"
            except Exception as exc:
                verdict = f"- 结论：VLM 处理失败（{exc}）\n- 问题：无法解析图像\n- 建议：人工复核"
            reviews.append(
                f"#### 图 {rec.get('label') or _slugify_label(rec['filename'])} — {rec['filename']}\n"
                f"引用建议章节：{semantics.get('recommended_section', 'Results')}\n"
                f"caption/提示：{caption_hint}\n{verdict}\n"
            )

        if not reviews:
            return "VLM 评审：未能生成有效的图表点评。"
        return "\n".join(reviews)

    async def _generate_text_review(self, big_model, draft_body: str, context: Dict[str, Any]) -> str:
        """调用 LLM 生成结构化文字评审（创新性、技术充分性、清晰度等）。"""
        if not big_model or not draft_body:
            return "LLM 评审：缺少模型或正文，暂无法生成。"

        body_excerpt = draft_body.strip()
        if len(body_excerpt) > 20000:
            body_excerpt = body_excerpt[:20000]

        task = context.get('task', {})
        prompt = (
            "You are an experienced ML conference reviewer. Read the LaTeX body excerpt below and generate a structured review."
            " Provide Markdown with the following headers: Summary, Strengths, Weaknesses, Novelty, Technical Soundness, Clarity, Questions, Suggestions, Overall Recommendation (accept/weak accept/borderline/weak reject/reject) with a brief justification, and Confidence (0-1)."
            " Keep claims grounded in the text and highlight missing evidence when necessary.\n\n"
            f"Task context: {task.get('description','')}\n\nLaTeX body excerpt:\n<<<\n{body_excerpt}\n>>>"
        )
        try:
            if hasattr(big_model, 'generate'):
                resp = await self._maybe_await(big_model.generate(prompt))
                text = resp.get('text') if isinstance(resp, dict) else str(resp)
            elif hasattr(big_model, 'call'):
                resp = await self._maybe_await(big_model.call(prompt))
                text = str(resp)
            else:
                text = "LLM 评审：当前模型不支持 generate/call 接口。"
        except Exception as exc:
            text = f"LLM 评审失败：{exc}"
        return text or "LLM 评审：模型返回空结果。"

    async def _apply_review_feedback(
        self,
        draft_body: str,
        llm_review: str,
        vlm_review: str,
        big_model,
        figure_records: List[Dict[str, Any]],
        figure_descriptions: Dict[str, str],
    ) -> str:
        """基于 LLM/VLM 评审意见驱动正文自动改写。"""
        if not draft_body or not big_model:
            return draft_body
        review_text = "\n\n".join(filter(None, [llm_review, vlm_review])).strip()
        if not review_text:
            return draft_body

        feedback_excerpt = review_text[:8000]
        figure_summary = []
        for rec in figure_records[:6]:
            desc = figure_descriptions.get(rec['filename']) or rec.get('semantics', {}).get('caption_focus', '')
            figure_summary.append(
                f"- {rec.get('label') or _slugify_label(rec['filename'])}: {desc}"
            )
        figure_block = "\n".join(figure_summary)

        try:
            revised = await self._apply_review_feedback_sectional(
                draft_body,
                big_model,
                feedback_excerpt,
                figure_block,
            )
        except Exception as exc:
            logger.warning(f"Sectional review revision failed: {exc}")
            revised = None

        if not revised or revised.strip() == draft_body.strip():
            return draft_body
        return self._sanitize_latex_body(revised)

    async def _apply_review_feedback_sectional(
        self,
        draft_body: str,
        big_model,
        review_excerpt: str,
        figure_block: str,
    ) -> Optional[str]:
        """按章节多轮吸收审稿意见，以增量补充段落为主。"""
        sections = self._extract_section_spans(draft_body)
        if not sections:
            return draft_body

        priority_norms = [
            'method',
            'experimentalsetup',
            'experiments',
            'introduction',
            'relatedwork',
            'background',
            'conclusion',
        ]

        def _priority(section):
            norm = section.get('norm') or ''
            if norm in priority_norms:
                return priority_norms.index(norm)
            return len(priority_norms)

        ordered_sections = sorted(sections, key=_priority)
        updated_body = draft_body
        additions_made = False

        figure_tips = figure_block or "(no additional figure constraints)"
        review_header = (
            "Reviewer feedback excerpt (trimmed):\n<<<\n"
            + review_excerpt
            + "\n>>>\n\n"
        )

        for base_meta in ordered_sections:
            current_sections = self._extract_section_spans(updated_body)
            match = next(
                (s for s in current_sections if self._normalize_section_name(s.get('title')) == base_meta['norm']),
                None,
            )
            if not match:
                continue

            current_content = updated_body[match['content_start']:match['content_end']].strip()
            if not current_content:
                continue

            title = match.get('title') or ''
            prompt = (
                f"You are revising a LaTeX manuscript after reviewer feedback.\n"
                f"Focus on the section titled '{title}'. Leave all existing sentences intact;"
                " instead, append 1-3 concise paragraphs that directly address reviewer concerns relevant"
                " to this section. Cite figures/tables consistently and avoid inventing results.\n"
                "If reviewers did not raise issues for this section, reply with NONE.\n\n"
                f"{review_header}"
                f"Figure context (keep labels consistent):\n{figure_tips}\n\n"
                "Current section body:<<SECTION>>\n"
                f"<<SECTION>>\n{current_content}\n<<SECTION>>\n"
                "Return ONLY the additional paragraphs to append (no section heading, no fences)."
            )

            try:
                if hasattr(big_model, 'generate'):
                    resp = await self._maybe_await(big_model.generate(prompt))
                    addition = resp.get('text') if isinstance(resp, dict) else str(resp)
                elif hasattr(big_model, 'call'):
                    resp = await self._maybe_await(big_model.call(prompt))
                    addition = str(resp)
                else:
                    addition = None
            except Exception as exc:
                logger.warning(f"Review appendix failed for section {match.get('title')}: {exc}")
                addition = None

            if not addition:
                continue
            addition_clean = self._sanitize_latex_body(addition).strip()
            if not addition_clean or addition_clean.strip().upper() in {"NONE", "NO CHANGE", "UNCHANGED"}:
                continue

            insert_pos = match['content_end']
            updated_body = (
                updated_body[:insert_pos].rstrip()
                + "\n\n"
                + addition_clean
                + "\n\n"
                + updated_body[insert_pos:]
            )
            additions_made = True

        if not additions_made:
            return draft_body
        return updated_body

    @staticmethod
    def _build_figure_prompt_block(figure_records: List[Dict[str, Any]], descriptions: Dict[str, str]) -> str:
        """构造供写作提示引用的图片及描述清单。"""
        if not figure_records:
            return ''
        lines = []
        for rec in figure_records[:30]:
            desc = descriptions.get(rec['filename']) or ''
            rel = rec.get('latex_rel_path') or rec['filename']
            semantics = rec.get('semantics') or _infer_figure_semantics(rec)
            label = rec.get('label') or _slugify_label(rec['filename'])
            base = (
                f"[{semantics['recommended_section']} | pos={semantics['placement']} | width={semantics['width_cmd']} | label={label}] "
                f"{rec['idea']} / {rec['run']} / {rel}"
            )
            detail = desc or semantics['caption_focus']
            lines.append(f"{base}: {detail}")
        return "\n".join(lines)

    def _resolve_vlm_model(self, params: Dict[str, Any]):
        """解析 VLM 模型实例，允许配置或参数覆盖。"""
        vlm_spec = params.get('vlm_model')
        if vlm_spec is None:
            vlm_spec = self.config.get('vlm_model')
        if not vlm_spec:
            logger.info("[PaperGenerationAgent] No VLM model configured; skipping image description")
            return None
        if isinstance(vlm_spec, dict):
            try:
                default_conf = self._load_default_agent_config()
                vlm_model = self._build_model_from_spec(vlm_spec, default_conf)
                if vlm_model:
                    logger.info(
                        "[PaperGenerationAgent] vlm_model initialized from spec: %s",
                        type(vlm_model).__name__,
                    )
                else:
                    logger.warning("[PaperGenerationAgent] Failed to build vlm_model from spec")
                return vlm_model
            except Exception as exc:
                logger.warning(f"Failed to instantiate VLM model: {exc}")
                return None
        logger.info(
            "[PaperGenerationAgent] vlm_model provided as instance: %s",
            type(vlm_spec).__name__,
        )
        return vlm_spec

    # --- LaTeX compilation --------------------------------------------------
    def _compile_latex(self, latex_root: str, timeout: int = 60) -> Dict[str, Any]:
        """在指定目录内运行 pdflatex/bibtex 流程，返回是否成功及日志等元信息。"""
        main_tex = os.path.join(latex_root, "main.tex")
        if not os.path.exists(main_tex):
            # try main.tex vs other names
            # fall back to creating main.tex that inputs generated body.tex
            return {"success": False, "error": "main.tex missing"}

        # Run pdflatex twice, then bibtex if .aux contains citations
        cmd_pdflatex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
        env = os.environ.copy()
        logs = []
        try:
            for i in range(2):
                proc = subprocess.run(cmd_pdflatex, cwd=latex_root, env=env, capture_output=True, timeout=timeout)
                logs.append(proc.stdout.decode(errors="ignore"))
                if proc.returncode != 0:
                    logs.append(proc.stderr.decode(errors="ignore"))
                    return {"success": False, "error": f"pdflatex failed on pass {i+1}", "logs": "\n".join(logs)}

            # If references.bib has entries, try bibtex
            bib_path = os.path.join(latex_root, "references.bib")
            with open(bib_path, "r", encoding="utf-8") as bf:
                bib_content = bf.read()
            if "@" in bib_content:
                # derive aux file name
                aux = "main.aux"
                if os.path.exists(os.path.join(latex_root, aux)):
                    proc_bib = subprocess.run(["bibtex", "main"], cwd=latex_root, capture_output=True, timeout=timeout)
                    logs.append(proc_bib.stdout.decode(errors="ignore"))
                    if proc_bib.returncode != 0:
                        logs.append(proc_bib.stderr.decode(errors="ignore"))
                        return {"success": False, "error": "bibtex failed", "logs": "\n".join(logs)}

                # run pdflatex twice more to resolve citations
                for i in range(2):
                    proc = subprocess.run(cmd_pdflatex, cwd=latex_root, env=env, capture_output=True, timeout=timeout)
                    logs.append(proc.stdout.decode(errors="ignore"))
                    if proc.returncode != 0:
                        logs.append(proc.stderr.decode(errors="ignore"))
                        return {"success": False, "error": f"pdflatex failed on post-bib pass {i+1}", "logs": "\n".join(logs)}

            pdf_path = os.path.join(latex_root, "main.pdf")
            if os.path.exists(pdf_path):
                return {"success": True, "pdf_path": pdf_path, "logs": "\n".join(logs)}
            else:
                return {"success": False, "error": "PDF not produced", "logs": "\n".join(logs)}

        except FileNotFoundError as fe:
            # pdflatex or bibtex not installed
            logger.warning(f"LaTeX toolchain not found: {fe}")
            return {"success": False, "error": "latex_toolchain_missing", "logs": ""}
        except subprocess.TimeoutExpired as te:
            logger.warning(f"LaTeX compilation timed out: {te}")
            return {"success": False, "error": "timeout", "logs": ""}

    # --- Helper utilities for execute() ------------------------------------
    def _load_default_agent_config(self) -> Dict[str, Any]:
        """懒加载 config/default_config.yaml，供模型解析等逻辑复用。"""
        if self._default_agent_conf is not None:
            return self._default_agent_conf

        default_conf_path = os.path.join(os.getcwd(), 'config', 'default_config.yaml')
        if not os.path.exists(default_conf_path):
            self._default_agent_conf = {}
            return self._default_agent_conf

        try:
            import yaml
            with open(default_conf_path, 'r', encoding='utf-8') as cf:
                self._default_agent_conf = yaml.safe_load(cf) or {}
        except Exception as exc:
            logger.warning(f"Failed to load default_config.yaml: {exc}")
            self._default_agent_conf = {}
        return self._default_agent_conf

    def _build_model_from_spec(self, spec: Any, global_conf: Optional[Dict[str, Any]] = None):
        """根据模型配置字典调用工厂创建实例，允许继承全局配置。"""
        if not spec:
            return None
        if not isinstance(spec, dict):
            return spec
        try:
            from ..models.model_factory import ModelFactory
            cfg = dict(global_conf or {})
            cfg.update(spec)
            return ModelFactory.create_model(cfg)
        except Exception as exc:
            logger.warning(f"Failed to create model from spec: {exc}")
            return None

    def _resolve_small_model(self, params: Dict[str, Any]):
        """解析小模型配置，按参数 > 实例属性 > 默认配置的优先级获取实例。"""
        small_spec = params.get('small_model') or self.config.get('small_model')
        small_model = self._build_model_from_spec(small_spec)
        if small_model:
            return small_model

        default_conf = self._load_default_agent_config()
        paper_conf = (default_conf.get('agents') or {}).get('paper_generation', {})
        return self._build_model_from_spec(paper_conf.get('small_model'), default_conf)

    def _resolve_big_model(self, params: Dict[str, Any]):
        """解析大模型实例，必要时回退到智能体自身绑定的模型或默认配置。"""
        big_spec = params.get('big_model') or self.config.get('big_model')
        big_model = self._build_model_from_spec(big_spec)
        if big_model:
            logger.info(
                "[PaperGenerationAgent] big_model initialized from params/config: %s",
                type(big_model).__name__,
            )
            return big_model

        # Fall back to agent's primary model
        if isinstance(self.model, dict):
            fallback = self._build_model_from_spec(self.model)
            if fallback:
                logger.info(
                    "[PaperGenerationAgent] big_model initialized from agent-bound spec: %s",
                    type(fallback).__name__,
                )
                return fallback
        if self.model:
            logger.info(
                "[PaperGenerationAgent] big_model using existing agent model instance: %s",
                type(self.model).__name__,
            )
            return self.model

        default_conf = self._load_default_agent_config()
        paper_conf = (default_conf.get('agents') or {}).get('paper_generation', {})
        fallback = self._build_model_from_spec(paper_conf.get('big_model'), default_conf)
        if fallback:
            logger.info(
                "[PaperGenerationAgent] big_model initialized from default_config: %s",
                type(fallback).__name__,
            )
        else:
            logger.warning("[PaperGenerationAgent] big_model resolution failed; continuing without big model")
        return fallback

    async def _collect_and_append_citations(self, small_model, context: Dict[str, Any], latex_root: str) -> List[Dict[str, Any]]:
        """封装引用检索与写入 BibTeX 的流程，返回最终选中的引用列表。"""
        num_cite_rounds = int(self.config.get("num_cite_rounds", 2))
        selected_papers = await self._collect_citations_with_small_model(small_model, context, num_rounds=num_cite_rounds)
        # selected_papers = await self._collect_citations_with_small_model(small_model, context, num_rounds=2)
        bib_path = os.path.join(latex_root, "references.bib")
        self._append_bib_entries(bib_path, selected_papers)
        return selected_papers

    async def _generate_draft_and_abstract(self, context: Dict[str, Any], big_model) -> Dict[str, Any]:
        """利用大模型生成正文草稿与摘要，失败时对正文回退到本地兜底方案。"""
        prompt = self._compose_prompt_for_draft(context)
        if big_model and hasattr(big_model, 'generate'):
            try:
                resp = await self._maybe_await(big_model.generate(prompt))
                draft_body = resp.get('text') if isinstance(resp, dict) else str(resp)
            except Exception:
                draft_body = await self._ask_model_for_draft(prompt)
        else:
            draft_body = await self._ask_model_for_draft(prompt)

        draft_body = self._sanitize_latex_body(draft_body)

        try:
            abstract_text = await self._generate_abstract(context, big_model)
        except Exception:
            abstract_text = ""

        return {
            'draft_body': draft_body,
            'abstract': abstract_text,
        }

    def _write_body_file(self, latex_root: str, draft_body: str) -> str:
        """将最新正文写入 body.tex 并返回文件路径。"""
        body_path = os.path.join(latex_root, "body.tex")
        filtered_body = self._remove_missing_figures(latex_root, draft_body)
        try:
            with open(body_path, "w", encoding="utf-8") as bf:
                bf.write(filtered_body)
        except Exception:
            logger.warning('Failed to write body.tex')
        return body_path

    @staticmethod
    def _normalize_section_name(name: Optional[str]) -> str:
        if not name:
            return ''
        return re.sub(r"[^a-z]", "", name.lower())

    def _extract_section_spans(self, draft_body: str) -> List[Dict[str, Any]]:
        """解析正文中的 \section 段落范围，供按章节定向改写。"""
        if not draft_body:
            return []
        matches = list(re.finditer(r"(\\section\{([^}]*)\})", draft_body, flags=re.IGNORECASE))
        sections: List[Dict[str, Any]] = []
        for idx, match in enumerate(matches):
            title = (match.group(2) or '').strip()
            content_start = match.end()
            content_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(draft_body)
            sections.append({
                'title': title,
                'norm': self._normalize_section_name(title),
                'header_start': match.start(),
                'content_start': content_start,
                'content_end': content_end,
            })
        return sections

    async def _expand_sections_iteratively(
        self,
        draft_body: str,
        big_model,
        context: Dict[str, Any],
        section_sequence: Optional[List[str]] = None,
        passes: Optional[int] = None,
    ) -> str:
        """多轮按章节扩写正文，每次仅聚焦单个 section 以获得更长内容。"""
        if not draft_body or not big_model:
            return draft_body

        num_passes = passes if passes is not None else self.config.get('section_expansion_passes', 2)
        try:
            num_passes = int(num_passes)
        except Exception:
            num_passes = 0
        if num_passes <= 0:
            return draft_body
        # Ensure at least two full passes to provide sufficient amplification even when configuration omits it.
        num_passes = max(2, num_passes)

        default_sequence = [
            'Introduction',
            'Related Work',
            'Background',
            'Method',
            'Experimental Setup',
            'Experiments',
            'Conclusion',
        ]
        sequence = section_sequence or self.config.get('section_expansion_sequence') or default_sequence
        normalized_targets = [self._normalize_section_name(name) for name in sequence if name]
        if not normalized_targets:
            return draft_body

        body = draft_body
        for round_idx in range(num_passes):
            sections = self._extract_section_spans(body)
            if not sections:
                break
            for target_norm in normalized_targets:
                matched = next((s for s in sections if s['norm'] == target_norm), None)
                if not matched:
                    continue
                updated = await self._rewrite_single_section(body, matched, big_model, context)
                if updated:
                    body = updated
                    sections = self._extract_section_spans(body)
        return body

    async def _rewrite_single_section(
        self,
        body: str,
        section_meta: Dict[str, Any],
        big_model,
        context: Dict[str, Any],
    ) -> Optional[str]:
        """调用大模型放大指定 section 内容，返回更新后的正文。"""
        if not body or not section_meta:
            return None

        task = context.get('task', {}) if isinstance(context, dict) else {}
        task_desc = task.get('description') or task.get('name') or ''
        current_content = body[section_meta['content_start']:section_meta['content_end']].strip()
        # 略过已经较长的段落，避免无谓改写
        if len(current_content.split()) >= int(self.config.get('section_expansion_min_words', 180)):
            return None

        prompt = (
            f"You are editing the LaTeX section titled '{section_meta['title']}'.\n"
            "Expand this section with multiple paragraphs covering motivation, methodology details,"
            " concrete quantitative evidence, limitations, and future directions relevant to the section's scope.\n"
            "Preserve any math, figure references, and citations already present.\n"
            "Return ONLY the section body content that should appear immediately after the \\section command, without repeating the heading.\n"
            "Paper context: " + task_desc + "\n"
            "Current section body:<<SECTION>>\n"
            f"<<SECTION>>\n{current_content}\n<<SECTION>>"
        )

        try:
            if hasattr(big_model, 'generate'):
                resp = await self._maybe_await(big_model.generate(prompt))
                new_content = resp.get('text') if isinstance(resp, dict) else str(resp)
            elif hasattr(big_model, 'call'):
                resp = await self._maybe_await(big_model.call(prompt))
                new_content = str(resp)
            else:
                return None
        except Exception as exc:
            logger.warning(f"Section expansion failed for {section_meta['title']}: {exc}")
            return None

        new_content = self._sanitize_latex_body(new_content or '').strip()
        if not new_content:
            return None

        before = body[:section_meta['content_start']]
        after = body[section_meta['content_end']:]
        updated_body = before.rstrip() + "\n\n" + new_content + "\n\n" + after.lstrip('\n')
        return updated_body

    def _ensure_figures_in_body(
        self,
        draft_body: str,
        figure_records: List[Dict[str, Any]],
        descriptions: Dict[str, str],
    ) -> str:
        """若正文未插入图片，则根据已收集的 figure metadata 自动补充 figure 环境。"""
        if not draft_body or not figure_records:
            return self._enforce_section_float_barriers(draft_body)

        text = draft_body
        existing_labels = set(re.findall(r"\\label\{([^}]+)\}", text))
        missing_blocks: List[Dict[str, Any]] = []
        max_auto = int(self.config.get('max_auto_figures', 6))
        max_auto = min(max_auto, 6)

        def _norm_section(name: Optional[str]) -> str:
            return re.sub(r"[^a-z]", "", (name or '').lower())

        existing_figures = re.findall(r"\\begin\{figure\*?\}", text, flags=re.IGNORECASE)
        available_slots = max(0, 6 - len(existing_figures))
        if available_slots == 0:
            return self._enforce_section_float_barriers(draft_body)

        def _caption_is_meaningful(raw: Optional[str]) -> bool:
            if not raw or not raw.strip():
                return False
            lowered = raw.strip().lower()
            # Skip strings that still look like raw filenames.
            if re.search(r"\.(png|jpg|jpeg|gif|bmp|svg|pdf)\b", lowered):
                return False
            return True

        for rec in figure_records:
            label = rec.get('label') or _slugify_label(rec.get('filename') or '')
            if not label or label in existing_labels:
                continue

            rel_path = (rec.get('latex_rel_path') or rec.get('filename') or '').replace('\\', '/')
            if not rel_path:
                continue

            semantics = rec.get('semantics') or _infer_figure_semantics(rec)
            placement = '[H]'
            width_cmd = semantics.get('width_cmd') or '0.9\\linewidth'
            caption_source = descriptions.get(rec.get('filename'))
            if not _caption_is_meaningful(caption_source):
                fallback = semantics.get('caption_focus') or ''
                idea_name = rec.get('idea') or 'experiment'
                run_name = rec.get('run') or rec.get('filename') or 'run'
                fallback_detail = (
                    f"Experimental setup artifact from idea {idea_name} (run {run_name}), "
                    f"highlighting {fallback or 'the observed procedure'}."
                )
                caption_source = fallback_detail
            if caption_source:
                idea_name = rec.get('idea') or 'experiment'
                run_name = rec.get('run') or rec.get('filename') or 'run'
                semantic_focus = semantics.get('caption_focus') or 'key measurements from the setup'
                enrichment = (
                    f"This experimental setup output (idea {idea_name}, run {run_name}) "
                    f"captures {semantic_focus}."
                )
                # Append additional context when the caption is too terse to be informative.
                if len(caption_source.split()) < 12:
                    caption_source = f"{caption_source.strip()} {enrichment}"
            caption = self._sanitize_caption_text(caption_source)
            section_hint = semantics.get('recommended_section') or 'Experimental Setup'
            norm_hint = _norm_section(section_hint)
            if norm_hint not in FIGURE_ALLOWED_SECTION_NORMS:
                section_hint = 'Experimental Setup'
                norm_hint = 'experimentalsetup'

            block = (
                f"\\begin{{figure}}{placement}\n"
                "\\centering\n"
                f"\\includegraphics[width={width_cmd}]{{{rel_path}}}\n"
                f"\\caption{{{caption}}}\n"
                f"\\label{{{label}}}\n"
                "\\end{figure}\n"
            )
            missing_blocks.append({
                'block': block,
                'section': section_hint,
                'norm_section': norm_hint,
            })
            existing_labels.add(label)

            if len(missing_blocks) >= min(max_auto, available_slots):
                break

        if not missing_blocks:
            return self._enforce_section_float_barriers(draft_body)

        section_matches = list(re.finditer(r"(\\section\{([^}]*)\})", text, flags=re.IGNORECASE))
        section_spans = [
            {
                'start': m.start(),
                'end': m.end(),
                'insert_at': m.end(),
                'title': m.group(2),
                'norm': _norm_section(m.group(2)),
            }
            for m in section_matches
        ]

        def _recompute_sections(updated_text: str) -> Tuple[List[re.Match], List[Dict[str, Any]]]:
            matches = list(re.finditer(r"(\\section\{([^}]*)\})", updated_text, flags=re.IGNORECASE))
            spans = [
                {
                    'start': m.start(),
                    'end': m.end(),
                    'insert_at': m.end(),
                    'title': m.group(2),
                    'norm': _norm_section(m.group(2)),
                }
                for m in matches
            ]
            return matches, spans

        if not any(sp['norm'] == 'experimentalsetup' for sp in section_spans):
            insert_pos = None

            def _find_span(norm: str) -> Optional[Dict[str, Any]]:
                return next((sp for sp in section_spans if sp['norm'] == norm), None)

            for anchor_norm in ('method', 'background'):
                anchor = _find_span(anchor_norm)
                if anchor:
                    insert_pos = anchor['insert_at']
                    break

            if insert_pos is None:
                for anchor_norm in ('experiments', 'results'):
                    anchor = _find_span(anchor_norm)
                    if anchor:
                        insert_pos = anchor['start']
                        break

            if insert_pos is None:
                insert_pos = len(text)

            insertion = "\n\section{Experimental Setup}\n"
            text = text[:insert_pos] + insertion + text[insert_pos:]
            section_matches, section_spans = _recompute_sections(text)

        insert_events: List[Tuple[int, str, int]] = []
        allowed_fallback_order = ['experimentalsetup', 'experiments', 'method', 'conclusion']
        for order, block_info in enumerate(missing_blocks):
            target_norm = block_info.get('norm_section') or _norm_section(block_info.get('section') or 'Experimental Setup') or 'experimentalsetup'
            if target_norm not in FIGURE_ALLOWED_SECTION_NORMS:
                target_norm = 'experimentalsetup'
            fallback_norms = [target_norm] + [norm for norm in allowed_fallback_order if norm != target_norm]
            insert_idx = None
            for norm in fallback_norms:
                span = next((sp for sp in section_spans if sp['norm'] == norm), None)
                if span:
                    insert_idx = span['insert_at']
                    break
            if insert_idx is None and section_spans:
                insert_idx = section_spans[-1]['insert_at']
            if insert_idx is None:
                insert_idx = len(text)
            insert_events.append((insert_idx, block_info['block'], order))

        insert_events.sort(key=lambda item: (item[0], item[2]))
        pieces: List[str] = []
        cursor = 0
        for idx, block, _ in insert_events:
            idx = max(0, min(len(text), idx))
            pieces.append(text[cursor:idx])
            pieces.append("\n\n" + block + "\n")
            cursor = idx
        pieces.append(text[cursor:])
        return self._enforce_section_float_barriers("".join(pieces))

    def _enforce_section_float_barriers(self, text: str) -> str:
        """在新的 \section 之前插入 \FloatBarrier，避免浮动越节。"""
        if not text:
            return text

        section_pattern = re.compile(r"(\\section\{[^}]+\})")
        matches = list(section_pattern.finditer(text))
        if not matches:
            return text

        pieces: List[str] = []
        last_idx = 0
        first_section = True
        barrier = "\n\\FloatBarrier\n\n"

        for match in matches:
            start = match.start()
            segment = text[last_idx:start]
            if not first_section:
                if not segment.rstrip().endswith("\\FloatBarrier"):
                    segment = segment.rstrip() + barrier
            pieces.append(segment)
            pieces.append(text[start:match.end()])
            last_idx = match.end()
            first_section = False

        pieces.append(text[last_idx:])
        return "".join(pieces)

    def _prepare_template_files(self, latex_root: str, task: Dict[str, Any], draft_body: str, params: Dict[str, Any]) -> str:
        """准备 main.tex：复制模版或生成最小骨架，正文只通过 body.tex 注入。"""
        template_tex = os.path.join(latex_root, "template.tex")
        main_tex = os.path.join(latex_root, "main.tex")

        if os.path.exists(template_tex):
            try:
                shutil.copy2(template_tex, main_tex)
            except Exception:
                try:
                    with open(template_tex, 'r', encoding='utf-8') as tf:
                        templ_content = tf.read()
                    with open(main_tex, 'w', encoding='utf-8') as mf:
                        mf.write(templ_content)
                except Exception:
                    logger.warning('Failed to mirror template.tex into main.tex')
        else:
            minimal = (
                "\\documentclass{article}\n"
                "\\usepackage[utf8]{inputenc}\n"
                "\\usepackage{graphicx}\n"
                "\\usepackage{float}\n"
                "\\usepackage{placeins}\n"
                "\\begin{document}\n"
                "\\title{<TITLE>}\\n"
                "\\author{InternAgent}\\n"
                "\\maketitle\\n"
                "\\input{body.tex}\\n"
                "\\bibliographystyle{plain}\\n"
                "\\bibliography{references}\\n"
                "\\end{document}\n"
            )
            if not os.path.exists(main_tex):
                with open(main_tex, "w", encoding="utf-8") as mf:
                    mf.write(minimal)

        self._inject_sections_into_template(latex_root, draft_body, params)

        return main_tex

    async def _run_refinement_loop(
        self,
        draft_body: str,
        body_path: str,
        latex_root: str,
        main_tex: str,
        params: Dict[str, Any],
        big_model,
        vlm_model,
        root_output_dir: str,
        max_reflections: Optional[int] = None,
    ) -> Dict[str, Any]:
        """循环执行编译-诊断-反思更新流程，收集每次成功的 PDF 版本路径。"""
        if max_reflections is not None:
            n_ref = int(max_reflections)
        else:
            n_ref = int(self.config.get("n_writeup_reflections", 4))
        pdf_versions = []
        compile_attempt = 0

        for i in range(n_ref + 1):
            draft_body = self._ensure_figures_in_body(
                draft_body,
                params.get('_figure_records') or [],
                params.get('_figure_descriptions') or {},
            )
            draft_body = self._remove_missing_figures(latex_root, draft_body)
            with open(body_path, "w", encoding="utf-8") as bf:
                bf.write(draft_body)

            compile_timeout = int(self.config.get("compile_timeout", 60))
            compile_result = self._compile_latex(latex_root, timeout=compile_timeout)
            compile_attempt += 1
            if compile_result.get('pdf_path'):
                vname = os.path.join(latex_root, f"main_v{compile_attempt}.pdf")
                try:
                    shutil.copy2(compile_result['pdf_path'], vname)
                    pdf_versions.append(vname)
                except Exception:
                    pass

            chktex_output = ""
            try:
                chktex_proc = subprocess.run(["chktex", main_tex], cwd=latex_root, capture_output=True, text=True, timeout=30)
                chktex_output = chktex_proc.stdout + "\n" + chktex_proc.stderr
            except Exception:
                chktex_output = "chktex not available or failed"

            impact_info = None
            try:
                pdfp = compile_result.get('pdf_path')
                if pdfp and os.path.exists(pdfp):
                    txt_out = os.path.join(latex_root, f"main_v{compile_attempt}.txt")
                    subprocess.run(["pdftotext", pdfp, txt_out], capture_output=True, timeout=30)
                    if os.path.exists(txt_out):
                        txt = open(txt_out, 'r', encoding='utf-8', errors='ignore').read()
                        if 'Impact Statement' in txt:
                            impact_info = 'Impact Statement found in PDF text.'
            except Exception:
                impact_info = None

            figure_records = params.get('_figure_records') or []
            if not figure_records:
                fig_dir = os.path.join(latex_root, 'figures')
                if os.path.exists(fig_dir):
                    figure_records = [{
                        'idea': 'unknown',
                        'run': 'unknown',
                        'path': os.path.join(fig_dir, fname),
                        'filename': fname,
                        'latex_rel_path': os.path.join('figures', fname).replace('\\', '/'),
                        'source': os.path.join(fig_dir, fname),
                    } for fname in os.listdir(fig_dir) if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.svg', '.pdf'))]
                    for rec in figure_records:
                        rec['label'] = _slugify_label(rec['filename'])
                        rec['semantics'] = _infer_figure_semantics(rec)

            figure_descriptions = params.get('_figure_descriptions') or {}
            if not figure_descriptions and vlm_model:
                figure_descriptions = await self._describe_figures_with_vlm(vlm_model, figure_records)
                params['_figure_descriptions'] = figure_descriptions

            if i == n_ref:
                break

            draft_body = await self._run_sectional_reflections(
                draft_body,
                big_model,
                chktex_output,
                compile_result,
                impact_info,
                figure_records,
                figure_descriptions,
            )

        return {
            'draft_body': draft_body,
            'pdf_versions': pdf_versions,
        }

    async def _run_sectional_reflections(
        self,
        draft_body: str,
        big_model,
        chktex_output: str,
        compile_result: Dict[str, Any],
        impact_info: Optional[str],
        figure_records: List[Dict[str, Any]],
        figure_descriptions: Dict[str, str],
    ) -> str:
        """按章节多次请求改写反思，每次聚焦一个 section 进行针对性扩展。"""
        if not draft_body or not big_model:
            return draft_body

        diagnostics_header = (
            "You are an assistant asked to improve a LaTeX manuscript.\n"
            "Work section by section; return only the updated body text (no fences).\n\n"
            "Diagnostics you must address globally:\n"
            "- Length: expand underspecified parts, especially Method, Experimental Setup, and Experiments.\n"
            "- Include detailed step-by-step explanations, ablation insights, and grounded interpretations of figures/tables.\n"
            f"- Target 70-90% of a {self.config.get('page_limit', 20)}-page double-column budget while remaining truthful to logs/results.\n\n"
            f"CHKTeX output:\n{chktex_output}\n\n"
            f"Compilation logs (truncated):\n{(compile_result.get('logs') or '')[:6000]}\n\n"
            f"Impact info: {impact_info}\n\n"
        )

        figure_hint = "Figure descriptions (figures/...):\n"
        for rec in figure_records:
            rel = rec.get('latex_rel_path') or rec['filename']
            desc = figure_descriptions.get(rec['filename']) or ''
            figure_hint += f"- {rel}: {desc}\n"
        figure_hint += "\n"

        sections = self._extract_section_spans(draft_body)
        if not sections:
            return draft_body

        priority_norms = [
            'method',
            'experimentalsetup',
            'experiments',
            'introduction',
            'relatedwork',
            'background',
            'conclusion',
        ]

        def _priority(section):
            norm = section.get('norm') or ''
            if norm in priority_norms:
                return priority_norms.index(norm)
            return len(priority_norms)

        ordered_sections = sorted(sections, key=_priority)

        updated_body = draft_body
        for section_meta in ordered_sections:
            current_content = updated_body[section_meta['content_start']:section_meta['content_end']].strip()
            if len(current_content.split()) >= int(self.config.get('section_expansion_min_words', 180)):
                continue

            prompt = (
                diagnostics_header
                + figure_hint
                + f"Focus now on the section titled '{section_meta['title']}'.\n"
                "Rewrite this section with multiple paragraphs that expand methodology, empirical evidence, limitations, and future directions relevant to the heading.\n"
                "Preserve existing equations, citations, and figure/table references; insert new figure calls only if justified.\n"
                "Return ONLY the text that should appear immediately after the \\section command (do not repeat the heading).\n"
                "Current section body:<<SECTION>>\n"
                f"<<SECTION>>\n{current_content}\n<<SECTION>>"
            )

            try:
                if hasattr(big_model, 'generate'):
                    resp = await self._maybe_await(big_model.generate(prompt))
                    rewrite = resp.get('text') if isinstance(resp, dict) else str(resp)
                elif hasattr(big_model, 'call'):
                    resp = await self._maybe_await(big_model.call(prompt))
                    rewrite = str(resp)
                else:
                    rewrite = None
            except Exception as exc:
                logger.warning(f"Sectional reflection failed for {section_meta['title']}: {exc}")
                rewrite = None

            rewrite = self._sanitize_latex_body((rewrite or '').strip())
            if not rewrite or rewrite.strip() == current_content.strip():
                continue

            before = updated_body[:section_meta['content_start']]
            after = updated_body[section_meta['content_end']:]
            updated_body = before.rstrip() + "\n\n" + rewrite + "\n\n" + after.lstrip('\n')
            sections = self._extract_section_spans(updated_body)
            match = next((s for s in sections if self._normalize_section_name(s.get('title')) == section_meta['norm']), None)
            if match:
                section_meta = match

        return updated_body

    # --- Main execution ----------------------------------------------------
    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """串联整条论文生成流水线，返回标题、摘要、LaTeX 目录、编译结果等信息。"""
        context = dict(context or {})
        task = context.get("task", {})
        ideas = context.get("ideas", []) or []
        results = context.get("results", []) or []
        session_traj = context.get("session_traj", {}) or {}

        root_output_dir = (
            params.get("root_output_dir")
            or self.config.get("output_dir")
            or _default_results_root(params.get("task_name"))
        )
        iteration = params.get("iteration") or "manual"
        iteration_suffix = f"iteration_{iteration}" if iteration else "manual"

        latex_root = self._init_latex_workspace(root_output_dir, iteration_suffix)
        small_model = self._resolve_small_model(params)
        big_model = self._resolve_big_model(params)
        vlm_model = self._resolve_vlm_model(params)
        params['_vlm_model'] = vlm_model

        idea_result_dirs = self._resolve_iteration_result_dirs(root_output_dir, params)
        figure_records = self._copy_iteration_figures(idea_result_dirs, root_output_dir, latex_root)
        params['_figure_records'] = figure_records
        figure_descriptions = await self._describe_figures_with_vlm(vlm_model, figure_records)
        params['_figure_descriptions'] = figure_descriptions
        figure_prompt_block = self._build_figure_prompt_block(figure_records, figure_descriptions)
        if figure_prompt_block:
            context['figure_prompt_block'] = figure_prompt_block

        logger.info(
            "[PaperGenerationAgent] Collecting citations (small_model=%s, num_cite_rounds=%s)",
            type(small_model).__name__ if small_model else None,
            self.config.get("num_cite_rounds", 2),
        )
        selected_papers = await self._collect_and_append_citations(small_model, context, latex_root)
        logger.info(
            "[PaperGenerationAgent] Citation collection done. references.bib entries=%d",
            len(selected_papers),
        )

        draft_info = await self._generate_draft_and_abstract(context, big_model)
        draft_body = draft_info['draft_body']
        draft_body = self._ensure_figures_in_body(
            draft_body,
            figure_records,
            figure_descriptions or {},
        )
        draft_body = self._remove_missing_figures(latex_root, draft_body)
        draft_body = await self._expand_sections_iteratively(
            draft_body,
            big_model,
            context,
            section_sequence=params.get('section_expansion_sequence'),
            passes=params.get('section_expansion_passes'),
        )
        draft_body = self._remove_missing_figures(latex_root, draft_body)
        if draft_info.get('abstract'):
            params['_generated_abstract'] = draft_info['abstract']
        else:
            params['_generated_abstract'] = params.get('_generated_abstract') or ''

        params['_paper_title'] = task.get('name', 'Paper')

        body_path = self._write_body_file(latex_root, draft_body)
        main_tex = self._prepare_template_files(latex_root, task, draft_body, params)

        refinement = await self._run_refinement_loop(
            draft_body=draft_body,
            body_path=body_path,
            latex_root=latex_root,
            main_tex=main_tex,
            params=params,
            big_model=big_model,
            vlm_model=vlm_model,
            root_output_dir=root_output_dir,
        )
        draft_body = refinement['draft_body']
        pdf_versions = refinement['pdf_versions']

        feedback_llm = await self._generate_text_review(big_model, draft_body, context)
        feedback_vlm = await self._review_figures_with_vlm(vlm_model, figure_records, figure_descriptions)
        revised_body = await self._apply_review_feedback(
            draft_body,
            feedback_llm,
            feedback_vlm,
            big_model,
            figure_records,
            figure_descriptions or {},
        )
        if revised_body.strip() != draft_body.strip():
            logger.info("Applying review-driven revisions to manuscript body")
            draft_body = revised_body
            draft_body = self._ensure_figures_in_body(
                draft_body,
                figure_records,
                figure_descriptions or {},
            )
            draft_body = self._remove_missing_figures(latex_root, draft_body)
            refinement_after_review = await self._run_refinement_loop(
                draft_body=draft_body,
                body_path=body_path,
                latex_root=latex_root,
                main_tex=main_tex,
                params=params,
                big_model=big_model,
                vlm_model=vlm_model,
                root_output_dir=root_output_dir,
                max_reflections=0,
            )
            draft_body = refinement_after_review['draft_body']
            pdf_versions.extend(refinement_after_review.get('pdf_versions', []))
        else:
            logger.info("Review feedback matched current draft; no extra revision pass applied")

        self._ensure_placeholder_bib_entries(draft_body, latex_root)

        compile_timeout = int(self.config.get("compile_timeout", 60))
        compile_result = self._compile_latex(latex_root, timeout=compile_timeout)

        llm_review = await self._generate_text_review(big_model, draft_body, context)
        vlm_review = await self._review_figures_with_vlm(vlm_model, figure_records, figure_descriptions)
        combined_review = "### LLM Review\n" + (llm_review or "(none)") + "\n\n### VLM Review\n" + (vlm_review or "(none)")
        review_path = os.path.join(latex_root, "review_summary.md")
        try:
            with open(review_path, 'w', encoding='utf-8') as rf:
                rf.write("# Review Summary\n\n" + combined_review)
        except Exception:
            review_path = None

        meta = {
            "task": task,
            "iteration": iteration,
            "latex_root": latex_root,
            "compile_result": compile_result,
            "papers_collected": len(selected_papers),
            "generated_at": datetime.now().isoformat(),
            "pdf_versions": pdf_versions,
            "reviews": {
                "llm": llm_review,
                "vlm": vlm_review,
                "review_path": review_path,
            },
        }
        meta_path = os.path.join(latex_root, "paper_meta.json")
        try:
            with open(meta_path, "w", encoding="utf-8") as mf:
                json.dump(meta, mf, indent=2, ensure_ascii=False)
        except Exception:
            logger.warning("Failed to write paper_meta.json")

        result = {
            "title": task.get("name") or "Paper Proposal",
            "abstract": params.get("_generated_abstract") or "",
            "latex_dir": latex_root,
            "pdf_path": compile_result.get("pdf_path") if isinstance(compile_result, dict) else None,
            "compile_logs": compile_result.get("logs") if isinstance(compile_result, dict) else None,
            "meta_path": meta_path,
            "reviews": {
                "llm": llm_review,
                "vlm": vlm_review,
                "review_path": review_path,
            },
        }

        return result
