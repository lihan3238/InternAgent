人机交互：
orchsertration_agent.py
stage.py

绘图的功能补充在：
提示词CODER_PROMPT_AIDER
stage.py
experiments_utils_aider.py
launcher.sh

论文生成智能体：
1、paper_generation_agent.py
2、agent_factory.py新增论文智能体的注册
3、launch_discovery.py加入人机交互的功能



变量控制：
idea数目 top ideas orchestration_agent.py  self.top_ideas_count = workflow_config.get("top_ideas_count", 3)以及deafult.config当中也要改成2
runs的次数 MAX_RUNS 

论文智能体完整的执行流程：
我现在希望每次iteration或者说一次summary调用结束之后都会生成一个论文，将一次iteration当中的所有idea的实验整合生成论文。
1、初始化和环境准备，删除历史生成的 latex 文件夹和 PDF 文件，避免干扰。
从模板（blank_icml_latex）复制基础 LaTeX 结构到新的 latex 文件夹，作为论文编写的初始框架。
2、引用，引用（Citations）收集（多轮搜索）
通过多轮迭代补充参考文献，确保全面性：
循环执行搜索（最多 num_cite_rounds 轮）：
确定缺失引用：LLM 分析当前论文草稿和已有引用，识别需要补充的引用类型（如相关工作、方法依据等），生成搜索关键词。
学术论文搜索：调用 search_for_papers 函数（基于 Semantic Scholar API），根据关键词获取相关论文。
筛选与添加引用：LLM 从搜索结果中选择最合适的论文，提取其 BibTeX 格式引用，清理格式（如去除特殊字符、规范引用键），并添加到 references.bib 文件。
终止条件：当 LLM 判断 “无需更多引用” 或达到最大轮次时停止。
3、论文撰写（LaTeX 生成与优化）
基于实验数据和引用，生成并优化 LaTeX 论文内容：
提示构建：
整合研究主题、实验总结、图表信息、绘图脚本（用于理解图表逻辑）和使用 VLM 分析实验生成的图表描述。
提供会议格式要求（如双栏布局、page_limit 页面限制）和各部分写作规范（标题、摘要、方法等）。
多轮迭代生成：
初始生成：LLM 根据提示生成初稿 LaTeX 代码，确保涵盖所有必要部分（引言、方法、实验、结论等）。
反思优化：通过 n_writeup_reflections 轮迭代，LLM 自我检查并完善内容，修正逻辑漏洞、格式错误（如重复标签、未转义特殊字符），确保图表引用正确。
关键约束：
如实报告实验结果（包括阴性或不确定结果）。
合理组织图表（如需组合子图），并使用正确的文件名引用。
4、 LaTeX 编译与 PDF 生成
将生成的 LaTeX 代码编译为最终 PDF：
编译步骤：
执行一系列 LaTeX 命令（pdflatex、bibtex 等），确保引用、图表正确嵌入。
处理编译超时或错误（如输出日志调试信息）。
PDF 生成：
编译成功后，将 template.pdf 重命名为与实验主题相关的文件名，保存到根目录。
若编译失败（如缺失 “Impact Statement” 等），尝试检测问题并重新编译。

做的很好，还有一些需要更改。1、首先就是我们已经有template的模板了，你可以从这个地方去复制。	强制复制一个固定的 blank_icml_latex 模板。
2、多轮、交互式搜索。LLM 先决定需要什么引文，然后执行搜索，再由 LLM 从结果中选择最相关的。生成标准的 BibTeX。
3、多阶段、多模型: 先用小模型处理引文，再用大模型（如 o1）进行初稿生成，最后进入复杂的反思迭代循环。
4、应该是基于编译结果、chktex 语法检查、页面限制、图表使用情况等进行多维度的反思和优化。LLM 会收到具体的错误和建议。
5、你可以使用pdftotext, chktex 等系统工具。
6、需要输出多个中间 PDF 版本、最终 PDF。
7、你需要使用vlm对上述的iteration当中的多个idea结果进行分析。
接下来，我将给你一个示例，你学习一下，将它的实现过程移植到我们的paper_generation_agent当中

1、首先请你解决这个小模型的配置问题，WARNING:__main__:Could not remove existing latex dir: /mnt/c/git_repos/interns/InternAgent/results/AutoForecast/iteration_1/latex_test/latex_iteration_test1
INFO:__main__:Copied LaTeX template from /mnt/c/git_repos/interns/InternAgent/blank_icml_latex to /mnt/c/git_repos/interns/InternAgent/results/AutoForecast/iteration_1/latex_test/latex_iteration_test1
INFO:__main__:Copied template.tex to main.tex in /mnt/c/git_repos/interns/InternAgent/results/AutoForecast/iteration_1/latex_test/latex_iteration_test1
INFO:__main__:No small model provided; skipping citation collection
2、摘要部分不要从 Introduction 中自动截取前 1-2 句作为摘要；不要这样写，让大模型生成一段独立、完整的摘要段落，摘要是非常重要的一部分，一个文章的精华，需要你单独去对话生成这一部分

