"""
Scholar Agent for InternAgent

This module implements the Scholar Agent, which interfaces with external
tools and databases to gather evidence relevant to research hypotheses.
"""

import logging
import os
from typing import Dict, Any, List, Tuple

from .base_agent import BaseAgent, AgentExecutionError
from ..tools.literature_search import LiteratureSearch, PaperMetadata
from ..tools.utils import download_pdf, extract_text_from_pdf, download_pdf_by_doi, replace_and_with_or

logger = logging.getLogger(__name__)


class ScholarAgent(BaseAgent):
    """
    Scholar Agent gathers external evidence for research hypotheses.

    This agent connects with external tools, databases, and literature
    to find supporting or contradicting evidence for hypotheses and
    ground them in established research.
    """
    
    def __init__(self, model, config: Dict[str, Any]):
        """
        Initialize the scholar agent.
        
        Args:
            model: Language model to use
            config: Configuration dictionary
        """
        super().__init__(model, config)
        
        # Load agent-specific configuration
        self.max_papers = config.get("max_papers", 5)
        self.search_depth = config.get("search_depth", "moderate")  # shallow, moderate, deep
        self.evidence_threshold = config.get("evidence_threshold", 0.6)  # Minimum relevance score
        self.sources = config.get("sources", ["pubmed", "arxiv", "semantic_scholar"])
        
        # Initialize tools
        tools_config = config.get("_global_config", {}).get("tools", {})
        self.literature_search = None
        self._init_literature_search(tools_config.get("literature_search", {}))
        self.deep_read = config.get("deep_read",False)
        self.temperature = config.get("temperature", None)

    def _init_literature_search(self, config: Dict[str, Any]) -> None:
        """
        Initialize the literature search tool.
        
        Args:
            config: Literature search configuration
        """
        email = config.get("email", "researcher@example.com")
        api_keys = config.get("api_keys", {})
        
        try:
            self.literature_search = LiteratureSearch(
                email=email,
                api_keys=api_keys
            )
            logger.info("Literature search tool initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize literature search: {str(e)}")
        
    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gather external evidence for a research hypothesis.
        
        Args:
            context: Dictionary containing:
                - goal: Research goal information
                - hypothesis: The hypothesis to gather evidence for
                - iteration: Current iteration number
            params: Dictionary containing optional configuration overrides
                
        Returns:
            Dictionary containing:
                - evidence: List of evidence items
                - references: List of references
                - relevance_summary: Summary of relevance to hypothesis
        """
        # Extract parameters
        goal = context.get("goal", {})
        hypothesis = context.get("hypothesis", {})
        # feedback = context.get("feedback", [])
        feedback = []
        
        if not goal or not hypothesis:
            raise AgentExecutionError("Research goal and hypothesis are required for scholar search")
        
        # Extract text from hypothesis
        hypothesis_text = hypothesis.get("text", "")
        if not hypothesis_text:
            raise AgentExecutionError("Hypothesis text is required for scholar search")
            
        # Extract optional parameters
        iteration = context.get("iteration", 0)
        max_papers = params.get("max_papers", self.max_papers)
        search_depth = params.get("search_depth", self.search_depth)
        method_phase = params.get("method_phase", False)
        
        # Prepare search queries
        search_queries = await self._generate_search_queries(
            goal=goal,
            hypothesis=hypothesis,
            search_depth=search_depth,
            feedback=feedback
        )
        
        # Gather evidence from literature
        evidence, references = await self._gather_literature_evidence(
            search_queries=search_queries,
            hypothesis=hypothesis,
            max_papers=max_papers,
            method_phase=method_phase
        )
        
        # Generate relevance summary
        relevance_summary = await self._generate_relevance_summary(
            hypothesis=hypothesis,
            evidence=evidence
        )
        
        # Build the result
        result = {
            "evidence": evidence,
            "references": references,
            "relevance_summary": relevance_summary,
            "metadata": {
                "hypothesis_id": hypothesis.get("id", ""),
                "search_queries": search_queries,
                "search_depth": search_depth,
                "sources": self.sources
            }
        }
        
        return result
    
    async def _generate_search_queries(self,
                                    goal: Dict[str, Any],
                                    hypothesis: Dict[str, Any],
                                    feedback: List[Dict[str, Any]],
                                    search_depth: str) -> List[str]:
        """
        Generate search queries based on the hypothesis.
        
        Args:
            goal: Research goal dictionary
            hypothesis: Hypothesis dictionary
            search_depth: Search depth (shallow, moderate, deep)
            
        Returns:
            List of search queries
        """
        # Extract text
        goal_description = goal.get("description", "")
        hypothesis_text = hypothesis.get("text", "")
        
        # Create a JSON schema for the expected output
        output_schema = {
            "type": "object",
            "properties": {
                "search_queries": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of search queries for literature databases"
                },
                "ratinale": {
                    "type": "string",
                    "description": "Rationale for the search queries"
                }
            },
            "required": ["search_queries"]
        }
        
        # Build the prompt
        # Add task description
        if not hypothesis.get("method_critiques", ""):
            prompt = f"# Research Goal\n{goal_description}\n\n"
            prompt += f"# Hypothesis\n{hypothesis_text}\n\n"
            prompt += "# Task\n"
            prompt += "Generate effective search queries to find scientific literature that could provide evidence related to the hypothesis above. "
            
            if search_depth == "shallow":
                prompt += "Generate 1-2 focused, specific queries targeting the most central aspect of the hypothesis."
                num_queries = 2
            elif search_depth == "deep":
                prompt += "Generate 4-6 diverse queries covering different aspects, mechanisms, and implications of the hypothesis."
                num_queries = 6
            else:  # moderate
                prompt += "Generate 2-4 balanced queries covering the main aspects of the hypothesis."
                num_queries = 4
                
            prompt += "\n\nFor each query, focus on scientific terminology likely to appear in academic publications. "
            # prompt += "Use Boolean operator (you can ONLY use 'OR' operator) and special syntax when helpful."
            prompt += "Use Boolean operator (you can ONLY use 'OR' operator) when helpful"

            # Add recent feedback
            if feedback:
                prompt += "# Scientist Feedback\n"
                # Sort by iteration and take the most recent
                recent_feedback = sorted(
                    feedback, 
                    key=lambda x: x.get("iteration", 0),
                    reverse=True
                )[:3]
                
                for entry in recent_feedback:
                    feedback_text = entry.get("text", "")
                    feedback_iter = entry.get("iteration", 0)
                    
                    if feedback_text:
                        prompt += f"From iteration {feedback_iter}: {feedback_text}\n\n"
            
            # Call the model
            system_prompt = """You are a scientific literature search specialist.
    Your task is to formulate effective search queries for academic databases based on scientific hypotheses.

    Guidelines:
    - Create queries using scientific terminology likely to appear in research papers
    - Use Boolean operator (you can ONLY use 'OR' operator) when helpful
    - Be specific enough to find relevant papers but not so narrow that important evidence is missed
    - Consider different aspects of the hypothesis that might be explored in separate literature
    - Prioritize search terms likely to yield empirical evidence rather than theoretical papers
    """
        else:
            # 修改后基于方法详情和评价的搜索查询prompt
            prompt = f"# Research Goal\n{goal_description}\n\n"
            prompt += f"# Hypothesis\n{hypothesis_text}\n\n"

            # 添加方法详情
            method_details = hypothesis["method_details"]
            method_critiques = hypothesis["method_critiques"]
            prompt += "# Method Details\n"
            method_overview = method_details["description"]
            method_statement = method_details["statement"] 
            method_explanation = method_details["method"] 

            prompt += f"## Overview\n{method_overview}\n\n"
            prompt += f"## Statement\n{method_statement}\n\n"
            prompt += f"## Detailed Explanation\n{method_explanation}\n\n"

            prompt += "# Method Critiques\n"
            
            # 过滤优先级高的方法相关评价
            high_priority_critiques = []
            for critique in method_critiques:
                category = critique.get("category", "")
                point = critique.get("point", "")
                severity = critique.get("severity", "minor")
                
                # 关注方法本身的技术问题，特别是severity为major和moderate的
                if severity in ["major", "moderate"] and category.lower() not in ["data processing", "evaluation", "testing"]:
                    high_priority_critiques.append({
                        "category": category,
                        "point": point,
                        "severity": severity
                    })
            
            # 添加高优先级的问题到prompt
            if high_priority_critiques:
                for i, critique in enumerate(high_priority_critiques):
                    prompt += f"## Critique {i+1}\n"
                    prompt += f"Category: {critique['category']}\n"
                    prompt += f"Severity: {critique['severity']}\n"
                    prompt += f"Point: {critique['point']}\n\n"

            # 添加任务描述
            prompt += "# Task\n"
            prompt += "Generate effective search queries to find scientific literature that could help address the specific methodological challenges and improve the proposed method. "

            # 根据search_depth调整搜索查询的数量和范围
            if search_depth == "shallow":
                prompt += "Generate 1-2 focused, specific queries targeting the most critical methodological issues identified in the critiques."
                num_queries = 2
            elif search_depth == "deep":
                prompt += "Generate 4-6 diverse queries covering different technical aspects of the method that need improvement, alternative approaches, and potential solutions to the identified issues."
                num_queries = 6
            else:  # moderate
                prompt += "Generate 2-4 balanced queries covering the main methodological challenges and potential solutions."
                num_queries = 4
                
            prompt += "\n\nFor each query, focus on scientific and technical terminology likely to appear in academic publications related to the specific method components that need improvement. "
            prompt += "Use Boolean operator (you can ONLY use 'OR' operator) when helpful. Prioritize searches that would yield papers with concrete techniques, algorithms, or mathematical formulations that could address the identified issues."

            system_prompt = """You are a scientific literature search specialist with expertise in methodology and algorithm development.
Your task is to formulate effective search queries for academic databases that can help address specific methodological challenges.

Guidelines:
- Create queries targeting scientific literature that addresses the specific technical issues identified in the method critiques
- Focus on technical terminology related to algorithms, mathematical formulations, and methodological approaches
- Use Boolean operator (you can ONLY use 'OR' operator) when helpful
- Balance specificity (to find directly relevant papers) with breadth (to discover alternative approaches)
- Prioritize search terms that would yield:
  * Papers with solutions to similar technical challenges
  * Alternative mathematical formulations or algorithmic approaches
  * Theoretical foundations that could strengthen the method
  * State-of-the-art techniques in the relevant domain
- For each query, briefly explain what technical aspect it targets and how the results might help improve the method
"""
        try:
            response = await self._call_model(
                prompt=prompt,
                system_prompt=system_prompt,
                schema=output_schema,
                temperature=self.temperature
            )
            
            # Extract queries
            queries = response.get("search_queries", [])
            loose_queries = []
            for query in queries:
                if "AND" in query:
                    query = replace_and_with_or(query, 1)
                loose_queries.append(query)
                
            queries = loose_queries    
            # Limit the number of queries based on search depth
            queries = queries[:num_queries]
            
            if not queries:
                # Fallback if no queries were generated
                queries = [hypothesis_text]
            else:
                queries = [q.replace('"', '') for q in queries]
                
            return queries
            
        except Exception as e:
            logger.error(f"Error generating search queries: {str(e)}")
            # Fallback
            return [hypothesis_text]
    
    async def _gather_literature_evidence(self,
                                       search_queries: List[str],
                                       hypothesis: Dict[str, Any],
                                       max_papers: int,
                                       method_phase: bool) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Gather evidence from scientific literature.
        
        Args:
            search_queries: List of search queries
            hypothesis: Hypothesis dictionary
            max_papers: Maximum number of papers to retrieve
            
        Returns:
            Tuple of (evidence items, references)
        """
        evidence = []
        references = []
        
        # Check if literature search is available
        if not self.literature_search:
            logger.warning("Literature search tool not available")
            return evidence, references
        
        try:
            # Gather papers from multiple sources
            all_papers = []
            
            # Execute each query
            for query in search_queries:
                try:
                    # Search across multiple sources
                    results = await self.literature_search.multi_source_search(
                        query=query,
                        sources=self.sources,
                        max_results=max_papers
                    )
                    
                    # Extract papers from results
                    for source, papers in results.items():
                        all_papers.extend(papers)
                        
                except Exception as e:
                    logger.error(f"Error searching with query '{query}': {str(e)}")
            
            # Remove duplicates (by DOI or title)
            unique_papers = []
            seen_dois = set()
            seen_titles = set()
            
            for paper in all_papers:
                doi = paper.doi
                title = paper.title.lower()
                
                if doi and doi in seen_dois:
                    continue
                if title in seen_titles:
                    continue
                    
                if doi:
                    seen_dois.add(doi)
                seen_titles.add(title)
                unique_papers.append(paper)
            
            # Limit to max papers
            unique_papers = unique_papers[:max_papers]
            
            if not unique_papers:
                logger.warning(f"No papers found for the search queries {search_queries} {all_papers}")
                return evidence, references
            
            # Evaluate relevance of each paper
            relevant_papers = await self._evaluate_paper_relevance(
                papers=unique_papers,
                hypothesis=hypothesis
            )
            
            # Create evidence items from relevant papers
            read_paper_method_count = 0
            for paper, relevance_score, relevance_note in relevant_papers:
                if relevance_score >= self.evidence_threshold:
                    method = None
                    # Only attempt to extract the method if both self.deep_read and method_phase are True
                    if self.deep_read and method_phase:
                        if read_paper_method_count < 3:
                            try:
                                method = await self.paper_extract_method(paper)
                            except Exception as e:
                                logger.error(f"Error extracting method for {paper.title}: {str(e)}")
                                method = "Methodology extraction failed"
                            read_paper_method_count += 1  

                    # Build the evidence item with the method field only if it was set
                    evidence_item = {
                        "source": "literature",
                        "title": paper.title,
                        "authors": ", ".join(paper.authors[:3]) + ("..." if len(paper.authors) > 3 else ""),
                        "year": paper.year or "Unknown",
                        "content": paper.abstract,  # paper.abstract[:300] + "..." if len(paper.abstract) > 300 else paper.abstract
                        "relevance": relevance_note,
                        "relevance_score": relevance_score,
                        "url": paper.url or "",
                        "doi": paper.doi or ""
                    }

                    # Add the 'method' field only if method was extracted
                    if method is not None:
                        evidence_item["method"] = method
                    
                    evidence.append(evidence_item)
                
                # Add as reference
                ref_item = {
                    "title": paper.title,
                    "authors": paper.authors,
                    "year": paper.year,
                    "journal": paper.journal,
                    "doi": paper.doi,
                    "url": paper.url,
                    "citation": paper.to_citation(format_type="apa")
                }
                references.append(ref_item)
            
            return evidence, references
            
        except Exception as e:
            logger.error(f"Error gathering literature evidence: {str(e)}")
            return evidence, references
    
    async def _evaluate_paper_relevance(self,
                                     papers: List[PaperMetadata],
                                     hypothesis: Dict[str, Any]) -> List[Tuple[PaperMetadata, float, str]]:
        """
        Evaluate the relevance of papers to the hypothesis.
        
        Args:
            papers: List of papers
            hypothesis: Hypothesis 
            
        Returns:
            List of tuples (paper, relevance_score, relevance_note)
        """

        hypothesis_text = hypothesis.get("text", "")

        if not papers:
            return []
            
        # Prepare batches to avoid too large prompts
        batch_size = 3
        paper_batches = [papers[i:i+batch_size] for i in range(0, len(papers), batch_size)]
        
        all_results = []
        
        for batch in paper_batches:
            # Create a JSON schema for the expected output
            output_schema = {
                "type": "object",
                "properties": {
                    "paper_evaluations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "paper_index": {
                                    "type": "integer",
                                    "description": "Index of the paper in the list"
                                },
                                "relevance_score": {
                                    "type": "number",
                                    "description": "Relevance score from 0.0 to 1.0"
                                },
                                "relevance_note": {
                                    "type": "string",
                                    "description": "Brief explanation of relevance"
                                },
                                "supports_or_contradicts": {
                                    "type": "string",
                                    "enum": ["supports", "contradicts", "neutral", "unclear"],
                                    "description": "Whether the paper supports or contradicts the hypothesis"
                                }
                            },
                            "required": ["paper_index", "relevance_score", "relevance_note", "supports_or_contradicts"]
                        }
                    }
                },
                "required": ["paper_evaluations"]
            }
            
            # Build the prompt
            prompt = f"# Hypothesis\n{hypothesis_text}\n\n"
            prompt += "# Scientific Papers\n"
            
            if hypothesis.get("method_details", ""):
                method_details = hypothesis["method_details"]

                prompt += f"## Overview\n{method_details['description']}\n\n"
                prompt += f"## Statement\n{method_details['statement']}\n\n"
                prompt += f"## Detailed Explanation\n{method_details['method']}\n\n"

                prompt += "# Method Critiques\n"
                method_critiques = hypothesis.get("method_critiques", [])
                high_priority_critiques = []
                for critique in method_critiques:
                    category = critique.get("category", "")
                    point = critique.get("point", "")
                    severity = critique.get("severity", "minor")
                    
                    if severity in ["major", "moderate"] and category.lower() not in ["data processing", "evaluation", "testing"]:
                        high_priority_critiques.append({
                            "category": category,
                            "point": point,
                            "severity": severity
                        })
                
                    if high_priority_critiques:
                        for i, critique in enumerate(high_priority_critiques):
                            prompt += f"## Critique {i+1}\n"
                            prompt += f"Category: {critique['category']}\n"
                            prompt += f"Severity: {critique['severity']}\n"
                            prompt += f"Point: {critique['point']}\n\n"

            for i, paper in enumerate(batch):
                prompt += f"\n## Paper {i+1}\n"
                prompt += f"Title: {paper.title}\n"
                prompt += f"Authors: {', '.join(paper.authors)}\n"
                prompt += f"Year: {paper.year or 'Unknown'}\n"
                if paper.journal:
                    prompt += f"Journal: {paper.journal}\n"
                prompt += f"Abstract: {paper.abstract}\n"
            
            # Add task description
            prompt += "\n# Task\n"
            prompt += "Evaluate the relevance of each paper to the hypothesis. For each paper:\n"
            prompt += "1. Assign a relevance score from 0.0 (not relevant) to 1.0 (highly relevant)\n"
            prompt += "2. Provide a brief explanation of why the paper is relevant or not\n"
            prompt += "3. Indicate whether the paper supports, contradicts, or is neutral toward the hypothesis\n"
            
            # Call the model
            system_prompt = """You are a scientific research evaluator.
Your task is to assess the relevance of scientific papers to a given hypothesis.

Guidelines:
- Focus on the scientific content and findings, not just keyword matches
- Consider methodological relevance and theoretical frameworks
- Identify whether papers provide supporting or contradicting evidence
- Be objective and precise in your evaluations
- Provide specific details about how each paper relates to the hypothesis
"""
            
            try:
                response = await self._call_model(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    schema=output_schema
                )
                
                # Process evaluations
                evaluations = response.get("paper_evaluations", [])
                
                for eval_data in evaluations:
                    paper_idx = eval_data.get("paper_index", 1) - 1  # Convert to 0-indexed
                    if 0 <= paper_idx < len(batch):
                        paper = batch[paper_idx]
                        score = eval_data.get("relevance_score", 0.0)
                        note = eval_data.get("relevance_note", "")
                        supports = eval_data.get("supports_or_contradicts", "neutral")
                        
                        # Enhance note with support information
                        enhanced_note = f"{note} [{supports.capitalize()}]"
                        
                        all_results.append((paper, score, enhanced_note))
                
            except Exception as e:
                logger.error(f"Error evaluating paper relevance: {str(e)}")
                # Add papers with default values in case of error
                for paper in batch:
                    all_results.append((paper, 0.5, "Relevance uncertain due to evaluation error"))
        
        # Sort by relevance score (descending)
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        return all_results
    
    async def _generate_relevance_summary(self,
                                       hypothesis: Dict[str, Any],
                                       evidence: List[Dict[str, Any]]) -> str:
        """
        Generate a summary of evidence relevance to the hypothesis.
        
        Args:
            hypothesis: Hypothesis dictionary
            evidence: List of evidence items
            
        Returns:
            Relevance summary string
        """
        if not evidence:
            return "No relevant evidence found in the literature."
            
        hypothesis_text = hypothesis.get("text", "")
        
        # Build the prompt
        prompt = f"# Hypothesis\n{hypothesis_text}\n\n"
        prompt += "# Evidence from Literature\n"
        
        for i, item in enumerate(evidence, 1):
            prompt += f"\n## Evidence {i}\n"
            prompt += f"Source: {item.get('title', 'Unknown paper')}\n"
            prompt += f"Authors: {item.get('authors', 'Unknown')}\n"
            prompt += f"Year: {item.get('year', 'Unknown')}\n"
            prompt += f"Content: {item.get('content', '')}\n"
            prompt += f"Relevance: {item.get('relevance', '')}\n"
        
        # Add task description
        prompt += "\n# Task\n"
        prompt += "Synthesize the collected evidence and provide a concise summary of how it relates to the hypothesis. "
        prompt += "Address whether the evidence generally supports, contradicts, or provides a mixed picture for the hypothesis. "
        prompt += "Highlight any significant gaps in the evidence."
        
        # Call the model
        system_prompt = """You are a scientific evidence synthesizer.
Your task is to summarize how a collection of evidence relates to a scientific hypothesis.

Guidelines:
- Be objective and balanced in your assessment
- Synthesize across different pieces of evidence to identify patterns
- Highlight both supporting and contradicting evidence
- Identify gaps or limitations in the available evidence
- Keep your summary concise and focused on relevance to the hypothesis
"""
        
        try:
            response = await self._call_model(
                prompt=prompt,
                system_prompt=system_prompt
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating relevance summary: {str(e)}")
            return "Unable to generate evidence summary due to an error." 
    
    async def paper_extract_method(self, paper: PaperMetadata) -> str:
        methodology_prompt = """
**Core Tasks**
Analyze the methodology of the provided research paper with an emphasis on uncovering the authors' thought process, reasoning, and logical progression. Your analysis should detail not only what methods were used but also the reasoning behind these choices, the evolution of ideas, and how experimental evidence supports or refutes their hypotheses. Base your analysis strictly on the paper's content while providing relevant context where necessary.

To complete this core task, You are responsible for analyzing the **Methodological Reasoning and Evolution** section of the research paper, focusing on how the authors developed and refined their approach. Your response should include:

1. **Step-by-Step Refinement**  
   - Identify any experiments or analyses that led to modifications in their approach.  
   - How did the authors logically connect initial observations to the design of specific algorithms, architectures, or training paradigms?  

2. **Formal Representations**  
   - Extract and analyze key equations or mathematical formalisms that illustrate critical methodological steps.  
   - Use LaTeX formatting where necessary to present equations clearly.

3. **Logical Progression**  
   - Detail the progression of ideas, showing how each step builds upon the previous one.  
   - Highlight any major shifts in methodology and the reasoning behind them.

### Additional Guidelines:
- Ensure your analysis is deeply rooted in the paper’s text; do not generate technical details that are not explicitly mentioned.
- Focus on the chain of thought, step-by-step reasoning, and logical evolution of ideas as conveyed by the authors.
- When referring to specific experimental findings (e.g., a table, section, or figure), provide detailed descriptions and analyses rather than summarizing with just a label.
- While the structure can be flexible, your response should clearly illustrate how the authors moved from initial observations to conclusions through systematic reasoning and experimental validation.
- Use technical precision, and where applicable, include relevant mathematical notation and comparisons to prior work as reported in the paper.
- Present your analysis in Markdown format for clarity and readability. Use a level-one heading (# ) at the beginning to emphasize your current analysis topic, but do not use level-one headings elsewhere.
- The Previous Analysis of the research paper are also provided for your reference.

# Input Paper:
{input_paper}

"""
        if not paper.url:
            return "No PDF URL available for methodology extraction"
        try:   
            base_dir = 'tmp'
            if paper.url:
                pdf_dir = os.path.join(base_dir, "pdf")
                if not os.path.exists(pdf_dir):
                    os.makedirs(pdf_dir)
            if "arxiv" in paper.url:
                url = paper.url.replace("abs", "pdf")
            else:
                url = paper.url
            pdf_path = download_pdf(url, save_folder=pdf_dir)

            if pdf_path is None and paper.doi:
                pdf_path = download_pdf_by_doi(paper.doi, pdf_dir)
            
            text = extract_text_from_pdf(pdf_path)
            input_paper = text
            response = await self._call_model(
                prompt = methodology_prompt.format(input_paper=input_paper)
            )

        except Exception as e:
            logger.error(f"Method extraction failed for {paper.title}: {str(e)}")
            return f"Methodology analysis error: {str(e)}"
        return response
    
