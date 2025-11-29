"""
Survey Agent for InternAgent

This module implements the Survey Agent, which performs comprehensive literature
surveys on research topics. The agent generates intelligent search queries, retrieves
relevant academic papers from multiple sources, scores papers based on relevance,
and performs deep reading analysis to extract methodological details from top papers.
This agent supports automated, iterative literature review with query refinement.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple, Union
import os
from .base_agent import BaseAgent, AgentExecutionError
from ..tools.paper_survey import PaperSurvey
from ..tools.utils import PaperMetadata, parse_io_description, format_papers_for_printing_next_query,\
    download_pdf, extract_text_from_pdf, download_pdf_by_doi, select_papers

logger = logging.getLogger(__name__)


class SurveyAgent(BaseAgent):
    """
    Survey Agent conducts comprehensive literature surveys for research topics.

    This agent performs intelligent literature search by:
    - Generating context-aware search queries based on research topics
    - Retrieving papers from multiple academic sources (Semantic Scholar, arXiv, PubMed)
    - Iteratively refining search queries to expand paper coverage
    - Scoring papers based on relevance, novelty, and methodological quality
    - Performing deep reading analysis on top-ranked papers to extract methodological details

    The agent employs an iterative search strategy that starts with keyword queries
    and progressively diversifies using paper similarity and reference-based queries
    to build a comprehensive literature bank.
    """
    
    def __init__(self, model, config: Dict[str, Any]):
        """
        Initialize the survey agent.
        
        Args:
            model: Language model to use
            config: Configuration dictionary
        """
        super().__init__(model, config)
        
        # Load agent-specific configuration
        self.max_papers = config.get("max_papers", 5)
        self.search_depth = config.get("search_depth", "moderate")  # shallow, moderate, deep
        self.sources = config.get("sources", ["pubmed", "arxiv", "semantic_scholar"])
        
        # Initialize tools
        tools_config = config.get("_global_config", {}).get("tools", {})
        self.paper_survey = None
        self._init_paper_survey(tools_config.get("paper_survey", {}))
        
    def _init_paper_survey(self, config: Dict[str, Any]) -> None:
        """
        Initialize the literature search tool.
        
        Args:
            config: Literature search configuration
        """
        max_results = config.get("max_results", 10)
        sort = config.get("sort", "relevance")
        try:
            self.paper_survey = PaperSurvey(max_results, sort)
            logger.info("Paper survey tool initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize literature search: {str(e)}")
        
    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        """
        papers, _ = await self.advanced_query_paper(context=context)
        
        return papers

    async def advanced_query_paper(self, context) -> Dict[str, Any]:
        
        search_queries = []
        
        goal_description = context.get("description", {})
        domain = context.get("domain", "")
        
        output_schema_paper_score={
            "type": "object",
            "Properties": {
                "^[a-zA-Z0-9_]+$": {
                    "type": "number",
                    "minimum": 1,
                    "maximum": 10
                }
            },
            "description": "A dictionary where each key is a paperID and each value is a score between 1 and 10."
        }
        output_schema_paper_details={
            "type": "object",
            "properties": {
                "background": {
                    "type": "string",
                    "description": "Core problem context and motivation"
                },
                "contributions": {
                    "type": "string",
                    "description": "Novel contributions to the field"
                },
                "methods": {
                    "type": "string",
                    "description": "Key technical approaches/methods used"
                },
                "challenges": {
                    "type": "string",
                    "description": "Limitations or challenges mentioned"
                }
            },
            "required": ["background", "contributions", "methods", "challenges"]
        }
        
        ###
        define_task_attribute_prompt = f"You are a researcher doing research on the topic of {domain}. You should define the task attribute such as the model input and output of the topic for better searching relevant papers. Formulate the input and output as: Attribute(\"attribute\"). For example, Input(\"input\"), Output(\"output\"). The attribute: (just return the task attribute itself with no additional text):"
        try:
            response = await self._call_model(
                prompt=define_task_attribute_prompt
            )
            io_description = parse_io_description(response)

        except Exception as e:
            logger.error(f"Error defining task attribute: {str(e)}")
            raise AgentExecutionError("Failed to define task attribute")
        
        ###
        init_keyword_query_prompt = f"You are a researcher doing literature review on the topic of {goal_description}.\n You should propose some keywords for using the Semantic Scholar API to find the most relevant papers to this topic.Formulate your query as: KeywordQuery(\"keyword\"). \n Just give me one query, with the most important keyword, the keyword can be a concatenation of multiple keywords (just put a space between every word) but please be concise and try to cover all the main aspects.\nYour query (just return the query itself with no additional text):"        
        try:
            response = await self._call_model(
                prompt=init_keyword_query_prompt
            )
            init_query = response

        except Exception as e:
            logger.error(f"Error generating initial keyword query: {str(e)}")
            raise AgentExecutionError("Failed to generate initial keyword query")
        
        init_paper_lst = self.paper_survey.query_route(init_query, 10)
        search_queries.append(init_query)
        
        # make paper bank 
        if init_paper_lst:
            flattened_papers = []
            for source, papers in init_paper_lst.items():
                if isinstance(papers, list):
                    flattened_papers.extend(papers)
                elif isinstance(papers, dict) and "data" in papers:
                    flattened_papers.extend(papers["data"])
                    
            paper_bank = {str(i): paper for i, paper in enumerate(flattened_papers)}
        else:
            # init_paper_lst = []
            logger.warning("No papers found for the initial query")
            paper_bank = {}

        # make advanced query
        grounding_k = 10
        iteration = 0
        while len(paper_bank) < self.max_papers and iteration < 10:
            ## select the top k papers with highest scores for grounding
            data_list = [{'id': id, **info} for id, info in paper_bank.items()]
            grounding_papers = data_list[: grounding_k]
            grounding_papers_str = format_papers_for_printing_next_query(grounding_papers)
            if io_description is not None:
                new_query_prompt = f"You are a researcher doing literature review on the topic of {domain}.\n You should propose some queries for using the Semantic Scholar API to find the most relevant papers to this topic.\n The input and output of the queries should be same with: input: {io_description[0]}, output: {io_description[1]}\n(1) KeywordQuery(\"keyword\"): find most relevant papers to the given keyword (the keyword shouldn't be too long and specific, otherwise the search engine will fail; it is ok to combine a few short keywords with spaces, such as \"lanaguage model reasoning\").\n(2) PaperQuery(\"paperId\"): find the most similar papers to the given paper (as specified by the paperId).\n(3) GetReferences(\"paperId\"): get the list of papers referenced in the given paper (as specified by the paperId).\nRight now you have already collected the following relevant papers: \n{grounding_papers_str}\nYou can formulate new search queries based on these papers. And you have already asked the following queries:\n{search_queries}\nPlease formulate a new query to expand our paper collection with more diverse and relevant papers (you can do so by diversifying the types of queries to generate and minimize the overlap with previous queries). Directly give me your new query without any explanation or additional text, just the query itself:"
            else:
                new_query_prompt = f"You are a researcher doing literature review on the topic of {domain}.\n You should propose some queries for using the Semantic Scholar API to find the most relevant papers to this topic.\n(1) KeywordQuery(\"keyword\"): find most relevant papers to the given keyword (the keyword shouldn't be too long and specific, otherwise the search engine will fail; it is ok to combine a few short keywords with spaces, such as \"lanaguage model reasoning\").\n(2) PaperQuery(\"paperId\"): find the most similar papers to the given paper (as specified by the paperId).\n(3) GetReferences(\"paperId\"): get the list of papers referenced in the given paper (as specified by the paperId).\nRight now you have already collected the following relevant papers: \n{grounding_papers_str}\nYou can formulate new search queries based on these papers. And you have already asked the following queries:\n{search_queries}\nPlease formulate a new query to expand our paper collection with more diverse and relevant papers (you can do so by diversifying the types of queries to generate and minimize the overlap with previous queries). Directly give me your new query without any explanation or additional text, just the query itself:"
            try: 
                response = await self._call_model(
                    prompt=new_query_prompt
                )
                new_query = response

                search_queries.append(new_query)
            except Exception as e:
                logger.error(f"Error generating new query: {str(e)}")
                raise AgentExecutionError("Failed to generate new query")
            
            try:
                logger.info(f"Searching new query {new_query}")
                new_paper_lst = self.paper_survey.query_route(new_query, 10)
            except Exception as e:
                logger.error(f"survey error: {e}")
            
            if new_paper_lst:
                flattened_papers = []
                for source, papers in new_paper_lst.items():
                    if isinstance(papers, list):
                        flattened_papers.extend(papers)
                    elif isinstance(papers, dict) and "data" in papers:
                        flattened_papers.extend(papers["data"])
                existing_titles = {paper['title'] for paper in paper_bank.values()}
                new_papers = [paper for paper in flattened_papers if paper['title'] not in existing_titles]
                logger.info(f"Size of new_papers after filtering: {len(new_papers)}")
                if new_papers:
                    # Assign new unique indices to new papers
                    start_index = len(paper_bank)
                    new_paper_bank = {str(start_index + i): paper for i, paper in enumerate(new_papers)}

                    # Update paper_bank with new papers
                    paper_bank.update(new_paper_bank)
                else:
                    logger.info("No NEW papers found for the query")
            else:
                logger.info("No papers found for the query")
            
            iteration += 1
        
        data_list = [{'id': id, **info} for id, info in paper_bank.items()]
        paper_bank = data_list[:]
        BATCH_SIZE = 10
        
        for batch_index in range(0, len(paper_bank), BATCH_SIZE):
            batch = paper_bank[batch_index:batch_index + BATCH_SIZE]
            abs_batch = [{'id': paper['id'], 'title': paper['title'], 'abstract': paper['abstract']} for paper in batch]
            if io_description is not None:
                paper_score_prompt = f"You are a helpful literature review assistant whose job is to read the below set of papers and score each paper.The criteria for scoring are: \n (1) The paper is directly relevant to the topic of: {domain}. Note that it should be specific to solve the problem of focus, rather than just generic methods. \n (2) The input and output of the proposed method in this paper is same with input: {io_description[0]}, output: {io_description[1]}. Note that if the input and output are not match, the paper should get a low score. \n (3) The paper is an empirical paper that proposes a novel method and conducts computational experiments to show improvement over baselines (position or opinion papers, review or survey papers, and analysis papers should get low scores for this purpose). \n (4) The paper is interesting, exciting, and meaningful, with potential to inspire many new projects. \n The papers are: \n {abs_batch} \n Please score each paper from 1 to 10. \n Write the response in JSON format with \"paperID: score\" as the key and value for each paper. \n\n ONLY output the JSON dict with NO additional text. DO NOT output newline characters. DO NOT output any markdown modifier so that we can call json.loads() on the output later."
            else:
                paper_score_prompt = f"You are a helpful literature review assistant whose job is to read the below set of papers and score each paper.The criteria for scoring are: \n (1) The paper is directly relevant to the topic of: {domain}. Note that it should be specific to solve the problem of focus, rather than just generic methods. \n (2) The paper is an empirical paper that proposes a novel method and conducts computational experiments to show improvement over baselines (position or opinion papers, review or survey papers, and analysis papers should get low scores for this purpose). \n (3) The paper is interesting, exciting, and meaningful, with potential to inspire many new projects. \n The papers are: \n {abs_batch} \n Please score each paper from 1 to 10. \n MUST Write the response in JSON format with \"paperID: score\" as the key and value for each paper. \n\n ONLY output the JSON dict with NO additional text. DO NOT output newline characters. DO NOT output any markdown modifier so that we can call json.loads() on the output later."
         
            try:
                response = await self._call_model(
                    prompt=paper_score_prompt,
                    schema=output_schema_paper_score
                )
            except Exception as e:
                logger.error(f"Failed to score papers: {e} {response}")
                raise AgentExecutionError(f"Failed to score papers{response}")

            for key, score in response.items():
                # actual_paper_id = batch_index + int(key)
                actual_paper_id = int(key)
                if 0 <= actual_paper_id < len(paper_bank):
                    paper_bank[actual_paper_id]['score'] = score
                else:
                    print(f"Warning: Index '{actual_paper_id}' out of range in paper_bank.")
        
        logger.info(f"Number of papers in paper_bank: {len(paper_bank)}")
        logger.debug("Final paper_bank: ", paper_bank)
        
        rag_read_depth = 3
        selected_for_deep_read = select_papers(paper_bank, self.max_papers, rag_read_depth)
        
        for paper in selected_for_deep_read:
            paper_id = paper["id"]
            url = None
            if paper['source'] in ['arXiv', 'pubmed']:
                url = paper.get('url') or paper.get('doi')
            elif paper['source'] == 'semantic_scholar':
                if paper.get('isOpenAccess', False):
                    url = paper['openAccessPdf']['url']
            
            print("paper_id:", paper_id, "url:", url)
            base_dir = 'tmp'
            if url:
                pdf_dir = os.path.join(base_dir, "pdf")
                if not os.path.exists(pdf_dir):
                    os.makedirs(pdf_dir)

                if paper['source'] in ["semantic_scholar", "arXiv"]:
                    pdf_path = download_pdf(url, save_folder=pdf_dir)
                elif paper['source'] == "pubmed":
                    pdf_path = download_pdf_by_doi(doi=url, download_dir=pdf_dir)
                
                if pdf_path:
                    text = extract_text_from_pdf(pdf_path)
                    if text:
                        get_detail_prompt = f"Analyze the following paper text and extract structured information:{text}\nExtract:\n- Background: Core problem context and motivation\n- Contributions: Novel contributions to the field\n- Methods: Key technical approaches/methods used\n- Challenges: Limitations or challenges mentioned\n\nReturn JSON format with keys: methods, contributions, background, challenges. Use concise technical language.\n\n Using JSON for response format: \"background: ...\", \"contributions: ...\",\"methods: ...\", \"challenges: ...\" ONLY output the JSON dict with NO additional text. DO NOT output newline characters. DO NOT output any markdown modifier so that we can call json.loads() on the output later."
                        try:
                            response = await self._call_model(
                                prompt=get_detail_prompt,
                                schema=output_schema_paper_details
                            )
                            details = response
                        except Exception as e:
                            logger.error(f"survey error: {e}")
                            raise AgentExecutionError("Failed to get paper details")
                        
                        if details:
                            paper["background"] = details.get("background", "")                  
                            paper["contributions"] = details.get("contributions", "")
                            paper["methods"] = details.get("methods", "")
                            paper["challenges"] = details.get("challenges", "")
                        else:
                            paper["background"] = None
                            paper["contributions"] = None
                            paper["methods"] = None
                            paper["challenges"] = None
      
        for paper in paper_bank:
            paper['is_deep_read'] = paper['id'] in [p['id'] for p in selected_for_deep_read]
            
        # Extract optional parameters
        
        
        return paper_bank, search_queries

    
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
        prompt = f"# Research Goal\n{goal_description}\n\n"
        prompt += f"# Hypothesis\n{hypothesis_text}\n\n"
        
        # Add task description
        prompt += "# Task\n"
        prompt += "Generate effective search queries to find scientific literature that could provide evidence "
        prompt += "related to the hypothesis above. "
        
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
        prompt += "Use Boolean operators (AND, OR) and special syntax when helpful."
        
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
- Use Boolean operators (AND, OR) and special syntax appropriately
- Be specific enough to find relevant papers but not so narrow that important evidence is missed
- Consider different aspects of the hypothesis that might be explored in separate literature
- Prioritize search terms likely to yield empirical evidence rather than theoretical papers
"""
        
        try:
            response = await self._call_model(
                prompt=prompt,
                system_prompt=system_prompt,
                schema=output_schema
            )
            
            # Extract queries
            queries = response.get("search_queries", [])
            
            # Limit the number of queries based on search depth
            queries = queries[:num_queries]
            
            if not queries:
                # Fallback if no queries were generated
                queries = [hypothesis_text]
                
            return queries
            
        except Exception as e:
            logger.error(f"Error generating search queries: {str(e)}")
            # Fallback
            return [hypothesis_text]
    
    async def _gather_literature_evidence(self,
                                       search_queries: List[str],
                                       hypothesis: Dict[str, Any],
                                       max_papers: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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
                logger.warning("No papers found for the search queries")
                return evidence, references
            
            # Evaluate relevance of each paper
            hypothesis_text = hypothesis.get("text", "")
            relevant_papers = await self._evaluate_paper_relevance(
                papers=unique_papers,
                hypothesis=hypothesis_text
            )
            
            # Create evidence items from relevant papers
            for paper, relevance_score, relevance_note in relevant_papers:
                if relevance_score >= self.evidence_threshold:
                    # Add as evidence
                    evidence_item = {
                        "source": "literature",
                        "title": paper.title,
                        "authors": ", ".join(paper.authors[:3]) + ("..." if len(paper.authors) > 3 else ""),
                        "year": paper.year or "Unknown",
                        "content": paper.abstract[:300] + "..." if len(paper.abstract) > 300 else paper.abstract,
                        "relevance": relevance_note,
                        "relevance_score": relevance_score,
                        "url": paper.url or "",
                        "doi": paper.doi or ""
                    }
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
                                     hypothesis: str) -> List[Tuple[PaperMetadata, float, str]]:
        """
        Evaluate the relevance of papers to the hypothesis.
        
        Args:
            papers: List of papers
            hypothesis: Hypothesis text
            
        Returns:
            List of tuples (paper, relevance_score, relevance_note)
        """
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
            prompt = f"# Hypothesis\n{hypothesis}\n\n"
            prompt += "# Scientific Papers\n"
            
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
                    paper_idx = eval_data.get("paper_index", 0) - 1  # Convert to 0-indexed
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
