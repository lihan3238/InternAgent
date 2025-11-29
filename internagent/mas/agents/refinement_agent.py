"""
Refinement Agent for InternAgent

Implements the method refinement agent that improves developed methods by addressing
critiques, incorporating literature insights, and strengthening theoretical foundations.
Produces enhanced methods with improved mathematical rigor, algorithmic clarity, and
implementation feasibility while maintaining 1-2 core contributions.
"""

import logging
import os
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent, AgentExecutionError

logger = logging.getLogger(__name__)


class RefinementAgent(BaseAgent):
    """
    Agent that refines methods by addressing critiques and integrating literature.

    Enhances developed methods by systematically addressing critical issues, incorporating
    relevant literature insights, strengthening mathematical foundations, and optimizing
    algorithmic structure. Produces improved methods that resolve identified limitations
    while preserving core innovations and maintaining focus on 1-2 key contributions.

    Attributes:
        improvement_focus (List[str]): Focus areas for improvement
    """
    
    def __init__(self, model, config: Dict[str, Any]):
        """
        Initialize the refinement agent with model and configuration.

        Args:
            model (BaseModel): Language model for method refinement
            config (Dict[str, Any]): Configuration with keys:
                - improvement_focus (List[str]): Focus areas (default: theoretical/practical/algorithmic)
        """
        super().__init__(model, config)
        
        # Load agent-specific configuration
        self.improvement_focus = config.get("improvement_focus", ["theoretical", "practical", "algorithmic"])
        
    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine method by addressing critiques and integrating literature.

        Produces improved method version that systematically resolves identified issues,
        incorporates relevant literature insights, strengthens theoretical foundations,
        and enhances practical implementability. Returns gracefully with original method
        if refinement fails.

        Args:
            context (Dict[str, Any]): Execution context with keys:
                - goal (Dict): Research goal and domain
                - hypothesis (Dict): Hypothesis with method_details and method_critiques
                - literature (List[Dict]): Relevant scientific literature (optional)
                - feedback (List[Dict]): Scientist feedback (optional)
                - iteration (int): Current iteration number
            params (Dict[str, Any]): Runtime parameters (currently unused)

        Returns:
            Dict[str, Any]: Refinement results containing:
                - refined_method (Dict): Improved method with name/title/description/statement/method
                - metadata (Dict): Refinement context and error status

        Raises:
            AgentExecutionError: If goal/hypothesis/method_details missing
        """
        # Extract parameters
        goal = context.get("goal", {})
        hypothesis = context.get("hypothesis", {})
        literature = context.get("literature", [])
        feedback = context.get("feedback", [])
        
        if not goal or not hypothesis:
            raise AgentExecutionError("Research goal and hypothesis are required for method refinement")
        
        # Extract method details from hypothesis
        method_details = hypothesis.get("method_details", {})
        method_critiques = hypothesis.get("method_critiques", [])
            
        if not method_details:
            raise AgentExecutionError("Method details are required for method refinement")
            
        # Extract optional parameters
        iteration = context.get("iteration", 0)
        
        # Create a JSON schema for the expected output
        output_schema = {
            "type": "object",
            "properties": {
                "refined_method": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "A concise name for the improved method"
                        },
                        "title": {
                            "type": "string",
                            "description": "A descriptive title for the improved method"
                        },
                        "description": {
                            "type": "string",
                            "description": "High-level overview of the improved approach"
                        },
                        "statement": {
                            "type": "string",
                            "description": "Improved statement of novelty and theoretical contributions"
                        },
                        "method": {
                            "type": "string",
                            "description": "Detailed explanation of the improved method with key enhancements highlighted"
                        }
                    },
                    "required": ["name", "title", "description", "statement", "method"]
                }
            },
            "required": ["refined_method"]
        }
        
        # Build the prompt
        prompt = self._build_refinement_prompt(
            goal=goal,
            hypothesis=hypothesis,
            method_details=method_details,
            method_critiques=method_critiques,
            literature=literature,
            feedback=feedback,
            iteration=iteration
        )
        
        # Call the model
        system_prompt = self._build_system_prompt()
        
        try:
            response = await self._call_model(
                prompt=prompt,
                system_prompt=system_prompt,
                schema=output_schema
            )
            
            # Process the response
            refined_method = response.get("refined_method", {})
            
            if not refined_method or not refined_method.get("method"):
                logger.warning("Refinement agent returned incomplete refined method")
                
            # Build the result
            result = {
                "refined_method": refined_method,
                "metadata": {
                    "hypothesis_id": hypothesis.get("id", ""),
                    "iteration": iteration
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Refinement agent execution failed: {str(e)}")
            
            # Return a minimal response to avoid breaking the workflow
            return {
                "refined_method": method_details,
                "metadata": {
                    "hypothesis_id": hypothesis.get("id", ""),
                    "iteration": iteration,
                    "error": str(e)
                }
            }
    
    def _build_refinement_prompt(self,
                               goal: Dict[str, Any],
                               hypothesis: Dict[str, Any],
                               method_details: Dict[str, Any],
                               method_critiques: List[Dict[str, Any]],
                               literature: List[Dict[str, Any]],
                               feedback: List[Dict[str, Any]],
                               iteration: int) -> str:
        """
        Construct comprehensive prompt for method refinement.

        Builds structured prompt with original method, identified critiques, relevant
        literature, and detailed improvement requirements covering critical issues,
        mathematical foundations, algorithmic structure, and implementation feasibility.

        Args:
            goal (Dict[str, Any]): Research goal and domain
            hypothesis (Dict[str, Any]): Hypothesis with text/rationale
            method_details (Dict[str, Any]): Original method to refine
            method_critiques (List[Dict[str, Any]]): Identified issues to address
            literature (List[Dict[str, Any]]): Literature for integration
            feedback (List[Dict[str, Any]]): Scientist feedback entries
            iteration (int): Current iteration for context

        Returns:
            str: Structured refinement prompt with improvement guidelines
        """
        # Extract information
        goal_description = goal.get("description", "")
        domain = goal.get("domain", "")
        
        hypothesis_text = hypothesis.get("text", "")
        hypothesis_rationale = hypothesis.get("rationale", "")
        
        # Build the prompt
        prompt = f"# Research Goal\n{goal_description}\n\n"
        
        # Add domain if available
        if domain:
            prompt += f"# Domain\n{domain}\n\n"
            
        # Add the hypothesis
        prompt += f"# Hypothesis\n{hypothesis_text}\n\n"
        
        # Add the rationale if available
        if hypothesis_rationale:
            prompt += f"# Hypothesis Rationale\n{hypothesis_rationale}\n\n"
            
        # Add method details
        prompt += "# Original Method\n"
        
        if isinstance(method_details, dict):
            # Extract key method components
            method_name = method_details["name"]  
            method_title = method_details["title"]
            method_description = method_details["description"]
            method_statement = method_details["statement"]
            method_text = method_details["method"] 
            
            if method_name:
                prompt += f"## Name\n{method_name}\n\n"
            if method_title:
                prompt += f"## Title\n{method_title}\n\n"
            if method_description:
                prompt += f"## Description\n{method_description}\n\n" 
            if method_statement:
                prompt += f"## Statement of Novelty\n{method_statement}\n\n"
            if method_text:
                prompt += f"## Method Details\n{method_text}\n\n"
        else:
            # Handle string method details
            prompt += f"{method_details}\n\n"
        
        # Add method critiques
        if method_critiques:
            prompt += "# Method Critiques\n"
            for i, critique in enumerate(method_critiques, 1):
                category = critique.get("category", "General")
                point = critique.get("point", "")
                severity = critique.get("severity", "minor")
                
                prompt += f"## Critique {i}\n"
                prompt += f"Category: {category}\n"
                prompt += f"Severity: {severity}\n"
                prompt += f"Point: {point}\n\n"
            
        # Add literature context if available
        if literature:
            prompt += "# Relevant Literature\n"
            for i, item in enumerate(literature, 1):
                if isinstance(item, dict):
                    title = item.get("title", "")
                    content = item.get("content", "")
                    relevance = item.get("relevance", "")
                    
                    prompt += f"## Source {i}: {title}\n"
                    prompt += f"{content}\n"
                    if relevance:
                        prompt += f"Relevance: {relevance}\n"
                    prompt += "\n"
                else:
                    # Handle string items
                    prompt += f"## Source {i}\n{item}\n\n"
            
        # Add task description
        prompt += '''# Task
Refine and improve the original method based on the critiques and relevant literature to create an enhanced version with stronger theoretical foundations and practical implementation.

Your output should include:
1. A concise name for the improved method
2. A descriptive title
3. A high-level overview of the improved approach (1-2 paragraphs)
4. An enhanced statement of novelty and theoretical contributions 
5. A detailed explanation of the final method

## Improvement Requirements

### Address Critical Issues
- Directly address each major critique from the method critiques section
- Prioritize fixing theoretical flaws and mathematical inconsistencies
- Resolve algorithmic inefficiencies and implementation challenges
- Strengthen weak components while preserving effective elements

### Incorporate Literature Insights
- Integrate relevant concepts and techniques from the provided literature
- Adapt established approaches to enhance the method's effectiveness
- Bridge theoretical gaps using insights from related work
- Properly attribute and contextualize borrowed concepts

### Enhance Mathematical Foundations
- Improve mathematical formulations for clarity and correctness
- Strengthen theoretical guarantees and convergence properties
- Ensure consistent notation and well-defined variables
- Add formal proofs or justifications where beneficial

### Refine Algorithmic Structure
- Optimize algorithmic workflow for efficiency and robustness
- Improve pseudocode clarity and implementation guidance
- Provide sufficient algorithmic details to guide future implementation
- Balance mathematical rigor with practical implementability

### Implementation Feasibility
- Ensure the method description contains sufficient information for future code implementation
- Include clear algorithmic workflows or pseudocode where appropriate
- Define key data structures and computational procedures conceptually
- Consider computational complexity and practical constraints

**It should be emphasized that only **1-2** key, innovative, and clearly defined contributions are required, while removing any redundant, unimportant, or superfluous parts.**

Note: Only 1 or 2 clear contributions are required. Do not exceed this amount limit and do not stack contributions together.

Note that the original method information is no longer used, so you need to output the complete information of improved method so that researchers can understand it without the original method information.

'''

        return prompt
    
    def _build_system_prompt(self) -> str:
        """
        Build system prompt for method refinement specialist.

        Creates instructions for improving methods through critique-driven enhancements,
        literature integration, and strengthening foundations while maintaining core
        innovations and focusing on 1-2 key contributions.

        Returns:
            str: System prompt with refinement guidelines
        """
        return """You are a method-refinement specialist in a multi-agent scientific ideation system with expertise in improving algorithms, mathematical models, and theoretical frameworks. Your task is to enhance and refine a scientific method based on critiques and relevant literature.

Guidelines:
- Focus on direct improvements that address specific critiques
- Maintain the core insights and innovations of the original method
- Integrate relevant concepts from scientific literature appropriately
- Strengthen mathematical formulations and theoretical foundations
- Improve algorithmic clarity and implementation guidance
- Enhance the method's novelty statement and theoretical contributions

Avoid common pitfalls:
- Don't introduce unnecessary complexity that doesn't address critiques
- Don't make vague improvements without clear justification
- Don't ignore major critiques or focus only on minor issues
- Don't add literature connections that aren't meaningfully integrated
- Don't lose sight of the method's core purpose and innovation

**It should be emphasized that only 1-2 key, innovative, and clearly defined contributions are required, while removing any redundant, unimportant, or superfluous parts.**
Note: Only 1 or 2 clear contributions are required. Do not exceed this amount limit and do not stack contributions together.
Your goal is to produce a refined method with stronger theoretical foundations, clearer algorithmic specifications, and enhanced practical utility that directly addresses the identified limitations of the original approach.
"""
