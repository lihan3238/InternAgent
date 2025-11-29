"""
Reflection Agent for InternAgent

Implements the critical evaluation agent that analyzes research hypotheses and
method details for logical consistency, scientific plausibility, testability, and
technical soundness. Provides structured critiques with severity ratings and
actionable improvement suggestions for iterative refinement.
"""

import logging
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent, AgentExecutionError

logger = logging.getLogger(__name__)


class ReflectionAgent(BaseAgent):
    """
    Agent that performs critical evaluation of hypotheses and methods.

    Provides rigorous but constructive criticism across multiple evaluation dimensions
    including logical consistency, scientific rigor, testability, and technical soundness.
    Supports both hypothesis-level and method-level critiques with severity ratings
    (minor/moderate/major) and specific improvement suggestions.

    Attributes:
        critique_categories (List[str]): Evaluation dimensions to assess
        detail_level (str): Critique depth: low/medium/high
    """
    
    def __init__(self, model, config: Dict[str, Any]):
        """
        Initialize the reflection agent with model and configuration.

        Args:
            model (BaseModel): Language model for critique generation
            config (Dict[str, Any]): Configuration with keys:
                - critique_categories (List[str]): Evaluation dimensions
                - detail_level (str): Critique depth (default: "medium")
        """
        super().__init__(model, config)
        
        # Load agent-specific configuration
        self.critique_categories = config.get("critique_categories", [
            "logical_consistency", 
            "scientific_plausibility", 
            "testability", 
            "novelty", 
            "goal_alignment"
        ])
        self.detail_level = config.get("detail_level", "medium")  # low, medium, high
        
    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Critically evaluate a hypothesis or method with structured feedback.

        Analyzes the hypothesis or method details across multiple dimensions,
        identifying weaknesses with severity ratings and providing actionable
        improvement suggestions. Adapts evaluation criteria based on whether
        method details are present.

        Args:
            context (Dict[str, Any]): Execution context with keys:
                - goal (Dict): Research goal and constraints
                - hypothesis (Dict): Hypothesis with text/rationale/method_details
                - iteration (int): Current iteration number
                - feedback (List[Dict]): Previous feedback (optional)
            params (Dict[str, Any]): Runtime parameters:
                - detail_level (str): Override critique depth (optional)

        Returns:
            Dict[str, Any]: Evaluation results containing:
                - critiques (List[Dict]): Issues with category/point/severity
                - strengths (List[str]): Identified strengths (hypothesis only)
                - overall_assessment (str): Summary (hypothesis only)
                - improvement_suggestions (List[str]): Actionable advice (hypothesis only)
                - metadata (Dict): Evaluation context

        Raises:
            AgentExecutionError: If goal/hypothesis missing or critique fails
        """
        # Extract parameters
        goal = context.get("goal", {})
        hypothesis = context.get("hypothesis", {})
        feedback = context.get("feedback", [])
        
        if not goal or not hypothesis:
            raise AgentExecutionError("Research goal and hypothesis are required for reflection")
        
        # Extract text from hypothesis
        hypothesis_text = hypothesis.get("text", "")
        if not hypothesis_text:
            raise AgentExecutionError("Hypothesis text is required for reflection")
            
        # Extract optional parameters
        iteration = context.get("iteration", 0)
        detail_level = params.get("detail_level", self.detail_level)
        
        # Create a JSON schema for the expected output
        if not hypothesis.get("method_details", {}):
            output_schema = {
                "type": "object",
                "properties": {
                    "critiques": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "category": {
                                    "type": "string",
                                    "description": "Category of critique"
                                },
                                "point": {
                                    "type": "string",
                                    "description": "Critique point"
                                },
                                "severity": {
                                    "type": "string",
                                    "enum": ["minor", "moderate", "major"],
                                    "description": "Severity of the issue"
                                }
                            },
                            "required": ["category", "point", "severity"]
                        }
                    },
                    "strengths": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "Strength of the hypothesis"
                        }
                    },
                    "overall_assessment": {
                        "type": "string",
                        "description": "Overall assessment of the hypothesis"
                    },
                    "improvement_suggestions": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "Suggestion for improvement"
                        }
                    }
                },
                "required": ["critiques", "strengths", "overall_assessment", "improvement_suggestions"]
            }
        else:
            output_schema = {
                "type": "object",
                "properties": {
                    "critiques": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "category": {
                                    "type": "string",
                                    "description": "Category of critique"
                                },
                                "point": {
                                    "type": "string",
                                    "description": "Critique point"
                                },
                                "severity": {
                                    "type": "string",
                                    "enum": ["minor", "moderate", "major"],
                                    "description": "Severity of the issue"
                                }
                            },
                            "required": ["category", "point", "severity"]
                        }
                    }
                },
                "required": ["critiques"]
            }
            
        # Build the prompt
        prompt = self._build_reflection_prompt(
            goal=goal,
            hypothesis=hypothesis,
            detail_level=detail_level,
            iteration=iteration,
            feedback=feedback,
        )
        
        # Call the model
        if not hypothesis.get("method_details", {}):
            system_prompt = self._build_system_prompt()
            try:
                response = await self._call_model(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    schema=output_schema
                )
                
                # Process the response
                critiques = response.get("critiques", [])
                strengths = response.get("strengths", [])
                overall = response.get("overall_assessment", "")
                suggestions = response.get("improvement_suggestions", [])
                
                # Build the result
                result = {
                    "critiques": critiques,
                    "strengths": strengths,
                    "overall_assessment": overall,
                    "improvement_suggestions": suggestions,
                    "metadata": {
                        "hypothesis_id": hypothesis.get("id", ""),
                        "iteration": iteration,
                        "detail_level": detail_level
                    }
                }
                
                return result
                
            except Exception as e:
                logger.error(f"Reflection agent execution failed: {str(e)}")
                raise AgentExecutionError(f"Failed to critique hypothesis: {str(e)}")
        else:
            system_prompt = self._build_system_prompt_method()
            try:
                response = await self._call_model(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    schema=output_schema
                )
                
                # Process the response
                critiques = response.get("critiques", [])

                # Build the result
                result = {
                    "critiques": critiques,
                    "metadata": {
                        "hypothesis_id": hypothesis.get("id", ""),
                        "iteration": iteration,
                        "detail_level": detail_level
                    }
                }
                
                return result
                
            except Exception as e:
                logger.error(f"Reflection agent execution failed: {str(e)}")
                raise AgentExecutionError(f"Failed to critique hypothesis: {str(e)}")
            
        
    
    def _build_reflection_prompt(self,
                               goal: Dict[str, Any],
                               hypothesis: Dict[str, Any],
                               feedback: List[Dict[str, Any]],
                               detail_level: str,
                               iteration: int) -> str:
        """
        Construct evaluation prompt tailored to hypothesis or method critique.

        Builds different prompts based on whether method details exist. Hypothesis
        prompts focus on plausibility and novelty, while method prompts emphasize
        technical soundness, mathematical correctness, and implementability.

        Args:
            goal (Dict[str, Any]): Research goal with domain and constraints
            hypothesis (Dict[str, Any]): Hypothesis including method_details if present
            feedback (List[Dict[str, Any]]): Previous feedback entries
            detail_level (str): Critique depth (low/medium/high)
            iteration (int): Current iteration for context

        Returns:
            str: Structured evaluation prompt with criteria and guidelines
        """
        # Extract information
        goal_description = goal.get("description", "")
        domain = goal.get("domain", "")
        constraints = goal.get("constraints", [])
        
        hypothesis_text = hypothesis.get("text", "")
        hypothesis_rationale = hypothesis.get("rationale", "")
        baseline_summary = hypothesis.get("baseline_summary", "")
        
        method_details = hypothesis.get("method_details", {})

        # Build the prompt
        if method_details:
            prompt = f"# Research Goal\n{goal_description}\n\n"
            
            # Add domain if available
            if domain:
                prompt += f"# Domain\n{domain}\n\n"
                
            # Add constraints if available
            if constraints:
                prompt += "# Constraints\n"
                for constraint in constraints:
                    prompt += f"- {constraint}\n"
                prompt += "\n"
            
            # Add the hypothesis
            prompt += f"# Hypothesis\n{hypothesis_text}\n\n"
            
            # Add method details
            prompt += "# Method Details\n"
            method_overview = method_details["description"]
            method_statement = method_details["statement"]
            method_explanation = method_details["method"]
            
            prompt += f"## Overview\n{method_overview}\n\n"
            prompt += f"## Statement\n{method_statement}\n\n"
            prompt += f"## Detailed Explanation\n{method_explanation}\n\n"
            
            # Add baseline for comparison if available
            if baseline_summary:
                prompt += f"# Baseline Summary\n{baseline_summary}\n\n"
                prompt += "When evaluating the method, consider how it compares to and improves upon the baseline approaches described above.\n\n"

            # Task description specific to method details evaluation
            prompt += "# Task\n"
            prompt += "Critically evaluate the proposed method details for the following aspects:\n\n"
            
            prompt += "## Technical Soundness\n"
            prompt += "1. Mathematical formulations: Are all formulations correct, complete, and clearly presented?\n"
            prompt += "2. Symbol definitions: Are all variables, parameters, and symbols clearly defined before use?\n"
            prompt += "3. Theoretical justification: Are key design choices properly justified with mathematical reasoning?\n"
            
            prompt += "## Algorithmic Clarity\n"
            prompt += "1. Execution flow: Is the end-to-end algorithmic workflow clearly presented?\n"
            prompt += "2. Pseudocode quality: Is the pseudocode at an appropriate abstraction level (neither too abstract nor too implementation-specific)?\n"
            prompt += "3. Module integration: Is the interaction between different components explicitly explained?\n"
            
            prompt += "## Innovation and Contribution\n"
            prompt += "1. Novelty claims: Does the method clearly articulate what makes it novel?\n"
            prompt += "2. Theoretical advances: What new algorithmic or mathematical contributions does this method make?\n"
            prompt += "3. Differentiation: How specifically does it differ from and improve upon baseline approaches?\n\n"
            
            prompt += "## Technical Details\n"
            prompt += "1. Reproducibility: Could another researcher implement this method based solely on the description provided?\n"
            prompt += "2. Parameter initialization: Are strategies for initializing key parameters described?\n"
            prompt += "3. It is necessary to pay attention to whether the technical details and key modules of the method are fully explained, and the steps are clearly indicated for other researchers to understand\n"
            
            # Detail level specific instructions
            if detail_level == "high":
                prompt += "Provide a comprehensive, detailed critique identifying specific mathematical errors, missing definitions, unclear algorithm steps, and gaps in theoretical justification. Include concrete suggestions for addressing each issue.\n"
            elif detail_level == "low":
                prompt += "Provide a concise critique highlighting the most critical issues that would prevent successful implementation or undermine the method's theoretical soundness.\n"
            else:  # medium
                prompt += "Provide a balanced critique focusing on key issues that should be addressed to improve the method's theoretical soundness and implementability.\n"

            prompt += "\nFor each identified issue, provide specific, actionable suggestions for improvement. Your evaluation should be constructive yet rigorous, helping to transform this method into one that is mathematically sound, clearly described, and practically implementable."
                
        else: 
            prompt = f"# Research Goal\n{goal_description}\n\n"
            
            # Add domain if available
            if domain:
                prompt += f"# Domain\n{domain}\n\n"
                
            # Add constraints if available
            if constraints:
                prompt += "# Constraints\n"
                for constraint in constraints:
                    prompt += f"- {constraint}\n"
                prompt += "\n"
                
            # Add the hypothesis
            prompt += f"# Hypothesis\n{hypothesis_text}\n\n"
            
            # Add the rationale if available
            if hypothesis_rationale:
                prompt += f"# Rationale\n{hypothesis_rationale}\n\n"
            
            if baseline_summary:
                prompt += f"# Baseline Summary\n{baseline_summary}\n\n"
                prompt += "When evaluating the novelty of a hypothesis, it is important to consider the Baseline Summary. The proposed hypothesis must differ from the methods described in the Baseline Summary and should have the potential to surpass the methods described in the Baseline Summary."
            
            if feedback:
                prompt += "# Scientist Feedback\n"
                prompt += f"{feedback}\n\n"

            # Add task description
            prompt += "# Task\n"
            prompt += "Critically evaluate the hypothesis for the following aspects:\n"
            prompt += "1. Logical consistency: Is the hypothesis internally consistent?\n"
            prompt += "2. Scientific plausibility: Is it consistent with established scientific knowledge?\n"
            prompt += "3. Testability: Can the hypothesis be empirically tested?\n"
            prompt += "4. Novelty: Does it offer a new perspective or approach?\n"
            prompt += "5. Goal alignment: How well does it address the research goal?\n\n"
            
            if detail_level == "high":
                prompt += "Provide a comprehensive, detailed critique with specific examples and references where relevant.\n"
            elif detail_level == "low":
                prompt += "Provide a concise critique highlighting only the most important points.\n"
            else:  # medium
                prompt += "Provide a balanced critique with adequate detail on key points.\n"
                
            # Additional guidance for later iterations
            if iteration > 0:
                prompt += f"\nThis is iteration {iteration}, so focus on more nuanced aspects that could be improved.\n"
            
        return prompt
    
    def _build_system_prompt(self) -> str:
        """
        Build system prompt for hypothesis-level critique.

        Creates instructions for constructive criticism of research hypotheses,
        emphasizing balance between rigor and encouragement.

        Returns:
            str: System prompt for hypothesis evaluation
        """
        return """You are a scientific critic working with a researcher. 
Your task is to provide constructive criticism of research hypotheses.

Guidelines:
- Evaluate the hypothesis objectively, being neither overly harsh nor too lenient
- Identify both strengths and weaknesses
- Provide specific, actionable feedback that can guide improvements
- Consider scientific rigor, logical consistency, and alignment with research goals
- Be thorough in your analysis but focus on substantive issues
- Suggest concrete ways to address each weakness identified
- Use a constructive tone that encourages refinement rather than dismissal

Remember: Your goal is to help strengthen the hypothesis, not just criticize it.
Scientific progress comes through iterative refinement and addressing weaknesses.
""" 

    def _build_system_prompt_method(self) -> str:
        """
        Build system prompt for method-level critique.

        Creates instructions for rigorous technical evaluation of method details,
        focusing on mathematical correctness, algorithmic clarity, and reproducibility.

        Returns:
            str: System prompt for method evaluation
        """
        return """You are a scientific critic with expertise in theoretical computer science, algorithm design, and mathematical modeling. 
Your task is to provide rigorous, constructive criticism of proposed research methods.

Guidelines for evaluating method details:
- Evaluate the mathematical formulations for correctness, clarity, and completeness
- Check that all variables, symbols, and parameters are properly defined before use
- Assess whether algorithm descriptions provide a clear execution flow from input to output
- Evaluate whether different components of the method are properly integrated with explicit interactions
- Analyze the appropriate abstraction level of pseudocode (neither too abstract nor too implementation-specific)
- Assess the strength and clarity of novelty claims and theoretical contributions
- Identify gaps in the method description that would hinder implementation
- Consider how well the method addresses the research hypothesis and goal

Your feedback should be:
- Technically precise, identifying specific mathematical or algorithmic issues
- Constructive, suggesting concrete improvements for each weakness
- Focused on making the method more sound, implementable, and innovative

Remember: Your goal is to help strengthen the method through rigorous critique. Scientific progress comes through iterative refinement and addressing technical weaknesses.
"""
