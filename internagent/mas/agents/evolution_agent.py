"""
Evolution Agent for InternAgent

Implements the hypothesis refinement agent that evolves and improves research hypotheses
by addressing critiques, incorporating evidence, and responding to feedback. Generates
multiple improved versions with configurable creativity levels for iterative refinement.
"""

import logging
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent, AgentExecutionError

logger = logging.getLogger(__name__)


class EvolutionAgent(BaseAgent):
    """
    Agent that refines hypotheses through iterative evolution.

    Generates improved hypothesis versions by systematically addressing critiques,
    incorporating supporting evidence, and responding to scientist feedback. Produces
    multiple evolution candidates with documented changes and improvements. Supports
    creativity control from conservative refinement to bold restructuring.

    Attributes:
        evolution_count (int): Number of evolved versions per hypothesis
        min_improvement_threshold (float): Minimum improvement required
        creativity_level (float): Evolution creativity 0-1
        temperature (float): Model sampling temperature
    """
    
    def __init__(self, model, config: Dict[str, Any]):
        """
        Initialize the evolution agent with model and configuration.

        Args:
            model (BaseModel): Language model for hypothesis evolution
            config (Dict[str, Any]): Configuration with keys:
                - evolution_count (int): Evolved versions count (default: 2)
                - min_improvement_threshold (float): Min improvement (default: 0.3)
                - creativity_level (float): Creativity 0-1 (default: 0.6)
                - temperature (float): Sampling temperature (optional)
        """
        super().__init__(model, config)
        
        # Load agent-specific configuration
        self.evolution_count = config.get("evolution_count", 2)  # Number of evolutions per hypothesis
        self.min_improvement_threshold = config.get("min_improvement_threshold", 0.3)
        self.creativity_level = config.get("creativity_level", 0.6)
        self.temperature = config.get("temperature", None)
        
    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evolve hypothesis by addressing critiques and incorporating feedback.

        Generates improved hypothesis versions that systematically address identified
        weaknesses while preserving core strengths. Incorporates evidence and feedback
        to produce scientifically stronger, more testable hypotheses. Documents specific
        improvements and changes made during evolution.

        Args:
            context (Dict[str, Any]): Execution context with keys:
                - goal (Dict): Research goal and constraints
                - hypothesis (Dict): Hypothesis to evolve with text/rationale
                - critiques (List[Dict]): Identified weaknesses to address
                - evidence (List[Dict]): Supporting evidence (optional)
                - feedback (List[Dict]): Scientist feedback (optional)
                - iteration (int): Current iteration number
            params (Dict[str, Any]): Runtime parameters:
                - evolution_count (int): Override evolution count (optional)

        Returns:
            Dict[str, Any]: Evolution results containing:
                - evolved_hypotheses (List[Dict]): Improved versions with text/rationale/improvements
                - reasoning (str): Evolution strategy explanation
                - changes (List[Dict]): Documented changes by type
                - metadata (Dict): Evolution context

        Raises:
            AgentExecutionError: If goal/hypothesis missing or evolution fails
        """
        # Extract parameters
        goal = context.get("goal", {})
        hypothesis = context.get("hypothesis", {})
        critiques = context.get("critiques", [])
        evidence = context.get("evidence", [])
        feedback = context.get("feedback", [])
        
        if not goal or not hypothesis:
            raise AgentExecutionError("Research goal and hypothesis are required for evolution")
        
        # Extract text from hypothesis
        hypothesis_text = hypothesis.get("text", "")
        if not hypothesis_text:
            raise AgentExecutionError("Hypothesis text is required for evolution")
            
        # Extract optional parameters
        iteration = context.get("iteration", 0)
        evolution_count = params.get("evolution_count", self.evolution_count)
        
        # Create a JSON schema for the expected output
        output_schema = {
            "type": "object",
            "properties": {
                "evolved_hypotheses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The evolved hypothesis statement"
                            },
                            "rationale": {
                                "type": "string",
                                "description": "Reasoning for the evolved hypothesis"
                            },
                            "improvements": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "description": "Specific improvement made"
                                }
                            }
                        },
                        "required": ["text", "rationale", "improvements"]
                    }
                },
                "reasoning": {
                    "type": "string",
                    "description": "Explanation of the evolution approach"
                },
                "changes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "description": "Type of change (e.g., 'specification', 'generalization', 'mechanism')"
                            },
                            "description": {
                                "type": "string",
                                "description": "Description of the change"
                            }
                        },
                        "required": ["type", "description"]
                    }
                }
            },
            "required": ["evolved_hypotheses", "reasoning", "changes"]
        }
        
        # Build the prompt
        prompt = self._build_evolution_prompt(
            goal=goal,
            hypothesis=hypothesis,
            critiques=critiques,
            evidence=evidence,
            feedback=feedback,
            iteration=iteration,
            count=evolution_count
        )
        
        # Call the model
        system_prompt = self._build_system_prompt(self.creativity_level)
        
        try:
            response = await self._call_model(
                prompt=prompt,
                system_prompt=system_prompt,
                schema=output_schema,
                temperature=self.temperature
            )
            
            # Process the response
            evolved_hypotheses = response.get("evolved_hypotheses", [])
            reasoning = response.get("reasoning", "")
            changes = response.get("changes", [])
            
            if not evolved_hypotheses:
                logger.warning("Evolution agent returned no evolved hypotheses")
                
            # Build the result
            result = {
                "evolved_hypotheses": evolved_hypotheses,
                "reasoning": reasoning,
                "changes": changes,
                "metadata": {
                    "original_hypothesis_id": hypothesis.get("id", ""),
                    "iteration": iteration,
                    "count": len(evolved_hypotheses)
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Evolution agent execution failed: {str(e)}")
            raise AgentExecutionError(f"Failed to evolve hypothesis: {str(e)}")
    
    def _build_evolution_prompt(self,
                              goal: Dict[str, Any],
                              hypothesis: Dict[str, Any],
                              critiques: List[Dict[str, Any]],
                              evidence: List[Dict[str, Any]],
                              feedback: List[Dict[str, Any]],
                              iteration: int,
                              count: int) -> str:
        """
        Construct comprehensive prompt for hypothesis evolution.

        Builds structured prompt incorporating original hypothesis, identified critiques,
        supporting evidence, and feedback. Adapts guidance based on iteration number for
        appropriate evolution focus (major fixes early, refinement later).

        Args:
            goal (Dict[str, Any]): Research goal with domain and constraints
            hypothesis (Dict[str, Any]): Original hypothesis with text/rationale
            critiques (List[Dict[str, Any]]): Weaknesses to address
            evidence (List[Dict[str, Any]]): Supporting evidence to incorporate
            feedback (List[Dict[str, Any]]): Scientist feedback entries
            iteration (int): Current iteration for adaptation
            count (int): Number of evolved versions to generate

        Returns:
            str: Structured evolution prompt with task guidelines
        """
        # Extract information
        goal_description = goal.get("description", "")
        domain = goal.get("domain", "")
        constraints = goal.get("constraints", [])
        
        hypothesis_text = hypothesis.get("text", "")
        hypothesis_rationale = hypothesis.get("rationale", "")
        
        # Build the prompt
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
            
        # Add the original hypothesis
        prompt += f"# Original Hypothesis\n{hypothesis_text}\n\n"
        
        # Add the rationale if available
        if hypothesis_rationale:
            prompt += f"# Original Rationale\n{hypothesis_rationale}\n\n"
            
        # Add critiques
        if critiques:
            prompt += "# Critiques\n"
            for i, critique in enumerate(critiques, 1):
                if isinstance(critique, dict):
                    category = critique.get("category", "")
                    point = critique.get("point", "")
                    severity = critique.get("severity", "")
                    
                    prompt += f"{i}. "
                    if category:
                        prompt += f"[{category}] "
                    prompt += point
                    if severity:
                        prompt += f" (Severity: {severity})"
                    prompt += "\n"
                else:
                    # Handle string critiques
                    prompt += f"{i}. {critique}\n"
            prompt += "\n"
            
        # Add evidence if available
        if evidence:
            prompt += "# Relevant Evidence\n"
            for i, item in enumerate(evidence, 1):
                if isinstance(item, dict):
                    source = item.get("source", "")
                    content = item.get("content", "")
                    relevance = item.get("relevance", "")
                    
                    prompt += f"{i}. "
                    if source:
                        prompt += f"[{source}] "
                    prompt += content
                    if relevance:
                        prompt += f" (Relevance: {relevance})"
                    prompt += "\n"
                else:
                    # Handle string evidence
                    prompt += f"{i}. {item}\n"
            prompt += "\n"
            
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
        
        # Add task description
        prompt += "# Task\n"
        prompt += f"Create {count} improved versions of the original hypothesis that address the critiques, "
        prompt += "incorporate relevant evidence, and align with the scientist's feedback.\n\n"
        
        prompt += "For each evolved hypothesis:\n"
        prompt += "1. Refine the hypothesis statement to be more precise, testable, and aligned with the research goal\n"
        prompt += "2. Address specific weaknesses identified in the critiques\n"
        prompt += "3. Incorporate relevant evidence to strengthen the hypothesis\n"
        prompt += "4. Provide a clear rationale explaining the improvement\n"
        
        # Additional guidance based on iteration
        if iteration == 0:
            prompt += "\nThis is the first iteration, so focus on addressing major weaknesses while preserving the core insight.\n"
        elif iteration < 3:
            prompt += f"\nThis is iteration {iteration}, so focus on incremental improvements and refinements.\n"
        else:
            prompt += f"\nThis is a later iteration ({iteration}), so focus on nuanced improvements and polishing.\n"
            
        return prompt
    
    def _build_system_prompt(self, creativity_level: float) -> str:
        """
        Build system prompt adapted to creativity level.

        Creates instructions for hypothesis evolution with approach tailored to
        creativity setting: conservative (minimal changes), balanced (moderate
        innovation), or creative (bold restructuring).

        Args:
            creativity_level (float): Creativity 0-1 where:
                <0.3: conservative minimal changes
                <0.7: balanced moderate innovation
                >=0.7: creative bold restructuring

        Returns:
            str: System prompt with evolution guidelines
        """
        base_prompt = """You are a scientific hypothesis refinement specialist working with a researcher. 
Your task is to evolve and improve research hypotheses based on critiques and feedback.

Guidelines:
- Address specific weaknesses identified in critiques while preserving strengths
- Make each evolved hypothesis more precise, testable, and aligned with research goals
- Incorporate relevant evidence to strengthen the scientific foundation
- Provide clear explanations for how and why you've made each change
- Ensure evolved hypotheses remain coherent and logically sound
"""

        if creativity_level < 0.3:
            # Conservative evolution
            base_prompt += """
Take a conservative approach to refinement. Make minimal necessary changes to address 
critical weaknesses. Focus on clarification, specification, and logical consistency 
rather than introducing new elements."""
        elif creativity_level < 0.7:
            # Balanced evolution
            base_prompt += """
Balance addressing critiques with moderate innovation. Consider alternate mechanisms 
or explanations when helpful, but maintain the core thesis. Look for opportunities to 
strengthen the hypothesis while resolving inconsistencies."""
        else:
            # Creative evolution
            base_prompt += """
Be bold in your refinements while addressing critiques. Consider alternative mechanisms, 
unexpected connections, or paradigm shifts if they could strengthen the hypothesis. 
Don't hesitate to substantially reshape the hypothesis if doing so creates a stronger 
scientific proposition."""
            
        return base_prompt 
