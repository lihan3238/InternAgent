"""
Ranking Agent for InternAgent

Implements the hypothesis evaluation agent that scores hypotheses across multiple
weighted criteria (novelty, plausibility, testability, alignment) to identify the
most promising candidates. Supports batch scoring and configurable selection strategies.
"""

import logging
from typing import Dict, Any, List, Optional
from tqdm import tqdm
from .base_agent import BaseAgent, AgentExecutionError

logger = logging.getLogger(__name__)


class RankingAgent(BaseAgent):
    """
    Agent that evaluates and ranks hypotheses using weighted criteria.

    Scores hypotheses across multiple evaluation dimensions with configurable weights,
    producing rankings that identify the most scientifically valuable candidates.
    Supports batch processing for efficiency and multiple selection strategies
    (default or distinct for diverse parent groups).

    Attributes:
        criteria (Dict): Evaluation criteria with descriptions and weights
        top_n (int): Number of top hypotheses to select
        strategy (str): Selection strategy (default/distinct)
    """
    
    def __init__(self, model, config: Dict[str, Any]):
        """
        Initialize the ranking agent with model and configuration.

        Args:
            model (BaseModel): Language model for hypothesis evaluation
            config (Dict[str, Any]): Configuration with keys:
                - criteria (Dict): Evaluation criteria with weights
                - _global_config.workflow.top_ideas_count (int): Top N to select
                - strategy (str): Selection strategy (default: "default")
        """
        super().__init__(model, config)
        
        # Load agent-specific configuration
        raw_criteria = config.get("criteria", {
            "novelty": 0.25,
            "plausibility": 0.25,
            "testability": 0.25,
            "alignment": 0.25
        })
        
        # self.top_n = config.get("top_n", 3)
        self.top_n = config.get("_global_config").get("workflow").get("top_ideas_count", 3)
        self.strategy = config.get("strategy", "default")
        
        # Convert flat criteria format to nested dictionary format if needed
        self.criteria = {}
        for key, value in raw_criteria.items():
            if isinstance(value, (int, float)):
                # Convert flat format (weight only) to nested dictionary
                self.criteria[key] = {
                    "description": self._get_default_description(key),
                    "weight": float(value)
                }
            else:
                # Already in nested dictionary format
                self.criteria[key] = value
        
        # Calculate total weight to ensure proper normalization
        total_weight = sum(c.get("weight", 0.0) for c in self.criteria.values())
        if abs(total_weight - 1.0) > 0.01:  # Allow for small floating point errors
            logger.warning(f"Criteria weights do not sum to 1.0 (sum: {total_weight}). Normalizing.")
            for criterion in self.criteria:
                self.criteria[criterion]["weight"] /= total_weight
        
    def _get_default_description(self, criterion: str) -> str:
        """
        Get default description for evaluation criterion.

        Provides standard descriptions for common criteria when not explicitly
        defined in configuration.

        Args:
            criterion (str): Criterion name

        Returns:
            str: Default description or generic fallback
        """
        descriptions = {
            "novelty": "Degree to which the hypothesis offers new ideas or approaches",
            "plausibility": "Scientific plausibility and grounding in established knowledge",
            "testability": "Ease of empirical testing and falsifiability",
            "alignment": "Alignment with the research goal and context",
        }
        return descriptions.get(criterion, f"Evaluation of {criterion}")
        
    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rank hypotheses using weighted multi-criteria scoring.

        Evaluates each hypothesis across configured criteria (novelty, plausibility,
        testability, alignment), computes weighted overall scores, and selects top
        candidates. Processes hypotheses in batches for efficiency. Supports distinct
        selection strategy to ensure diversity across hypothesis families.

        Args:
            context (Dict[str, Any]): Execution context with keys:
                - goal (Dict): Research goal and constraints
                - hypotheses (List[Dict]): Hypotheses to rank with id/text/rationale
                - iteration (int): Current iteration number
                - feedback (List[Dict]): Scientist feedback (optional)
            params (Dict[str, Any]): Runtime parameters (currently unused)

        Returns:
            Dict[str, Any]: Ranking results containing:
                - ranked_hypotheses (List[Dict]): Scored hypotheses sorted by score
                - scoring_explanation (str): Overall scoring rationale
                - top_hypotheses (List[str]): Top N hypothesis IDs
                - metadata (Dict): Ranking context

        Raises:
            AgentExecutionError: If goal/hypotheses missing or ranking fails
        """
        # Extract parameters
        goal = context.get("goal", {})
        hypotheses = context.get("hypotheses", [])
        
        if not goal or not hypotheses:
            raise AgentExecutionError("Research goal and hypotheses are required for ranking")
        
        if len(hypotheses) == 0:
            raise AgentExecutionError("At least one hypothesis is required for ranking")
            
        # Extract optional parameters
        iteration = context.get("iteration", 0)
        feedback = context.get("feedback", [])
        
        # Create a JSON schema for the expected output
        output_schema = {
            "type": "object",
            "properties": {
                "scored_hypotheses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "ID of the hypothesis"
                            },
                            "overall_score": {
                                "type": "number",
                                "description": "Overall score (0.0-10.0)"
                            },
                            "criteria_scores": {
                                "type": "object",
                                "description": "Scores for individual criteria (0.0-10.0)",
                                "additionalProperties": {
                                    "type": "number"
                                }
                            },
                            "scoring_rationale": {
                                "type": "string",
                                "description": "Rationale for the scores"
                            }
                        },
                        "required": ["id", "overall_score", "criteria_scores", "scoring_rationale"]
                    }
                },
                "scoring_explanation": {
                    "type": "string",
                    "description": "Overall explanation of the scoring process"
                },
            },
            "required": ["scored_hypotheses", "scoring_explanation"]
        }
        
        hyp_id2parent_id = {}
        for hyp in hypotheses:
            hyp_id = hyp.get("id", "")
            hyp_id2parent_id[hyp_id] = hyp.get("parent_id", "")
        
        
        # Call the model
        system_prompt = self._build_system_prompt()
        
        try:
            SCORE_BATCH_SIZE = 5
            # Split hypotheses into batches
            batches = [hypotheses[i:i + SCORE_BATCH_SIZE] for i in range(0, len(hypotheses), SCORE_BATCH_SIZE)]
            scored_hypotheses = []
            scoring_explanation = ""
            logger.info(f"Ranking {len(hypotheses)} hypotheses in {len(batches)} batches")
            # Iterate over batches
            for batch in tqdm(batches, desc="Ranking hypotheses", unit="batch"):
                # Extract the current batch of hypotheses
                batch_hypotheses = batch
                # Build the prompt
                prompt = self._build_ranking_prompt(
                    goal=goal,
                    hypotheses=batch_hypotheses,
                    criteria=self.criteria,
                    iteration=iteration,
                    feedback=feedback
                )
                
                response = await self._call_model(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    schema=output_schema
                )
                batch_scored_hypotheses = response.get("scored_hypotheses", [])
                if len(batch_scored_hypotheses) != len(batch_hypotheses):
                    logger.warning(f"Expected {len(batch_hypotheses)} scored hypotheses, but got {len(batch_scored_hypotheses)}")
                # Process the response
                scored_hypotheses.extend(batch_scored_hypotheses)
                # Merge the scoring explanation
                scoring_explanation += response.get("scoring_explanation", "")
                # Merge the scored hypotheses
                            
            # prepare hypotheses for ranking
            ranked_hypotheses = []
            for hypo in scored_hypotheses:
                hypo_id = hypo.get("id", "")
                parent_id = hyp_id2parent_id.get(hypo_id, "")
                
                overall_score = hypo.get("overall_score", 0.0)
                criteria_scores = hypo.get("criteria_scores", {})
                scoring_rationale = hypo.get("scoring_rationale", "")
                
                ranked_hypotheses.append({
                    "id": hypo_id,
                    "parent_id": parent_id,
                    "overall_score": overall_score,
                    "criteria_scores": criteria_scores,
                    "scoring_rationale": scoring_rationale
                })
        
            if self.strategy == "distinct":
                # For leaf nodes with the same root node, only the one with the highest overall score is kept, and then all leaf nodes are top-n sorted according to the overall score
                # Group by parent_id
                parent_groups = {}
                for hypo in ranked_hypotheses:
                    parent_id = hypo["parent_id"]
                    if parent_id not in parent_groups:
                        parent_groups[parent_id] = []
                    parent_groups[parent_id].append(hypo)
                # Select the highest scored hypothesis for each parent_id
                selected_hypotheses = []
                for parent_id, group in parent_groups.items():
                    # Sort by overall score and select the top one
                    group.sort(key=lambda x: x["overall_score"], reverse=True)
                    selected_hypotheses.append(group[0])
                # Sort the selected hypotheses by overall score
                selected_hypotheses.sort(key=lambda x: x["overall_score"], reverse=True)
                ranked_hypotheses = selected_hypotheses
                top_n = min(len(ranked_hypotheses), self.top_n)
                if top_n != self.top_n:
                    logger.warning(f"Only {top_n} hypotheses were selected from {ranked_hypotheses}")
                top_hypotheses = [hypo["id"] for hypo in selected_hypotheses[:top_n]]
            else:
                # Select the top N hypotheses based on overall score
                ranked_hypotheses.sort(key=lambda x: x["overall_score"], reverse=True)
                top_n = min(len(ranked_hypotheses), self.top_n)
                top_hypotheses = [hypo["id"] for hypo in ranked_hypotheses[:top_n]]
            
            if not ranked_hypotheses:
                logger.warning("Ranking agent returned no ranked hypotheses")
                
            # Build the result
            result = {
                "ranked_hypotheses": ranked_hypotheses,
                "scoring_explanation": scoring_explanation,
                "top_hypotheses": top_hypotheses,
                "metadata": {
                    "iteration": iteration,
                    "criteria": list(self.criteria.keys()),
                    "hypothesis_count": len(hypotheses)
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Ranking agent execution failed: {str(e)}")
            raise AgentExecutionError(f"Failed to rank hypotheses: {str(e)}")
    
    def _build_ranking_prompt(self,
                            goal: Dict[str, Any],
                            hypotheses: List[Dict[str, Any]],
                            criteria: Dict[str, Dict[str, Any]],
                            iteration: int,
                            feedback: List[Dict[str, Any]]) -> str:
        """
        Construct evaluation prompt with criteria and hypotheses.

        Builds structured prompt with research goal, weighted evaluation criteria,
        recent feedback, and hypotheses to score. Provides clear scoring instructions
        and consistency guidelines.

        Args:
            goal (Dict[str, Any]): Research goal with domain/constraints
            hypotheses (List[Dict[str, Any]]): Hypotheses to evaluate
            criteria (Dict[str, Dict[str, Any]]): Criteria with descriptions/weights
            iteration (int): Current iteration for context
            feedback (List[Dict[str, Any]]): Recent feedback entries

        Returns:
            str: Structured ranking prompt with scoring guidelines
        """
        # Extract information
        goal_description = goal.get("description", "")
        domain = goal.get("domain", "")
        constraints = goal.get("constraints", [])
        
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
            
        # Add evaluation criteria
        prompt += "# Evaluation Criteria\n"
        for criterion, details in criteria.items():
            description = details.get("description", "")
            weight = details.get("weight", 0.0)
            prompt += f"- {criterion.upper()} (weight: {weight:.2f}): {description}\n"
        prompt += "\n"
        
        # Add recent feedback if available
        if feedback:
            prompt += "# Scientist Feedback\n"
            recent_feedback = sorted(
                feedback, 
                key=lambda x: x.get("iteration", 0),
                reverse=True
            )[:2]  # Only the 2 most recent feedback entries
            
            for entry in recent_feedback:
                feedback_text = entry.get("text", "")
                feedback_iter = entry.get("iteration", 0)
                
                if feedback_text:
                    prompt += f"From iteration {feedback_iter}: {feedback_text}\n\n"
        
        # Add the hypotheses to evaluate
        prompt += "# Hypotheses to Evaluate\n"
        for i, hypothesis in enumerate(hypotheses, 1):
            hyp_id = hypothesis.get("id", f"hyp{i}")
            text = hypothesis.get("text", "")
            rationale = hypothesis.get("rationale", "")
            
            prompt += f"\n## Hypothesis {i} [ID: {hyp_id}]\n"
            prompt += f"Text: {text}\n"
            if rationale:
                prompt += f"Rationale: {rationale}\n"
                
        # Add task description
        prompt += "\n# Task\n"
        prompt += "Evaluate each hypothesis according to the criteria provided. For each hypothesis:\n"
        prompt += "1. Assign a score from 0.0 to 10.0 for each criterion\n"
        prompt += "2. Calculate a weighted overall score based on the criterion weights\n"
        prompt += "3. Provide a brief rationale for the scores\n"
        prompt += "Ensure consistent and fair evaluation across all hypotheses."
        
        if iteration > 0:
            prompt += f"\nThis is iteration {iteration}, so consider how the hypotheses have evolved and improved."
        
        return prompt
    
    def _build_system_prompt(self) -> str:
        """
        Build system prompt for objective hypothesis evaluation.

        Creates instructions emphasizing consistent application of criteria,
        fair scoring (0-10 scale), and focus on scientific merit.

        Returns:
            str: System prompt with evaluation guidelines
        """
        return """You are a scientific hypothesis evaluator working with a researcher. 
Your task is to objectively evaluate and rank research hypotheses based on specific criteria.

Guidelines:
- Apply the same evaluation standards consistently across all hypotheses
- Score each criterion on a scale from 0.0 (lowest) to 10.0 (highest)
- Calculate overall scores by applying the weights provided for each criterion
- Provide clear, specific rationales for your scoring decisions
- Consider both the strengths and weaknesses of each hypothesis
- Avoid being unduly influenced by writing style over substance
- Be particularly attentive to scientific merit, testability, and alignment with research goals

Focus on helping the researcher identify the most promising hypotheses to pursue.
Your evaluation should help guide the research process toward scientifically valuable outcomes.
""" 

#####
"""
You are a scientific hypothesis evaluator working with a researcher. 
Your task is to objectively evaluate and rank research hypotheses based on specific criteria.

Guidelines:
- Apply the same evaluation standards consistently across all hypotheses
- Score each criterion on a scale from 0.0 (lowest) to 10.0 (highest)
- Calculate overall scores by applying the weights provided for each criterion
- Provide clear, specific rationales for your scoring decisions
- Consider both the strengths and weaknesses of each hypothesis
- Avoid being unduly influenced by writing style over substance
- Be particularly attentive to scientific merit, testability, and alignment with research goals
- Rank hypotheses based on their overall scores

Focus on helping the researcher identify the most promising hypotheses to pursue.
Your evaluation should help guide the research process toward scientifically valuable outcomes.
"""
