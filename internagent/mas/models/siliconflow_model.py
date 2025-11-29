"""
SiliconFlow Model Adapter for InternAgent

SiliconFlow 提供兼容 OpenAI 的 API 接口，支持多种开源大模型。
官网: https://siliconflow.cn
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any, Union
from json_repair import repair_json

from openai import AsyncOpenAI

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class SiliconFlowModel(BaseModel):
    """SiliconFlow implementation of the BaseModel interface.
    
    SiliconFlow 是一个国内的 AI 模型服务平台，提供兼容 OpenAI 的 API。
    支持多种开源模型，如 Qwen、DeepSeek、GLM 等。
    """
    
    # SiliconFlow 支持的一些常用模型
    SUPPORTED_MODELS = {
        # Qwen 系列
        "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5-14b-instruct": "Qwen/Qwen2.5-14B-Instruct",
        "qwen2.5-32b-instruct": "Qwen/Qwen2.5-32B-Instruct",
        "qwen2.5-72b-instruct": "Qwen/Qwen2.5-72B-Instruct",
        "qwen-max": "Qwen/Qwen2.5-72B-Instruct",
        # Qwen3 VL 系列（视觉语言模型）
        "qwen3-vl-32b": "Qwen/Qwen3-VL-32B-Instruct",
        "qwen3-vl-72b": "Qwen/Qwen3-VL-72B-Instruct",
        
        # DeepSeek 系列
        "deepseek-v3": "deepseek-ai/DeepSeek-V3",
        "deepseek-r1": "deepseek-ai/DeepSeek-R1",
        "deepseek-coder": "deepseek-ai/deepseek-coder-33b-instruct",
        
        # GLM 系列
        "glm-4-9b": "THUDM/glm-4-9b-chat",
        
        # Yi 系列
        "yi-lightning": "01-ai/Yi-Lightning",
        "yi-large": "01-ai/Yi-Large",
        
        # Llama 系列
        "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
    }
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                model_name: str = "qwen2.5-72b-instruct", 
                max_tokens: int = 4096,
                temperature: float = 0.7,
                timeout: int = 120,
                base_url: Optional[str] = None):
        """
        Initialize the SiliconFlow model adapter.
        
        Args:
            api_key: SiliconFlow API key (defaults to SILICONFLOW_API_KEY environment variable)
            model_name: Model identifier to use (e.g., "qwen2.5-72b-instruct")
            max_tokens: Maximum tokens to generate by default
            temperature: Default temperature setting (0 to 1)
            timeout: Timeout in seconds for API calls
            base_url: Custom base URL (defaults to SiliconFlow's API endpoint)
        """
        super().__init__()
        
        self.api_key = api_key or os.environ.get("SILICONFLOW_API_KEY")
        self.base_url = base_url or os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
        
        if not self.api_key:
            logger.warning("SiliconFlow API key not provided. Please set SILICONFLOW_API_KEY environment variable.")
        
        # 解析模型名称：如果是简写，转换为完整名称
        if model_name in self.SUPPORTED_MODELS:
            self.model_name = self.SUPPORTED_MODELS[model_name]
            logger.info(f"Using model alias: {model_name} -> {self.model_name}")
        else:
            self.model_name = model_name
        
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        
        try:
            self.client = AsyncOpenAI(
                api_key=self.api_key, 
                base_url=self.base_url, 
                timeout=self.timeout
            )
            logger.info(f"SiliconFlow client initialized with model: {self.model_name}")
        except TypeError as e:
            logger.warning(f"Error initializing SiliconFlow client: {e}")
            self.client = None
    
    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None,
                      stop_sequences: Optional[List[str]] = None,
                      **kwargs) -> str:
        """
        Generate text based on the provided prompt using SiliconFlow API.
        
        Args:
            prompt: The user prompt to send to the model
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: List of sequences at which to stop generation
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text response from the model
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating response from SiliconFlow: {e}")
            raise
    
    async def generate_with_json_output(self, 
                                       prompt: str, 
                                       json_schema: Dict[str, Any],
                                       system_prompt: Optional[str] = None,
                                       temperature: Optional[float] = None,
                                       **kwargs) -> Dict[str, Any]:
        """
        Generate a response formatted as JSON according to the provided schema.
        
        Args:
            prompt: The user prompt to send to the model
            json_schema: JSON schema defining the expected response structure
            system_prompt: Optional system prompt to guide the model
            temperature: Controls randomness (0 to 1)
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON response matching the provided schema
        """
        if system_prompt:
            enhanced_system_prompt = f"{system_prompt}\n\n请以JSON格式回复，匹配以下schema:\n{json.dumps(json_schema, ensure_ascii=False)}"
        else:
            enhanced_system_prompt = f"请以JSON格式回复，匹配以下schema:\n{json.dumps(json_schema, ensure_ascii=False)}"
        
        try:
            # 某些 SiliconFlow 模型支持 response_format，某些不支持
            # 我们先尝试使用，如果失败则回退到普通模式
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": enhanced_system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature if temperature is not None else self.temperature,
                    response_format={"type": "json_object"},
                    **kwargs
                )
            except Exception as format_error:
                logger.warning(f"Model doesn't support response_format, falling back: {format_error}")
                # 回退：不使用 response_format
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": enhanced_system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature if temperature is not None else self.temperature,
                    **kwargs
                )
            
            result_text = response.choices[0].message.content
            
            # 尝试提取 JSON（可能包含在 markdown 代码块中）
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(1)
            
            try:
                result_dict = json.loads(result_text)
            except json.JSONDecodeError:
                logger.warning(f"Model returned invalid JSON, attempting repair: {result_text[:200]}...")
                result_text_repair = repair_json(result_text)
                if result_text_repair:
                    try:
                        result_dict = json.loads(result_text_repair)
                        logger.info("Successfully repaired JSON")
                    except json.JSONDecodeError:
                        logger.error(f"Repaired JSON still invalid: {result_text_repair[:200]}...")
                        raise ValueError("Model did not return valid JSON after repair")
                else:
                    logger.error("Failed to repair JSON response")
                    raise ValueError("Model did not return valid JSON")
            
            return result_dict
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            raise ValueError(f"Model did not return valid JSON: {e}")
        except Exception as e:
            logger.error(f"Error generating JSON response from SiliconFlow: {e}")
            raise
    
    async def generate_json(self, 
                          prompt: str, 
                          schema: Dict[str, Any],
                          system_prompt: Optional[str] = None,
                          temperature: Optional[float] = None,
                          default: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """
        Generate JSON output from the model.
        
        Args:
            prompt: User prompt to generate from
            schema: JSON schema that the output should conform to
            system_prompt: System prompt (instructions for the model)
            temperature: Sampling temperature (0.0 to 1.0)
            default: Default JSON to return if generation fails
            **kwargs: Additional model-specific parameters
            
        Returns:
            JSON output as a Python dictionary
            
        Raises:
            ModelError: If generation fails and no default is provided
        """
        try:
            return await self.generate_with_json_output(
                prompt=prompt,
                json_schema=schema,
                system_prompt=system_prompt,
                temperature=temperature,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error in generate_json: {e}")
            if default is not None:
                logger.warning(f"Returning default JSON due to error: {e}")
                return default
            raise
    
    async def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for the given text(s).
        
        注意: SiliconFlow 的 embedding 支持可能有限，建议使用专门的 embedding 模型。
        
        Args:
            text: Text string or list of strings to embed
            
        Returns:
            Embedding vector or list of embedding vectors
        """
        try:
            text_list = [text] if isinstance(text, str) else text
            
            # SiliconFlow 支持的 embedding 模型
            # 例如: BAAI/bge-large-zh-v1.5
            response = await self.client.embeddings.create(
                model="BAAI/bge-large-zh-v1.5",  # 默认中文 embedding 模型
                input=text_list
            )
            
            embeddings = [item.embedding for item in response.data]
            
            return embeddings[0] if isinstance(text, str) else embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings from SiliconFlow: {e}")
            logger.warning("SiliconFlow embedding support may be limited. Consider using a dedicated embedding service.")
            raise
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SiliconFlowModel':
        """
        Create a SiliconFlow model instance from a configuration dictionary.
        
        Args:
            config: Configuration dictionary with model settings
            
        Returns:
            Configured SiliconFlowModel instance
        """
        return cls(
            api_key=config.get("api_key"),
            model_name=config.get("model_name", "qwen2.5-72b-instruct"),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.7),
            timeout=config.get("timeout", 120),
            base_url=config.get("base_url")
        )
    
    @classmethod
    def list_supported_models(cls) -> Dict[str, str]:
        """
        列出支持的模型别名和完整名称。
        
        Returns:
            Dict: 模型别名到完整名称的映射
        """
        return cls.SUPPORTED_MODELS.copy()

