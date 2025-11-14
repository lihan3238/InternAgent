import dataclasses
from enum import auto, Enum
from typing import List, Any, Dict, Union, Tuple
import re
import base64
from io import BytesIO
from PIL import Image
from transformers import AutoTokenizer


class SeparatorStyle(Enum):
    """Different separator style."""

    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    CHATML = auto()
    LLAMA_2 = auto()
    LLAMA_3 = auto()
    QWEN = auto()
    GEMMA = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    tokenizer_id: str = ""
    tokenizer: Any = None
    # Stop criteria (the default one is EOS token)
    stop_str: Union[str, List[str]] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    skip_next: bool = False


conv_vicuna_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. " "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=[
        ["Human", "What are the key differences between renewable and non-renewable energy sources?"],
        [
            "Assistant",
            "Renewable energy sources are those that can be replenished naturally in a relatively "
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
            "Non-renewable energy sources, on the other hand, are finite and will eventually be "
            "depleted, such as coal, oil, and natural gas. Here are some key differences between "
            "renewable and non-renewable energy sources:\n"
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
            "energy sources are finite and will eventually run out.\n"
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
            "and other negative effects.\n"
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
            "have lower operational costs than non-renewable sources.\n"
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
            "locations than non-renewable sources.\n"
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
            "non-renewable sources are not, and their depletion can lead to economic and social instability.\n",
        ],
    ],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_vicuna_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. " "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llama_2 = Conversation(
    system="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_llava_llama_2 = Conversation(
    system="You are a helpful language and vision assistant. " "You are able to understand the visual content that the user provides, " "and assist the user with a variety of tasks using natural language.",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

def safe_load_tokenizer(tokenizer_id):
    try:
        return AutoTokenizer.from_pretrained(tokenizer_id)
    except Exception:
        return None

conv_llava_llama_3 = Conversation(
    system="You are a helpful language and vision assistant. " "You are able to understand the visual content that the user provides, " "and assist the user with a variety of tasks using natural language.",
    roles=("user", "assistant"),
    version="llama_v3",
    messages=[],
    offset=0,
    sep="<|eot_id|>",
    sep_style=SeparatorStyle.LLAMA_3,
    tokenizer_id="meta-llama/Meta-Llama-3-8B-Instruct",
    tokenizer=safe_load_tokenizer("meta-llama/Meta-Llama-3-8B-Instruct"),
    stop_token_ids=[128009],
)

conv_mistral_instruct = Conversation(
    system="",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="",
    sep2="</s>",
)

conv_llava_llama_2_simple = Conversation(
    system="Answer the questions about the visual content that the user provides.",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_llava_llama_2_mmtag = Conversation(
    system="Answer the questions about the visual content that the user provides." "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("USER", "ASSISTANT"),
    version="llama_v2_mmtag",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_mpt = Conversation(
    system="""<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_qwen = Conversation(
    system="""<|im_start|>system
You are a helpful assistant.""",
    roles=("<|im_start|>user", "<|im_start|>assistant"),
    version="qwen",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.CHATML,
    sep="<|im_end|>",
)

conv_gemma_instruct = Conversation(system="", roles=("<start_of_turn>user\n", "<start_of_turn>model\n"), version="gemma", messages=[], offset=0, sep_style=SeparatorStyle.GEMMA, sep="<end_of_turn>\n")

conv_llava_plain = Conversation(
    system="",
    roles=("", ""),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)

conv_llava_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. " "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_llava_v0_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
    "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
    version="v0_mmtag",
)

conv_llava_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. " "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llava_v1_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
    "The visual content will be provided with the following format: <Image>visual content</Image>.",
    roles=("USER", "ASSISTANT"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
    version="v1_mmtag",
)

conv_mistral_orca = Conversation(
    system="""<|im_start|>system
You are MistralOrca, a large language model trained by Alignment Lab AI. Write out your reasoning step-by-step to be sure you get the right answers!""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_mistral_zephyr = Conversation(
    system="""<|system|>
You are a helpful AI assistant.""",
    roles=("<|user|>\n", "<|assistant|>\n"),
    version="mpt",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="</s>",
)

conv_mistral_direct = Conversation(
    system="""<|im_start|>system
Answer the questions.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_chatml_direct = Conversation(
    system="""<|im_start|>system
Answer the questions.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

default_conversation = conv_vicuna_v0
conv_templates = {
    "default": conv_vicuna_v0,
    "v0": conv_vicuna_v0,
    "v1": conv_vicuna_v1,
    "vicuna_v1": conv_vicuna_v1,
    "llama_2": conv_llama_2,
    "mistral_instruct": conv_mistral_instruct,
    "mistral_orca": conv_mistral_orca,
    "mistral_zephyr": conv_mistral_zephyr,
    "mistral_direct": conv_mistral_direct,
    "plain": conv_llava_plain,
    "v0_plain": conv_llava_plain,
    "chatml_direct": conv_chatml_direct,
    "llava_v0": conv_llava_v0,
    "llava_v0_mmtag": conv_llava_v0_mmtag,
    "llava_v1": conv_llava_v1,
    "llava_v1_mmtag": conv_llava_v1_mmtag,
    "llava_llama_2": conv_llava_llama_2,
    "llava_llama_3": conv_llava_llama_3,
    "llava_llama_2_simple": conv_llava_llama_2_simple,
    "llava_llama_2_mmtag": conv_llava_llama_2_mmtag,
    "llava_mistral_instruct": conv_mistral_instruct,
    "mpt": conv_mpt,
    "qwen_1_5": conv_qwen,
    "qwen_2": conv_qwen,
    "gemma_instruct": conv_gemma_instruct,
}


if __name__ == "__main__":
    print(default_conversation.get_prompt())
