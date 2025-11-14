import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from PIL import Image
import requests
from io import BytesIO
from cog import  Input, Path

class Predictor():
    def __init__(self, model_path) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        disable_torch_init()
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, model_name="moce_qwen_vlm", model_base=None, load_8bit=False, load_4bit=False)
        self.tokenizer.add_tokens(["<image>"], special_tokens=True)

    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(description="Prompt to use for text generation"),
    ):
        """Run a single prediction on the model"""
        image_data = load_image(image)
        image_tensor = self.image_processor.preprocess(image_data, return_tensors="pt")["pixel_values"].half().cuda()

        # just one turn, always prepend image token
        if DEFAULT_IMAGE_TOKEN not in prompt:
            prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt

        messages = [
            {"role": "system", "content": "You are a helpful assistant. You are given a math problem image, please solve the problem step by step."},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to("cuda")
        image_token_index = self.tokenizer.convert_tokens_to_ids("<image>")
        model_inputs["input_ids"][model_inputs["input_ids"]==image_token_index] = IMAGE_TOKEN_INDEX

        with torch.inference_mode():
            output = self.model.generate(
                inputs=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                images=image_tensor,
                max_new_tokens=4096, 
                do_sample=False,
                use_cache=True,
            )
        response = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        return response


def load_image(image_file):
    
    if isinstance(image_file, Image.Image):
        return image_file
    elif image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup()
    image = "1604.png"

    prompt = """you are given a math problem image, please solve the problem step by step.
Question:The degree measure of a minor arc and a major arc are x and y respectively. If m \angle A B C = 70, find y.
Choices:
(A) 110
(B) 180
(C) 250
(D) 270"""
    response = predictor.predict(image=image, prompt=prompt)
    print("===1===")
    print(response)

    response = predictor.predict(image=image, prompt=prompt)
    print("===2===")
    print(response)
