import warnings
warnings.filterwarnings('ignore')
import torch
model_path_or_name="upstage/TinySolar-248m-4k-py"
from transformers import AutoModelForCausalLM
tiny_custom_model=AutoModelForCausalLM.from_pretrained(
        model_path_or_name,
        device_map="cpu",
        torch_dtype=torch.bfloat16
        )
from transformers import AutoTokenizer
tiny_custom_tokenizer=AutoTokenizer.from_pretrained(
    model_path_or_name
        )
prompt="def find_max(numbers):"
inputs=tiny_custom_tokenizer(prompt, return_tensors="pt").to(tiny_custom_model.device)
from transformers import TextStreamer
streamer=TextStreamer(
    tiny_custom_tokenizer,
    skip_prompt=True,
    skip_special_tokens=True
        )
outputs=tiny_custom_model.generate(
    **inputs,
    streamer=streamer,
    use_cache=True,
    max_new_tokens=128,
    do_sample=False,
    repetition_penalty=1.1
        )

