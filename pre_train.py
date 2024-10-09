import warnings
warnings.filterwarnings('ignore')
import torch
model_path_or_name="upstage/TinySolar-248m-4k"
from transformers import AutoModelForCausalLM
tiny_general_model=AutoModelForCausalLM.from_pretrained(
        model_path_or_name,
        device_map="cpu",
        torch_dtype=torch.bfloat16
        )
from transformers import AutoTokenizer
tiny_general_tokenizer=AutoTokenizer.from_pretrained(
    model_path_or_name
        )
prompt="I am an engineer, I Love"
inputs=tiny_general_tokenizer(prompt, return_tensors="pt")
from transformers import TextStreamer
streamer=TextStreamer(
    tiny_general_tokenizer,
    skip_prompt=True,
    skip_special_tokens=True
        )
outputs=tiny_general_model.generate(
    **inputs,
    streamer=streamer,
    use_cache=True,
    max_new_tokens=128,
    do_sample=False,
    temperature=0.0,
    repetition_penalty=1.1
        )

