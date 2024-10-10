import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import TextStreamer
model_name="upstage/TinySolar-248m-4k-py"
tiny_model=AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.bfloat16
        )
tiny_tokenizer=AutoTokenizer.from_pretrained(
    model_name
        )
prompt="def find_max(numbers):"
inputs=tiny_tokenizer(prompt, return_tensors="pt")
streamer=TextStreamer(
    tiny_tokenizer,
    skip_prompt=True,
    skip_special_tokens=True
        )
outputs=tiny_model.generate(
    **inputs,
    streamer=streamer,
    use_cache=True,
    max_new_tokens=128,
    do_sample=False,
    temperature=0.0,
    repetition_penalty=1.1
        )

