import os
import warnings
import transformers
import torch
import logging
import time
#import bitsandbytes as bnb -- not needed but activate for quantisation on a less powerful GPU 
from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
import random
import numpy as np
import json

from tqdm import tqdm

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

model_id = "/work/m24047/m24047flhg/Llama-3.3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
# Add padding token configuration
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
  ]
pad_token_id = 18610  # This corresponds to `#***`
tokenizer.pad_token_id = pad_token_id 
tokenizer.padding_side = "right"  # Optional but recommended for most cases

SYSTEM_PROMPT = '''
    You are an auto-regressive AI model designed to predict the next value in a sequence of input-output pairs that follow a linear function f(x) = ax + b.
    Your task is to analyze the given input-output pairs and predict the output for the final input value.
    
    Here are some examples to illustrate the task:

    Example 1:
    CONTEXT: [1, 3, 2, 5, 3, 7, 4]
    #Answer: 9
    

    Example 2:
    CONTEXT: [0, 1, 2, 5, 4, 9, 6]
    #Answer: 13

    Example 3:
    CONTEXT: [1, -1, 2, -3, 3, -5, 4]
    #Answer: -7


    Example 4:
    CONTEXT: [0, 0, 2, 6, 4, 12, 6]
    #Answer: 18

    Example 5:
    CONTEXT: [-2, -3, 0, 1, 2, 5, 4]
    #Answer: 9

    Now, given a new sequence of input-output pairs where the last output is missing, predict the final value.
    DO NOT include any explanations in your response.
    DO NOT use any PYTHON code in your response.
    GIVE JUST THE NUMERICAL OUTPUT AS THE ANSWER.
    '''

model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        use_cache=True,
        attn_implementation="flash_attention_2",  # Enable Flash Attention 2
    )

print("Model with flash attention 2 and Tokenizer loaded!!")

def generate_llama_output(input_instruction):
    messages = [
        {
            'role':'system',
            'content':SYSTEM_PROMPT
        },
        {
            'role':'user',
            'content':input_instruction.strip()
        }
    ]
    
    input_ids = tokenizer.apply_chat_template(messages,add_generation_prompt=True,return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=10000,
        do_sample=True,
        eos_token_id=terminators,
        temperature=0.6,
        top_p=0.9,
    )

    response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    #response_lines = [line.strip() for line in response.split('\n') if line.strip().startswith('place(')]

    #return '\n'.join(response_lines)
    return response

def alternate_lists(list1, list2):
    if not isinstance(list1, list) or not isinstance(list2, list):
        list2 = [list2]
    return [round(x, 2) if x in list2 else x for pair in zip(list1, list2) for x in pair]

def gen_test_data(sigma1=1, sigma2=1,seed=42):
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    weights = [round(num, 2) for num in np.random.normal(loc=0, scale=sigma1, size=2)]
    a = weights[0]
    b = weights[1]
    xs = [round(num, 2) for num in np.random.normal(loc=0, scale=sigma1, size=40)]
    # Calculate ys using list comprehension
    ys = [a * x + b for x in xs]
    
    merged = alternate_lists(xs,ys)
    new_input = merged[:-1]
    new_output = merged[-1:]
    return new_input,new_output

def run_inference(output_file_path=""):
    predictions = {}
    total_sigma1 = len(range(1, 30, 1))
    total_sigma2 = len(range(1, 30, 1))
    total_iterations = total_sigma1 * total_sigma2
    key = 1
    for sigma1 in tqdm(range(1,30,1), desc="sigma_x", total=total_sigma1):
        #print("length", length)
        for sigma2 in tqdm(range(1,30,1), desc="sigma_w", total=total_sigma2):
            #print("sigma",sigma)
            seed = 100*total_sigma1+total_sigma2 +10000
            input,output = gen_test_data(sigma1=sigma1,sigma2=sigma2,seed=seed)
            try:
                
                #print("nums",nums)
                user_prompt = f"CONTEXT: {input}"
                pred_output = generate_llama_output(user_prompt)
                #print("output",output)
                predictions[key] = {
                    'input':input,
                    'output':pred_output,
                    'gold':output
                }
            except Exception as e:
                #print("exception", e)
                predictions[key] = {
                    'input':input,
                    'output':"[]",
                    'gold':output
                }
            #break
            key += 1
        #break
    with open(output_file_path,'w') as f:
        json.dump(predictions,f,indent=4)

if __name__ == "__main__":
    run_inference(output_file_path="/tmpdir/m24047nmmr/finetuning/outputs_lf/lf_70b_2.json")


