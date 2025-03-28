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
    You are an auto-regressive AI model designed to evaluate whether each sublist of a list of numbers has a positive element. 
    Your task is to process the list incrementally, verifying if there exist a positive element in each sublist one by one. 
    A sublist is considered "TRUE" if it has a positive element. 
    Although you process the list step-by-step, you will only output the final list of booleans once all sublists have been evaluated.

    Here are some examples to illustrate the task:

    Example 1:
    CONTEXT: [1, 1, 2, 3, -1, 2, 1]
    #Answer: 
    [True, True, True, True, True, True, True]

    Example 2:
    CONTEXT: [-0.1, -9, -0.11, 5, 0, 3.5]
    #Answer: 
    [False, False, False, True, True, True]

    Example 3:
    CONTEXT: [-1, -2, -3, -4, -5]
    #Answer: 
    [False, False, False, False, False]

    Example 4:
    CONTEXT: [-0.5, 1.5, -0.5, 2.5, -2.5]
    #Answer: 
    [False, True, True, True, True]

    Example 5:
    CONTEXT: [-10, -10, -3, 0.01, -0.01]
    #Answer: 
    [False, False, False, True, True]

    Now, given a new list of numbers, perform the same task and provide the final output in the specified format.
    DO NOT include any other text in your response.
    DO NOT use any PYTHON code in your response, GIVE JUST THE OUTPUT LIST AS THE ANSWER.
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

def gen_test_data(sigma=30,length=200,seed=42):
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    nums = [round(num, 2) for num in np.random.normal(loc=0, scale=sigma, size=length)]
    return nums

def run_inference(output_file_path=""):
    predictions = {}
    start_length = 10
    end_length = 151
    end_sigma = 30
    total_lengths = len(range(start_length, end_length, 10))
    total_sigmas = len(range(1, end_sigma, 1))
    total_iterations = total_lengths * total_sigmas
    key = 1
    for length in tqdm(range(start_length,end_length,10), desc="Lengths", total=total_lengths):
        #print("length", length)
        for sigma in tqdm(range(1,end_sigma,1), desc="Sigmas", total=total_sigmas):
            #print("sigma",sigma)
            try:
                seed = 33*length+sigma*8 + 8
                nums = gen_test_data(sigma=sigma,length=length,seed=seed)
                #print("nums",nums)
                user_prompt = f"CONTEXT: {nums}"
                output = generate_llama_output(user_prompt)
                #print("output",output)
                predictions[key] = {
                    'nums':nums,
                    'output':output
                }
            except Exception as e:
                #print("exception", e)
                predictions[key] = {
                    'nums':nums,
                    'output':"[]"
                }
            #break
            key += 1
        #break
    with open(output_file_path,'w') as f:
        json.dump(predictions,f,indent=4)

if __name__ == "__main__":
    run_inference(output_file_path="/tmpdir/m24047nmmr/finetuning/outputs/llama_70b_OR_output1_150_rerun2.json")


