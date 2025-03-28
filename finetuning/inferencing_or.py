import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from tqdm import tqdm
import numpy as np


def naive_prompt(example):
    prompt  = f'''
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an AI assistant that will generate a boolean outputs from inputs, where the task is "OR" logic function, where positive values represent "TRUE" and negative ones represent "FALSE". 
        If the input is of type: [x1,..,xn] the output is [y1,..,yn] where y1 is true if x1 is true, y2 is true if x1 or x2 are true, like this for other yi.
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        CONTEXT: {example}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>#Answer: 
        '''
    return prompt

def gen_test_data(sigma=30,length=200,seed=42):
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    nums = [round(num, 2) for num in np.random.normal(loc=0, scale=sigma, size=length)]
    return nums

def load_finetuned_model():
    device_map = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        "/tmpdir/naim/llama3.1.8b/Llama-3.1-8B-Instruct",
        return_dict=True,       
        torch_dtype=torch.float16,
        device_map=device_map,  
    )
    print("BaseLine Model Loaded !!")
    print("-------------------------------------")
    model = PeftModel.from_pretrained(model, "/tmpdir/naim/finetuning/model/results_2_epoch_dummy_OR/checkpoint-13120", device_map=device_map)
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained("/tmpdir/naim/llama3.1.8b/Llama-3.1-8B-Instruct", use_fast=True,trust_remote_code=True)
    tokenizer.pad_token_id = 18610
    tokenizer.padding_side = "right"
    print("Fine tuned Model and tokenizer Loaded Locally !!")

    print("We are trying to get outputs for the following data: ")
    #nums = [1,2,-1,3,4]
    key = 1
    outputs = {}
    # Calculate total iterations for the outer progress bar
    nb_len = 151
    nb_sigma = 30
    total_lengths = len(range(10, nb_len, 10))
    total_sigmas = len(range(1, nb_sigma, 1))
    total_iterations = total_lengths * total_sigmas
    #with tqdm(total=total_iterations, desc="Overall Progress") as pbar:
    for length in tqdm(range(10,nb_len,10), desc="Lengths", total=total_lengths):
        for sigma in tqdm(range(1,nb_sigma,1), desc="Sigmas", total=total_sigmas):
            seed = 100*length+sigma
            nums = gen_test_data(sigma=sigma,length=length,seed=seed)
            prompt = naive_prompt(nums)
                #print(f"Prompt passed to the model: {prompt}")
            try:
                model_inputs = tokenizer(prompt,return_tensors = "pt").to("cuda")
                output = model.generate(**model_inputs , max_length = 10000, pad_token_id= tokenizer.eos_token_id,eos_token_id= tokenizer.eos_token_id)
                question_to_claims = tokenizer.decode(output[0], skip_special_tokens=True)
                outputs[key] = question_to_claims
                    
                    
            except Exception as e:
                outputs[key] = "[]"
                    #print(f"Error: {e}")
                    #continue
            key += 1
            #pbar.update(1)
    print("Dumping to Output file")
    with open("/tmpdir/naim/finetuning/data/output/outputs_OR_finetuned_1_150.json", "w") as f:
        json.dump(outputs, f, indent=4)

    print("-------------------------------------")
if __name__ == "__main__":
    load_finetuned_model()

    