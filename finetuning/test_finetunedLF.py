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
        You are an AI assistant that will generate an output for an input-output pairs sequence of type (x1,f(x1),...,xn) and you role is to find the value f(xn).        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        CONTEXT: {example}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>#Answer: 
    
        '''
    return prompt

# def gen_test_data(sigma=30,length=200,seed=42):
#     # Set seeds for reproducibility
#     torch.manual_seed(seed)
#     np.random.seed(seed)
#     nums = [round(num, 2) for num in np.random.normal(loc=0, scale=sigma, size=length)]
#     return nums

def alternate_lists(list1, list2):
    if not isinstance(list1, list) or not isinstance(list2, list):
        list2 = [list2]
    return [x if x in list2 else x for pair in zip(list1, list2) for x in pair]

def gen_test_data(sigma1=1, sigma2=1,seed=42):
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    #weights = [round(num, 2) for num in np.random.normal(loc=0, scale=sigma1, size=2)]
    weights = [round(num, 2) for num in np.random.normal(loc=0, scale=sigma2, size=2)]
    a = weights[0]
    b = weights[1]
    #xs = [round(num, 2) for num in np.random.normal(loc=0, scale=sigma1, size=40)]
    xs = [round(num, 2) for num in np.random.normal(loc=0, scale=sigma1, size=40)]
    # Calculate ys using list comprehension
    ys = [round(a * x + b, 2) for x in xs]
    
    merged = alternate_lists(xs,ys)
    new_input = merged[:-1]
    new_output = merged[-1:]
    return new_input,new_output

def load_finetuned_model():
    device_map = "auto"
    model = AutoModelForCausalLM.from_pretrained(
        "/tmpdir/user_name/llama3.1.8b/Llama-3.1-8B-Instruct",
        return_dict=True,       
        torch_dtype=torch.float16,
        device_map=device_map,  
    )
    print("BaseLine Model Loaded !!")
    print("-------------------------------------")
    model = PeftModel.from_pretrained(model, "/tmpdir/user_name/finetuning/model/results_3_epoch_dummy_LF_10_3/checkpoint-14400", device_map=device_map)
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained("/tmpdir/user_name/llama3.1.8b/Llama-3.1-8B-Instruct", use_fast=True,trust_remote_code=True)
    tokenizer.pad_token_id = 18610
    tokenizer.padding_side = "right"
    print("Fine tuned Model and tokenizer Loaded Locally !!")

    print("We are trying to get outputs for the following data: ")
    #nums = [1,2,-1,3,4]
    key = 1
    outputs = {}
    # Calculate total iterations for the outer progress bar
    maxsigma1 = 11
    maxsigma2 = 11
    total_sigma1 = len(range(1, maxsigma1, 1))
    total_sigma2 = len(range(1, maxsigma2, 1))
    total_iterations = total_sigma1 * total_sigma2
    #with tqdm(total=total_iterations, desc="Overall Progress") as pbar:
    for sigma1 in tqdm(range(1,maxsigma1,1), desc="sigma1", total=total_sigma1):
        for sigma2 in tqdm(range(1,maxsigma2,1), desc="sigma2", total=total_sigma2):
            seed = 100*sigma1+sigma2
            input,g_output = gen_test_data(sigma1=sigma1,sigma2=sigma2,seed=seed)
            #user_prompt = f"CONTEXT: {input}"
            #pred_output = generate_llama_output(user_prompt)
            #nums = gen_test_data(sigma=sigma,length=length,seed=seed)
            prompt = naive_prompt(input)
                #print(f"Prompt passed to the model: {prompt}")
            try:
                model_inputs = tokenizer(prompt,return_tensors = "pt").to("cuda")
                output = model.generate(**model_inputs , max_length = 10000, pad_token_id= tokenizer.eos_token_id,eos_token_id= tokenizer.eos_token_id)
                question_to_claims = tokenizer.decode(output[0], skip_special_tokens=True)
 
                outputs[key] = {
                    'input':input,
                    'predictions':question_to_claims,
                    'gold':g_output
                }
                      
                    
            except Exception as e:
                outputs[key] = {
                    'input':input,
                    'predictions':[],
                    'gold':g_output
                }
                print("-"*100)
                print("error in sigma1", sigma1)
                print("exactly in sigma2", sigma2)
                print(f"Error: {e}")
                print("-"*100)
                    
                    #continue
            key += 1
            #pbar.update(1)
    print("Dumping to Output file")
    with open("/tmpdir/user_name/finetuning/data/output/outputs_LFtest_Finally.json", "w") as f:
        json.dump(outputs, f, indent=4)

    print("-------------------------------------")
if __name__ == "__main__":
    load_finetuned_model()

    
