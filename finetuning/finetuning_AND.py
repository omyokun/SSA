from process_data import return_datasets
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
import ast
import torch
from peft import LoraConfig, PeftModel
from datasets import Dataset
import pandas as pd
import json
import os
os.environ["WANDB_DISABLED"] = "True"
           
# QLoRA parameters
# LoRA attention dimension

#input_data_path = "/tmpdir/naim/finetuning/data/omar_data_smallest_2.json"
#input_data_path = "/tmpdir/naim/finetuning/linear_functions/input/train_LF.json"
#input_data_csv = "/tmpdir/naim/finetuning/data/omar_data_smallest_41_rounded.csv"
#input_data_csv = "/tmpdir/naim/finetuning/linear_functions/input/linear_functions_data_10_1-4.csv"
input_data_csv = "/tmpdir/naim/finetuning/linear_functions/input/linear_functions_data_10_3.csv"

lora_r = 64
lora_alpha = 16
lora_dropout = 0.1
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
output_dir = "/tmpdir/naim/finetuning/model/results_3_epoch_dummy_LF_10_3"
#output_dir = "/tmpdir/naim/finetuning/model/results_3_epoch_dummy_AND"
num_train_epochs = 3
fp16 = False
bf16 = False
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 0
logging_steps = 50

# SFT parameters
#max_seq_length = 1000
max_seq_length = 512
max_seq_length = None
packing = False
device_map = {"cuda": 0}


def formatting_prompts_func(example):
    # Initialize the dictionary to store formatted texts
    formatted_dict = {'text': []}
    for i in range(len(example['input'])):
        # text  = f'''
        # <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        # You are an AI assistant that will generate a boolean outputs from inputs, where the task is "AND" logic function, where positive values represent "TRUE" and negative ones represent "FALSE".        <|eot_id|>
        # <|start_header_id|>user<|end_header_id|>
        # CONTEXT: {example["input"][i]}
        # <|eot_id|>
        # <|start_header_id|>assistant<|end_header_id|>#Answer: 
        # {example["output"][i]}
        # '''
        text  = f'''
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an AI assistant that will generate an output for an input-output pairs sequence of type (x1,f(x1),...,xn) and you role is to find the value f(xn).        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        CONTEXT: {example["input"][i]}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>#Answer: 
        {example["output"][i]}
        '''
        formatted_dict['text'].append(text)

    return formatted_dict


def prepare_dataset_for_sft():
    # Read the CSV file
    df = pd.read_csv('output.csv')
    
    df['input'] = df['input'].apply(ast.literal_eval)
    df['output'] = df['output'].apply(ast.literal_eval)
    
    df['input'] = df['input'].apply(lambda x: str(x))
    df['output'] = df['output'].apply(lambda x: str(x))
    
    # Convert DataFrame to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    
    # Apply the formatting function
    formatted_dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=dataset.column_names  # Remove original columns
    )
    
    return formatted_dataset

def convert_to_pandas(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    rows = []
    for entry in data:
        for item in entry:
            rows.append(
                {
                    'input': item['input'],
                    'output': item['output']
                }
            )
    df = pd.DataFrame(rows)
    #print(df.head())
    df.to_csv(json_path.replace('.json', '.csv'), index=False)
    return df

def main():
    #train_dataset,test_dataset,dev_dataset = return_datasets()
    model_name = "/tmpdir/naim/llama3.1.8b/Llama-3.1-8B-Instruct"
    # Fine-tuned model name
    #new_model = "Llama-3-8B-SNLI"
    #new_model = "Llama-3-8B-AND-3-epoch"
    new_model = "Llama-31-8B-LF-3-epoch"

    #train_dataset = return_datasets()
    #json2pandas = convert_to_pandas(input_data_path)
    dataset = Dataset.from_pandas(pd.read_csv(input_data_csv))
    print("Dataset Loaded")
    formatted_dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=dataset.column_names
    )
    print("Datasets generated !!")
    #response_template = " ### Answer:"
    response_template = "#Answer:"

    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)
    
    print("Loading Model and Tokenizer ")
    #model = AutoModelForCausalLM.from_pretrained("/tmpdir/bhar/llama3-8B-Instruct-hf")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"The device is:{model.device}")
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    tokenizer = AutoTokenizer.from_pretrained("/tmpdir/naim/llama3.1.8b/Llama-3.1-8B-Instruct",trust_remote_code=True ,truncation = True , add_eos_token = True)
    pad_token_id = 18610  # This corresponds to `#_***`
    tokenizer.pad_token_id = pad_token_id
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
    
    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Set training parameters
    training_arguments = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        max_seq_length=max_seq_length,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        packing=packing,
        report_to=None
    )
    #    report_to=None
    #)
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    #print("All good until here !!")
    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        peft_config=peft_config,
        data_collator=collator,
        tokenizer=tokenizer,
        args=training_arguments,
        
    )
    """
    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        args=SFTConfig(output_dir="/tmpdir/bhar/codes/snli_training/tmp"),
        formatting_func=formatting_prompts_func,
        data_collator=collator,
    )
    """
    print("Trainer Loaded !!")
    print("Starting Training !!")
    trainer.train()
    print("Training Loop Ended")
    print("Saving Model...")
    trainer.model.save_pretrained(new_model)
    print("Model Saved Sucessfully")

if __name__ == "__main__":
    main()
    