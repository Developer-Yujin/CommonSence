import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import set_seed
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp

# Define your evaluation function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1)
    return {
        'accuracy': (predictions == labels).float().mean().item()
    }

def generate_dict(df):
    prompt_q = "Provide me an answer to the following question."
    prompt_ca = "You are given the following multiple choices and supposed to output the index of the correct answer."
    prompt_ex = "For instance, if the choices are ['a', 'b', 'c', 'd'] and the answer is 'b' the correct output should be 1."
    
    instruction_list = [[prompt_q + ' ' + prompt_ca + ' ' + prompt_ex] for _ in range(len(df))]
    question_list = df['문제']
    choices_list = df['선택지']
    dataset_dict = {'instruction': instruction_list, 'question': question_list, 'choices': choices_list}
    dataset = Dataset.from_dict(dataset_dict)
    
    return dataset

def create_datasets(df, tokenizer):
    """ 
    Customized function for converting dataframes to huggingface datasets 
    """
    def preprocess(samples):
        batch = []
        PROMPT_DICT = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context.\n"
                "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
                "Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n"
                "### Instruction(명령어):\n{instruction}\n\n### Input(입력):\n{input}\n\n### Response(응답):"
            ),
        }
        for instruction, question, choices in zip(samples["instruction"], samples["question"], samples["choices"]):
            user_input = question + '' + choices  
            conversation = PROMPT_DICT['prompt_input'].replace('{instruction}', instruction[0]).replace('{input}', user_input)
            batch.append(conversation)

        return {"content": batch}

    dataset = generate_dict(df)
    
    raw_datasets = DatasetDict()
    raw_datasets["test"] = dataset

    raw_datasets = raw_datasets.map(
        preprocess,
        batched=True,
        remove_columns=raw_datasets["test"].column_names,
    )

    test_data = raw_datasets["test"]
    print(
        f"Size of the test set: {len(test_data)}"
    )
    print(f"A sample of test dataset: {test_data[1]}")

    return test_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', required=False, default=42, help='add seed number')
    parser.add_argument('--response_split', required=False, default='\nThe answer is', help='add response splitter')
    parser.add_argument('--model_path', required=False, default='MLP-KTLim/llama-3-Korean-Bllossom-8B', help='add pretrained model path')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank for distributed training')

    args = parser.parse_args()
    
    # Initialize distributed training
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    # set seed for reproducibility
    set_seed(args.seed)
    
    model_path = args.model_path
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    df = pd.read_csv('data/test_data.csv', encoding='utf-8')
    
    test_dataset = create_datasets(df, tokenizer)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model = DDP(model, device_ids=[args.local_rank])

    # Set up DataLoader
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)

    df_submission = pd.DataFrame()
    id_list, answer_list = list(), list()

    model.eval()
    for i, batch in enumerate(tqdm(test_dataloader)): 
        text = batch['content'][0]
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        
        # Remove 'token_type_ids' if present 
        model_inputs.pop('token_type_ids', None)

        with torch.no_grad():
            generated_ids = model.module.generate(
                **model_inputs,
                max_new_tokens=9,
                eos_token_id=tokenizer.eos_token_id, 
                pad_token_id=tokenizer.pad_token_id
            )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
        
        # create submission.csv 
        answer = response.split(args.response_split)[1].strip()
        id_list.append(i)
        answer_list.append(answer)

    if dist.get_rank() == 0:
        df_submission['id'] = id_list
        df_submission['answer'] = answer_list
        df_submission.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main()
