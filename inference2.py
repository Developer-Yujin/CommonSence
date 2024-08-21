import os
import torch
from tqdm import tqdm
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, DatasetDict
import argparse
from transformers import set_seed
from torch.utils.data import DataLoader, DistributedSampler
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

def create_datasets(df, tokenizer, apply_chat_template=False):
    def preprocess(samples):
        batch = []
        PROMPT_DICT = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context.\n"
                "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
                "Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n"
                "### Instruction(명령어):\n{instruction}\n\n### Input(입력):\n{input}\n\n### Response(응답):"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task.\n"
                "아래는 작업을 설명하는 명령어입니다.\n\n"
                "Write a response that appropriately completes the request.\n명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n\n"
                "### Instruction(명령어):\n{instruction}\n\n### Response(응답):"
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
    print(f"Size of the test set: {len(test_data)}")
    print(f"A sample of test dataset: {test_data[1]}")

    return test_data

def main(rank, world_size, args):
    # Set up distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[rank])

    # Load data
    df = pd.read_csv(os.path.join(BASE_DIR, 'data/test_data.csv'), encoding='utf-8')
    test_dataset = create_datasets(df, tokenizer, apply_chat_template=False)

    # Create DataLoader with DistributedSampler
    sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(test_dataset, sampler=sampler, batch_size=1)

    device = torch.device("cuda", rank)

    # Run inference
    df_submission = pd.DataFrame()
    id_list, answer_list = [], []

    for i, test_data in enumerate(tqdm(dataloader)):
        text = test_data['content'][0]
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

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

    # Only save the submission file on the main process
    if rank == 0:
        df_submission['id'] = id_list
        df_submission['answer'] = answer_list
        df_submission.to_csv(os.path.join(BASE_DIR, 'submission.csv'), index=False)

    dist.destroy_process_group()

def run_ddp(num_nodes, num_gpus_per_node, args):
    world_size = num_nodes * num_gpus_per_node
    mp.spawn(main, args=(world_size, args), nprocs=num_gpus_per_node, join=True)

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(__file__)

    # Confirm which GPUs are visible
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', required=False, default=42, help='add seed number')
    parser.add_argument('--response_split', required=False, default='\nThe answer is', help='add response splitter')
    parser.add_argument('--model_path', required=True, help='add pretrained model path')
    parser.add_argument('--num_nodes', type=int, default=1, help='number of nodes')
    parser.add_argument('--num_gpus_per_node', type=int, default=torch.cuda.device_count(), help='number of GPUs per node')

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Run distributed inference
    run_ddp(args.num_nodes, args.num_gpus_per_node, args)
