import os
from tqdm import tqdm
import transformers
import pandas as pd 
import torch
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from transformers import BitsAndBytesConfig

import argparse 
from transformers import set_seed
from datasets import Dataset, DatasetDict
#from transformers import LlamaForCausalLM, LlamaDecoderLayer
import tensor_parallel as tp
from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec

import deepspeed

from huggingface_hub import login
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

login(token='')



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
            "prompt_no_input": (
                "Below is an instruction that describes a task.\n"
                "아래는 작업을 설명하는 명령어입니다.\n\n"
                "Write a response that appropriately completes the request.\n명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n\n"
                "### Instruction(명령어):\n{instruction}\n\n### Response(응답):"
            ),
        }
        for instruction, question, choices in zip(samples["instruction"], samples["question"], samples["choices"]):
            user_input = question + '<|sep|>' + choices  
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

import collections

import json

def create_device_map_from_index(weight_map, num_gpus=4):
    """weight_map 정보를 기반으로 device_map을 생성하는 함수"""

    # 1. 파일별 레이어 그룹화
    layer_groups = collections.defaultdict(list)
    for layer_name, file_name in weight_map.items():
        layer_groups[file_name].append(layer_name)

    # 2. GPU 할당
    gpu_assignments = {i: [] for i in range(num_gpus)} 
    for i, group in enumerate(layer_groups.values()):
        gpu_assignments[i % num_gpus].extend(group) 

    # 'lm_head' 레이어를 마지막 GPU에 할당
    lm_head_layer = 'lm_head.weight'
    if lm_head_layer in gpu_assignments[0]:
        gpu_assignments[0].remove(lm_head_layer)
        gpu_assignments[num_gpus - 1].append(lm_head_layer)

    # 3. device_map 딕셔너리 생성
    device_map = {}
    for gpu_id, layer_names in gpu_assignments.items():
        for layer_name in layer_names:
            device_map[layer_name] = f'cuda:{gpu_id}'

    return device_map
# weight_map 정보를 활용하여 device_map 생성



if __name__ == "__main__":

    # set base directory 
    BASE_DIR = os.path.dirname(__file__)
    
    # Confirm which GPUs are visible
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', required=False, default=42, help='add seed number')
    parser.add_argument('--response_split', required=False, default='\nThe answer is', help='add response splitter')
    parser.add_argument('--model_path', required=False, default='', help='add pretrained model path')

    args = parser.parse_args()
   
    # set seed for reproducibility
    set_seed(args.seed)

    index_file_path = 'model.safetensors.index.json'
    
    
    model_path = os.path.join(BASE_DIR, args.model_path)
    #model_path = args.model_path
    model_name = 'meta-llama/Meta-Llama-3.1-70B-Instruct'

    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,  # 4-bit quantization 사용
    #     bnb_4bit_quant_type='nf4',  # quantization type 설정 (nf4, fp4 등)
    #     bnb_4bit_use_double_quant=True,  # double quantization 사용 (선택 사항)
    #     bnb_4bit_compute_dtype=torch.float16  # 계산에 사용할 데이터 타입
    # )


    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16, attn_implementation="eager", device_map='auto')
    #device_map = create_device_map_from_index(model.config.weight_map) 
    with open(index_file_path, 'r') as f:
        index_data = json.load(f)
    device_map = create_device_map_from_index(index_data["weight_map"])

    print(device_map)

# tensor_parallel 적용
    #model = tp.tensor_parallel(model, device_map)
    #model = tp.tensor_parallel(model, ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print(model)
    df = pd.read_csv(os.path.join(BASE_DIR, 'data/test_data.csv'), encoding='utf-8')
    
    test_dataset = create_datasets(
        df,
        tokenizer,
        apply_chat_template=False
    )
    num_stages = 4 
    layer_specs = [
        LayerSpec(transformers.LlamaDecoderLayer, 20),  # 예시: 첫 10개 레이어를 첫 번째 stage에 할당
        LayerSpec(transformers.LlamaDecoderLayer, 20),  # 예시: 다음 10개 레이어를 두 번째 stage에 할당
        LayerSpec(transformers.LlamaDecoderLayer, 20),  # 예시: 다음 10개 레이어를 세 번째 stage에 할당
        TiedLayerSpec(transformers.LlamaDecoderLayer, 20)   # 예시: 마지막 10개 레이어와 'lm_head'를 네 번째 stage에 할당
    ]

    # PipelineModule 생성 및 DeepSpeed 초기화
    model = PipelineModule(
        layers=model.transformer.h,  # 모델의 레이어들을 전달
        num_stages=num_stages,  # stage 개수 (GPU 개수와 일치해야 함)
        loss_fn=model.loss_fn,  # 손실 함수 전달
        topology=deepspeed.pipe.PipelineTopology(num_stages, 'cuda'),  # GPU 토폴로지 설정
        layer_specs=layer_specs
    )

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config_params="ds_config.json" 
    )


        
    device = "cuda" if torch.cuda.is_available else "cpu"
    model = model.to(device)
    
    # inference 
    df_submission = pd.DataFrame()
    id_list, answer_list = list(), list()

    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        args=args,  # 커맨드 라인 인자 전달
        model=model,
        model_parameters=model.parameters(),
        training_data=test_dataset,  # 데이터셋 전달
        config_params="ds_config.json"  # DeepSpeed 설정 파일 경로
    )
    
    for i, test_data in enumerate(tqdm(test_dataset)): 
        text = test_data['content']
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        
        # Remove 'token_type_ids' if present 
        model_inputs.pop('token_type_ids', None)

        with torch.no_grad():
            generated_ids = model.generate(
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
        
        
    df_submission['id'] = id_list
    df_submission['answer'] = answer_list
    df_submission.to_csv(os.path.join(BASE_DIR, 'submission.csv'), index=False)
    