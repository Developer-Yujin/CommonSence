import os
import re
from tqdm import tqdm

import pandas as pd 
import torch
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset

import argparse 
from transformers import set_seed
from datasets import Dataset, DatasetDict
from ast import literal_eval

from huggingface_hub import login

from vllm import LLM, SamplingParams

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

login(token="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")

def generate_dict2(df):
    prompt_t = "문제에 주어진 답안 중 가장 적절한 답안을 선택하시오. 답안은 0,1,2,3 중 하나로만 선택하시오."

    ins_list = [ prompt_t for _ in range(len(df))] 
    q_list = df['문제']
    choice_tmp_list = []

    choices_list = df['선택지']

    for choi in choices_list:
        choi_list= literal_eval(choi)
        choice_tmp_list.append(choi_list)
    
    choi_tmp_list2 = pd.DataFrame(choice_tmp_list)
    choice1_list =choi_tmp_list2[0]
    choice2_list =choi_tmp_list2[1]
    choice3_list =choi_tmp_list2[2]
    choice4_list =choi_tmp_list2[3]

    dataset_dict = {'instruction': ins_list, 'question': q_list, 'choice1': choice1_list, 'choice2': choice2_list, 'choice3': choice3_list, 'choice4': choice4_list,}
    dataset = Dataset.from_dict(dataset_dict)

    return dataset


def create_datasets2(df):
    """ 
    Customized function for converting dataframes to huggingface datasets 
    """
    def preprocess2(samples):
        batch = []
        PROMPT_DICT = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context.\n"
                "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
                "Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n"
                "### Example\n\n"
                "## Instruction:\n다음 중 빈칸에 들어갈 의미상 가장 알맞은 단어를 고르세요.\n\n"
                "## Question:\n일출이나 일몰의 붉은색은 주로 _______ 않은 빛에 기인합니다.\n\n"
                "## Choice:\n0) 흡수된\n1) 전달된\n2) 산란된\n3) 편광된\n\n"
                "## Answer: 답은 2\n\n"
                # "## Instruction:\n다음 중 알맞은 보기를 고르세요.\n\n"
                # "## Question:\n계획과 성과 사이에는 명확한 연관성이 없다'는 입장을 취한 사람은 누구입니까?\n\n"
                # "## Choice:\n0) 마이클 포터\n1) 밀턴 프리드먼\n2) 게리 하멜\n3) 헨리 민츠버그\n\n"
                # "## Answer: 답은 3\n\n"
                "## Instruction:\n이어질 알맞은 문장을 고르세요. \n\n"
                "## Question:\n나이 든 근로자와 젊은 근로자가 직장을 잃었을 때, 나이 든 근로자는 일반적으로\n\n"
                "## Choice:\n0) 더 많은 경험으로 인해 더 빨리 직장을 찾는다\n1) 새 직장을 찾는 데 더 오래 걸린다\n2) 새 직장을 찾기보다는 은퇴한다\n3) 고령자 고용 차별법(ADEA)을 이용하여 회사를 고소한다\n\n"
                "## Answer: 답은 1\n\n"
                # "## Instruction:\n다음 문장을 이해해 알맞은 수치를 계산하시오. \n\n"
                # "## Question:\n현재 가처분 소득이 $10000이고 소비 지출이 $8000이라고 가정합니다. 가처분 소득이 $100 증가할 때마다 저축이 $10 증가합니다. 이 정보를 바탕으로,\n\n"
                # "## Choice:\n0) 한계 소비 성향은 0.80입니다.\n1) 한계 저축 성향은 0.20입니다.'\n2) 한계 저축 성향은 0.10입니다.'\n3) 한계 저축 성향은 0.90입니다.\n\n"
                # "## Answer: 1\n\n"
                "## Instruction:\n다음을 계산하세요.. \n\n"
                "## Question:\n방정식 x - 10.5 = -11.6을 해결하십시오.\n\n"
                "## Choice:\n0) -22.1\n1) 1.1\n2) 22.1\n3) -1.1\n\n"
                "## Answer: 답은 3\n\n"
                # "## Instruction:\n다음을 계산하세요.\n\n"
                # "##Question:\n40세 여성 비서가 2개월 동안 피로감, 전반적인 통증 및 사지의 근위근 약화로 인해 병원에 왔습니다. 환자는 처음에는 차에 타고 내릴 때만 약화를 느꼈지만, 지난 2주 동안 약화가 진행되어 이제는 머리를 빗는 것도 어려워졌습니다. 증상이 시작된 이후로 그녀는 손 관절의 통증도 있었으며, 이부프로펜에 부분적으로 반응했습니다. 그녀는 입양되었고 가족력은 알 수 없습니다. 그녀는 건강한 십대 자녀 두 명이 있습니다. 그녀는 불편해 보입니다. 키는 170 cm(5 ft 7 in)이고 체중은 68 kg(150 lb)이며, BMI는 24 kg/m2입니다. 활력 징후는 체온 37.7°C(99.8°F), 맥박 90회/분, 호흡 20회/분, 혈압 110/70 mm Hg입니다. 환자는 의식이 명료하고 완전히 지남력이 있습니다. 신체 검사에서는 몇몇 손가락 끝과 측면을 포함한 양손의 피부 갈라짐이 발견되었습니다. 상완과 다리 근육은 압력에 약간 민감합니다. 혈청 검사 결과 크레아틴 키나아제 농도는 600 U/L이고, 젖산 탈수소 효소 농도는 800 U/L입니다. 전체 혈구 수치는 기준 범위 내에 있습니다. 다음 중 가장 가능성이 높은 진단은 무엇입니까?\n\n"
                # "##Choice:\n0)섬유근육통\n1)중증 근무력증\n2)다발성 근염\n3)경피증\n\n"
                # "##Answer: 2\n\n"

                "## Instruction:\n{instruction}\n\n## Question:\n{question}\n\n## Choice:\n0) {choice1}\n1) {choice2}\n2) {choice3}\n3) {choice4}\n\n"
                "## Answer : 답은 "
            ),
        }
        for instruction, question, choice1, choice2, choice3, choice4 in zip(samples["instruction"], samples["question"], samples["choice1"], samples["choice2"], samples["choice3"], samples["choice4"]):
            conversation = PROMPT_DICT['prompt_input'].replace('{instruction}', instruction).replace('{question}', question).replace('{choice1}', choice1).replace('{choice2}', choice2).replace('{choice3}', choice3).replace('{choice4}', choice4)
            batch.append(conversation)

        return {"content": batch}

    dataset = generate_dict2(df)
    
    raw_datasets = DatasetDict()
    raw_datasets["test"] = dataset

    raw_datasets = raw_datasets.map(
        preprocess2,
        batched=True,
        remove_columns=raw_datasets["test"].column_names,
    )

    test_data = raw_datasets["test"]
    # print(
    #     f"Size of the test set: {len(test_data)}"
    # )
    # print(f"A sample of test dataset: {test_data[1]}")

    return test_data

if __name__ == "__main__":

    torch.cuda.empty_cache()

    # set base directory 
    BASE_DIR = os.path.dirname(__file__)

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', required=False, default=42, help='add seed number')
    parser.add_argument('--response_split', required=False, default='\nThe answer is', help='add response splitter')
    parser.add_argument('--model_path', required=False, default='', help='add pretrained model path')

    args = parser.parse_args()

    # set seed for reproducibility
    set_seed(args.seed)

    df = pd.read_csv(os.path.join(BASE_DIR, 'data/test_data.csv'), encoding='utf-8')

    test_dataset = create_datasets2(
        df
    )



    # 모델 생성
    model = LLM(model="Qwen/Qwen2-72B-Instruct-AWQ", 
                dtype="half", 
                trust_remote_code=True,
                quantization="AWQ", 
                # load_format="bitsandbytes",
                tensor_parallel_size=4,
                max_model_len = 1024, 
                max_num_seqs = 400,
                enforce_eager=True
                )

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    err_cnt = 0

    id_list, answer_list = list(), list()
    df_submission = pd.DataFrame()

    for i, test_data in enumerate(tqdm(test_dataset)): 
        input = [test_data["content"]]
        res = model.generate(prompts=input, sampling_params=sampling_params)
        response = res[0].outputs[0].text
        print(f"응답 : {response}")

        try: 

            first_number = re.findall(r'[0-9]+', response)

            if len(first_number) == 0:
                print(f"answer : no")
                
                err_cnt += 1
                print(f"err_cnt : {err_cnt}")
                id_list.append(i)
                answer_list.append(0)
            
            else:
                ans = first_number[0]
            
                print(f"answer : {ans}")
                if int(ans) > 3 or int(ans) < 0:
                    err_cnt += 1
                    print(f"err_cnt : {err_cnt}")
                    id_list.append(i)
                    answer_list.append(0)
                else:
                    # 0~ 3 사이로 추론한 값만 넣는다.
                    id_list.append(i)
                    answer_list.append(int(ans))
        
        except Exception as e:
            # 답을 내지 못한 문제는 일단 0으로 처리
            err_cnt += 1
            print(f"err_cnt : {err_cnt}")
            id_list.append(i)
            answer_list.append(0)

    print(f"총 에러 발생 수 : {err_cnt}")
    df_submission['id'] = id_list
    df_submission['answer'] = answer_list
    df_submission.to_csv(os.path.join(BASE_DIR, 'submission.csv'), index=False)

