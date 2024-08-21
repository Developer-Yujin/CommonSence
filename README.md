##### 2차 모의경진대회 - 한국어 일반 상식 객관식 문제 풀이 ######

## Directory Structure 
├── data/
│   ├── train_data.csv
│   └── test_data.csv
│
├── configs/
│   └── config.yaml
│
├── utils.py
├── train.py
├── inference.py
├── run.sh
└── sample_submission.csv

## Description
- `data/`: 학습데이터/평가데이터 csv 파일 
- `configs/`: accelerate 및 fsdp config 파일
- `utils.py`: 학습 및 추론하는데 필요한 함수들 정리한 python script
- `train.py`: LLM 모델 학습하는 python script
- `inference.py`: LLM 모델 추론하는 python script
- `run.sh`: LLM 모델 config 값 설정하여 train.py 실행하는 shell script
- `sample_submission.csv`: 제출 csv 파일 예시

## Getting Started
- 도커 컨테이너 접속 : 
    - 로컬 호스트에서 로컬 포트를 원격 호스트로 포워딩 
        `ssh -i ~/.ssh/student_keypair.pem -L 2222:localhost:2222 ubuntu@원격_호스트_IP`
    - 새로운 셸을 열어 포트를 지정해주는 방식으로 ssh 연결
        `ssh -p 2222 root:localhost`
- 환경 설정 : `cat requirements.txt | xargs -n 1 pip install`
- 학습 : config 수정 후, `sh run.sh` 명령어 실행 
- 추론 : `CUDA_VISIBLE_DEVICES=0 python3 inference.py --model_path MODEL_PATH`
  (MODEL_PATH = results/...)

## GPU 확인하는 법
- 한번만 볼 때 : nvidia-smi
- 계속 띄워서 보고 싶을 때 : watch -n 1 nvidia-smi

## 돌고 있는 프로세스 보는 법
ps -aux -> 내가 돌린 프로세스 pid 확인 -> kill -9 pid 

## ssh 환경에서 동시에 돌리면 바로 안되기 때문에 업무 분배가 확실히 필요할 듯
