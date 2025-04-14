# -*- coding: utf-8 -*-
"""
TSC RFT (Reinforcement Fine-Tuning) 训练脚本
基于 Llama 3.1 8B 模型的 GRPO (Generative Reward-Penalized Optimization) 训练
"""

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset, Dataset
import re
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams
from huggingface_hub import HfApi

def setup_model(max_seq_length=2048, lora_rank=32):
    """
    设置和初始化模型
    Args:
        max_seq_length: 最大序列长度
        lora_rank: LoRA 秩
    Returns:
        model: 初始化后的模型
        tokenizer: 分词器
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.6,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    return model, tokenizer

def prepare_dataset(num_samples=None):
    """
    准备训练数据集
    Args:
        num_samples: 要使用的样本数量，如果为None则使用全部样本
    Returns:
        dataset: 处理后的数据集
    """
    def process_data(data):
        return {
            'prompt': [
                {'role': 'system', 'content': data['instruction']},
                {'role': 'user', 'content': data['input']}
            ],
            'answer': data['output']
        }

    # 加载数据集
    dataset = load_dataset('jiam/tsc-datasets', data_files='tsc_rl_dataset.json')['train']
    total_samples = len(dataset)
    print(f"数据集总共有 {total_samples} 条样本")
    
    if num_samples is not None:
        # 随机选择指定数量的样本
        dataset = dataset.shuffle(seed=42).select(range(min(num_samples, total_samples)))
        print(f"已选择 {len(dataset)} 条数据进行训练")
    else:
        print(f"使用全部 {total_samples} 条数据进行训练")
    
    dataset = dataset.map(process_data)
    return dataset

def setup_reward_functions():
    """
    设置奖励函数
    Returns:
        list: 奖励函数列表
    """
    def extract_xml_answer(text: str) -> str:
        """从XML格式中提取答案"""
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()

    def extract_xml_reasoning(text: str) -> str:
        """从XML格式中提取推理过程"""
        reasoning = text.split("<reasoning>")[-1]
        reasoning = reasoning.split("</reasoning>")[0]
        return reasoning.strip()

    def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
        """检查答案是否正确"""
        responses = [completion[0]['content'] for completion in completions]
        q = prompts[0][-1]['content']
        extracted_responses = [extract_xml_answer(r) for r in responses]
        print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
        return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

    def reasoning_quality_reward_func(completions, **kwargs) -> list[float]:
        """评估推理过程的质量"""
        responses = [completion[0]['content'] for completion in completions]
        reasoning_list = [extract_xml_reasoning(r) for r in responses]
        
        rewards = []
        for reasoning in reasoning_list:
            # 检查推理过程是否包含关键要素
            score = 0.0
            if "分析" in reasoning:
                score += 0.5
            if "评估" in reasoning:
                score += 0.5
            if "选择" in reasoning:
                score += 0.5
            if "根据" in reasoning:
                score += 0.5
            rewards.append(score)
        return rewards

    def format_reward_func(completions, **kwargs) -> list[float]:
        """检查输出格式是否正确"""
        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, r) for r in responses]
        return [0.5 if match else 0.0 for match in matches]

    def phase_validity_reward_func(completions, **kwargs) -> list[float]:
        """检查相位ID是否有效"""
        responses = [completion[0]['content'] for completion in completions]
        extracted_responses = [extract_xml_answer(r) for r in responses]
        
        rewards = []
        for answer in extracted_responses:
            try:
                phase_id = int(answer)
                # 检查相位ID是否在有效范围内（0-3）
                if 0 <= phase_id <= 3:
                    rewards.append(0.5)
                else:
                    rewards.append(0.0)
            except ValueError:
                rewards.append(0.0)
        return rewards

    return [
        format_reward_func,
        reasoning_quality_reward_func,
        phase_validity_reward_func,
        correctness_reward_func,
    ]

def setup_trainer(model, tokenizer, dataset, max_seq_length=2048, max_prompt_length=1024):
    """
    设置训练器
    Args:
        model: 模型
        tokenizer: 分词器
        dataset: 数据集
        max_seq_length: 最大序列长度
        max_prompt_length: 最大提示长度
    Returns:
        trainer: 训练器实例
    """
    training_args = GRPOConfig(
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=10,
        per_device_train_batch_size=6,
        gradient_accumulation_steps=1,
        num_generations=6,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_seq_length - max_prompt_length,
        max_steps=250,
        save_steps=250,
        max_grad_norm=0.1,
        report_to="none",
        output_dir="outputs",
    )

    return GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=setup_reward_functions(),
        args=training_args,
        train_dataset=dataset,
    )

def test_model(model, tokenizer, test_prompt, lora_path=None):
    """
    测试模型
    Args:
        model: 模型
        tokenizer: 分词器
        test_prompt: 测试提示
        lora_path: LoRA 权重路径
    Returns:
        str: 模型输出
    """
    # 构建完整的提示
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "你是一位交通管理专家。请根据给定的交通场景和状态，分析并预测下一个最优信号相位。\n\n关键概念说明：\n1. 信号相位：是指交通信号灯的一组状态组合，包括不同方向的车流通行权分配。\n2. 车道与相位关系：每个相位控制特定车道的通行权，需要考虑车道的交通流量和排队长度。\n3. 相位选择原则：\n   - 优先考虑排队长度最长的车道\n   - 考虑车道的交通流量和等待时间\n   - 注意交通拥堵主要由早期排队车辆决定\n   - 结合各个相位的车辆速度\n   - 不必紧急考虑远距离路段的车辆\n\n分析步骤：\n1. 分析当前交通状态\n2. 识别关键拥堵点\n3. 评估各相位优先级\n4. 选择最优相位\n\n请按以下格式回答：\n<reasoning>详细的分析推理过程</reasoning><answer>选择的相位ID</answer>"},
            {"role": "user", "content": test_prompt}
        ],
        tokenize=False,
        add_generation_prompt=True
    )

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=2048,
    )

    output = model.fast_generate(
        [text],
        sampling_params=sampling_params,
        lora_request=model.load_lora(lora_path) if lora_path else None,
    )[0].outputs[0].text

    return output

def upload_to_huggingface(model, tokenizer, repo_id, token):
    """
    将模型上传到 Hugging Face Hub
    Args:
        model: 模型
        tokenizer: 分词器
        repo_id: Hugging Face 仓库 ID (例如: "username/model-name")
        token: Hugging Face API token
    """
    # 保存模型到临时目录
    temp_dir = "temp_model"
    model.save_pretrained_merged(temp_dir, tokenizer, save_method="merged_16bit")
    
    # 上传到 Hugging Face
    api = HfApi(token=token)
    api.create_repo(repo_id, exist_ok=True, private=False)
    api.upload_folder(
        folder_path=temp_dir,
        repo_id=repo_id,
        ignore_patterns=["*.md", "*.txt"]
    )
    
    print(f"模型已成功上传到: https://huggingface.co/{repo_id}")

def main():
    """
    主函数
    """
    # 设置模型
    model, tokenizer = setup_model()
    
    # 准备数据集（使用全部样本）
    dataset = prepare_dataset()  # 不传入num_samples参数，将使用全部样本
    
    # 设置训练器
    trainer = setup_trainer(model, tokenizer, dataset)
    
    # 开始训练
    trainer.train()
    
    # 测试模型
    test_prompt = """路口场景描述：该路口有4个相位，分别是[0,1,2,3]，有11个车道，分别是[0, 1, 2, 3, 4, 5, 6, 7]，其中相位0控制车道[0, 1, 2, 6, 7]，相位1控制车道[6, 7, 8]，相位2控制车道[3, 4, 5]，相位3控制车道[9, 10]，车道(0)的可观测范围为103.71米，(1)的可观测范围为103.71米，(2)的可观测范围为103.71米，(3)的可观测范围为250米，(4)的可观测范围为250米，(5)的可观测范围为250米，(6)的可观测范围为250米，(7)的可观测范围为250米，(8)的可观测范围为250米，(9)的可观测范围为91.76米，(10)的可观测范围为91.76米，
交通状态描述：目前该交叉口的当前相位为3，当前相位持续时间为20。
相位(0)控制的车道的平均车辆数量为1.00，排队车辆为0.80，平均车速为0.09m/s，车辆到路口的平均距离为8.35米。
相位(1)控制的车道的平均车辆数量为0.33，排队车辆为0.00，平均车速为0.01m/s，车辆到路口的平均距离为0.40米。
相位(2)控制的车道的平均车辆数量为3.33，排队车辆为3.00，平均车速为0.00m/s，车辆到路口的平均距离为8.67米。
相位(3)控制的车道的平均车辆数量为0.00，排队车辆为0.00，平均车速为0.00m/s，车辆到路口的平均距离为0.00米。"""
    
    output = test_model(model, tokenizer, test_prompt)
    print("Model output:", output)

    # 上传到 Hugging Face
    repo_id = ""  # 替换为你的仓库名
    hf_token = ""  # 替换为你的 Hugging Face token
    upload_to_huggingface(model, tokenizer, repo_id, hf_token)

if __name__ == "__main__":
    main()