# -*- coding: utf-8 -*-
"""
TSC模型微调脚本
用于加载预训练模型并进行微调的主要脚本文件
"""

from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from huggingface_hub import create_repo, HfApi

def load_base_model(max_seq_length=2048):
    """
    加载基础预训练模型
    
    Args:
        max_seq_length (int): 模型处理文本的最大长度
    
    Returns:
        tuple: (model, tokenizer) 预训练模型和分词器
    """
    # 加载预训练模型，使用4位量化以节省内存
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    return model, tokenizer

def prepare_training_data(tokenizer, num_samples=1000, dataset_name="jiam/tsc-datasets"):
    """
    准备训练数据集
    
    Args:
        dataset_name (str): Hugging Face上的数据集名称
        num_samples (int): 要使用的样本数量
    
    Returns:
        Dataset: 处理后的数据集
    """
    # 定义提示模板
    train_prompt_style = """
### 指令：
{}

### 输入：
{}

### 输出：
{}"""

    # 从Hugging Face下载数据集
    full_dataset = load_dataset(dataset_name, data_files="tsc_sft_dataset.json", split="train")
    
    # 随机选择指定数量的样本
    dataset = full_dataset.shuffle(seed=42).select(range(min(num_samples, len(full_dataset))))
    print(f"已选择 {len(dataset)} 条数据进行训练")
    
    # 处理数据
    def formatting_prompts_func(examples):
        """格式化数据集中的每条记录"""
        texts = []
        for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
            text = train_prompt_style.format(
                instruction,
                input_text,
                output
            ) + tokenizer.eos_token
            texts.append(text)
        return {"text": texts}  # 返回字典格式
    
    # 转换为Dataset格式
    processed_dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return processed_dataset

def setup_model_for_training(model, r=16):
    """
    配置模型用于训练
    
    Args:
        model: 预训练模型
        r (int): LoRA的秩
    
    Returns:
        模型: 配置好的用于训练的模型
    """
    FastLanguageModel.for_training(model)
    
    return FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

def train_model(model, tokenizer, dataset, max_seq_length=2048, num_epochs=1):
    """
    训练模型
    
    Args:
        model: 准备训练的模型
        tokenizer: 分词器
        dataset: 训练数据集
        max_seq_length (int): 最大序列长度
        num_epochs (int): 训练轮数
    """
    batch_size = 2 * 4  # per_device_batch_size * gradient_accumulation_steps
    max_steps = (len(dataset) * num_epochs) // batch_size
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=max_steps // 10,  # 预热步数设为总步数的10%
            max_steps=max_steps,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",
        ),
    )
    
    return trainer.train()

def inference(model, tokenizer, question):
    """
    使用模型进行推理
    
    Args:
        model: 训练好的模型
        tokenizer: 分词器
        question (str): 输入问题
    
    Returns:
        str: 模型生成的回答
    """
    prompt_style = """
### 指令：
你是一位交通管理专家。你可以运用你的交通常识知识来解决交通信号控制任务。根据给定的交通场景和状态，预测下一个信号相位。

### 问题：
{}

### 回答：
{}"""

    FastLanguageModel.for_inference(model)
    inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1200,
        use_cache=True,
    )
    
    return tokenizer.batch_decode(outputs)[0]

def save_model(model, tokenizer, repo_id, token=None):
    """
    保存模型并上传到Hugging Face Hub
    
    Args:
        model: 要保存的模型
        tokenizer: 分词器
        repo_id (str): Hugging Face Hub上的仓库ID，格式为"用户名/仓库名"
        token (str, optional): Hugging Face的访问令牌
    """
    try:
        # 创建仓库（如果不存在）
        create_repo(repo_id, token=token, exist_ok=True)
        print(f"仓库 {repo_id} 准备就绪")
        
        # 将模型上传到Hub
        model.push_to_hub_gguf(
            repo_id,
            tokenizer,
            token=token
        )
        print(f"模型已成功上传到 {repo_id}")
        
    except Exception as e:
        print(f"上传模型时发生错误: {str(e)}")
        raise

def main():
    """
    主函数，用于演示完整的训练流程
    """
    # 1. 加载基础模型
    model, tokenizer = load_base_model()
    
    # 2. 准备训练数据（使用1000条样本）
    dataset = prepare_training_data(tokenizer, num_samples=1000)
    
    # 3. 配置模型用于训练
    model = setup_model_for_training(model)
    
    # 4. 训练模型
    train_stats = train_model(model, tokenizer, dataset)
    
    # 5. 测试推理
    test_question = "路口场景描述：该路口有4个相位，分别是[0,1,2,3]，有8个车道，分别是[0, 1, 2, 3, 4, 5, 6, 7]，其中相位0控制车道[0, 4]，相位1控制车道[1, 5]，相位2控制车道[2, 6]，相位3控制车道[3, 7]，车道(0)的可观测范围为81.36米，(1)的可观测范围为81.36米，(2)的可观测范围为197.34米，(3)的可观测范围为197.34米，(4)的可观测范围为73.07米，(5)的可观测范围为73.07米，(6)的可观测范围为224.53米，(7)的可观测范围为224.53米，\n交通状态描述：目前该交叉口的当前相位为0，当前相位持续时间为10。\n相位(0)控制的车道的平均车辆数量为0.0，排队车辆为0.0，平均车速为0.0m/s，车辆到路口的平均距离为0.0米。\n相位(1)控制的车道的平均车辆数量为0.0，排队车辆为0.0，平均车速为0.0m/s，车辆到路口的平均距离为0.0米。\n相位(2)控制的车道的平均车辆数量为1.5，排队车辆为1.5，平均车速为0.0m/s，车辆到路口的平均距离为2.5013094176369917米。\n相位(3)控制的车道的平均车辆数量为0.5，排队车辆为0.5，平均车速为0.0m/s，车辆到路口的平均距离为0.5005051909216293米。"
    response = inference(model, tokenizer, test_question)
    print(f"问题: {test_question}")
    print(f"回答: {response}")
    
    # 6. 上传模型到Hugging Face Hub
    # 注意：需要设置你的Hugging Face用户名和访问令牌
    repo_id = ""  # 例如 "username/tsc-model"
    token = ""  # 你需要从Hugging Face获取访问令牌
    save_model(model, tokenizer, repo_id, token)

if __name__ == "__main__":
    main() 