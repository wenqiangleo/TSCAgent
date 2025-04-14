import json
import time
import random
import requests
import dashscope
from openai import OpenAI
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import os

class OpenRouterClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
        }
        self.last_request_time = 0
        self.min_request_interval = 5  # OpenRouter的最小请求间隔（秒）

    def generate_response(self, messages: List[Dict[str, str]]) -> Optional[str]:
        try:
            # 确保请求间隔
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last_request)
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json={
                    "model": "deepseek/deepseek-r1:free",
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 1000,
                }
            )
            response.raise_for_status()
            self.last_request_time = time.time()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if "rate_limit" in str(e).lower():
                print("OpenRouter达到速率限制，等待30秒...")
                time.sleep(30)  # 遇到限制时等待30秒
                return self.generate_response(messages)  # 重试
            print(f"OpenRouter生成响应时出错: {e}")
            return None

class QwenClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        dashscope.api_key = api_key
        self.last_request_time = 0
        self.min_request_interval = 10  # Qwen的最小请求间隔（秒）

    def generate_response(self, messages: List[Dict[str, str]]) -> Optional[str]:
        try:
            # 确保请求间隔
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last_request)
            
            response = dashscope.Generation.call(
                model="qwq-plus-0305",
                messages=messages,
                stream=True,
            )
            
            reasoning_content = ""
            answer_content = ""
            is_answering = False
            
            for chunk in response:
                if (chunk.output.choices[0].message.content == "" and 
                    chunk.output.choices[0].message.reasoning_content == ""):
                    continue
                    
                if (chunk.output.choices[0].message.reasoning_content != "" and 
                    chunk.output.choices[0].message.content == ""):
                    reasoning_content += chunk.output.choices[0].message.reasoning_content
                elif chunk.output.choices[0].message.content != "":
                    answer_content += chunk.output.choices[0].message.content
            
            self.last_request_time = time.time()
            return f"<reasoning>{reasoning_content}</reasoning>\n<answer>{answer_content}</answer>"
        except Exception as e:
            if "rate_limit" in str(e).lower():
                print("Qwen达到速率限制，等待30秒...")
                time.sleep(30)  # 遇到限制时等待30秒
                return self.generate_response(messages)  # 重试
            print(f"Qwen生成响应时出错: {e}")
            return None

class KimiClient:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1",
        )
        self.last_request_time = 0
        self.min_request_interval = 20  # 最小请求间隔（秒）

    def generate_response(self, messages: List[Dict[str, str]]) -> Optional[str]:
        try:
            # 确保请求间隔
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last_request)
            
            completion = self.client.chat.completions.create(
                model="moonshot-v1-8k",
                messages=messages,
                temperature=0.7,
            )
            self.last_request_time = time.time()
            return completion.choices[0].message.content
        except Exception as e:
            if "rate_limit_reached_error" in str(e):
                print("Kimi达到速率限制，等待60秒...")
                time.sleep(60)  # 遇到限制时等待1分钟
                return self.generate_response(messages)  # 重试
            print(f"Kimi生成响应时出错: {e}")
            return None

def load_sft_dataset(file_path: str) -> List[Dict]:
    """加载SFT数据集"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_enhanced_instruction() -> str:
    """创建增强版的指令"""
    return """你是一位交通管理专家。请根据给定的交通场景和状态，分析并预测下一个最优信号相位。

关键概念说明：
1. 信号相位：是指交通信号灯的一组状态组合，包括不同方向的车流通行权分配。
2. 车道与相位关系：每个相位控制特定车道的通行权，需要考虑车道的交通流量和排队长度。
3. 相位选择原则：
   - 优先考虑排队长度最长的车道
   - 考虑车道的交通流量和等待时间
   - 注意交通拥堵主要由早期排队车辆决定
   - 结合各个相位的车辆速度
   - 不必紧急考虑远距离路段的车辆

分析步骤：
1. 分析当前交通状态
2. 识别关键拥堵点
3. 评估各相位优先级
4. 选择最优相位

请按以下格式回答：
<reasoning>详细的分析推理过程</reasoning><answer>选择的相位ID</answer>"""

def process_samples(samples: List[Dict], client, model_name: str, temp_output_path: str) -> List[Dict]:
    """处理一组样本"""
    results = []
    total = len(samples)
    
    for i, sample in enumerate(samples, 1):
        messages = [
            {"role": "system", "content": create_enhanced_instruction()},
            {"role": "user", "content": sample["input"]}
        ]
        
        # 添加重试机制
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            response = client.generate_response(messages)
            if response:
                # 创建新的样本
                new_sample = {
                    "instruction": create_enhanced_instruction(),
                    "input": sample["input"],
                    "output": response
                }
                results.append(new_sample)
                
                # 每收集一条数据就保存一次
                try:
                    # 如果文件已存在，先读取现有数据
                    existing_data = []
                    if os.path.exists(temp_output_path):
                        with open(temp_output_path, 'r', encoding='utf-8') as f:
                            existing_data = json.load(f)
                    
                    # 添加新数据并保存
                    existing_data.append(new_sample)
                    with open(temp_output_path, 'w', encoding='utf-8') as f:
                        json.dump(existing_data, f, ensure_ascii=False, indent=2)
                    print(f"已保存第 {len(existing_data)} 条数据")
                except Exception as e:
                    print(f"保存数据时出错: {e}")
                
                break
            else:
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(2)  # 重试前等待2秒
    
    return results

def load_processed_indices(file_path: str = "processed_indices.json") -> set:
    """加载已处理的样本索引"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        return set()
    except Exception as e:
        print(f"加载已处理索引时出错: {e}")
        return set()

def save_processed_indices(indices: set, file_path: str = "processed_indices.json"):
    """保存已处理的样本索引"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(list(indices), f)
        print(f"已保存 {len(indices)} 个已处理索引")
    except Exception as e:
        print(f"保存已处理索引时出错: {e}")

def generate_rl_dataset(
    sft_data: List[Dict],
    openrouter_key: str,
    qwen_key: str,
    kimi_key: str,
    total_samples: int = 6
) -> List[Dict]:
    """生成强化学习数据集"""
    # 加载已处理的索引
    processed_indices = load_processed_indices()
    
    # 获取可用的样本索引
    available_indices = [i for i in range(len(sft_data)) if i not in processed_indices]
    
    if not available_indices:
        print("警告：所有样本都已被处理过！")
        return []
    
    # 随机选择样本
    selected_indices = random.sample(available_indices, min(total_samples, len(available_indices)))
    selected_samples = [sft_data[i] for i in selected_indices]
    
    # 立即更新并保存已处理的索引
    processed_indices.update(selected_indices)
    save_processed_indices(processed_indices)
    print(f"已选择 {len(selected_indices)} 个新样本，当前共处理 {len(processed_indices)} 个样本")
    
    # 设置模型权重
    weights = {
        "openrouter": 0.0,
        "qwen": 1.0,
        "kimi": 0.0
    }
    
    # 根据权重分配样本数量
    samples_per_model = {
        "openrouter": int(len(selected_samples) * weights["openrouter"]),
        "qwen": int(len(selected_samples) * weights["qwen"]),
        "kimi": len(selected_samples) - int(len(selected_samples) * (weights["openrouter"] + weights["qwen"]))
    }
    
    # 初始化客户端
    clients = {
        "openrouter": OpenRouterClient(openrouter_key),
        "qwen": QwenClient(qwen_key),
        "kimi": KimiClient(kimi_key)
    }
    
    rl_dataset = []
    temp_output_path = "./tsc_rl_dataset_temp.json"
    
    # 如果临时文件已存在，先删除它
    if os.path.exists(temp_output_path):
        os.remove(temp_output_path)
        print("已清除旧的临时文件")
    
    # 创建一个空的JSON文件
    with open(temp_output_path, 'w', encoding='utf-8') as f:
        json.dump([], f)
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        
        # 为每个模型分配样本
        current_idx = 0
        for model_name, client in clients.items():
            num_samples = samples_per_model[model_name]
            model_samples = selected_samples[current_idx:current_idx + num_samples]
            current_idx += num_samples
            
            futures.append(
                executor.submit(process_samples, model_samples, client, model_name, temp_output_path)
            )
        
        # 等待所有任务完成并收集结果
        for future in as_completed(futures):
            try:
                results = future.result()
                if results:  # 只添加非空结果
                    rl_dataset.extend(results)
            except Exception as e:
                print(f"处理样本时出错: {e}")
    
    # 最后读取临时文件中的所有数据
    try:
        with open(temp_output_path, 'r', encoding='utf-8') as f:
            final_dataset = json.load(f)
            print(f"最终数据集包含 {len(final_dataset)} 条数据")
            return final_dataset
    except Exception as e:
        print(f"读取最终数据集时出错: {e}")
        return rl_dataset  # 如果读取失败，返回内存中的数据集

def save_dataset(dataset: List[Dict], output_path: str, append: bool = True):
    """保存数据集到JSON文件
    Args:
        dataset: 要保存的数据集
        output_path: 输出文件路径
        append: 是否追加到现有文件，默认为True
    """
    try:
        existing_data = []
        if append and os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        
        # 合并现有数据和新数据
        final_data = existing_data + dataset
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)
        print(f"数据集已保存到: {output_path}，共 {len(final_data)} 条数据")
    except Exception as e:
        print(f"保存数据集时出错: {e}")

def main():
    # 配置参数
    openrouter_key = ""
    qwen_key = ""
    kimi_key = ""
    
    sft_dataset_path = "./tsc_sft_dataset.json"
    output_path = "./tsc_rl_dataset.json"
    temp_output_path = "./tsc_rl_dataset_temp.json"
    total_samples = 20  # 总共要生成的样本数
    
    try:
        # 加载SFT数据集
        print("加载SFT数据集...")
        sft_data = load_sft_dataset(sft_dataset_path)
        
        # 生成RL数据集
        print("开始生成RL数据集...")
        rl_dataset = generate_rl_dataset(
            sft_data, 
            openrouter_key,
            qwen_key,
            kimi_key,
            total_samples
        )
        
        # 保存最终数据集
        if len(rl_dataset) > 0:
            print(f"生成完成，新增 {len(rl_dataset)} 条数据")
            save_dataset(rl_dataset, output_path, append=True)
        else:
            print("警告：没有成功生成任何数据")
            
    except Exception as e:
        print(f"程序执行出错: {e}")
        # 尝试加载临时文件
        try:
            with open(temp_output_path, 'r', encoding='utf-8') as f:
                temp_data = json.load(f)
                if len(temp_data) > 0:
                    print(f"已恢复临时保存的 {len(temp_data)} 条数据")
                    save_dataset(temp_data, output_path, append=True)
        except:
            print("无法恢复临时数据")

if __name__ == "__main__":
    main() 