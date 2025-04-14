import pickle
import json
from untils.description_generator import generate_traffic_descriptions

def generate_training_data(data_path, output_path, last_n_scenes=5, last_n_steps=30):
    """
    生成训练数据集
    
    Args:
        data_path (str): 数据文件路径
        output_path (str): 输出JSON文件路径
        last_n_scenes (int): 使用最后多少个场景
        last_n_steps (int): 使用每个场景最后多少个决策步骤
    """
    # 读取数据
    with open(data_path, 'rb') as file:
        data = pickle.load(file)
    
    # 所有交叉口ID
    train_id = ['1159176756', '1159176757', '1916386495', '1916386553', '2712421554', 
                '2852225599', '3523298662', '3562004404', '3780408667', 
                '432429372', '432429376', '4779165760', '5514165402', '671587213',
                'cluster_1916386555_432429395', 'cluster_2712421553_5213237283', 
                'cluster_3028924422_314614336', 'cluster_432429373_5213238455', 'cluster_5213238458_J22', 
                'cluster_5213238478_5628750476_5628750477_J18', 'cluster_5436607838_J47_J69_J76_J78']
    test_id = ['314655170', '314635491', '8013476764', '432452987', '1159176227', 'J54']
    ts_ids = train_id + test_id
    
    # 准备训练数据
    training_data = []
    
    # 获取最后n个场景
    scenes = data[0][-last_n_scenes:]  # 状态数据
    actions = data[2][-last_n_scenes:]  # 动作数据
    
    # 遍历每个场景
    for scene_idx in range(len(scenes)):
        scene = scenes[scene_idx]
        scene_actions = actions[scene_idx]
        
        # 获取最后n个决策步骤
        steps = scene[last_n_steps:]
        step_actions = scene_actions[last_n_steps:]
        
        # 遍历每个决策步骤
        for step_idx in range(len(steps)):
            step = steps[step_idx]
            step_action = step_actions[step_idx]
            
            # 遍历每个交叉口
            for ts_id in ts_ids:
                if ts_id in step:
                    # 获取交叉口状态
                    obs_dict = step[ts_id]
                    
                    # 生成描述
                    scene_desc, state_desc = generate_traffic_descriptions(obs_dict)
                    
                    # 获取下一个动作
                    next_action = step_action[ts_id]
                    
                    # 构建训练样本
                    sample = {
                        "instruction": "你是一位交通管理专家。你可以运用你的交通常识知识来解决交通信号控制任务。根据给定的交通场景和状态，预测下一个信号相位。你必须直接回答：下一个信号相位是={你预测的相位}",
                        "input": f"路口场景描述：{scene_desc}\n交通状态描述：{state_desc}",
                        "output": f"下一个信号相位：{next_action}"
                    }
                    
                    training_data.append(sample)
    
    # 保存为JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    print(f"已生成{len(training_data)}条训练样本，保存在{output_path}")

if __name__ == "__main__":
    # 测试代码
    data_path = 'history_data./sac_data.pkl'
    output_path = './tsc_sft_dataset.json'
    generate_training_data(data_path, output_path, last_n_scenes=5, last_n_steps=30) 