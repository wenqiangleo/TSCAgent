def generate_traffic_descriptions(obs_dict):
    """
    根据观察字典生成交通场景描述和交通状态描述
    
    Args:
        obs_dict (dict): 包含交通状态信息的字典
        
    Returns:
        tuple: (场景描述, 状态描述)
    """
    # 提取基本信息
    cur_phase = obs_dict['tsc_feat'][0]
    phase_num = obs_dict['tsc_feat'][1]
    dur_phase = obs_dict['tsc_feat'][3]
    lane_num = len(obs_dict['lane_feat'])
    
    # 生成相位-车道映射
    phase_lane_mapping = {}
    for lane_id, phase_id in zip(obs_dict['lane-phase'][0], obs_dict['lane-phase'][1]):
        if phase_id not in phase_lane_mapping:
            phase_lane_mapping[phase_id] = []
        phase_lane_mapping[phase_id].append(lane_id)
    
    # 生成场景描述
    scene_desc = f"该路口有{phase_num}个相位，分别是[0,1,2,3]，有{lane_num}个车道，分别是[0, 1, 2, 3, 4, 5, 6, 7]，"
    scene_desc += "其中"
    for phase in range(phase_num):
        lanes = phase_lane_mapping[phase]
        scene_desc += f"相位{phase}控制车道{lanes}，"
    
    scene_desc += "车道"
    for i, lane in enumerate(obs_dict['lane_feat']):
        scene_desc += f"({i})的可观测范围为{lane[0]}米，"
    
    # 生成状态描述
    state_desc = f"目前该交叉口的当前相位为{cur_phase}，当前相位持续时间为{dur_phase}。"
    
    # 获取所有相位的交通状态
    for phase in range(phase_num):
        phase_state = obs_dict['phase_feat'][phase]
        state_desc += f"\n相位({phase})控制的车道的平均车辆数量为{phase_state[3]:.2f}，"
        state_desc += f"排队车辆为{phase_state[4]:.2f}，"
        state_desc += f"平均车速为{phase_state[5]:.2f}m/s，"
        state_desc += f"车辆到路口的平均距离为{phase_state[6]:.2f}米。"
    
    return scene_desc, state_desc

# 测试代码
if __name__ == "__main__":
    test_dict = {
        'tsc_feat': [0, 4, 0, 0],
        'phase_feat': [[1, 0.5, 239.405, 0.0, 0.0, 0.0, 0.0],
                      [0, 0.5, 239.405, 0.0, 0.0, 0.0, 0.0],
                      [0, 0.5, 244.09, 0.0, 0.0, 0.0, 0.0],
                      [0, 0.5, 244.09, 0.0, 0.0, 0.0, 0.0]],
        'lane_feat': [[250, 0, 0, 0.0, 0.0],
                     [250, 0, 0, 0.0, 0.0],
                     [250, 0, 0, 0.0, 0.0],
                     [250, 0, 0, 0.0, 0.0],
                     [238.18, 0, 0, 0.0, 0.0],
                     [238.18, 0, 0, 0.0, 0.0],
                     [228.81, 0, 0, 0.0, 0.0],
                     [228.81, 0, 0, 0.0, 0.0]],
        'phase-tsc': [[0, 1, 2, 3], [0, 0, 0, 0]],
        'lane-phase': [[0, 1, 2, 3, 4, 5, 6, 7], [2, 3, 0, 1, 2, 3, 0, 1]]
    }
    
    scene_desc, state_desc = generate_traffic_descriptions(test_dict)
    print("路口场景描述：")
    print(scene_desc)
    print("\n交通状态描述：")
    print(state_desc) 