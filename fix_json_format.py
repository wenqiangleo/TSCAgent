import json
import re

def is_valid_output(output):
    # 检查是否同时包含reasoning和answer标签
    has_reasoning = bool(re.search(r'<reasoning>.*?</reasoning>', output, re.DOTALL))
    has_answer = bool(re.search(r'<answer>\d+</answer>', output))
    return has_reasoning and has_answer

def check_and_fix_format(data):
    modified = False
    
    # 检查instruction格式
    if not data['instruction'].strip().endswith('<answer>选择的相位ID</answer>'):
        data['instruction'] = data['instruction'].strip() + ' <answer>选择的相位ID</answer>'
        modified = True
    
    # 检查output格式
    output = data['output'].strip()
    
    # 删除重复的reasoning部分
    reasoning_parts = re.findall(r'<reasoning>.*?</reasoning>', output, re.DOTALL)
    if len(reasoning_parts) > 1:
        # 只保留第一个reasoning部分
        output = reasoning_parts[0]
        modified = True
    
    # 清理reasoning后面的answer部分
    # 1. 先提取reasoning部分
    reasoning_match = re.search(r'(<reasoning>.*?</reasoning>)', output, re.DOTALL)
    if reasoning_match:
        reasoning_text = reasoning_match.group(1)
        
        # 2. 从output中提取最后一个有效的数字（可能在answer标签中或文本中）
        all_numbers = re.findall(r'相位(\d+)|<answer>(\d+)</answer>', output)
        final_number = None
        
        # 从所有匹配中找到最后一个非空的数字
        for num_tuple in all_numbers:
            num = next((n for n in num_tuple if n), None)
            if num:
                final_number = num
        
        if not final_number:
            final_number = '0'
        
        # 3. 构建新的output，只包含一个answer标签
        new_output = f'{reasoning_text}\n<answer>{final_number}</answer>'
        
        if new_output != output:
            output = new_output
            modified = True
    
    data['output'] = output
    return modified

def process_json_file(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_modified = 0
        original_count = len(data)
        
        # 过滤并修复数据
        filtered_data = []
        for item in data:
            if is_valid_output(item['output']):
                if check_and_fix_format(item):
                    total_modified += 1
                filtered_data.append(item)
        
        filtered_count = original_count - len(filtered_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        
        print(f'处理完成！')
        print(f'原始数据条数: {original_count}')
        print(f'过滤掉的数据条数: {filtered_count}')
        print(f'保留的数据条数: {len(filtered_data)}')
        print(f'修改格式的数据条数: {total_modified}')
        print(f'新文件已保存为: {output_file}')
        
    except Exception as e:
        print(f'处理文件时出错: {str(e)}')

if __name__ == '__main__':
    input_file = 'tsc_rl_dataset.json'
    output_file = 'tsc_rl_dataset_fixed.json'
    process_json_file(input_file, output_file) 