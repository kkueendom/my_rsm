import os
import re
import shutil

# 设置你的query_graph目录路径
query_graph_dir = 'dataset/yeast/query_graph'

# 正则表达式，匹配文件名中的节点数
pattern = re.compile(r'_(\d+)_')

for filename in os.listdir(query_graph_dir):
    if filename.endswith('.graph'):
        match = pattern.search(filename)
        if match:
            node_num = match.group(1)
            target_dir = os.path.join(query_graph_dir, node_num)
            os.makedirs(target_dir, exist_ok=True)
            src_path = os.path.join(query_graph_dir, filename)
            dst_path = os.path.join(target_dir, filename)
            print(f"移动 {filename} 到 {target_dir}")
            shutil.move(src_path, dst_path)
        else:
            print(f"未识别节点数，跳过：{filename}")

print("分类完成！")
