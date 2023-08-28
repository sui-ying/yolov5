import os
import codecs


def count_lines_in_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            total_lines = 0
            # 判断文件是否为文本文件
            if file_path.endswith('.txt') or file_path.endswith('.csv'):
                try:
                    with codecs.open(file_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                        total_lines += len(lines)
                except UnicodeDecodeError:
                    print(f"无法解析文件：{file_path}")
            if total_lines != 4:
                print(file_name)

    return


# 指定要统计行数的文件夹路径
folder_path = '/cv/all_training_data/container/type/dataset1/info'

# 调用函数并打印结果
count_lines_in_folder(folder_path)
