import os

root_path = '/data1/home/zhangzhedong/tts_dataset'
def check_file_existence_in_txt(input_file):
    """
    读取文本文件中的每一行，判断每一行中的文件路径是否存在
    :param input_file: 输入的 .txt 文件路径
    """
    with open(input_file, 'r') as file:
        for line in file:
            # 去掉行末的换行符并去除空白字符
            file_path = line.strip()
            file_path = line.split('|')[0]
            file_path = os.path.join(root_path, file_path)
            # 检查文件是否存在
            if os.path.isfile(file_path):
                # print(f"{file_path}: Exists")
                continue
            else:
                print(f"{file_path}: Does not exist")

# 示例用法
if __name__ == "__main__":
    input_file_path = 'Data/val_list_LibriTTS460.txt'  # 替换为你的输入文件路径
    check_file_existence_in_txt(input_file_path)
