from tqdm import tqdm
def process_line(line):
    """
    对每行内容进行修改的示例函数
    :param line: 原始行内容
    :return: 修改后的行内容
    """
    # 在这里进行你需要的修改，示例为将行内容转换为大写
    path, text = line.strip("\n").split("|")
    speaker = path.split("/")[2]
    new_line = "{}|{}|{}".format(path, text, speaker)
    return new_line.strip()  # 去掉换行符并转换为大写

def read_and_modify_file(input_file, output_file):
    """
    读取文本文件，修改内容后写入新文件
    :param input_file: 输入的 .txt 文件路径
    :param output_file: 输出的新 .txt 文件路径
    """
    modified_lines = []

    # 按行读取输入文件
    with open(input_file, 'r') as infile:
        for line in tqdm(infile):
            modified_line = process_line(line)  # 调用处理函数
            modified_lines.append(modified_line)

    # 将修改后的内容写入新文件
    with open(output_file, 'w') as outfile:
        for line in tqdm(modified_lines):
            outfile.write(line + '\n')  # 添加换行符

    print(f"Modified content written to {output_file}")

# 示例用法
if __name__ == "__main__":
    input_file_path = 'Data/val_list_libritts.txt'  # 替换为你的输入文件路径
    output_file_path = 'val_list_Libri.txt'  # 输出的新文件路径
    read_and_modify_file(input_file_path, output_file_path)
