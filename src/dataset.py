from transformers import GPT2Tokenizer
from datasets import load_dataset

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# 初始化分词器
# GPT2Tokenizer.from_pretrained('gpt2')：
# 这是 Hugging Face 提供的用于 GPT-2 模型的预训练分词器。
# 它能够将输入的文本数据转化为 token 序列（即模型可以处理的数字格式）。

def tokenize_function(examples):
    inputs = [f"Instruction: {instr}\nInput: {inp}\nOutput:" for instr, inp in zip(examples['instruction'], examples['input'])]
    targets = examples['output']
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
    labels = tokenizer(targets, padding="max_length", truncation=True, max_length=512)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

"""
tokenize_function：
这个函数将输入的 examples 数据进行预处理，并将文本数据转换为 GPT-2 模型所需的格式。

inputs 这个列表推导式将 instruction 和 input 字段拼接成一个完整的输入字符串
其格式为 "Instruction:{instr}\nInput:{inp}\nOutput:" 以便模型根据这些信息生成输出

model_inputs：通过调用 tokenizer，
将 inputs 转换为 token 序列，并设置 padding="max_length" 和 truncation=True，
确保所有输入序列的长度一致（512个 token），并在必要时截断过长的序列。

labels：将 targets（即 output 字段）也转换为 token 序列。

model_inputs["labels"]：将生成的 token 序列作为模型的标签，
这样在训练过程中，模型可以对比预测结果与实际标签来计算损失。
"""

def get_datasets():
    dataset = load_dataset('csv', data_files={'train': 'data/train.csv', 'val': 'data/val.csv'})
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets['train'], tokenized_datasets['val']

"""

load_dataset 从 csv 文件中加载数据集，这个函数使得 datasets 库中加载 csv 格式的文件
并创建一个包含 train 和 val 数据集的字典。

data_files={'train': 'data/train.csv', 'val': 'data/val.csv'}：
指定了数据文件的位置。train.csv 是训练集，val.csv 是验证集。
你需要确保这些文件路径正确，并且文件格式符合要求。

dataset.map(tokenize_function, batched=True)：
将 tokenize_function 应用于数据集中的每一批数据（batched=True），
这会加速处理过程，并确保数据集中的每一条记录都被正确分词和格式化。

返回值：这个函数最终返回处理好的训练集和验证集，以供后续的模型训练使用。

"""

"""

为了确保这个代码能够正确运行，train.csv 和 val.csv 文件应该有三个字段（列）：

instruction：用于指定模型应该执行的任务，如 "Translate to French"。
input：输入数据，比如 "I love programming"。
output：期望的输出，比如 "J'aime programmer"。

"""