from transformers import Trainer, TrainingArguments
from model import GPT2WithLoRA
from dataset import get_datasets

"""

Trainer 用于封装训练和评估逻辑
TrainerArguments 用于定义训练过程中的各种参数，例如批处理大小，训练次数，学习率预热
GPT2WithLoRA 这是我们从 model.py 文件中导入的自定义模型类
get_datasets 从 datasets.py 文件中导入的函数，用于加载并处理数据集

"""

# 加载数据集，一个用于训练的数据集，一个用于验证
train_dataset, val_dataset = get_datasets()


# 加载使用LoRA优化的模型
model = GPT2WithLoRA(model_name='gpt2', r=8, alpha=1.0)

"""
model_name='gpt2' 指定使用预训练的 GPT2-2 模型作为基础模型
r=8 表示 LoRA 中低秩矩阵的秩
alpha=1.0 LoRA 的缩放因子，用于控制低秩矩阵的输出
"""

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./models/lora-gpt2',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./results/logs',
)

"""

output_dir：指定模型训练完成后保存的目录。
num_train_epochs：指定训练的轮次，这里设置为 3 轮。
per_device_train_batch_size：每个设备上的训练批处理大小，这里设置为 8。
per_device_eval_batch_size：每个设备上的验证批处理大小，这里设置为 8。
warmup_steps：指定学习率预热的步数，通常在训练开始时逐步增加学习率，以避免突然的大梯度更新。
weight_decay：权重衰减率，用于防止模型过拟合。
logging_dir：指定日志文件的保存目录。

"""

# 创建Trainer实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Trainer 会自动处理训练和验证的逻辑，比如前向传播，反向椽笔，梯度更新等。
# 他还支持分布式训练和自动化的日志记录。

"""

model 传递前面初始化的模型
args 传递训练参数
train_dataset 传递训练数据集
eval_dataset 传递验证数据集

"""




# 开始训练
trainer.train()