from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = FastAPI()

# 加载模型和分词器
model = GPT2LMHeadModel.from_pretrained('../models/lora-gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('../models/lora-gpt2')

# 设置模型为评估模式
model.eval()


# 定义请求体的数据模型
class InputData(BaseModel):
    instruction: str
    input: str


@app.post("/predict/")
async def predict(data: InputData):
    # 构建模型输入序列
    input_sequence = f"Instruction: {data.instruction}\nInput: {data.input}\nOutput:"
    inputs = tokenizer(input_sequence, return_tensors="pt")

    # 使用模型生成输出
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=100)

    # 解码生成的输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"output": generated_text}


# 用于测试FastAPI是否启动成功
@app.get("/")
async def root():
    return {"message": "FastAPI is running!"}