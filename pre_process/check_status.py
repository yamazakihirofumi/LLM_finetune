

from openai import OpenAI
client = OpenAI(
    api_key="*", 
    base_url="https://api.siliconflow.cn/v1"
)

batch = client.batches.retrieve("batch_qhbzyaalvx")
print(batch)