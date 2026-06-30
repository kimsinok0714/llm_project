from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-not-needed" # vLLM 기본 설정상 아무 문자나 넣어도 됩니다.
)

response = client.chat.completions.create(
    model="my-lora-model", # --lora-modules에서 정한 이름 지정!
    messages=[
        {"role": "user", "content": "소극적 손해에 대해"}
    ],
    temperature=0.7, 
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)