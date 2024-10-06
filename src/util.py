import os
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
import re

# OpenAI API 키 설정
client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

def get_openai_response(user_question, best_answer):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # 또는 gpt-4
        messages=[
            {"role": "system", "content": "네이버 스마트스토어 FAQ에 대한 답변을 해주세요. 관련없는 질문일 경우 '저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다.' 라고 말해주세요."},
            {"role": "assistant", "content": [{ "type": "text", "text": f"{best_answer}" }]},
            {"role": "user", "content": f"{user_question}\n\n사용자가 궁금해 할 만한 질문 두가지도 제안해 주세요."}
        ],
        stream=True
    )
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")

def clean_text(text):
    patterns = [
        r"위 도움말이 도움이 되었나요\?\s*(별점[0-9]점\s*)+소중한 의견을 남겨주시면 보완하도록 노력하겠습니다\.\s*보내기",
        r"\s*도움말 닫기"
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, "", text)
    
    return text.strip()

def extract_category(text):
    match = re.match(r'^\s*\[(.*?)\]', text)
    if match:
        return match.group(1)
    return "None"