from dotenv import load_dotenv
load_dotenv()
import pickle
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# model load
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize
client = PersistentClient(
    path="./resource",
)

if client.list_collections():
    collection = client.get_collection(name="smart_store_faq")
else:
    # pkl file load
    with open('./resource/final_result.pkl', 'rb') as file:
        smart_store_faq = pickle.load(file)
    
    # create collection
    collection = client.create_collection(name="smart_store_faq")

    # Embed the loaded data
    for idx, (question, answer) in enumerate(smart_store_faq.items()):
        doc_id = f"doc_{idx}"  # 고유 ID 생성
        if doc_id not in collection.get()['ids']:
            print(f"add count_{idx + 1}")
            embedding = model.encode(question).tolist()  # 리스트로 변환하여 Chroma에 추가
            collection.add(
                ids=doc_id,
                documents=question,  # 질문을 문서로 사용
                metadatas={"answer":answer},  # 답변을 메타데이터로 사용
                embeddings=embedding  # 임베딩 추가
            )

# 사용자의 질문에 대해 가장 관련성 높은 답변 찾기
def find_best_answer(user_question, model, collection):
    question_embedding = model.encode(user_question)
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=1
    )
    
    return results['metadatas'][0][0]['answer']

# 사용자 질문 입력 받기 및 답변 출력
user_question = input("Please enter your question: ")  # 사용자 질문 입력 받기
best_answer = find_best_answer(user_question, model, collection)  # 가장 관련성 높은 답변 찾기
print(best_answer)