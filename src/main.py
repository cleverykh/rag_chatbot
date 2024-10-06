from dotenv import load_dotenv
load_dotenv()
import pickle
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from util import get_openai_response, clean_text, extract_category
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# model load
# model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# model = SentenceTransformer('sentence-transformers/bert-large-nli-mean-tokens')
# model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


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
        doc_id = f"doc_{idx}"  # ID 생성
        if doc_id not in collection.get()['ids']:
            print(f"add count_{idx + 1}")
            print(f"질문: {question}\n답변: {clean_text(answer)}")
            print(f"category:{extract_category(question)}\n\n\n\n")
            embedding = model.encode(question).tolist()  # 리스트로 변환하여 Chroma에 추가
            collection.add(
                ids=doc_id,
                documents=[f"질문: {question}\n답변: {clean_text(answer)}"],  # 질문과 답변을 documents로 사용
                metadatas=[{"category":extract_category(question)}],  # 카테고리를 metadatas로 사용
                embeddings=embedding
            )

# 사용자의 질문과 가장 관련성 높은 답변 찾기
def find_best_answer(user_question, model, collection):
    question_embedding = model.encode(user_question)
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )
    return results['documents'][0]

# 사용자 질문 및 답변 출력
def chat():
    print("챗봇에 질문하세요. 종료하려면 'exit' 또는 '종료' 입력하세요.")
    while True:
        user_question = input("사용자 질문: ")
        if user_question.lower() in ['exit', 'quit', '종료', '끝']:
            print("챗봇을 종료합니다.")
            break

        best_answer = find_best_answer(user_question, model, collection)
        print("챗봇의 답변:")
        get_openai_response(user_question, best_answer)
        print("\n\n")

chat()