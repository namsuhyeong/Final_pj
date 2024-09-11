import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
from PIL import Image
import os
import base64
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import nest_asyncio
import openai
from rank_bm25 import BM25Okapi
import re

# 환경 변수 설정 (OpenAI API Key)
os.environ["OPENAI_API_KEY"] = ""

# OpenAI API 설정
openai.api_key = os.environ["OPENAI_API_KEY"]

def get_openai_embedding(text,model):
    try:
        response = openai.embeddings.create(
            model = model,
            input=text
        )
        embedding = np.array(response.data[0].embedding)
        
        # Check if the embedding is empty
        if embedding.size == 0:
            return np.zeros((1,))  # Return a zero-vector (1D) if embedding is empty
        return embedding
    
    except Exception as e:
        print(f"Error in generating embedding: {e}")
        return np.zeros((1,))  # Return a zero-vector (1D) in case of any error
    
def get_minilm_embedding(text, model_name="all-MiniLM-L6-v2"):
    try:
        model = SentenceTransformer(model_name)
        return model.encode(text)
    except Exception as e:
        print(f"Error in generating embedding: {e}")
        return np.zeros((1,))  # Return a zero-vector (1D) in case of any error


qdrant = QdrantClient(
    url='https://119f2970-ba35-42b1-9e56-5826b62fd428.europe-west3-0.gcp.cloud.qdrant.io',  # Qdrant Cloud의 URL
    api_key=""  # Qdrant API 키
)

# BM25 필터링 추가
keywords = ['침수', '피해', '홍수', '물난리', '재해', '기상']  # 필터링할 키워드 목록

def apply_bm25_filtering(docs):
    """주어진 문서 리스트에 대해 BM25 필터링을 적용"""
    tokenized_docs = [doc['content'].split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    
    tokenized_keywords = keywords
    scores = bm25.get_scores(tokenized_keywords)
    
    filtered_docs = [doc for doc, score in zip(docs, scores) if score > 0]  # 점수가 0보다 큰 문서만 필터링
    return filtered_docs

def search_qdrant(query_text, collection_name, model):
    """Qdrant에서 관련 데이터를 검색하는 함수"""
    if model == "all-MiniLM-L6-v2":   
        search_result = qdrant.search(
            collection_name=collection_name,
            query_vector= get_minilm_embedding(query_text),
            limit=10
        )
    else:
        search_result = qdrant.search(
            collection_name=collection_name,
            query_vector= get_openai_embedding(query_text,model),
            limit=10
        )
    
    
    search_results = []
    for point in search_result:
        payload = point.payload
        search_results.append({
            "title": payload['제목'],
            "link": payload['링크'],
            "date": payload['날짜'],
            "source": payload['언론사'],
            "content": payload['content']
        })
    
    # BM25 필터링 적용
    filtered_results = apply_bm25_filtering(search_results)
    
    context_text = ""
    for result in filtered_results:
        context_text += f"제목: {result['title']}\n링크: {result['link']}\n날짜: {result['date']}\n출처: {result['source']}\n내용: {result['content']}\n\n"
    
    return context_text, filtered_results

def get_openai_response(context, user_query):
    """OpenAI API를 사용하여 Qdrant 결과를 바탕으로 질문에 대한 응답을 생성"""
    client = openai.OpenAI(api_key="")
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer in Korean."},
            {"role": "user", "content": f"Context: {context}\n\nUser Query: {user_query}\n\nPlease respond in Korean and include the title and content and links of the news_bot_openai_pre_total in your answer."}
        ]
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            n=1,
            stop=None,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"응답 생성 중 오류가 발생했습니다: {e}"
    

#%%
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. 데이터 로드
file_path = './data/top_100vocab_20percent.csv'
df = pd.read_csv(file_path)

''''''''''''''''''''''''''''''
''''''''''''''''''''''''''''''

# NDCG 계산 함수들
def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def calculate_ndcg(query_embedding, article_embeddings, relevance_scores, k=10):
    # Ensure that the query_embedding is 2D and has features
    if query_embedding.size == 0 or query_embedding.shape[0] == 1:
        print("Query embedding is empty or zero vector. NDCG will be 0.")
        return 0  # Return NDCG score of 0 if query embedding is empty or zero vector
    
    query_embedding = query_embedding.reshape(1, -1)
    
    # Ensure that article_embeddings is a 2D array and has features
    if article_embeddings.size == 0 or article_embeddings.shape[1] == 0:
        print("Article embeddings are empty or malformed. NDCG will be 0.")
        return 0  # Return NDCG score of 0 if article embeddings are empty or malformed
    
    similarities = cosine_similarity(query_embedding, article_embeddings).flatten()
    ranked_indices = np.argsort(-similarities)
    ndcg_score = ndcg_at_k(relevance_scores[ranked_indices], k)
    return ndcg_score


#%% 점수 할당 방법 2가지
# 1. 정규화 함수 정의
def normalize_scores(scores):
    min_score = scores.min()
    max_score = scores.max()
    return (scores - min_score) / (max_score - min_score)

# 2. 구간별 점수 할당 함수(구간 및 점수 배치는 임의 설정)
def assign_scores_by_range(df):
    df_sorted = df.sort_values(by='TF-IDF 점수 합', ascending=False)
    
    top_10_percent = int(len(df_sorted) * 0.1) # 상위 10%
    top_30_percent = int(len(df_sorted) * 0.3) # 상위 10%~30%
    top_70_percent = int(len(df_sorted) * 0.7) # 상위 30%~70%
    
    df_sorted['구간별 점수'] = 0.0
    df_sorted.iloc[:top_10_percent, df_sorted.columns.get_loc('구간별 점수')] = 1                 # 상위  10%  -> 1점(만점)
    df_sorted.iloc[top_10_percent:top_30_percent, df_sorted.columns.get_loc('구간별 점수')] = 0.7 # 상위 ~30%  -> 0.7점
    df_sorted.iloc[top_30_percent:top_70_percent, df_sorted.columns.get_loc('구간별 점수')] = 0.3 # 상위 ~70%  -> 0.3점
    
    return df_sorted['구간별 점수'].values


# 평가 함수
def evaluate_models(collection_name,embedding_model, option='range',printall = False):
    df = pd.read_csv('./data/top_100vocab_20percent.csv')
    df['정규화된 TF-IDF 점수'] = normalize_scores(df['TF-IDF 점수 합'])
    
    if option == 'range':
        relevance_scores = assign_scores_by_range(df)
    elif option == 'normalized':
        relevance_scores = df['정규화된 TF-IDF 점수'].values
    else:
        raise ValueError("Invalid option. Choose 'range' or 'normalized'")
    ndcg_scores = []
    
    
    if embedding_model == "all-MiniLM-L6-v2":

        for query in user_input:
            # 각 쿼리에 대해 검색하고 NDCG 계산
            context, filtered_results = search_qdrant(query, collection_name, embedding_model)
            query_embedding = get_minilm_embedding(query)
            article_embeddings = np.array([get_minilm_embedding(result['content']) for result in filtered_results])
            ndcg_score = calculate_ndcg(query_embedding, article_embeddings, relevance_scores, k=10)
            ndcg_scores.append(ndcg_score)
            
            # 각 쿼리별로 결과 출력
            if printall:
                print(f"사용자 쿼리: {query}")
                for idx, result in enumerate(filtered_results[:10], start=1):
                    print(f"  답변 {idx}: {result['title']}")
                print(f"  NDCG 점수: {ndcg_score:.4f}\n")
                
    else:
        for query in user_input:
            # 각 쿼리에 대해 검색하고 NDCG 계산
            context, filtered_results = search_qdrant(query, collection_name, embedding_model)
            query_embedding = get_openai_embedding(query,embedding_model)
            article_embeddings = np.array([get_openai_embedding(result['content'], embedding_model) for result in filtered_results])
            ndcg_score = calculate_ndcg(query_embedding, article_embeddings, relevance_scores, k=10)
            ndcg_scores.append(ndcg_score)
            
            # 각 쿼리별로 결과 출력
            if printall:
                print(f"사용자 쿼리: {query}")
                for idx, result in enumerate(filtered_results[:10], start=1):
                    print(f"  답변 {idx}: {result['title']}")
                print(f"  NDCG 점수: {ndcg_score:.4f}\n")
                
    
    # 모든 쿼리에 대한 평균 NDCG 점수 계산 및 출력
    avg_ndcg_score = np.mean(ndcg_scores)
    # print(f"총 NDCG 평균 점수: {avg_ndcg_score:.4f}\n")
    return avg_ndcg_score

# 사용자 질문 예시
user_input = [
    "강남역 문제점",
    "서울 집중호우 피해",
    "2022년 8월 서울 침수 피해",
    "강남역 침수",
    "한강 범람",
    "서울 침수 피해지역",
    "신림 침수 피해",
    "대림 침수 지역",
    "사당 침수 지역",
    "신길동 침수",
    "대림 침수",
    "사당 침수",
    "신월동 침수",
    "화곡동 침수",
    "방배동 침수",
    "봉천동 침수",
    "개봉동 침수",
    "신길동 침수",
    "상도동 침수",
    "비가 많이 올때 서울에서 침수가 자주 되는곳은?",
    "지금 비가 많이 오고있는데 비가 많이 왔을 때 일어난 사건이 있나요?",
    "강남구 침수에 대해 알려주세요.",
    "침수 상습 지역은?",
    "비가 많이 오면 주로 침수되는 지역은 어디인가요?",
    "100mm 이상의 강우 시 침수 피해 예상 지역을 알려주세요.",
    "서울시에서 침수 피해가 자주 발생하는 구역은 어디인가요?",
    "침수 피해를 예방하기 위한 주요 대책은 무엇인가요?",
    "침수 피해가 발생했을 때, 어떤 조치를 취해야 하나요?",
    "청담동의 침수 위험도는 어떻게 되나요?",
    "송도에 비가 많이오면 어떻게 되나요?",
    "집에 침수가 발생했을 때, 보상 받을 수 있는 방법은 무엇인가요?",
    "기상청의 집중호우 예보를 어떻게 활용해야 하나요?",
    "강수량이 일정 수준을 초과하면 자동으로 경고가 발생하나요?",
    "강남구와 송파구의 침수 위험 지역 비교는 어떻게 되나요?",
    "침수 피해에 대한 보험 상품이 있나요?",
    "역삼동과 청담동의 침수 위험도는 어떻게 되나요?",
    "서울의 하수도 시스템은 얼마나 효율적인가요?",
    "비 오는 날 저지대에 사는 경우, 특별히 주의할 점이 무엇인가요?",
    "침수 피해 발생 후, 피해 복구를 위한 정부 지원 프로그램은 무엇이 있나요?",
    "서울시가 최근에 개선한 침수 대책은 어떤 것들이 있나요?",
    "침수 피해를 줄이기 위한 가정에서 할 수 있는 준비 작업은 무엇이 있나요?",
    "침수 피해 지역에 거주하고 있는데, 이사 고려 시 유의사항은 무엇인가요?",
    "침수 피해를 모니터링할 수 있는 앱이나 웹사이트는 무엇이 있나요?",
    "침수 피해에 대한 책임은 누구에게 있나요?",
]



#%% 모델 성능 평가

#사용된 모델 dict
collection_names = {
    "news_bot_openai_small3_500_26" : "text-embedding-3-small",
    # "news_bot_openai_ada2_500_26" : "text-embedding-ada-002", 
    # "news_bot_openai_large3_500_26" : "text-embedding-3-large",
    # "news_bot_hugging_500_26" : "all-MiniLM-L6-v2",
    }

for collection_name, embedding_model in collection_names.items():
    print(f"Evaluating collection: {collection_name}")
    
    # 구간별 점수 할당 방식으로 평가
    print("### 구간별 점수 할당 방식 ###")
    ndcg_range_score = evaluate_models(collection_name,embedding_model, option='range')
    print(f"NDCG (Range) for {collection_name}: {ndcg_range_score:.4f}\n")
    
    # 정규화된 점수 사용 방식으로 평가
    print("### 정규화된 점수 사용 방식 ###")
    ndcg_normalized_score = evaluate_models(collection_name,embedding_model, option='normalized')
    print(f"NDCG (Normalized) for {collection_name}: {ndcg_normalized_score:.4f}\n")
    
    
#%% 모델 성능 평가
# user_input = input("쿼리입력하세요 : ")
# user_input = [
#     "서울 침수",
#     "서울 홍수",
#     "서울 물난리",
#     "서울 집중호우 피해",
    
    
#     "2024년 8월 서울 침수 피해",
#     "서울 강남역 침수",
#     "한강 범람으로 인한 서울 침수",
#     "서울 대형 폭우 피해",
    
    
#     # "서울 침수 원인 기후변화",
#     # "서울 침수 경제적 피해",
#     # "서울 폭우 후 교통 마비",
#     # "침수로 인한 서울 시민 대피",
    
    
#     # "서울, 또다시 침수…시민들 불안",
#     # "서울 대홍수, 복구 작업 난항",
#     # "서울 집중호우, 도심 침수 피해 속출",
#     # "서울 폭우, 하수도 문제로 도심 침수",
#     ]


# for user_chat in user_input:
#     context_text, search_results = search_qdrant(user_chat)
#     answer = get_openai_response(context_text, user_chat)
    
#     print(answer)
