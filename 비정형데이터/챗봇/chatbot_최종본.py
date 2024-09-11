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
import time
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# 페이지 설정 (전체 너비 사용)
st.set_page_config(layout="wide")

# asyncio 오류 방지
nest_asyncio.apply()

# 환경 변수 설정 (OpenAI API Key)
os.environ["OPENAI_API_KEY"] = ""

# OpenAI API 설정
openai.api_key = os.environ["OPENAI_API_KEY"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 동-구 매핑 데이터 예시
df = pd.read_csv('dong_to_gu.csv')
dong_to_gu = dict(zip(df['법정동'], df['자치구']))
gu_list = df['자치구'].unique()

# Qdrant 클라이언트 및 SentenceTransformer 설정
def get_openai_embedding(text):
    response = openai.Embedding.create(
        model="text-embedding-3-small",  # OpenAI의 모델 예시
        input=text
    )
    return response['data'][0]['embedding']

qdrant = QdrantClient(
    url='https://119f2970-ba35-42b1-9e56-5826b62fd428.europe-west3-0.gcp.cloud.qdrant.io',  # Qdrant Cloud의 URL
    api_key=""  # Qdrant API 키
)

# BM25 필터링 추가
keywords = ['침수', '피해', '홍수', '물난리', '재해', '기상']  # 필터링할 키워드 목록

def apply_ensemble_retriever(docs, query_vector):
    """BM25 점수와 OpenAI 임베딩 유사성을 결합한 Ensemble Retriever 적용"""
    
    # BM25 필터링
    tokenized_docs = [doc['content'].split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_keywords = keywords
    bm25_scores = bm25.get_scores(tokenized_keywords)
    
    # 이미 Qdrant에서 가져온 임베딩을 사용하여 코사인 유사성 계산
    doc_embeddings = np.array([doc['embedding'] for doc in docs])
    cosine_scores = cosine_similarity([query_vector], doc_embeddings)[0]
    
    # 정규화
    norm_bm25 = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores))
    norm_cosine = (cosine_scores - np.min(cosine_scores)) / (np.max(cosine_scores) - np.min(cosine_scores))
    
    # 가중치 설정
    bm25_weight = 0.1
    cosine_weight = 0.9
    
    # 가중 합 계산
    combined_scores = bm25_weight * norm_bm25 + cosine_weight * norm_cosine
    
    # 점수에 따라 문서 정렬
    ranked_docs = [doc for _, doc in sorted(zip(combined_scores, docs), key=lambda x: x[0], reverse=True)]
    
    return ranked_docs

def search_qdrant(query_text):
    """Qdrant에서 관련 데이터를 검색하는 함수"""
    query_vector = get_openai_embedding(query_text)
    search_result = qdrant.search(
        collection_name="open_ai_small3_0903_final",
        query_vector=query_vector,
        limit=5,
        with_vectors=True
    ) 
    
    search_results = []
    for point in search_result:
        payload = point.payload
        search_results.append({
            "title": payload['제목'],
            "link": payload['링크'],
            "date": payload['날짜'],
            "source": payload['언론사'],
            "content": payload['content'],
            "embedding": point.vector  # 임베딩 벡터를 함께 반환
        })
    
    # Ensemble Retriever 적용
    ranked_results = apply_ensemble_retriever(search_results, query_vector)
    
    context_text = ""
    for result in ranked_results:
        context_text += f"제목: {result['title']}\n링크: {result['link']}\n날짜: {result['date']}\n출처: {result['source']}\n내용: {result['content']}\n\n"
    
    return context_text, ranked_results

def summarize_article(content):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes Korean texts into a single sentence in Korean."},
                {"role": "user", "content": f"Please summarize this in Korean into one sentence: {content}"}
            ],
            max_tokens=150,
            temperature=0.5
        )
        summary = response.choices[0].message['content'].strip()
        return summary
    except Exception as e:
        return f"Error while summarizing: {e}"


# 'None'이 나오는 문제 해결 및 제목/내용을 한글로 변경
def get_openai_response_1(context_text, user_query, dong_name=None, gu_name=None, article_list=None):
    """dong_name과 gu_name의 None 처리 및 출력 형식 수정"""
    
    # 만약 동 이름이 없고 자치구 이름만 있을 경우
    if dong_name is None:
        location_name = gu_name  # 자치구 이름만 표시
    else:
        location_name = f"{dong_name}({gu_name})"  # 동과 자치구 이름 모두 표시

    # 침수 관련 정보를 사용자 쿼리에 맞춰 제공
    if "침수" in user_query:
        specific_info = (
            f"  - 물이 집 안으로 들어올 수 있는 곳을 차단하세요.\n"
            f"  - 전자기기 및 귀중품을 안전한 곳으로 옮기세요.\n"
            f"  - 하천이나 도로가 범람할 위험이 있으니 피하세요.\n"
            f"  - 안전한 장소로 대피하고, 비상 연락처를 확인하세요.\n"
        )
    elif "지하건물" in user_query or "지하 피해" in user_query:
        specific_info = (
            f"{location_name}의 지하 건물 피해를 예방하기 위해 다음을 참고하세요:\n"
            f"  - 지하실의 배수펌프가 정상 작동하는지 확인하세요.\n"
            f"  - 중요한 물품은 지하에 두지 말고 높은 곳으로 이동시키세요.\n"
            f"  - 침수 예보 시 지하 공간 사용을 자제하세요.\n"
        )
    else:
        specific_info = (
            f"{location_name}에 대한 추가 정보가 필요할 경우, 관련 기사를 확인하거나 구청에 문의하세요.\n"
        )

    try:
        # OpenAI 모델을 사용하여 응답 생성
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer in Korean. "
                    "Please focus on the specific user query and provide detailed, relevant information. "
                    "Ensure that any articles are summarized briefly with a link."
                    
                    )
            },
            {
                "role": "user",
                "content": user_query
            }
        ]

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1500,
            n=1,
            stop=None,
            temperature=0.7
        )
        openai_result = response.choices[0].message['content'].strip()

        # 응답을 형식에 맞게 정리하여, 기사가 위로 올라오고 추가 정보가 아래로 배치되게 수정
        return format_response(location_name, gu_name, article_list, openai_result, specific_info)

    except Exception as e:
        return f"질문 처리 중 오류가 발생했습니다: {e}"

def get_openai_response_2(context_text, user_query, article_list):
    """dong_name, gu_name이 없는 경우의 질문 처리"""
    
    system_prompt = (
        "You are a helpful assistant that answers questions about flood risks and regional information in Korea. "
        "You provide clear, concise, and relevant answers based on the given information. "
        "If the response includes a specific location (within Seoul), make sure to mention that location and related data, such as flood risk, green areas, or other relevant information. "
        "Always respond in Korean. Format the response with Markdown, use bullet points for lists, and ensure that if any articles are referenced, they are formatted as links with a brief summary."
    )
    
    user_prompt = (
        f"{user_query}\n\n"
        f"뉴스 기사에서 발견된 관련 정보:\n"
        f"{context_text}\n\n"
        f"#### 관련 뉴스 기사 목록:\n{article_list}\n\n"
        "해당 기사와 관련된 침수 지역 정보가 있으면 제공해주세요."
    )
    
    try:
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1500,
            n=1,
            stop=None,
            temperature=0.7
        )
        openai_result = response.choices[0].message['content'].strip()

        # 응답을 형식에 맞게 정리하여 반환
        return openai_result

    except Exception as e:
        return f"질문 처리 중 오류가 발생했습니다: {e}"
    

def format_articles(articles):
    """Qdrant에서 불러온 기사를 한글로 정리, 제목과 요약 내용을 각각 별도의 줄에 출력"""
    formatted_articles = ""
    for article in articles:
        summary = summarize_article(article['content'])  # 기사 요약
        formatted_articles += (
            f"<p style='font-size: 18px; font-weight: bold;'>"
            f"<a href='{article['link']}' style='color: #007BFF; text-decoration: underline;'>"
            f"📰 {article['title']}</a></p>"  # 제목을 하이퍼링크처럼 파란색과 밑줄로 스타일링
            f"<blockquote style='margin-left: 20px; font-style: italic; color: #555;'>"
            f"{summary}</blockquote>"  # 본문 요약을 인용 블록으로 들여쓰기
            "<hr style='border-top: 1px solid #bbb;' />\n\n"  # 구분선을 얇게
        )
    return formatted_articles






def format_response(location_name, gu_name, article_list, context_text, specific_info):
    """AI의 응답을 형식화하는 함수"""
    
    # 이미지 경로 또는 base64 이미지
    image_base64 = load_image_as_base64('로고2.png')  # 이미지를 base64로 인코딩

    return (
        f"#### {location_name}\n"
        "<hr style='border: 2px solid black;'>\n\n"  # 두꺼운 구분선
        f"- 해당 지역에서 침수와 관련된 여러 예측 결과가 있습니다.\n"
        f"- 왼쪽 지도는 {gu_name}의 침수 예측 지도입니다.\n"
        f"- 지도에서 <img src='data:image/png;base64,{image_base64}' alt='이미지' width='20'> 버튼을 클릭하시면 더 많은 정보를 확인하실 수 있습니다.\n"  # 이미지 삽입
        f"- 사이드바에서는 {gu_name}의 주요 정보를 확인하실 수 있습니다.\n"
        "<hr style='border: 2px solid black;'>\n\n"  # 두꺼운 구분선
        f"**침수 시 아래와 같은 조치를 취하세요:**\n\n"  # 진하게 표시
        f"{specific_info}\n\n"  # 사용자 쿼리와 관련된 정보 제공
        "<hr style='border: 2px solid black;'>\n\n"  # 두꺼운 구분선
        f"**{location_name} 정보**\n\n"  # 진하게 표시
        f"{context_text}\n\n"  # AI의 추가적인 답변도 기사보다 먼저 출력
        "<hr style='border: 2px solid black;'>\n\n"  # 두꺼운 구분선
        f"#### {location_name}의 과거 뉴스 기사로는 다음이 있습니다\n"
        f"&nbsp;&nbsp;&nbsp;&nbsp;"  # HTML 공백 추가
        f"{article_list}\n\n"
    )




    
def display_logo(image_path):
    """로고 이미지를 가운데 정렬하여 표시하는 함수"""
    try:
        image = Image.open(image_path)
        st.image(image, width=300)  # 이미지 크기 조정
    except Exception as e:
        st.error(f"이미지를 로드하는 중 오류가 발생했습니다: {e}")

def initialize_session_state():
    """세션 상태를 초기화하는 함수"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'start' not in st.session_state:
        st.session_state.start = False
    if 'input_text' not in st.session_state:
        st.session_state.input_text = ''

def apply_css_style():
    css = """
    <style>
    .st-emotion-cache-1vt4y43.ef3psqc13 {
        background-color: #007bff;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: block;
        width: fit-content;
        margin: 0 auto;
        font-size: 16px;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    
    .custom-button:hover {
        background-color: #0056b3;
    }
    
    .custom-button:active {
        background-color: #003d7a;
    }
 
    
    /* 챗봇 창에 스크롤 추가 */
    .st-chatbot {
        max-height: 400px; /* 챗봇 창의 최대 높이 설정 */
        overflow-y: auto; /* 수직 스크롤 추가 */
        background-color: #f8f9fa; /* 배경색 조정 */
        padding: 10px; /* 패딩 추가 */
        border-radius: 5px; /* 둥근 모서리 추가 */
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1); /* 그림자 추가 */
    }
    
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# 배경 이미지 설정 함수
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
    
# 사이드 바 배경화면 설정 함수
def add_sidebar_bg(image_path):
    bin_str = get_base64(image_path)
    css = f"""
    <style>
    [data-testid="stSidebar"] {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def load_image_as_base64(image_path):
    """이미지 파일을 base64로 인코딩하여 반환하는 함수"""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

def home_page():
    """홈페이지를 보여주는 함수"""
    # 이미지 파일을 base64로 인코딩
    image_base64 = load_image_as_base64('침수_로고_3.png')
    wave_image_base64 = load_image_as_base64('파도.png')  # 파도 이미지 파일을 base64로 인코딩

    # CSS를 사용하여 이미지 가운데 정렬 및 버튼 스타일 정의
    st.markdown(f"""
        <style>
        html, body {{
            margin: 0;
            padding: 0;
            overflow-x: hidden;  /* 가로 스크롤바를 숨김 */
        }}
        .centered-image {{
            display: block;
            margin-left: auto;
            margin-right: auto;
        }}
        .wave-footer {{
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;  /* 너비를 100%로 조정 */
            height: 400px;  /* 높이를 500px로 설정 */
            overflow: hidden;  /* 이미지가 넘칠 경우 숨김 처리 */
        }}
        </style>
        <img class="centered-image" src="data:image/png;base64,{image_base64}" width="500">
    """, unsafe_allow_html=True)
    
    apply_css_style()  # CSS 스타일 적용
    
    # 버튼 클릭 시 세션 상태 업데이트
    if st.button("시작하기"):
        st.session_state.start = True
        st.rerun()  # 페이지 새로고침

    # 페이지의 맨 아래에 파도 이미지 삽입
    st.markdown(f'<img class="wave-footer" src="data:image/png;base64,{wave_image_base64}">', unsafe_allow_html=True)


# 기존의 main_page 함수에서 else 부분 수정
def main_page():
    apply_css_style()
    initialize_session_state()  # 세션 상태 초기화

    # 침수 로고 이미지 불러오기
    image_base64 = load_image_as_base64('침수_로고.png')
    
    # 침수 로고와 AI 로고를 나란히 배치
    st.markdown(f"""
        <style>
        .centered-container {{
            display: flex;
            justify-content: center;
            align-items: center;
        }}
        .right-container {{
            display: flex;
            justify-content: flex-end;
            margin-right: 110px; /* 오른쪽 여백 조절 */
        }}
        </style>
        <div class="centered-container">
            <img src="data:image/png;base64,{image_base64}" width="400">
        </div>
        <div class="right-container">
            <img src="data:image/png;base64,{load_image_as_base64('AI_로고.png')}" width="150">
        </div>
    """, unsafe_allow_html=True)

    # 두 개의 열로 설정, 중앙에 더 넓은 공간
    col2, col3 = st.columns([8, 2.22])

    # 기본 분석 내용 파일 설정 (서울 전체 정보)
    default_info_html = "서울_info.html"
    # 세션 상태에서 정보 파일을 가져오거나 기본 정보 파일을 설정
    info_html = st.session_state.get("info_html", default_info_html)
    
    image_path = "info_로고.png"
    
    add_sidebar_bg('rain.png')
       
    # 정보 파일을 로드하는 함수
    def load_info(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        st.components.v1.html(html_content, height=1350, scrolling=False)

    # 기본 지도 파일 설정 (서울 전체 지도)
    default_map_html = "PCA1_Predicted_Depth_Map_침수예상지도.html"

    # 세션 상태에서 지도 파일을 가져오거나 기본 지도 파일을 설정
    map_html = st.session_state.get("map_html", default_map_html)

    # 지도 파일을 로드하는 함수
    def load_map(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        st.components.v1.html(html_content, height=800, scrolling=True)

    # 챗봇 UI와 로직
    with col3:
        st.markdown(
            """
            <style>
            div[data-testid="stExpanderDetails"] {
                height: 740px;
                overflow: auto;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        with st.expander("대화하기", expanded=True):
            user_input = st.text_input("현재 거주 중이신 동(법정동명) 혹은 자치구 이름을 포함하거나 침수와 관련된 질문을 입력하세요:", key="input", placeholder="여기에 질문을 입력하세요...")

            with st.container():  # 추가
                if user_input:
                    st.session_state.messages = []  # 이전 메시지 초기화
                    st.session_state.input_text = user_input  # 입력값 저장

                    if user_input:
                        dong_name = None
                        gu_name = None
                        
                    # 동 이름 확인
                    for dong in dong_to_gu.keys():
                        if dong in user_input:
                            dong_name = dong
                            gu_name = dong_to_gu[dong_name]
                            break
                    # 동 이름이 없을 경우 구 이름 확인
                    if not dong_name:
                        for gu in gu_list:  # gu_list는 자치구 이름 목록입니다.
                            if gu in user_input:
                                gu_name = gu
                                break
                    # 만약 동 이름이 없고 구 이름만 포함된 경우
                    if not dong_name and gu_name is None:
                        for gu in gu_list:
                            if gu[:-1] in user_input:  # "송파"와 같이 "구"가 생략된 경우도 포함
                                gu_name = gu
                                break
                    if gu_name:
                        with st.spinner(f"{gu_name}을(를) 기준으로 검색 중입니다."):
                            
                            user_query = f"{gu_name} 침수"

                            # 지도 파일 경로 업데이트
                            map_file_path = f"PCA1_Predicted_Depth_Map_침수예상지도_{gu_name}.html"
                            info_file_path = f"{gu_name}_info.html"
                            
                            if os.path.exists(map_file_path):
                                st.session_state.map_html = map_file_path
                            else:
                                st.session_state.map_html = "PCA1_Predicted_Depth_Map_침수예상지도.html"

                            if os.path.exists(info_file_path):
                                st.session_state.info_html = info_file_path
                            else:
                                st.session_state.info_html = "서울_info.html"

                            # Qdrant에서 기사 검색
                            context_text, search_results = search_qdrant(user_query)
                            article_list = format_articles(search_results)
                            
                            # 침수 사건 횟수 예시로 계산
                            event_count = len(search_results)

                            # OpenAI API를 사용하여 응답 생성
                            ai_response = get_openai_response_1(context_text, user_query, dong_name, gu_name, article_list)

                            # 스타일링된 응답 출력
                            st.markdown(ai_response, unsafe_allow_html=True)

                            # 지도 및 정보 섹션
                            st.write("**왼쪽 지도는 해당 지역의 침수 예측 지도입니다.**")
                            
                    else:
                        # 동, 구 이름이 포함되지 않은 경우
                        with st.spinner("침수 관련 정보를 검색 중입니다."):
                            
                            user_query = user_input

                            # Qdrant에서 기사 검색
                            context_text, search_results = search_qdrant(user_query)
                            article_list = format_articles(search_results)

                            # OpenAI API를 사용하여 응답 생성
                            ai_response = get_openai_response_2(context_text, user_query, article_list)

                            # 스타일링된 응답 출력
                            st.markdown(ai_response, unsafe_allow_html=True)

                        st.write("**서울 전체 침수 예측 지도는 왼쪽에 있습니다.**")
    
        st.markdown("""
        <p style='font-size: 10px;'>※현재 본 서비스의 chatbot은 서울시 침수 관련 뉴스 기사를 기반으로 대답을 해주고 있습니다. 침수와 관련없는 질문 시 엉뚱한 답변이 나올 수 있습니다.※</p>
        """, unsafe_allow_html=True)


    # 지도 파일을 로드 (초기 로드 시 및 자치구 변경 시 동일하게 사용)
    with col2:
        load_map(st.session_state.get("map_html", default_map_html))

    # 사이드바에서 info_html을 로드
    with st.sidebar:
        st.image(image_path, width=320)
        load_info(st.session_state.get("info_html", default_info_html))
        
def main():
    initialize_session_state()
    if st.session_state.start:
        main_page()
    else:
        home_page()

if __name__ == "__main__":
    main()
