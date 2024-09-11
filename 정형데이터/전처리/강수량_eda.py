# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:35:01 2024

@author: DW
"""
import pandas as pd
import geopandas as gpd
import folium
import requests
import json
import numpy as np
from folium import Choropleth
import branca.colormap as cm


############
# open API로 받아오기
# API_key = 'OA-1120'
# service = 'SearchFAQOfGUListService' 
# gu_url = f'http://openapi.seoul.go.kr:8088/{API_key}/json/{service}/1/5'
# gu_list = requests.get(gu_url).json()
# print(gu_list)

gu_list = {'SearchFAQOfGUListService': {'list_total_count': 25, 'RESULT': {'CODE': 'INFO-000', 'MESSAGE': '정상 처리되었습니다'}, 'row': [{'CODE': '301', 'CD_DESC': '중구'}]}}
# gu_list = {'SearchFAQOfGUListService': {'list_total_count': 25, 'RESULT': {'CODE': 'INFO-000', 'MESSAGE': '정상 처리되었습니다'}, 'row': [{'CODE': '300', 'CD_DESC': '종로구'}, {'CODE': '301', 'CD_DESC': '중구'}, {'CODE': '302', 'CD_DESC': '용산구'}, {'CODE': '303', 'CD_DESC': '성동구'}, {'CODE': '304', 'CD_DESC': '광진구'}, {'CODE': '305', 'CD_DESC': '동대문구'}, {'CODE': '306', 'CD_DESC': '중랑구'}, {'CODE': '307', 'CD_DESC': '성북구'}, {'CODE': '308', 'CD_DESC': '강북구'}, {'CODE': '309', 'CD_DESC': '도봉구'}, {'CODE': '310', 'CD_DESC': '노원구'}, {'CODE': '311', 'CD_DESC': '은평구'}, {'CODE': '312', 'CD_DESC': '서대문구'}, {'CODE': '313', 'CD_DESC': '마포구'}, {'CODE': '314', 'CD_DESC': '양천구'}, {'CODE': '315', 'CD_DESC': '강서구'}, {'CODE': '316', 'CD_DESC': '구로구'}, {'CODE': '317', 'CD_DESC': '금천구'}, {'CODE': '318', 'CD_DESC': '영등포구'}, {'CODE': '319', 'CD_DESC': '동작구'}, {'CODE': '320', 'CD_DESC': '관악구'}, {'CODE': '321', 'CD_DESC': '서초구'}, {'CODE': '322', 'CD_DESC': '강남구'}, {'CODE': '323', 'CD_DESC': '송파구'}, {'CODE': '324', 'CD_DESC': '강동구'}]}}
gu_list = {'SearchFAQOfGUListService': {'list_total_count': 25, 'RESULT': {'CODE': 'INFO-000', 'MESSAGE': '정상 처리되었습니다'}, 'row': [{'CODE': '300', 'CD_DESC': '종로구'}, {'CODE': '301', 'CD_DESC': '중구'}, {'CODE': '302', 'CD_DESC': '용산구'}, {'CODE': '303', 'CD_DESC': '성동구'}, {'CODE': '304', 'CD_DESC': '광진구'}, {'CODE': '305', 'CD_DESC': '동대문구'}, {'CODE': '306', 'CD_DESC': '중랑구'}, {'CODE': '307', 'CD_DESC': '성북구'}, {'CODE': '308', 'CD_DESC': '강북구'}, {'CODE': '309', 'CD_DESC': '도봉구'}, {'CODE': '310', 'CD_DESC': '노원구'}, {'CODE': '311', 'CD_DESC': '은평구'}, {'CODE': '312', 'CD_DESC': '서대문구'}, {'CODE': '313', 'CD_DESC': '마포구'}, {'CODE': '314', 'CD_DESC': '양천구'}, {'CODE': '315', 'CD_DESC': '강서구'}, {'CODE': '316', 'CD_DESC': '구로구'}, {'CODE': '317', 'CD_DESC': '금천구'}, {'CODE': '318', 'CD_DESC': '영등포구'}, {'CODE': '319', 'CD_DESC': '동작구'}, {'CODE': '320', 'CD_DESC': '관악구'}, {'CODE': '321', 'CD_DESC': '서초구'}, {'CODE': '322', 'CD_DESC': '강남구'}, {'CODE': '323', 'CD_DESC': '송파구'}, {'CODE': '324', 'CD_DESC': '강동구'}]}}




df_gu = pd.DataFrame(gu_list['SearchFAQOfGUListService']['row'])
df_gu

gu_json = []
vwolrd_key = '43CA47D5-0AEC-31E0-B596-F219E84702E5'
for gu in df_gu['CD_DESC']:
    url_vworld = f'https://api.vworld.kr/req/data?service=data&version=2.0&request=GetFeature&format=json&errorformat=json&size=10&page=1&data=LT_C_ADSIGG_INFO&attrfilter=sig_kor_nm:like:{gu}&columns=sig_cd,full_nm,sig_kor_nm,sig_eng_nm,ag_geom&geometry=true&attribute=true&key={vwolrd_key}&domain=https://localhost'
    result_dict = requests.get(url_vworld).json()
    gu_json.append(result_dict)
    
    
gu_json


# 서울시 25개 구의 경계 데이터 수집 및 합치기
features = []
for gu_data in gu_json:  # gu_json 25개 구의 API 응답 데이터 리스트
   gu_name = gu_data['response']['result']['featureCollection']['features'][0]['properties']['sig_kor_nm']
   feature = {
       "type": "Feature",
       "id": gu_name,  # 구명을 id로 추가
       "geometry": gu_data['response']['result']['featureCollection']['features'][0]['geometry'],
       "properties": {
           "name": gu_name
       }
   }
   features.append(feature)


geojson_data = {
   "type": "FeatureCollection",
   "features": features
}

# GeoJSON 파일 저장
with open('data/seoul_gu_boundaries.geojson', 'w', encoding='cp949') as f:
    json.dump(geojson_data, f, ensure_ascii=False)
    

###########################################
###########################################
###########################################


geo_data = gpd.read_file('data/seoul_gu_boundaries.geojson',encoding='cp949')

geo_data.columns = ['id','자치구','geometry']

rainfall_data_path = 'C:/mulcam/00_project/Final_Project/강수량/서울시_일강수량.csv'
rainfall_data = pd.read_csv(rainfall_data_path)

전체강수량 = rainfall_data.groupby('관측소명')['강수량'].sum().reset_index()

merged = geo_data.set_index('자치구').join(전체강수량.set_index('관측소명'))









# Folium 지도 객체 생성
m = folium.Map(
    location=[37.5651, 127], 
    zoom_start=12,
    # tiles=None
)


색범위 = np.linspace(전체강수량.강수량.min()-1000, 전체강수량.강수량.max()+1000, 100).tolist()




# Choropleth 생성
Choropleth(
    geo_data=merged.__geo_interface__,
    name='choropleth',
    data=merged,
    columns=['id', '강수량'],
    key_on='feature.properties.id',
    fill_color='YlGnBu',
    fill_opacity=0.8,
    line_opacity=0.2,
    legend_name='Total Rainfall (mm)',
    threshold_scale=색범위
).add_to(m)


for _, row in merged.iterrows():
    centroid = row.geometry.centroid
    popup_text = f"{row['id']}<br>({row['강수량']} mm)"
    folium.Marker(
        location=[centroid.y, centroid.x],
        icon=folium.DivIcon(html=f"""<div style="font-size: 10pt; color: black; white-space: nowrap; text-align: center">{popup_text}</div>""")
    ).add_to(m)

# 지도 저장 및 출력
m.save('seoul_rainfall_map.html')
m






















