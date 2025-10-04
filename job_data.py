import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
from itertools import combinations
import networkx as nx
import warnings

#직무데이터분석

# 사소한 경고 메시지는 무시
warnings.filterwarnings('ignore')

# ---  시각화 위한 한글 폰트 설정 ---
# 사용자 요청 및 오류 방지를 위해 폰트 설정 부분을 주석 처리합니다.
# 그래프의 한글이 깨져 보일 수 있으나, 분석 자체에는 영향을 주지 않습니다.
# try:
#     font_path = None
#     for font in fm.fontManager.ttflist:
#         if 'Nanum' in font.name or 'Malgun' in font.name:
#             font_path = font.get_file()
#             break
#     if font_path:
#         font_prop = fm.FontProperties(fname=font_path)
#         plt.rcParams['font.family'] = font_prop.get_name()
#         print("✅ 한글 폰트가 설정되었습니다.")
#     else:
#         print("⚠️ 한글 폰트를 찾을 수 없습니다. 시각화 시 글자가 깨질 수 있습니다.")
#     plt.rcParams['axes.unicode_minus'] = False
# except Exception as e:
#     print(f"폰트 설정 중 오류 발생: {e}")

# ---  데이터 불러오기 및 전처리 ---
file_path = 'job_code.csv'
try:
    # ⭐️ 핵심 수정 사항: 한글 CSV 파일이 깨질 때 'cp949' 인코딩을 지정하여 해결합니다.
    df = pd.read_csv(file_path, encoding='cp949')
    print(f"\n '{file_path}' 파일을 성공적으로 불러왔습니다.")

    # 'keywords_bert'가 비어있는 행은 분석할 수 없으므로 제거합니다
    df.dropna(subset=['keywords_bert'], inplace=True)
    # 처리를 쉽게 하기 위해 키워드를 리스트 형태로 변환합니다
    df['keywords_list'] = df['keywords_bert'].str.split(', ')

    print("\n---------- [ 데이터 정보 ] ----------")
    df.info()
    print("\n---------- [ 데이터 샘플 ] ----------")
    print(df.head())

except FileNotFoundError:
    print(f" 오류: '{file_path}' 파일을 찾을 수 없습니다. 파일 위치를 확인해주세요.")
    df = pd.DataFrame()  # 스크립트가 중단되지 않도록 빈 데이터프레임을 생성합니다
except UnicodeDecodeError:
    print(f" 오류: '{file_path}' 파일의 인코딩 문제일 수 있습니다. 'utf-8'로 다시 시도합니다.")
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"\n '{file_path}' 파일을 'utf-8'로 다시 시도하여 성공적으로 불러왔습니다.")
    except Exception as e:
        print(f" 오류: 파일 읽기에 최종적으로 실패했습니다. 원인: {e}")
        df = pd.DataFrame()


# --- 메인 분석 블록 (데이터가 성공적으로 로드된 경우에만 실행) ---
if not df.empty:

    # ==========================================================================
    #  분석 1: 전체 키워드 및 직무명 분석
    # ==========================================================================
    print("\n" + "=" * 50)
    print(" 분석 1: 전체 키워드 및 직무명 분석")
    print("=" * 50)
    try:
        all_keywords = df['keywords_list'].explode().tolist()
        top_20_keywords = Counter(all_keywords).most_common(20)
        print("\n 가장 빈번하게 등장하는 키워드 (상위 20개):")
        print(pd.DataFrame(top_20_keywords, columns=['키워드', '빈도']))

        title_words = df['Title_ko'].str.split().explode().tolist()
        top_10_title_words = Counter(title_words).most_common(10)
        print("\n🏷 직무명에 가장 많이 사용된 단어 (상위 10개):")
        print(pd.DataFrame(top_10_title_words, columns=['직무명 단어', '빈도']))
    except Exception as e:
        print(f"분석 1 오류: {e}")

    # ==========================================================================
    #  분석 2: 직무 특성 태깅
    # ==========================================================================
    print("\n" + "=" * 50)
    print(" 분석 2: 직무 특성 태깅")
    print("=" * 50)
    try:
        characteristic_dict = {
            '#저강도_신체활동': ['상담', '안내', '사무', '말벗', '실내 근무', '관람'],
            '#고강도_신체활동': ['순찰', '배달', '청소', '육체 활동', '미화', '조리', '정리'],
            '#실외근무': ['순찰', '외부 활동', '교통 안내', '환경미화'],
            '#실내근무': ['실내 근무', '사무', '상담', '돌봄', '매장'],
            '#사회활동성': ['상담', '안내', '말벗', '돌봄', '교육'],
            '#독립적업무': ['순찰', '정리', '단순 포장', '보안']
        }

        def create_tags(keywords):
            tags = []
            if not isinstance(keywords, str): return ''
            for tag, keyword_list in characteristic_dict.items():
                if any(keyword in keywords for keyword in keyword_list):
                    tags.append(tag)
            return ', '.join(tags)

        df['job_tags'] = df['keywords_bert'].apply(create_tags)
        print("\n✨ 직무 태깅 결과 (샘플):")
        print(df[['Title_ko', 'job_tags']].head())
    except Exception as e:
        print(f"분석 2 오류: {e}")

    # ==========================================================================
    #  분석 3: 토픽 모델링을 통한 직무 군집화
    # ==========================================================================
    print("\n" + "=" * 50)
    print(" 분석 3: 토픽 모델링을 통한 직무 군집화")
    print("=" * 50)
    try:
        vectorizer = CountVectorizer()
        dtm = vectorizer.fit_transform(df['keywords_bert'])
        num_topics = 5
        lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda.fit(dtm)

        print(f"\n🔍 발견된 직무 군집 (토픽):")
        feature_names = vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(lda.components_):
            top_keywords = " ".join([feature_names[i] for i in topic.argsort()[:-8:-1]])
            print(f"📂 군집 #{topic_idx + 1}: {top_keywords}")
    except Exception as e:
        print(f"분석 3 오류: {e}")

    # ==========================================================================
    #  4: 작업 군집에 대한 네트워크 분석
    # ==========================================================================
    print("\n" + "=" * 50)
    print(" 분석 4: 작업 군집에 대한 네트워크 분석")
    print("=" * 50)
    try:
        keyword_pairs = [pair for keywords in df['keywords_list'] if isinstance(keywords, list) for pair in
                         combinations(sorted(keywords), 2)]

        if keyword_pairs:
            pair_counts = Counter(keyword_pairs)
            print("\n🔗 가장 흔한 작업 쌍 (상위 10개):")
            print(pair_counts.most_common(10))
       # 시각화 주석처리
        #     G = nx.Graph()
        #     for pair, count in pair_counts.most_common(30):
        #         G.add_edge(pair[0], pair[1], weight=count)
        #     if G.nodes():
        #         plt.figure(figsize=(16, 12))
        #         pos = nx.spring_layout(G, k=0.9, iterations=50)
        #         d = dict(G.degree)
        #         nx.draw(G, pos, with_labels=True, node_color='skyblue',
        #                 node_size=[v * 100 for v in d.values()],
        #                 font_size=12, edge_color='gray', alpha=0.8)
        #         plt.title('작업 군집 네트워크 그래프', size=20)
        #         plt.savefig('task_network.png')
        #         print("\n✅ 네트워크 그래프가 'task_network.png'로 저장되었습니다.")
    except Exception as e:
        print(f"분석 4 오류: {e}")

print("\n 모든 분석이 완료.")