import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
from itertools import combinations
import networkx as nx
import warnings
import matplotlib.font_manager as fm  # <<< 수정: 폰트 관리를 위한 라이브러리 추가

# 사소한 경고 메시지는 무시
warnings.filterwarnings('ignore')

# --- 시각화 위한 한글 폰트 설정 ---
# <<< 수정: 주석 해제 및 시스템 폰트 자동 탐색 로직으로 변경
font_name = None
try:
    # 시스템에 설치된 폰트 중 'Malgun Gothic' 또는 'Nanum' 계열 폰트 탐색
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    if 'Malgun Gothic' in available_fonts:
        font_name = 'Malgun Gothic'
    elif any('Nanum' in font for font in available_fonts):
        # 'Nanum'을 포함하는 폰트 중 하나를 선택
        font_name = [font for font in available_fonts if 'Nanum' in font][0]

    if font_name:
        plt.rcParams['font.family'] = font_name
        print(f"✅ 한글 폰트 '{font_name}'가 성공적으로 설정되었습니다.")
    else:
        print("⚠️ 'Malgun Gothic' 또는 'Nanum' 폰트를 찾을 수 없습니다. 시각화 시 글자가 깨질 수 있습니다.")

    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 부호 깨짐 방지
except Exception as e:
    print(f"폰트 설정 중 오류 발생: {e}")

# ---  데이터 불러오기 및 전처리 ---
file_path = 'job_code.csv'
try:
    df = pd.read_csv(file_path, encoding='cp949')
    print(f"\n'{file_path}' 파일을 성공적으로 불러왔습니다.")

    df.dropna(subset=['keywords_bert'], inplace=True)
    df['keywords_list'] = df['keywords_bert'].str.split(', ')

    print("\n---------- [ 데이터 정보 ] ----------")
    df.info()
    print("\n---------- [ 데이터 샘플 ] ----------")
    print(df.head())

except FileNotFoundError:
    print(f" 오류: '{file_path}' 파일을 찾을 수 없습니다. 파일 위치를 확인해주세요.")
    df = pd.DataFrame()
except UnicodeDecodeError:
    print(f" 오류: '{file_path}' 파일의 인코딩 문제일 수 있습니다. 'utf-8'로 다시 시도합니다.")
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"\n'{file_path}' 파일을 'utf-8'로 다시 시도하여 성공적으로 불러왔습니다.")
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
            # keywords가 문자열이 아닌 경우(예: float 타입의 NaN)를 대비
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
        # 'keywords_bert'에 NaN 값이 있을 경우를 대비하여 문자열로 변환
        df['keywords_bert'].fillna('', inplace=True)
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
    #  분석 4: 작업 군집에 대한 네트워크 분석
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

            G = nx.Graph()
            for pair, count in pair_counts.most_common(30):
                G.add_edge(pair[0], pair[1], weight=count)

            if G.nodes():
                plt.figure(figsize=(16, 12))
                pos = nx.spring_layout(G, k=0.9, iterations=50)
                d = dict(G.degree)


                nx.draw(G, pos, with_labels=True, node_color='skyblue',
                        node_size=[v * 100 for v in d.values()],
                        font_size=12, edge_color='gray', alpha=0.8,
                        font_family=font_name)

                plt.title('작업 군집 네트워크 그래프', size=20)
                plt.savefig('task_network.png')
                print("\n✅ 네트워크 그래프가 'task_network.png'로 저장되었습니다.")
        else:
            print("\n⚠️ 분석할 키워드 쌍이 없어 네트워크 그래프를 생성할 수 없습니다.")

    except Exception as e:
        print(f"분석 4 오류: {e}")

print("\n모든 분석이 완료되었습니다.")