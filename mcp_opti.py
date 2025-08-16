from fastmcp import FastMCP
import os
import sys
import datetime
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


# MCP 서버 인스턴스 생성
# 서버의 역할을 명확히 나타내는 이름을 부여합니다.
mcp = FastMCP("Side Dish Production Optimizer")


# 설정 상수
# =====================================================================
DEFAULT_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
DEFAULT_NUM_LINES = 8
DEFAULT_MAX_TIME = 240
DEFAULT_BASE_CHANGEOVER_TIME = 2
DEFAULT_MAX_ADDITIONAL_TIME = 2
DEFAULT_UNKNOWN_COOKING_TIME = 3
UNIT_TIME_PER_QUANTITY = 0.01
OPTIMIZATION_DB_PATH = "./chroma_optimization_results_db"
OPTIMIZATION_TIME_LIMIT = 60


# 유틸리티 클래스
# =====================================================================
class StdoutCapture:
    """표준 출력 캡처를 위한 클래스"""
    
    def __init__(self):
        self.contents = []
        
    def write(self, data: str) -> None:
        self.contents.append(data)
        
    def flush(self) -> None:
        pass
        
    def get_output(self) -> str:
        return ''.join(self.contents)


# 전역 변수(Agent 연동용)
# =====================================================================
last_optimization_output = None
last_optimization_text = None
current_file_name = None


## 1. 벡터 임베딩 관련 함수 ##
# =====================================================================
# 1-1. 반찬명 벡터 임베딩 생성 함수
def create_dish_embeddings(df: pd.DataFrame, 
                          dish_column: str = '상품명',
                          model_name: str = DEFAULT_MODEL_NAME) -> Dict[str, Any]:
    
    model = SentenceTransformer(model_name)
    unique_dishes = df[dish_column].unique().tolist()
    embeddings = model.encode(unique_dishes, show_progress_bar=True)
    print(f"임베딩 완료 / 차원 : {embeddings.shape}")
    
    return {
        'dish_names': unique_dishes,
        'embeddings': embeddings,
        'embedding_dim': embeddings.shape[1],
        'model': model
    }

# 1-2. 전환시간 계산 함수
def calculate_changeover_matrix(embedding_result: Dict[str, Any],
                               base_time: int = DEFAULT_BASE_CHANGEOVER_TIME,
                               max_additional_time: int = DEFAULT_MAX_ADDITIONAL_TIME) -> pd.DataFrame:
    dish_names = embedding_result['dish_names']
    embeddings = embedding_result['embeddings']
    cosine_dist_matrix = cosine_distances(embeddings)
    changeover_matrix = base_time + (cosine_dist_matrix * max_additional_time)
    np.fill_diagonal(changeover_matrix, 0)
    
    changeover_df = pd.DataFrame(
        changeover_matrix,
        index=dish_names,
        columns=dish_names
    )
    
    print(f"전환 시간 범위: {changeover_matrix.min():.1f}분 ~ {changeover_matrix.max():.1f}분")
    return changeover_df


## 2. 조리시간 관련 함수 ##
# =====================================================================
# 2-1. 반찬별 조리시간 데이터
def get_dish_cooking_times() -> Dict[str, int]:
    return {
        '콩나물무침': 1, '미나리무침': 2, '무생채': 2, '시금치나물 - 90g': 3,
        '새콤달콤 유채나물무침': 3, '새콤달콤 방풍나물무침': 3, '닭가슴살 두부무침': 3,
        '새콤달콤 돌나물무침': 2, '새콤달콤 오징어무침': 3, '새콤달콤 오이달래무침': 2,
        '브로콜리 두부무침 - 100g': 3, '매콤 콩나물무침': 3, '오이부추무침': 2,
        '참깨소스 시금치무침': 3, '(gs재등록) 닭가슴살 참깨무침': 3, '무말랭이무침': 3,
        '오징어무말랭이무침 - 130g': 3, '참나물무침 - 80g': 2, '연근참깨무침': 3,
        '참깨소스 버섯무침 - 100g': 3, '톳두부무침': 3, '가지무침': 3,
        '숙주나물무침 - 90g': 3, '달래김무침': 2, '새콤 꼬시래기무침': 3,
        '오이부추무침 - 100g': 2, '참깨두부무침 - 200g': 3, '새콤 오이무생채': 3,
        '새콤달콤 오징어무침 - 110g': 3, '새콤달콤 도라지무침': 3, '콩나물무침 - 90g': 2,
        '무생채 - 100g': 2, '파래김무침': 2, '무나물 - 100g': 2,
        '물김치 - 350g': 2, '백김치 - 350g': 2, '양파고추 장아찌 - 150g': 2,
        '유자향 오이무피클 - 240g': 2, '깻잎 장아찌': 2, '셀러리 장아찌': 2,
        '깍두기': 3, '나박김치': 3, '총각김치': 3, '곰취 장아찌': 2,
        '볶음김치': 3, '볶음김치_대용량': 3,
        '아이들 된장국': 4, '감자국': 5, '계란국(냉동)': 3, '순한 오징어무국': 5,
        '시래기 된장국(냉동)': 5, '달래 된장찌개': 4, '근대 된장국(냉동)': 5,
        '된장찌개': 5, '동태알탕': 5, '맑은 콩나물국(냉동)': 4, '오징어 무국(냉동)': 5,
        '냉이 된장국(냉동)': 4, '한우 소고기 감자국': 5, '우리콩 강된장찌개': 5,
        '맑은 순두부찌개': 4, '계란 황태국(냉동)': 4, '오징어찌개': 5,
        '시금치 된장국(냉동)': 4, '김치콩나물국(냉동)': 5, '한우사골곰탕(냉동) - 600g': 5,
        '한우 소고기 무국(냉동) - 650g': 5, '한우 소고기 미역국(냉동) - 650g': 5,
        '맑은 동태국': 5, '콩나물 황태국(냉동)': 4, '배추 된장국(냉동)': 5,
        '한돈 돼지김치찌개': 7, '한돈 청국장찌개': 6, '동태찌개': 6,
        '한돈 돼지돼지 김치찌개_쿠킹박스': 7, '한돈 돼지고추장찌개': 7, '알탕': 8,
        '한우 무볶음': 4, '고추장 멸치볶음': 3, '야채 어묵볶음': 4,
        '느타리버섯볶음 - 90g': 3, '풋마늘 어묵볶음': 4, '애호박볶음': 3,
        '새우 애호박볶음 - 110g': 4, '한돈 가지볶음': 4, '들깨머위나물볶음': 3,
        '도라지볶음 - 80g': 3, '감자햄볶음': 4, '느타리버섯볶음': 3,
        '토마토 계란볶음': 3, '미역줄기볶음': 3, '건곤드레볶음': 4,
        '건고사리볶음 - 80g': 3, '호두 멸치볶음_대용량': 4, '미역줄기볶음_대용량': 4,
        '감자채볶음': 3, '건취나물볶음 - 80g': 3, '호두 멸치볶음': 4,
        '꼴뚜기 간장볶음': 5, '새우오이볶음': 3, '소고기 야채볶음_반조리': 5,
        '들깨시래기볶음 - 90g': 4, '보리새우 간장볶음': 4, '소고기 우엉볶음': 5,
        '한우오이볶음': 4, '건가지볶음': 3, '들깨고구마 줄기볶음 - 80g': 3,
        '한우오이볶음 - 100g': 4, '야채 어묵볶음 - 80g': 4, '감자채볶음 - 80g': 3,
        '매콤 어묵볶음': 4, '건피마자볶음': 3, '한우 무볶음 - 110g': 4,
        '감자햄볶음 - 80g': 4, '소고기 우엉볶음 - 80g': 5, '꽈리멸치볶음 - 60g': 3,
        '호두 멸치볶음 - 60g': 4, '미역줄기볶음 - 60g': 3, '꽈리멸치볶음_대용량': 4,
        '소고기 가지볶음': 5, '간장소스 어묵볶음': 4, '건호박볶음': 3,
        '고추장 멸치볶음_대용량': 4, '한돈 냉이 버섯볶음밥 재료': 5,
        '상하농원 케찹 소세지 야채볶음': 4, '상하농원 햄 어묵볶음': 4,
        '한돈 매콤 제육볶음_반조리 - 500g': 5, '주꾸미 한돈 제육볶음_반조리': 5,
        '한돈 김치두루치기_반조리': 5, '한돈 미나리 고추장불고기_반조리': 5,
        '한돈 대파 제육볶음_반조리': 5, '주꾸미 야채볶음_반조리': 5,
        '오징어 야채볶음_반조리': 4, '간장 오리 주물럭_반조리': 5,
        '한돈 콩나물불고기_반조리': 5, '한돈 간장 콩나물불고기_반조리': 5,
        '한돈 간장불고기_반조리': 4, '오리 주물럭_반조리': 5,
        '한돈 된장불고기_반조리': 5, '한돈 간장불고기_쿠킹박스': 4,
        '한돈 매콤 제육볶음_쿠킹박스': 5, '한돈 풋마늘 두루치기_반조리': 5,
        '메추리알 간장조림': 5, '소고기 장조림 - 180g': 5, '두부조림': 4,
        '알감자조림': 4, '케찹두부조림': 4, '매콤 닭가슴살 장조림': 5,
        '메추리알 간장조림_대용량': 5, '깻잎조림_대용량': 3, '소고기 장조림_대용량': 5,
        '한입 두부간장조림': 4, '검은콩조림': 5, '한입 두부간장조림 - 110g': 4,
        '표고버섯조림': 5, '케찹두부조림 - 120g': 4, '계란 간장조림': 4,
        '명란 장조림': 3, '국내산 땅콩조림': 5, '깻잎조림': 3,
        '간장 감자조림': 5, '마늘쫑 간장조림': 3, '메추리알 간장조림 - 110g': 5,
        '한우 장조림': 5, '우엉조림 - 100g': 5, '유자견과류조림': 4,
        '한돈 매콤 안심장조림': 5, '촉촉 간장무조림': 5, '미니새송이버섯조림': 4,
        '간장 코다리조림': 5, '매콤 코다리조림': 5, '고등어무조림': 5,
        '꽈리고추찜': 5, '야채 계란찜': 5, '계란찜': 5, '매운돼지갈비찜': 8,
        '순두부 계란찜': 5, '안동찜닭_반조리': 8,
        '소고기육전과 파채': 5, '참치깻잎전': 5, '냉이전 - 140g': 4,
        '매생이전': 4, '동태전': 5, '달콤 옥수수전 - 140g': 4,
        '반달 계란전': 4, '매콤김치전': 5,
        '간편화덕 고등어 순살구이': 4, '간편화덕 삼치 순살구이': 4,
        '간편화덕 연어 순살구이': 5, '한돈 너비아니(냉동)': 4,
        '오븐치킨_반조리(냉동)': 5, '한돈등심 치즈가스_반조리(냉동)': 4,
        '통등심 수제돈가스_반조리(냉동)': 4,
        '한돈 주먹밥': 3, '계란 두부소보로 주먹밥': 3, '멸치 주먹밥': 3,
        '참치마요 주먹밥': 3, '한우 주먹밥': 3, '햇반 발아현미밥': 2, '햇반 백미': 2,
        '한돈 토마토 덮밥': 3, '아이들 두부덮밥': 3, '사색 소보로 덮밥': 3,
        '새우 볶음밥 재료': 4, '닭갈비 볶음밥 재료': 4, '냉이 새우볶음밥 재료': 4,
        '상하농원 소세지 볶음밥 재료': 4, '감자볶음밥 재료': 4, '한돈 불고기볶음밥 재료': 4,
        '꼬막비빔밥': 3,
        '궁중 떡볶이_반조리 - 520g': 5, '우리쌀로 만든 기름떡볶이_반조리': 4,
        '뚝배기 불고기_반조리': 7, '서울식 불고기버섯전골_반조리': 8,
        '한우 파육개장(냉동)': 8, '소불고기_반조리 - 400g': 7,
        '한우 소불고기_반조리': 8, '모둠버섯 불고기_반조리': 6,
        '계란말이': 3, '야채계란말이': 3,
        '달래장': 1, '맛쌈장': 1, '양배추와 맛쌈장': 1, '사랑담은 돈가스소스': 1,
        '옥수수 버무리': 3, '상하농원 햄 메추리알 케찹볶음': 3, '무나물': 3,
        '수제비_요리놀이터': 3, '봄나물 샐러드': 3, '황태 보푸리': 3,
        '가지강정_대용량': 3, '가지강정': 3, '낙지젓': 3, '영양과채사라다': 3,
        '시래기 된장지짐': 3, '잡채 - 450g': 3, '해물잡채': 3,
        '바른 간장참치 - 130g': 3, '골뱅이무침_반조리': 3, '참깨소스 버섯무침': 3,
        '한우 계란소보로': 3, '꼬마김밥_요리놀이터': 3, '요리놀이터 꼬꼬마 김발': 3,
        '오징어젓': 3, '황기 닭곰탕(냉동)': 3, '불고기 잡채': 3,
        '우엉잡채 - 80g': 3, '만두속재료_요리놀이터': 3,
    }

# 2-2. 특정 반찬의 총 조리시간 계산 함수
def get_cooking_time(dish_name: str, quantity: int = 1) -> float:
    cooking_times = get_dish_cooking_times()
    if dish_name in cooking_times:
        base_time = cooking_times[dish_name]
    else:
        base_time = DEFAULT_UNKNOWN_COOKING_TIME
        print(f"⚠️ '{dish_name}' 조리시간을 찾을 수 없어 기본값 {base_time}분 사용")
    total_time = base_time + (quantity * UNIT_TIME_PER_QUANTITY)
    return total_time

# 2-3. 조리시간 DataFrame 생성 함수
def create_cooking_time_dataframe() -> pd.DataFrame:
    cooking_times = get_dish_cooking_times()
    df = pd.DataFrame([
        {'반찬명': dish, '기본조리시간(분)': time}
        for dish, time in cooking_times.items()
    ])
    return df.sort_values('기본조리시간(분)')


# 3. VRP 최적화 함수
# =====================================================================
def solve_dish_production_vrp(embedding_result: Dict[str, Any],
                             changeover_matrix: pd.DataFrame,
                             orders_df: pd.DataFrame,
                             num_lines: int = DEFAULT_NUM_LINES,
                             max_time: int = DEFAULT_MAX_TIME) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    dish_demands = orders_df.groupby('상품명')['수량'].sum().to_dict()
    ordered_dishes = list(dish_demands.keys())
    num_dishes = len(ordered_dishes)
    
    print(f"주문된 반찬: {num_dishes}개")
    print(f"총 생산량: {sum(dish_demands.values())}개")
    
    cooking_times = {dish: get_cooking_time(dish, dish_demands[dish]) for dish in ordered_dishes}
    print(f"조리 시간 범위: {min(cooking_times.values()):.1f}분 ~ {max(cooking_times.values()):.1f}분")
    
    num_depots = num_lines
    num_nodes = num_depots + num_dishes
    distance_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    
    for i in range(num_dishes):
        for j in range(num_dishes):
            dish_i = ordered_dishes[i]
            dish_j = ordered_dishes[j]
            changeover_time = int(changeover_matrix.loc[dish_i, dish_j]) if dish_i in changeover_matrix.index and dish_j in changeover_matrix.columns else DEFAULT_UNKNOWN_COOKING_TIME
            distance_matrix[num_depots + i][num_depots + j] = changeover_time
    
    manager = pywrapcp.RoutingIndexManager(num_nodes, num_lines, list(range(num_lines)), list(range(num_lines)))
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_index: int, to_index: int) -> int:
        return distance_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    def time_callback(from_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        return 0 if from_node < num_depots else int(cooking_times[ordered_dishes[from_node - num_depots]])
    
    time_callback_index = routing.RegisterUnaryTransitCallback(time_callback)
    routing.AddDimensionWithVehicleCapacity(time_callback_index, 0, [max_time] * num_lines, True, 'Time')
    for dish_idx in range(num_dishes):
        routing.AddDisjunction([manager.NodeToIndex(num_depots + dish_idx)], 1000000)
    
    time_dimension = routing.GetDimensionOrDie('Time')
    max_end_time = routing.AddVariableMinimizedByFinalizer(routing.solver().Max(
        [time_dimension.CumulVar(routing.End(line)) for line in range(num_lines)]
    ))
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(OPTIMIZATION_TIME_LIMIT)
    
    solution = routing.SolveWithParameters(search_parameters)
    
    if solution:
        print_solution(manager, routing, solution, ordered_dishes, cooking_times, num_depots)
        return manager, routing, solution
    else:
        print("해를 찾을 수 없습니다!")
        return None, None, None

# 3-2. 최적화 결과 출력 함수
def print_solution(manager: Any, routing: Any, solution: Any,
                  ordered_dishes: List[str], cooking_times: Dict[str, float],
                  num_depots: int) -> None:
    print("\n" + "="*50)
    print("최적화 결과")
    print("="*50)
    max_line_time = 0
    
    for line_id in range(routing.vehicles()):
        index = routing.Start(line_id)
        plan_output = f'생산라인 {line_id + 1}: '
        route_time = 0
        
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node >= num_depots:
                dish_idx = node - num_depots
                dish_name = ordered_dishes[dish_idx]
                cooking_time = cooking_times[dish_name]
                plan_output += f'{dish_name}({cooking_time:.1f}분) -> '
                route_time += cooking_time
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                if not routing.IsEnd(index):
                    route_time += routing.GetArcCostForVehicle(previous_index, index, line_id)
            else:
                index = solution.Value(routing.NextVar(index))
        
        plan_output += '완료'
        print(f'{plan_output}')
        print(f'⏱️  총 소요시간: {route_time:.1f}분')
        print('-' * 50)
        max_line_time = max(max_line_time, route_time)
    
    print(f"\n 전체 완료 시간 (Makespan): {max_line_time:.1f}분")
    print(f" 제한 시간 대비: {max_line_time/DEFAULT_MAX_TIME*100:.1f}%")
    if max_line_time <= DEFAULT_MAX_TIME:
        print("시간 제약 만족!")
    else:
        print("시간 제약 초과!")


# 4. 벡터 데이터베이스 저장 함수
# =====================================================================
def save_optimization_to_vectordb(manager, routing, solution, 
                                           ordered_dishes, cooking_times, 
                                           num_depots, file_name=None):
    try:
        embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        vector_store = Chroma(
            collection_name="optimization_results",
            persist_directory=OPTIMIZATION_DB_PATH,
            embedding_function=embeddings
        )
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        documents = []
        for line_id in range(routing.vehicles()):
            index = routing.Start(line_id)
            sequence = 1
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                if node >= num_depots:
                    dish_idx = node - num_depots
                    dish_name = ordered_dishes[dish_idx]
                    cooking_time = cooking_times[dish_name]
                    
                    doc_content = f"라인{line_id + 1} 순서{sequence}: {dish_name}"
                    documents.append(Document(
                        page_content=doc_content,
                        metadata={
                            "type": "production_step",
                            "line_no": line_id + 1,
                            "sequence": sequence,
                            "product_name": dish_name,
                            "cooking_time": round(cooking_time, 1),
                            "timestamp": timestamp,
                            "file_name": file_name or "unknown"
                        }
                    ))
                    sequence += 1
                index = solution.Value(routing.NextVar(index))
        vector_store.add_documents(documents)
        print(f"구조화된 최적화 결과 저장 완료 ({len(documents)}개 작업)")
    except Exception as e:
        print(f"벡터 DB 저장 중 오류: {str(e)}")


# 5. 메인 실행 함수들
# =====================================================================
def run_vrp_optimization(embedding_result: Dict[str, Any],
                        changeover_matrix: pd.DataFrame,
                        orders_df: pd.DataFrame,
                        num_lines: int = DEFAULT_NUM_LINES,
                        max_time: int = DEFAULT_MAX_TIME) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    print("생산 최적화를 시작합니다!")
    return solve_dish_production_vrp(
        embedding_result=embedding_result,
        changeover_matrix=changeover_matrix,
        orders_df=orders_df,
        num_lines=num_lines,
        max_time=max_time
    )

def run_full_optimization(file_path: str,
                         dish_column: str = '상품명',
                         num_lines: int = DEFAULT_NUM_LINES,
                         max_time: int = DEFAULT_MAX_TIME) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    global current_file_name
    current_file_name = os.path.basename(file_path)
    df = pd.read_excel(file_path)
    embedding_result = create_dish_embeddings(df, dish_column)
    changeover_df = calculate_changeover_matrix(embedding_result)
    manager, routing, solution = run_vrp_optimization(embedding_result, changeover_df, df, num_lines, max_time)
    
    if solution:
        dish_demands = df.groupby(dish_column)['수량'].sum().to_dict()
        ordered_dishes = list(dish_demands.keys())
        cooking_times = {dish: get_cooking_time(dish, dish_demands[dish]) for dish in ordered_dishes}
        save_optimization_to_vectordb(
            manager, routing, solution, ordered_dishes, cooking_times, 
            DEFAULT_NUM_LINES, current_file_name
        )
    return manager, routing, solution


# 6. 생산 최적화 도구
# =====================================================================
@mcp.tool()
def dish_optimization_tool(query: str) -> str:
    """엑셀 파일 경로를 입력받아 반찬 생산 최적화(VRP)를 실행하고 결과를 반환합니다."""
    global last_optimization_output, last_optimization_text
    try:
        file_path = query.split(',')[0].strip()
        if not os.path.exists(file_path):
            return f"파일을 찾을 수 없습니다: {file_path}"
        
        print(f"반찬 최적화 시작: {file_path}")
        
        captured_output = StdoutCapture()
        old_stdout = sys.stdout
        
        try:
            sys.stdout = captured_output
            optimization_result = run_full_optimization(file_path)
        finally:
            sys.stdout = old_stdout
        
        captured_text = captured_output.get_output()
        last_optimization_output = optimization_result
        last_optimization_text = captured_text
        print(captured_text)
        
        return "반찬 생산 최적화 완료! 상세 결과가 출력되었으며 벡터 DB에도 저장되었습니다."
        
    except Exception as e:
        return f"최적화 중 오류 발생: {str(e)}"


# 스크립트를 직접 실행할 경우 MCP 서버를 시작합니다.
if __name__ == "__main__":
    mcp.run()