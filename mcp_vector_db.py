from fastmcp import FastMCP
import os

# 전역 변수
_embeddings = None
_optimization_vector_store = None

# 파일 경로
optimization_db_location = "./chroma_optimization_results_db"


# 임베딩 모델 로드 함수
def get_embeddings():

    global _embeddings
    if _embeddings is None:
        from langchain_ollama import OllamaEmbeddings
        _embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    return _embeddings


# 최적화 결과 벡터 스토어 로드 함수
def get_optimization_vector_store():

    global _optimization_vector_store
    
    if _optimization_vector_store is None:
        from langchain_chroma import Chroma
        
        embeddings = get_embeddings()
        
        if os.path.exists(optimization_db_location):
            _optimization_vector_store = Chroma(
                collection_name="optimization_results",
                persist_directory=optimization_db_location,
                embedding_function=embeddings
            )
        else:
            return None
    
    return _optimization_vector_store

# MCP 서버 인스턴스 생성
# 서버의 역할을 명확히 나타내는 이름을 부여합니다.
mcp = FastMCP("Optimization Result Search Server")


# 최적화 결과 검색 Tool : Agent가 사용
# @mcp.tool() 데코레이터를 사용하여 이 함수를 MCP 툴로 등록합니다.
@mcp.tool()
def vector_search_tool(query: str) -> str:
    """최적화 결과 벡터 데이터베이스에서 주어진 질의에 대한 관련 정보를 검색합니다."""
    try:
        vector_store = get_optimization_vector_store()
        if vector_store is None:
            return "최적화 결과가 없습니다. 먼저 반찬 생산 최적화를 실행해주세요."
        
        # 벡터 검색
        results = vector_store.similarity_search(query, k=5)
        
        if not results:
            return f"'{query}' 관련 검색 결과가 없습니다."
        
        # production_step 타입만 필터링하고 구조화된 형태로 반환
        production_steps = []
        for doc in results:
            if doc.metadata.get("type") == "production_step":
                production_steps.append({
                    "line_no": doc.metadata.get("line_no"),
                    "sequence": doc.metadata.get("sequence"),
                    "product_name": doc.metadata.get("product_name"),
                    "cooking_time": doc.metadata.get("cooking_time")
                })
        
        # 라인별, 순서별로 정렬
        production_steps.sort(key=lambda x: (x["line_no"] or 0, x["sequence"] or 0))
        
        # 테이블 형태로 포맷팅
        if production_steps:
            result = "Line No | Seq | 제품명 | 생산시간(분)\n"
            result += "-" * 50 + "\n"
            for step in production_steps:
                result += f"{step['line_no']:7} | {step['sequence']:3} | {step['product_name']:20} | {step['cooking_time']:8.1f}\n"
            return result
        else:
            # production_step가 없으면 일반 결과 반환
            result_data = []
            for doc in results:
                result_data.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            return str(result_data)
        
    except Exception as e:
        return f"검색 오류: {str(e)}"


# 스크립트를 직접 실행할 경우 MCP 서버를 시작합니다.
if __name__ == "__main__":
    mcp.run()