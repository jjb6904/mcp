from fastmcp import FastMCP
import pandas as pd
import re
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.llms.ollama import Ollama
from sqlalchemy import create_engine, text

# MCP 서버 인스턴스 생성
# 서버의 역할을 명확히 나타내는 이름을 부여합니다.
mcp = FastMCP("SQL Query Tool")


# 설정
# ============================================================================
DB_URL = 'mysql+pymysql://langchain_user:1234@localhost:3306/langchain_db'
EXCEL_FILE = '/Users/jibaekjang/VS-Code/AI_Agent/product_data_all.xlsx'

# 전역 변수
db = None
llm = None


# 데이터베이스 연결 및 초기 설정
# ============================================================================
def setup_database():
    """서버 시작 시 데이터베이스 연결 및 테이블 설정을 한 번만 수행합니다."""
    global db, llm
    
    try:
        print("DB 연결 및 초기 설정 시작...")
        engine = create_engine(DB_URL)
        
        # 테이블 존재 확인
        with engine.connect() as conn:
            result = conn.execute(text("SHOW TABLES LIKE 'products'"))
            table_exists = result.fetchone() is not None
        
        if not table_exists:
            df = pd.read_excel(EXCEL_FILE)
            df.to_sql('products', engine, if_exists='replace', index=False)
            print(f"테이블 'products' 생성 완료 ({len(df)}행)")
        else:
            print("기존 테이블 'products' 사용")
        
        db = SQLDatabase(engine)
        llm = Ollama(model="gemma3:12b-it-qat")
        print("DB 연결 완료!")
        return True
        
    except Exception as e:
        print(f"DB 설정 실패: {e}")
        return False


# SQL 쿼리 도구 함수 : Agent에서 호출하는 함수
# ============================================================================
# @mcp.tool() 데코레이터를 사용하여 이 함수를 MCP 툴로 등록합니다.
@mcp.tool()
def sql_query_tool(question: str) -> str:
    """사용자의 질문을 SQL 쿼리로 변환하여 데이터베이스에서 정보를 검색합니다."""
    
    # 서버 시작 시 DB가 이미 설정되었으므로 별도 확인이 필요 없습니다.
    try:
        # SQL 쿼리 생성
        chain = create_sql_query_chain(llm, db)
        sql_query = chain.invoke({"question": question})
        
        # SQL 정리 - SQLQuery: 이후 부분만 추출
        if "SQLQuery:" in sql_query:
            clean_sql = sql_query.split("SQLQuery:")[-1].strip()
        else:
            clean_sql = sql_query
        
        # 마크다운, 세미콜론 제거
        clean_sql = re.sub(r'```sql|```|;', '', clean_sql).strip()
        
        print(f"실행 SQL: {clean_sql}")
        
        # 쿼리 실행
        result = db.run(clean_sql)
        
        return f"결과:\n{result}"
        
    except Exception as e:
        return f"쿼리 오류: {str(e)}"


# 스크립트를 직접 실행할 경우 MCP 서버를 시작합니다.
if __name__ == "__main__":
    if setup_database():
        mcp.run()
    else:
        print("서버를 시작할 수 없습니다. DB 설정을 확인해주세요.")