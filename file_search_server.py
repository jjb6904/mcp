# 함수자체가 돌아가는지 확인먼저 / MCP로 tool들 넘기기
import os
import subprocess
import platform
from fastmcp import FastMCP

mcp = FastMCP("File Search Server")

@mcp.tool()
def search_and_open_files(query):
    """파일을 검색하고 요약 후 시스템 기본 프로그램으로 열기"""
    found_files = []
    start_directory = "."
    
    # 파일 검색
    for root, dirs, files in os.walk(start_directory):
        for file in files:
            if str(query).lower() in file.lower():
                file_path = os.path.abspath(os.path.join(root, file))
                found_files.append(file_path)
    
    if not found_files:
        return "일치하는 파일을 찾을 수 없습니다."
    
    # 요약 정보 생성
    result = f"검색 결과: '{query}'와 일치하는 {len(found_files)}개 파일을 찾았습니다.\n\n"
    
    for i, file_path in enumerate(found_files, 1):
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        file_ext = os.path.splitext(file_name)[1]
        
        # 파일 크기를 읽기 쉽게 변환
        if file_size < 1024:
            size_str = f"{file_size} bytes"
        elif file_size < 1024 * 1024:
            size_str = f"{file_size / 1024:.1f} KB"
        else:
            size_str = f"{file_size / (1024 * 1024):.1f} MB"
        
        result += f"{i}. 📄 {file_name}\n"
        result += f"경로: {file_path}\n"
        result += f"크기: {size_str}\n"
        result += f"형식: {file_ext or '확장자 없음'}\n\n"
    
    # 첫 번째 파일을 자동으로 열기
    first_file = found_files[0]
    
    try:
        system = platform.system()
        if system == "Darwin":  # macOS
            subprocess.run(["open", first_file])
        elif system == "Windows":
            os.startfile(first_file)
        elif system == "Linux":
            subprocess.run(["xdg-open", first_file])
        
        result += f"첫 번째 파일을 자동으로 열었습니다: {os.path.basename(first_file)}"
        
    except Exception as e:
        result += f"파일 열기 실패: {str(e)}"
    
    return result
if __name__ == "__main__":
    mcp.run()

'''
import os
from fastmcp import FastMCP

mcp = FastMCP("Math Server")

@mcp.tool()
def add_numbers(numbers):
    """두 숫자를 더해줍니다. 예: '3,5' 또는 '10,20'"""
    try:
        # 입력을 문자열로 변환하고 쉼표로 분리
        nums = str(numbers).split(',')
        
        if len(nums) != 2:
            return "두 개의 숫자를 쉼표로 구분해서 입력하세요. 예: '3,5'"
        
        # 숫자로 변환
        num1 = float(nums[0].strip())
        num2 = float(nums[1].strip())
        
        # 덧셈 계산
        result = num1 + num2
        
        return f"{num1} + {num2} = {result}"
        
    except ValueError:
        return "숫자가 아닌 값이 입력되었습니다. 예: '3,5'"
    except Exception as e:
        return f"계산 오류: {str(e)}"

if __name__ == "__main__":
    mcp.run()
'''