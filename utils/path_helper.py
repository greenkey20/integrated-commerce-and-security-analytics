"""
경로 설정 헬퍼 유틸리티

모든 모듈에서 공통으로 사용할 Python 경로 설정
"""

import sys
import os


def setup_python_path():
    """Python 경로를 설정하여 프로젝트 모듈들을 import할 수 있도록 함"""
    # 프로젝트 루트 디렉토리 찾기
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))
    
    # Python 경로에 추가
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    return project_root


# 모듈이 import될 때 자동으로 경로 설정
setup_python_path()
