#!/bin/bash

# 🚀 Customer Segmentation - 빠른 실행 가이드

echo "🛍️ Customer Segmentation Analysis System"
echo "========================================"
echo ""

# 함수 정의
show_help() {
    echo "사용법:"
    echo "  ./quick_start.sh [옵션]"
    echo ""
    echo "옵션:"
    echo "  web       - 웹 애플리케이션 실행"
    echo "  api       - API 서버 실행"
    echo "  train     - 모델 훈련 실행"
    echo "  status    - 서버 상태 확인"
    echo "  install   - 의존성 설치"
    echo "  help      - 도움말 표시"
    echo ""
    echo "예시:"
    echo "  ./quick_start.sh web"
    echo "  ./quick_start.sh api"
    echo "  ./quick_start.sh train"
}

check_python() {
    if ! command -v python &> /dev/null; then
        echo "❌ Python이 설치되지 않았습니다."
        echo "Python 3.7 이상을 설치해주세요."
        exit 1
    fi
    
    python_version=$(python --version 2>&1 | awk '{print $2}')
    echo "✅ Python $python_version 확인됨"
}

install_dependencies() {
    echo "📦 필요한 패키지 설치 중..."
    
    if [ ! -f "requirements.txt" ]; then
        echo "❌ requirements.txt 파일이 없습니다."
        exit 1
    fi
    
    pip install -r requirements.txt
    
    if [ $? -eq 0 ]; then
        echo "✅ 의존성 설치 완료"
    else
        echo "❌ 의존성 설치 실패"
        exit 1
    fi
}

run_web_app() {
    echo "🌐 웹 애플리케이션 실행 중..."
    
    if [ ! -f "main_app.py" ]; then
        echo "❌ main_app.py 파일이 없습니다."
        exit 1
    fi
    
    echo "📱 브라우저에서 http://localhost:8501 접속하세요"
    echo "🛑 종료하려면 Ctrl+C를 누르세요"
    echo ""
    
    streamlit run main_app.py
}

run_api_server() {
    echo "🚀 API 서버 실행 중..."
    
    if [ ! -f "api_server.py" ]; then
        echo "❌ api_server.py 파일이 없습니다."
        exit 1
    fi
    
    echo "📡 API 문서: http://localhost:8000/docs"
    echo "🏥 상태 확인: http://localhost:8000/api/v1/health"
    echo "🛑 종료하려면 Ctrl+C를 누르세요"
    echo ""
    
    python api_server.py
}

train_models() {
    echo "🧠 모델 훈련 시작..."
    
    if [ ! -f "train_models.py" ]; then
        echo "❌ train_models.py 파일이 없습니다."
        exit 1
    fi
    
    echo "📊 모든 모델을 훈련합니다..."
    echo "⏱️  이 과정은 몇 분 정도 소요될 수 있습니다."
    echo ""
    
    python train_models.py
}

check_status() {
    echo "🔍 서버 상태 확인 중..."
    
    if [ -f "check_servers.sh" ]; then
        chmod +x check_servers.sh
        ./check_servers.sh
    else
        echo "📊 실행 중인 Python 프로세스:"
        ps aux | grep python | grep -v grep
        echo ""
        echo "🌐 포트 사용 현황:"
        lsof -i :8000 2>/dev/null || echo "포트 8000: 사용 중이지 않음"
        lsof -i :8501 2>/dev/null || echo "포트 8501: 사용 중이지 않음"
    fi
}

# 메인 실행 로직
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

case "$1" in
    "web")
        check_python
        run_web_app
        ;;
    "api")
        check_python
        run_api_server
        ;;
    "train")
        check_python
        train_models
        ;;
    "status")
        check_status
        ;;
    "install")
        check_python
        install_dependencies
        ;;
    "help")
        show_help
        ;;
    *)
        echo "❌ 알 수 없는 옵션: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
