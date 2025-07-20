"""
프로젝트 전역 로깅 설정

모든 모듈에서 일관된 로깅 사용을 위한 설정
"""

import logging
import logging.handlers
import os
from pathlib import Path
from config.settings import LoggingConfig


def setup_logger(name: str, log_file: str = None, level: str = None) -> logging.Logger:
    """
    프로젝트 표준 로거 설정
    
    Args:
        name: 로거 이름 (보통 __name__ 사용)
        log_file: 로그 파일 경로 (선택사항)
        level: 로그 레벨 (선택사항)
    
    Returns:
        설정된 로거 객체
    """
    logger = logging.getLogger(name)
    
    # 이미 핸들러가 설정된 경우 중복 방지
    if logger.handlers:
        return logger
    
    # 로그 레벨 설정
    log_level = level or LoggingConfig.LOG_LEVEL
    logger.setLevel(getattr(logging, log_level))
    
    # 포매터 설정
    formatter = logging.Formatter(LoggingConfig.LOG_FORMAT)
    
    # 콘솔 핸들러 (항상 추가)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (선택사항)
    if log_file:
        # 로그 디렉토리 생성
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 로테이팅 파일 핸들러
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=_parse_size(LoggingConfig.MAX_LOG_SIZE),
            backupCount=LoggingConfig.BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_security_logger() -> logging.Logger:
    """보안 전용 로거 반환"""
    return setup_logger('security', LoggingConfig.SECURITY_LOG)


def get_api_logger() -> logging.Logger:
    """API 전용 로거 반환"""
    return setup_logger('api', LoggingConfig.API_ACCESS_LOG)


def get_error_logger() -> logging.Logger:
    """에러 전용 로거 반환"""
    return setup_logger('error', LoggingConfig.ERROR_LOG)


def _parse_size(size_str: str) -> int:
    """크기 문자열을 바이트로 변환"""
    size_str = size_str.upper()
    if size_str.endswith('KB'):
        return int(size_str[:-2]) * 1024
    elif size_str.endswith('MB'):
        return int(size_str[:-2]) * 1024 * 1024
    elif size_str.endswith('GB'):
        return int(size_str[:-2]) * 1024 * 1024 * 1024
    else:
        return int(size_str)


# 편의를 위한 기본 로거들
default_logger = setup_logger('customer_segmentation')
security_logger = get_security_logger()
api_logger = get_api_logger()
error_logger = get_error_logger()
