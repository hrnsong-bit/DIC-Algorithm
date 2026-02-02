"""로깅 설정 모듈"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str = "speckle", 
                 level: int = logging.INFO,
                 log_file: bool = False) -> logging.Logger:
    """
    로거 설정
    
    Args:
        name: 로거 이름
        level: 로깅 레벨
        log_file: 파일 출력 여부
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # 포맷터
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (선택)
    if log_file:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / f"speckle_{datetime.now():%Y%m%d}.log",
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# 전역 로거
logger = setup_logger()
