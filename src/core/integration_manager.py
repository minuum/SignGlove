"""
SignGlove 통합 관리 시스템

외부 팀들의 구현체들을 통합하고 관리하는 핵심 모듈
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import importlib.util

# 외부 저장소 경로 설정
EXTERNAL_PATHS = {
    'klp_signglove': Path('external/KLP-SignGlove'),
    'signglove_hw': Path('external/SignGlove_HW')
}

class IntegrationManager:
    """외부 시스템 통합 관리자"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.external_modules = {}
        self._load_external_modules()
    
    def _load_external_modules(self):
        """외부 모듈들을 동적으로 로드"""
        for name, path in EXTERNAL_PATHS.items():
            if path.exists():
                self.logger.info(f"외부 모듈 로드: {name} from {path}")
                self.external_modules[name] = path
            else:
                self.logger.warning(f"외부 모듈을 찾을 수 없음: {name} at {path}")
    
    def get_external_module_path(self, module_name: str) -> Optional[Path]:
        """외부 모듈 경로 반환"""
        return self.external_modules.get(module_name)
    
    def list_available_modules(self) -> list:
        """사용 가능한 외부 모듈 목록 반환"""
        return list(self.external_modules.keys())
    
    def check_module_health(self, module_name: str) -> Dict[str, Any]:
        """외부 모듈 상태 확인"""
        if module_name not in self.external_modules:
            return {'status': 'not_found', 'error': f'Module {module_name} not found'}
        
        path = self.external_modules[module_name]
        health_info = {
            'status': 'healthy',
            'path': str(path),
            'exists': path.exists(),
            'readme_exists': (path / 'README.md').exists(),
            'requirements_exists': (path / 'requirements.txt').exists()
        }
        
        return health_info
    
    def get_module_info(self, module_name: str) -> Dict[str, Any]:
        """외부 모듈 정보 반환"""
        if module_name not in self.external_modules:
            return {}
        
        path = self.external_modules[module_name]
        readme_path = path / 'README.md'
        
        info = {
            'name': module_name,
            'path': str(path),
            'description': self._extract_description(readme_path)
        }
        
        return info
    
    def _extract_description(self, readme_path: Path) -> str:
        """README 파일에서 설명 추출"""
        if not readme_path.exists():
            return "설명 없음"
        
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # 첫 번째 줄에서 설명 추출
                lines = content.split('\n')
                for line in lines:
                    if line.strip() and not line.startswith('#'):
                        return line.strip()
                return "설명 없음"
        except Exception as e:
            self.logger.error(f"README 파일 읽기 오류: {e}")
            return "파일 읽기 오류"

# 싱글톤 인스턴스
integration_manager = IntegrationManager()
