"""
SignGlove 통합 시스템 테스트

외부 모듈 통합 및 API 게이트웨이 테스트
"""

import pytest
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.core.integration_manager import integration_manager
from src.integration.api_gateway import app
from fastapi.testclient import TestClient

client = TestClient(app)

class TestIntegrationManager:
    """통합 관리자 테스트"""
    
    def test_list_available_modules(self):
        """사용 가능한 모듈 목록 테스트"""
        modules = integration_manager.list_available_modules()
        assert isinstance(modules, list)
        assert len(modules) >= 0  # 외부 모듈이 없을 수도 있음
    
    def test_get_external_module_path(self):
        """외부 모듈 경로 반환 테스트"""
        # 존재하지 않는 모듈
        path = integration_manager.get_external_module_path("nonexistent_module")
        assert path is None
        
        # 존재하는 모듈 (외부 모듈이 있는 경우)
        modules = integration_manager.list_available_modules()
        if modules:
            module_name = modules[0]
            path = integration_manager.get_external_module_path(module_name)
            assert path is not None
            assert isinstance(path, Path)
    
    def test_check_module_health(self):
        """모듈 상태 확인 테스트"""
        # 존재하지 않는 모듈
        health = integration_manager.check_module_health("nonexistent_module")
        assert health['status'] == 'not_found'
        assert 'error' in health
        
        # 존재하는 모듈 (외부 모듈이 있는 경우)
        modules = integration_manager.list_available_modules()
        if modules:
            module_name = modules[0]
            health = integration_manager.check_module_health(module_name)
            assert 'status' in health
            assert 'path' in health
            assert 'exists' in health

class TestAPIGateway:
    """API 게이트웨이 테스트"""
    
    def test_root_endpoint(self):
        """루트 엔드포인트 테스트"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'message' in data
        assert 'external_modules' in data
    
    def test_health_endpoint(self):
        """헬스 체크 엔드포인트 테스트"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'message' in data
    
    def test_modules_endpoint(self):
        """모듈 목록 엔드포인트 테스트"""
        response = client.get("/modules")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        
        # 각 모듈 정보 확인
        for module in data:
            assert 'name' in module
            assert 'path' in module
            assert 'description' in module
            assert 'health' in module
    
    def test_module_info_endpoint(self):
        """특정 모듈 정보 엔드포인트 테스트"""
        # 존재하지 않는 모듈
        response = client.get("/modules/nonexistent_module")
        assert response.status_code == 404
        
        # 존재하는 모듈 (외부 모듈이 있는 경우)
        modules = integration_manager.list_available_modules()
        if modules:
            module_name = modules[0]
            response = client.get(f"/modules/{module_name}")
            assert response.status_code == 200
            data = response.json()
            assert 'name' in data
            assert 'path' in data
            assert 'health' in data
    
    def test_module_health_endpoint(self):
        """특정 모듈 상태 엔드포인트 테스트"""
        # 존재하지 않는 모듈
        response = client.get("/modules/nonexistent_module/health")
        assert response.status_code == 404
        
        # 존재하는 모듈 (외부 모듈이 있는 경우)
        modules = integration_manager.list_available_modules()
        if modules:
            module_name = modules[0]
            response = client.get(f"/modules/{module_name}/health")
            assert response.status_code == 200
            data = response.json()
            assert 'status' in data
    
    def test_sync_status_endpoint(self):
        """동기화 상태 엔드포인트 테스트"""
        response = client.get("/sync/status")
        assert response.status_code == 200
        data = response.json()
        assert 'last_sync' in data
        assert 'next_sync' in data
        assert 'status' in data
        assert 'external_repos' in data

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
