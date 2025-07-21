#!/usr/bin/env python3
"""
SignGlove 서버 헬스체크 스크립트
Docker 컨테이너의 서버 상태를 확인합니다.
"""

import sys
import httpx
import json
import time
from datetime import datetime


def check_main_server():
    """메인 FastAPI 서버 상태 확인"""
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get("http://localhost:8000/health")
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "healthy",
                    "details": data,
                    "response_time": response.elapsed.total_seconds()
                }
            else:
                return {
                    "status": "unhealthy",
                    "details": f"HTTP {response.status_code}: {response.text}",
                    "response_time": response.elapsed.total_seconds()
                }
                
    except Exception as e:
        return {
            "status": "unhealthy", 
            "details": str(e),
            "response_time": None
        }


def check_data_endpoints():
    """데이터 수집 엔드포인트 상태 확인"""
    endpoints = ["/data/stats", "/api/ksl/statistics"]
    results = {}
    
    try:
        with httpx.Client(timeout=3.0) as client:
            for endpoint in endpoints:
                try:
                    response = client.get(f"http://localhost:8000{endpoint}")
                    results[endpoint] = {
                        "status": "ok" if response.status_code == 200 else "error",
                        "code": response.status_code
                    }
                except Exception as e:
                    results[endpoint] = {
                        "status": "error",
                        "error": str(e)
                    }
                    
    except Exception as e:
        results["error"] = str(e)
        
    return results


def check_disk_space():
    """디스크 공간 확인"""
    import shutil
    
    try:
        # 데이터 디렉토리 용량 확인
        data_usage = shutil.disk_usage("/opt/signglove/data")
        backup_usage = shutil.disk_usage("/opt/signglove/backup")
        
        data_free_gb = data_usage.free / (1024**3)
        backup_free_gb = backup_usage.free / (1024**3)
        
        return {
            "data_free_gb": round(data_free_gb, 2),
            "backup_free_gb": round(backup_free_gb, 2),
            "status": "ok" if data_free_gb > 1.0 else "warning"  # 1GB 미만 경고
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


def check_log_files():
    """로그 파일 상태 확인"""
    import os
    
    log_files = [
        "/var/log/signglove/server.log",
        "/var/log/signglove/error.log", 
        "/var/log/signglove/access.log"
    ]
    
    results = {}
    
    for log_file in log_files:
        try:
            if os.path.exists(log_file):
                stat = os.stat(log_file)
                size_mb = stat.st_size / (1024**2)
                results[os.path.basename(log_file)] = {
                    "exists": True,
                    "size_mb": round(size_mb, 2),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "status": "warning" if size_mb > 100 else "ok"  # 100MB 이상 경고
                }
            else:
                results[os.path.basename(log_file)] = {
                    "exists": False,
                    "status": "warning"
                }
                
        except Exception as e:
            results[os.path.basename(log_file)] = {
                "exists": False,
                "error": str(e),
                "status": "error"
            }
            
    return results


def main():
    """헬스체크 실행"""
    print(f"🏥 SignGlove 헬스체크 시작 - {datetime.now().isoformat()}")
    
    # 전체 상태 결과
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "healthy",
        "checks": {}
    }
    
    # 1. 메인 서버 확인
    print("📡 메인 서버 상태 확인...")
    server_status = check_main_server()
    health_status["checks"]["main_server"] = server_status
    
    if server_status["status"] != "healthy":
        health_status["overall_status"] = "unhealthy"
        print(f"❌ 메인 서버 상태 이상: {server_status['details']}")
    else:
        print(f"✅ 메인 서버 정상 (응답시간: {server_status['response_time']:.3f}s)")
    
    # 2. 데이터 엔드포인트 확인
    print("📊 데이터 엔드포인트 확인...")
    endpoint_status = check_data_endpoints()
    health_status["checks"]["data_endpoints"] = endpoint_status
    
    unhealthy_endpoints = [ep for ep, status in endpoint_status.items() 
                          if isinstance(status, dict) and status.get("status") != "ok"]
    
    if unhealthy_endpoints:
        print(f"⚠️ 일부 엔드포인트 이상: {unhealthy_endpoints}")
        if health_status["overall_status"] == "healthy":
            health_status["overall_status"] = "degraded"
    else:
        print("✅ 모든 데이터 엔드포인트 정상")
    
    # 3. 디스크 공간 확인
    print("💾 디스크 공간 확인...")
    disk_status = check_disk_space()
    health_status["checks"]["disk_space"] = disk_status
    
    if disk_status["status"] == "warning":
        print(f"⚠️ 디스크 공간 부족: {disk_status['data_free_gb']}GB 남음")
        if health_status["overall_status"] == "healthy":
            health_status["overall_status"] = "warning"
    elif disk_status["status"] == "error":
        print(f"❌ 디스크 상태 확인 실패: {disk_status.get('error', 'Unknown error')}")
        health_status["overall_status"] = "unhealthy"
    else:
        print(f"✅ 디스크 공간 충분 (데이터: {disk_status['data_free_gb']}GB)")
    
    # 4. 로그 파일 확인
    print("📝 로그 파일 확인...")
    log_status = check_log_files()
    health_status["checks"]["log_files"] = log_status
    
    log_issues = [name for name, status in log_status.items() 
                  if status.get("status") in ["warning", "error"]]
    
    if log_issues:
        print(f"⚠️ 로그 파일 이슈: {log_issues}")
        if health_status["overall_status"] == "healthy":
            health_status["overall_status"] = "warning"
    else:
        print("✅ 모든 로그 파일 정상")
    
    # 결과 출력
    print(f"\n🎯 전체 상태: {health_status['overall_status'].upper()}")
    
    # JSON 형태로도 출력 (모니터링 도구 연동용)
    if len(sys.argv) > 1 and sys.argv[1] == "--json":
        print(json.dumps(health_status, indent=2))
    
    # 종료 코드 결정
    if health_status["overall_status"] == "healthy":
        sys.exit(0)  # 정상
    elif health_status["overall_status"] in ["warning", "degraded"]:
        sys.exit(1)  # 경고 (컨테이너는 유지하되 문제 표시)
    else:
        sys.exit(2)  # 심각한 오류 (컨테이너 재시작 필요)


if __name__ == "__main__":
    main() 