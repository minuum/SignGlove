"""
교수님 서버 연동 프록시 모듈
FastAPI 서버와 교수님 서버 간의 데이터 중계 및 변환을 담당합니다.
"""

import os
import asyncio
import httpx
from typing import Dict, Any, Optional
from loguru import logger
from datetime import datetime
import json

from server.models.sensor_data import SensorData, GestureData


class ProfessorServerProxy:
    """교수님 서버와의 통신을 담당하는 프록시 클래스"""
    
    def __init__(self):
        self.professor_url = os.getenv("PROFESSOR_SERVER_URL", "http://localhost:8080/api")
        self.professor_token = os.getenv("PROFESSOR_SERVER_TOKEN", "")
        self.enabled = os.getenv("ENABLE_PROFESSOR_PROXY", "false").lower() == "true"
        self.timeout = 10.0
        self.retry_count = 3
        
        if self.enabled:
            logger.info(f"교수님 서버 프록시 활성화: {self.professor_url}")
        else:
            logger.info("교수님 서버 프록시 비활성화 (로컬 모드)")
    
    def _get_headers(self) -> Dict[str, str]:
        """HTTP 요청 헤더 생성"""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "SignGlove-Server/1.0",
            "X-Timestamp": datetime.now().isoformat()
        }
        
        if self.professor_token:
            headers["Authorization"] = f"Bearer {self.professor_token}"
            
        return headers
    
    def _convert_sensor_data_format(self, sensor_data: SensorData) -> Dict[str, Any]:
        """센서 데이터를 교수님 서버 포맷으로 변환"""
        return {
            "device_id": sensor_data.device_id,
            "timestamp": sensor_data.timestamp.isoformat(),
            "sensors": {
                "flex": {
                    "finger1": sensor_data.flex_sensors[0],
                    "finger2": sensor_data.flex_sensors[1], 
                    "finger3": sensor_data.flex_sensors[2],
                    "finger4": sensor_data.flex_sensors[3],
                    "finger5": sensor_data.flex_sensors[4]
                },
                "imu": {
                    "accelerometer": {
                        "x": sensor_data.gyroscope.get("accel_x", 0),
                        "y": sensor_data.gyroscope.get("accel_y", 0),
                        "z": sensor_data.gyroscope.get("accel_z", 0)
                    },
                    "gyroscope": {
                        "x": sensor_data.gyroscope.get("gyro_x", 0),
                        "y": sensor_data.gyroscope.get("gyro_y", 0), 
                        "z": sensor_data.gyroscope.get("gyro_z", 0)
                    }
                }
            },
            "metadata": {
                "quality_score": sensor_data.quality_score,
                "battery_level": 100,  # 하드웨어에서 추후 구현
                "signal_strength": -50  # WiFi 신호 강도
            }
        }
    
    def _convert_gesture_data_format(self, gesture_data: GestureData) -> Dict[str, Any]:
        """제스처 데이터를 교수님 서버 포맷으로 변환"""
        return {
            "device_id": gesture_data.device_id,
            "timestamp": gesture_data.timestamp.isoformat(),
            "gesture": {
                "class": gesture_data.gesture_class,
                "confidence": gesture_data.confidence,
                "duration_ms": gesture_data.duration_ms,
                "sequence_id": gesture_data.session_id
            },
            "sensor_snapshot": {
                "flex_values": gesture_data.sensor_snapshot.get("flex_sensors", []),
                "imu_values": gesture_data.sensor_snapshot.get("gyroscope", {})
            }
        }
    
    async def send_sensor_data(self, sensor_data: SensorData) -> bool:
        """센서 데이터를 교수님 서버로 전송"""
        if not self.enabled:
            logger.debug("교수님 서버 프록시 비활성화됨 - 센서 데이터 전송 스킵")
            return True
            
        try:
            converted_data = self._convert_sensor_data_format(sensor_data)
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.professor_url}/sensor-data",
                    json=converted_data,
                    headers=self._get_headers()
                )
                response.raise_for_status()
                
                logger.debug(f"센서 데이터 전송 성공: {sensor_data.device_id}")
                return True
                
        except httpx.TimeoutException:
            logger.warning(f"교수님 서버 타임아웃: {sensor_data.device_id}")
        except httpx.HTTPStatusError as e:
            logger.error(f"교수님 서버 HTTP 오류 {e.response.status_code}: {e.response.text}")
        except Exception as e:
            logger.error(f"교수님 서버 전송 오류: {str(e)}")
            
        return False
    
    async def send_gesture_data(self, gesture_data: GestureData) -> bool:
        """제스처 데이터를 교수님 서버로 전송"""
        if not self.enabled:
            logger.debug("교수님 서버 프록시 비활성화됨 - 제스처 데이터 전송 스킵")
            return True
            
        try:
            converted_data = self._convert_gesture_data_format(gesture_data)
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.professor_url}/gesture-data", 
                    json=converted_data,
                    headers=self._get_headers()
                )
                response.raise_for_status()
                
                logger.info(f"제스처 데이터 전송 성공: {gesture_data.gesture_class}")
                return True
                
        except httpx.TimeoutException:
            logger.warning(f"교수님 서버 타임아웃: {gesture_data.gesture_class}")
        except httpx.HTTPStatusError as e:
            logger.error(f"교수님 서버 HTTP 오류 {e.response.status_code}: {e.response.text}")
        except Exception as e:
            logger.error(f"교수님 서버 전송 오류: {str(e)}")
            
        return False
    
    async def get_server_status(self) -> Dict[str, Any]:
        """교수님 서버 상태 확인"""
        if not self.enabled:
            return {"status": "disabled", "message": "프록시 비활성화"}
            
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self.professor_url}/health",
                    headers=self._get_headers()
                )
                response.raise_for_status()
                
                return {
                    "status": "connected",
                    "server_info": response.json(),
                    "response_time_ms": response.elapsed.total_seconds() * 1000
                }
                
        except Exception as e:
            return {
                "status": "error", 
                "message": str(e),
                "last_check": datetime.now().isoformat()
            }
    
    async def batch_send_data(self, sensor_batch: list, gesture_batch: list) -> Dict[str, int]:
        """배치 데이터 전송 (성능 최적화용)"""
        if not self.enabled:
            return {"sensor_sent": 0, "gesture_sent": 0, "total_skipped": len(sensor_batch) + len(gesture_batch)}
            
        results = {"sensor_sent": 0, "gesture_sent": 0, "sensor_failed": 0, "gesture_failed": 0}
        
        # 센서 데이터 배치 전송
        sensor_tasks = [self.send_sensor_data(data) for data in sensor_batch]
        sensor_results = await asyncio.gather(*sensor_tasks, return_exceptions=True)
        
        for result in sensor_results:
            if result is True:
                results["sensor_sent"] += 1
            else:
                results["sensor_failed"] += 1
                
        # 제스처 데이터 배치 전송  
        gesture_tasks = [self.send_gesture_data(data) for data in gesture_batch]
        gesture_results = await asyncio.gather(*gesture_tasks, return_exceptions=True)
        
        for result in gesture_results:
            if result is True:
                results["gesture_sent"] += 1
            else:
                results["gesture_failed"] += 1
                
        logger.info(f"배치 전송 완료: {results}")
        return results


# 전역 프록시 인스턴스
professor_proxy = ProfessorServerProxy()


async def send_to_professor_server(data_type: str, data: Any) -> bool:
    """교수님 서버로 데이터 전송하는 편의 함수"""
    if data_type == "sensor":
        return await professor_proxy.send_sensor_data(data)
    elif data_type == "gesture":
        return await professor_proxy.send_gesture_data(data)
    else:
        logger.error(f"알 수 없는 데이터 타입: {data_type}")
        return False 