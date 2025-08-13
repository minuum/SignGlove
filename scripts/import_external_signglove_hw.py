#!/usr/bin/env python3
"""
SignGlove_HW CSV → SignGlove 통합 스키마 변환 스크립트

지원 입력:
- WiFi CSV: ['timestamp(ms)', 'ax(g)', 'ay(g)', 'az(g)', 'pitch(°)', 'roll(°)', 'yaw(°)']
- UART CSV: ['timestamp(ms)', 'pitch(°)', 'roll(°)', 'yaw(°)', 'flex1', 'flex2', 'flex3', 'flex4', 'flex5']

출력 스키마(통합):
- timestamp_iso, timestamp_ms, device_id, source_repo, source_file
- flex_1..flex_5 (없으면 빈값)
- accel_x, accel_y, accel_z (m/s2, g→m/s2 변환)
- orientation_pitch, orientation_roll, orientation_yaw (deg)

사용법:
  python scripts/import_external_signglove_hw.py --input external/SignGlove_HW/imu_madgwick_*.csv
  python scripts/import_external_signglove_hw.py --dir external/SignGlove_HW
"""

import argparse
import glob
import os
from pathlib import Path
from typing import List

import pandas as pd

G_TO_MS2 = 9.80665


def detect_format(df: pd.DataFrame) -> str:
    """입력 CSV 포맷 감지"""
    cols = [c.strip().lower() for c in df.columns]
    if 'ax(g)' in df.columns or 'ax(g)' in [c for c in df.columns]:
        return 'wifi_imu'
    if 'flex1' in cols and 'pitch(°)' in [c.lower() for c in df.columns]:
        return 'uart_imu_flex'
    # 헤더가 없는 경우 길이로 추정
    if len(cols) == 7:
        return 'wifi_imu_noheader'
    if len(cols) == 9:
        return 'uart_imu_flex_noheader'
    return 'unknown'


def load_csv(input_file: Path) -> pd.DataFrame:
    """CSV 파일 로드 (인코딩 문제 대응)"""
    encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1']
    
    for encoding in encodings:
        try:
            # 헤더 있는 방식 시도
            df = pd.read_csv(input_file, encoding=encoding)
            return df
        except UnicodeDecodeError:
            continue
        except Exception:
            try:
                # 헤더 없는 방식 시도
                df = pd.read_csv(input_file, header=None, encoding=encoding)
                return df
            except Exception:
                continue
    
    raise RuntimeError(f"CSV 로드 실패: {input_file} (모든 인코딩 실패)")


def convert_wifi_imu(df: pd.DataFrame, input_file: Path) -> pd.DataFrame:
    # 기대 헤더: timestamp(ms), ax(g), ay(g), az(g), pitch(°), roll(°), yaw(°)
    cols = [c.strip() for c in df.columns]
    if len(cols) == 7 and cols[0].lower().startswith('timestamp'):
        df.columns = ['timestamp_ms', 'ax_g', 'ay_g', 'az_g', 'pitch_deg', 'roll_deg', 'yaw_deg']
    else:
        # 무헤더일 가능성
        df = df.rename(columns={0: 'timestamp_ms', 1: 'ax_g', 2: 'ay_g', 3: 'az_g', 4: 'pitch_deg', 5: 'roll_deg', 6: 'yaw_deg'})

    # 단위 변환 및 통합 스키마 구성
    df['timestamp_ms'] = pd.to_numeric(df['timestamp_ms'], errors='coerce')
    df['timestamp_iso'] = pd.to_datetime(df['timestamp_ms'], unit='ms', errors='coerce')
    df['accel_x'] = pd.to_numeric(df['ax_g'], errors='coerce') * G_TO_MS2
    df['accel_y'] = pd.to_numeric(df['ay_g'], errors='coerce') * G_TO_MS2
    df['accel_z'] = pd.to_numeric(df['az_g'], errors='coerce') * G_TO_MS2
    df['orientation_pitch'] = pd.to_numeric(df['pitch_deg'], errors='coerce')
    df['orientation_roll'] = pd.to_numeric(df['roll_deg'], errors='coerce')
    df['orientation_yaw'] = pd.to_numeric(df['yaw_deg'], errors='coerce')

    # 누락 열 생성
    for k in ['flex_1', 'flex_2', 'flex_3', 'flex_4', 'flex_5']:
        df[k] = pd.NA

    df['device_id'] = 'SIGNGLOVE_HW_WIFI'
    df['source_repo'] = 'SignGlove_HW'
    df['source_file'] = str(input_file)

    return df[['timestamp_iso', 'timestamp_ms', 'device_id', 'source_repo', 'source_file',
               'flex_1', 'flex_2', 'flex_3', 'flex_4', 'flex_5',
               'accel_x', 'accel_y', 'accel_z',
               'orientation_pitch', 'orientation_roll', 'orientation_yaw']]


def convert_uart_imu_flex(df: pd.DataFrame, input_file: Path) -> pd.DataFrame:
    # 기대 헤더: timestamp(ms), pitch(°), roll(°), yaw(°), flex1..flex5
    if len(df.columns) == 9 and isinstance(df.columns[0], str):
        df.columns = ['timestamp_ms', 'pitch_deg', 'roll_deg', 'yaw_deg', 'flex_1', 'flex_2', 'flex_3', 'flex_4', 'flex_5']
    else:
        df = df.rename(columns={0: 'timestamp_ms', 1: 'pitch_deg', 2: 'roll_deg', 3: 'yaw_deg',
                                 4: 'flex_1', 5: 'flex_2', 6: 'flex_3', 7: 'flex_4', 8: 'flex_5'})

    df['timestamp_ms'] = pd.to_numeric(df['timestamp_ms'], errors='coerce')
    df['timestamp_iso'] = pd.to_datetime(df['timestamp_ms'], unit='ms', errors='coerce')

    # 가속도/자이로 원시 값 없음 → 결측치로 두거나 후처리에서 파생 가능
    for k in ['accel_x', 'accel_y', 'accel_z']:
        df[k] = pd.NA

    df['orientation_pitch'] = pd.to_numeric(df['pitch_deg'], errors='coerce')
    df['orientation_roll'] = pd.to_numeric(df['roll_deg'], errors='coerce')
    df['orientation_yaw'] = pd.to_numeric(df['yaw_deg'], errors='coerce')

    # 플렉스 정수화
    for k in ['flex_1', 'flex_2', 'flex_3', 'flex_4', 'flex_5']:
        df[k] = pd.to_numeric(df[k], errors='coerce')

    df['device_id'] = 'SIGNGLOVE_HW_UART'
    df['source_repo'] = 'SignGlove_HW'
    df['source_file'] = str(input_file)

    return df[['timestamp_iso', 'timestamp_ms', 'device_id', 'source_repo', 'source_file',
               'flex_1', 'flex_2', 'flex_3', 'flex_4', 'flex_5',
               'accel_x', 'accel_y', 'accel_z',
               'orientation_pitch', 'orientation_roll', 'orientation_yaw']]


def convert_file(input_path: Path, output_dir: Path) -> Path:
    df = load_csv(input_path)
    fmt = detect_format(df)
    if fmt == 'wifi_imu' or fmt == 'wifi_imu_noheader':
        converted = convert_wifi_imu(df, input_path)
    elif fmt == 'uart_imu_flex' or fmt == 'uart_imu_flex_noheader':
        converted = convert_uart_imu_flex(df, input_path)
    else:
        raise RuntimeError(f"지원하지 않는 포맷: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"converted_{input_path.stem}.csv"
    converted.to_csv(out_file, index=False)
    return out_file


def main():
    parser = argparse.ArgumentParser(description='SignGlove_HW CSV → 통합 스키마 변환기')
    parser.add_argument('--input', type=str, help='단일 파일 경로(glob 가능)')
    parser.add_argument('--dir', type=str, help='디렉토리 내 모든 CSV 변환')
    parser.add_argument('--output', type=str, default='data/processed/external_converted', help='출력 디렉토리')
    args = parser.parse_args()

    output_dir = Path(args.output)
    converted_files: List[Path] = []

    if args.input:
        for fp in glob.glob(args.input):
            converted_files.append(convert_file(Path(fp), output_dir))
    elif args.dir:
        for fp in Path(args.dir).glob('*.csv'):
            converted_files.append(convert_file(fp, output_dir))
    else:
        parser.error('하나 이상의 입력 옵션이 필요합니다: --input 또는 --dir')

    print('✅ 변환 완료:')
    for f in converted_files:
        print(' -', f)


if __name__ == '__main__':
    main()


