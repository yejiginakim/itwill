#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1 — Face Landmarks (MediaPipe FaceMesh)

역할: 사진/폴더/웹캠/동영상에서 얼굴을 탐지하고 468(또는 478)개 랜드마크를 추출.
산출물: CSV/JSON(픽셀 좌표, 정규화 좌표) + 핵심 포인트(양눈 중심, 턱끝, 눈동자 반경 등)

환경 권장:
- mediapipe==0.10.21
- opencv-python>=4.9
- numpy>=1.24

사용 예시:
1) 단일 이미지 → JSON/CSV 저장
   python step1_face_landmarks_mediapipe.py \
     --image '/path/to/face.jpg' \
     --out_dir './out_landmarks'

2) 폴더 일괄 처리 (jpg/png/webp)
   python step1_face_landmarks_mediapipe.py \
     --dir '/path/to/faces' \
     --out_dir './out_landmarks' \
     --exts jpg png webp

3) 동영상 파일에서 추출(매 프레임 또는 N프레임 간격 저장)
   python step1_face_landmarks_mediapipe.py \
     --video '/path/to/video.mp4' \
     --out_dir './out_landmarks' \
     --frame_stride 5

4) 웹캠 미리보기(키보드 q 종료, s로 현재 프레임 저장)
   python step1_face_landmarks_mediapipe.py --webcam 0 --preview 1 --out_dir './out_landmarks'

출력 파일:
- *_landmarks.json : 전체 랜드마크(정규화+픽셀)
- *_landmarks.csv  : 픽셀 좌표만 테이블로 저장
- *_preview.jpg    : 미리보기용 랜드마크 오버레이 이미지(옵션)
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

# MediaPipe는 런타임 시 import (설치가 안 된 환경을 위해 메시지 친절화)
try:
    import mediapipe as mp
except Exception as e:  # pragma: no cover
    raise SystemExit('❌ mediapipe import 실패. `pip install mediapipe` 후 다시 실행하세요.')

# ===== 유틸 =====

def to_int_point(x: float, y: float) -> Tuple[int, int]:
    return int(round(x)), int(round(y))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ===== FaceMesh Wrapper =====

class FaceMeshExtractor:
    def __init__(self,
                 static_image_mode: bool = True,
                 max_num_faces: int = 1,
                 refine_landmarks: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.draw = mp.solutions.drawing_utils
        self.styles = mp.solutions.drawing_styles

        # 연결 세트(윤곽/눈/입 등)
        self.CONN_TESSEL = self.mp_face_mesh.FACEMESH_TESSELATION
        self.CONN_CONTOURS = self.mp_face_mesh.FACEMESH_CONTOURS
        self.CONN_IRISES = getattr(self.mp_face_mesh, 'FACEMESH_IRISES', frozenset())

    def process_rgb(self, rgb: np.ndarray):
        return self.face_mesh.process(rgb)

    @staticmethod
    def _normalized_to_pixel(landmark, img_w: int, img_h: int) -> Tuple[float, float, float]:
        x = landmark.x * img_w
        y = landmark.y * img_h
        z = landmark.z  # z는 상대값(카메라에서의 깊이 스케일)
        return x, y, z

    def extract_from_image(self, bgr: np.ndarray) -> Optional[Dict]:
        if bgr is None:
            return None
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self.process_rgb(rgb)
        if not res.multi_face_landmarks:
            return None

        # 단일 얼굴만 사용 (max_num_faces=1 기본)
        lms = res.multi_face_landmarks[0].landmark
        n = len(lms)
        px = np.zeros((n, 3), dtype=np.float32)
        for i, lm in enumerate(lms):
            px[i] = self._normalized_to_pixel(lm, w, h)

        out: Dict = {
            'image_width': int(w),
            'image_height': int(h),
            'num_landmarks': int(n),
            'landmarks_px': px.tolist(),  # [[x,y,z], ...]
            'landmarks_norm': [[float(lms[i].x), float(lms[i].y), float(lms[i].z)] for i in range(n)],
        }

        # ===== 파생 특징(눈 중심/반경, 턱끝 등) =====
        # refine_landmarks=True 일 때 홍채(iris) 5포인트씩 추가 → 중심/반경 추정에 유리
        left_iris_ids = list(range(473, 478)) if n >= 478 else []
        right_iris_ids = list(range(468, 473)) if n >= 478 else []

        def mean_point(ids: List[int]) -> Optional[Tuple[float, float]]:
            if not ids:
                return None
            arr = px[ids, :2]
            return float(arr[:, 0].mean()), float(arr[:, 1].mean())

        def iris_radius(ids: List[int], center_xy: Tuple[float, float]) -> Optional[float]:
            if not ids or center_xy is None:
                return None
            cx, cy = center_xy
            arr = px[ids, :2]
            d = np.sqrt(((arr[:, 0] - cx) ** 2) + ((arr[:, 1] - cy) ** 2))
            return float(np.median(d))

        left_center = mean_point(left_iris_ids)
        right_center = mean_point(right_iris_ids)
        left_r = iris_radius(left_iris_ids, left_center) if left_center else None
        right_r = iris_radius(right_iris_ids, right_center) if right_center else None

        # 턱끝(일반적으로 152번이 턱끝에 해당)
        chin_idx = 152 if n > 152 else None
        chin_xy = None
        if chin_idx is not None:
            cx, cy = px[chin_idx, 0], px[chin_idx, 1]
            chin_xy = (float(cx), float(cy))

        # 코 팁(라이브러리 버전에 따라 다를 수 있어 보수적으로 선택)
        # 방법: 코 영역 중 카메라에 가장 가까운 점(z가 가장 작은)을 코팁 근사값으로 사용
        nose_conns = getattr(self.mp_face_mesh, 'FACEMESH_NOSE', frozenset())
        nose_ids = sorted({i for pair in nose_conns for i in pair}) if nose_conns else list(range(1, min(20, n)))
        nose_tip_xy = None
        if nose_ids:
            zvals = px[nose_ids, 2]
            k = int(np.argmin(zvals))
            nose_tip_xy = (float(px[nose_ids[k], 0]), float(px[nose_ids[k], 1]))

        out['derived'] = {
            'left_iris_center_xy': left_center,
            'right_iris_center_xy': right_center,
            'left_iris_radius_px': left_r,
            'right_iris_radius_px': right_r,
            'chin_xy': chin_xy,
            'nose_tip_xy': nose_tip_xy,
        }
        return out

    def draw_overlay(self, bgr: np.ndarray, result_dict: Dict, draw_tessel: bool = False) -> np.ndarray:
        vis = bgr.copy()
        h, w = vis.shape[:2]

        # 랜드마크 점 찍기
        pts = np.array(result_dict['landmarks_px'], dtype=np.float32)
        for i in range(pts.shape[0]):
            x, y = to_int_point(pts[i, 0], pts[i, 1])
            cv2.circle(vis, (x, y), 1, (0, 255, 0), -1)

        # 컨투어/홍채 라인(선택)
        if draw_tessel:
            # Tessellation을 직접 그리려면 연결 정보를 이용해 선분을 그림
            for a, b in self.CONN_TESSEL:
                ax, ay = to_int_point(pts[a, 0], pts[a, 1])
                bx, by = to_int_point(pts[b, 0], pts[b, 1])
                cv2.line(vis, (ax, ay), (bx, by), (80, 80, 80), 1)

        # 핵심 포인트 시각화
        d = result_dict.get('derived', {})
        def draw_keypoint(name: str, xy: Optional[Tuple[float, float]]):
            if not xy:
                return
            x, y = to_int_point(xy[0], xy[1])
            cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)
            cv2.putText(vis, name, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

        draw_keypoint('L_iris', d.get('left_iris_center_xy'))
        draw_keypoint('R_iris', d.get('right_iris_center_xy'))
        draw_keypoint('Chin', d.get('chin_xy'))
        draw_keypoint('Nose', d.get('nose_tip_xy'))

        return vis


# ===== I/O Helpers =====

def save_outputs(base: Path, result: Dict, save_preview: bool, bgr: Optional[np.ndarray] = None, extractor: Optional[FaceMeshExtractor] = None):
    base.parent.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(base.with_suffix('.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # CSV (픽셀좌표만)
    pts = np.array(result['landmarks_px'], dtype=np.float32)
    header = 'idx,x_px,y_px,z_rel\n'
    lines = [header] + [f'{i},{pts[i,0]:.3f},{pts[i,1]:.3f},{pts[i,2]:.5f}\n' for i in range(pts.shape[0])]
    with open(base.with_suffix('.csv'), 'w', encoding='utf-8') as f:
        f.writelines(lines)

    # Preview 이미지
    if save_preview and bgr is not None and extractor is not None:
        vis = extractor.draw_overlay(bgr, result, draw_tessel=False)
        cv2.imwrite(str(base.with_suffix('.preview.jpg')), vis)


# ===== 메인 파이프라인 =====

def process_image_file(img_path: Path, out_dir: Path, extractor: FaceMeshExtractor, save_preview: bool = True) -> Optional[Path]:
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        print(f'⚠️  이미지 로드 실패: {img_path}')
        return None
    res = extractor.extract_from_image(bgr)
    if res is None:
        print(f'⚠️  얼굴 미검출: {img_path}')
        return None
    base = out_dir / f'{img_path.stem}_landmarks'
    save_outputs(base, res, save_preview=save_preview, bgr=bgr, extractor=extractor)
    print(f'✅ 저장: {base.with_suffix(".json").name}, {base.with_suffix(".csv").name}')
    return base


def process_dir(dir_path: Path, out_dir: Path, exts: List[str], extractor: FaceMeshExtractor, save_preview: bool = True):
    patterns = [f'**/*.{ext.lower()}' for ext in exts]
    files: List[Path] = []
    for pat in patterns:
        files.extend(sorted(dir_path.glob(pat)))

    if not files:
        print('⚠️  처리할 이미지가 없습니다.')
        return
    for p in files:
        process_image_file(p, out_dir, extractor, save_preview=save_preview)


def process_video(video_path: Path, out_dir: Path, extractor: FaceMeshExtractor, frame_stride: int = 1, save_preview: bool = False):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f'❌ 비디오 열기 실패: {video_path}')

    idx = 0
    saved = 0
    ensure_dir(out_dir)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % max(1, frame_stride) != 0:
            idx += 1
            continue
        res = extractor.extract_from_image(frame)
        if res:
            frame_name = f'{video_path.stem}_f{idx:06d}_landmarks'
            base = out_dir / frame_name
            save_outputs(base, res, save_preview=save_preview, bgr=frame, extractor=extractor)
            saved += 1
        idx += 1
    cap.release()
    print(f'✅ 프레임 저장 수: {saved}')


def run_webcam(cam_index: int, extractor: FaceMeshExtractor, preview: bool, out_dir: Optional[Path] = None):
    cap = cv2.VideoCapture(int(cam_index))
    if not cap.isOpened():
        raise SystemExit('❌ 웹캠 열기 실패')

    ensure_dir(out_dir) if out_dir else None
    print("[q] 종료, [s] 현재 프레임 저장")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = extractor.extract_from_image(frame)
        vis = frame.copy()
        if res:
            vis = extractor.draw_overlay(frame, res, draw_tessel=False)
        cv2.imshow('FaceMesh', vis if preview else frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s') and res and out_dir:
            base = out_dir / 'webcam_snapshot_landmarks'
            save_outputs(base, res, save_preview=True, bgr=frame, extractor=extractor)
            print('💾 스냅샷 저장')

    cap.release()
    cv2.destroyAllWindows()


# ===== CLI =====

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='MediaPipe FaceMesh Landmarks Extractor')
    p.add_argument('--image', type=str, help='단일 이미지 경로')
    p.add_argument('--dir', type=str, help='일괄 처리할 폴더 경로')
    p.add_argument('--exts', nargs='*', default=['jpg', 'jpeg', 'png', 'webp'], help='폴더 처리 시 허용 확장자')
    p.add_argument('--video', type=str, help='동영상 파일 경로')
    p.add_argument('--frame_stride', type=int, default=1, help='비디오 처리 시 프레임 간격')
    p.add_argument('--webcam', type=int, help='웹캠 인덱스 (예: 0)')
    p.add_argument('--preview', type=int, default=1, help='미리보기 오버레이 그리기 (1/0)')
    p.add_argument('--out_dir', type=str, default='./out_landmarks', help='출력 폴더')
    p.add_argument('--static', type=int, default=1, help='static_image_mode (1=정지 이미지, 0=스트리밍)')
    p.add_argument('--max_faces', type=int, default=1, help='최대 얼굴 수')
    p.add_argument('--refine', type=int, default=1, help='refine_landmarks (홍채 포함)')
    p.add_argument('--det_conf', type=float, default=0.5, help='min_detection_confidence')
    p.add_argument('--trk_conf', type=float, default=0.5, help='min_tracking_confidence')
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    extractor = FaceMeshExtractor(
        static_image_mode=bool(args.static),
        max_num_faces=int(args.max_faces),
        refine_landmarks=bool(args.refine),
        min_detection_confidence=float(args.det_conf),
        min_tracking_confidence=float(args.trk_conf),
    )  

    has_job = False
    if args.image:
        has_job = True
        process_image_file(Path(args.image), out_dir, extractor, save_preview=bool(args.preview))

    if args.dir:
        has_job = True
        process_dir(Path(args.dir), out_dir, [ext.lower() for ext in args.exts], extractor, save_preview=bool(args.preview))

    if args.video:
        has_job = True
        process_video(Path(args.video), out_dir, extractor, frame_stride=int(args.frame_stride), save_preview=bool(args.preview))

    if args.webcam is not None:
        has_job = True
        run_webcam(int(args.webcam), extractor, preview=bool(args.preview), out_dir=out_dir)

    if not has_job:
        print('아래 중 하나를 지정하세요: --image, --dir, --video, --webcam')


if __name__ == '__main__':
    main()
