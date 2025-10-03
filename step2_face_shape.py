#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from pathlib import Path
import numpy as np

# Key landmark indices (MediaPipe FaceMesh)
# Cheek (outer near eyes): 234 (left), 454 (right)
# Chin tip: 152
# Forehead (approx): 10
# Jawline approx (left/right): 205, 425
L_CHEEK, R_CHEEK = 234, 454
CHIN, FOREHEAD = 152, 10
L_JAW, R_JAW = 205, 425  # approximate left/right jawline points

def load_pts(json_path):
    data = json.loads(Path(json_path).read_text(encoding='utf-8'))
    pts = np.array(data['landmarks_px'], dtype=np.float32)  # (n,3) [x,y,z]
    return pts, data

def euclid(p, q):
    return float(np.linalg.norm(np.asarray(p) - np.asarray(q)))

def face_shape_rules(pts):
    # Width/height and derived ratios
    face_w = euclid(pts[L_CHEEK, :2], pts[R_CHEEK, :2])
    face_h = euclid(pts[FOREHEAD, :2], pts[CHIN, :2])

    # Cheek width (= face_w), jaw width approx using lower jawline points
    jaw_w = euclid(pts[L_JAW, :2], pts[R_JAW, :2])

    ratio_wh = face_w / (face_h + 1e-6)        # width / height
    ratio_jaw_cheek = jaw_w / (face_w + 1e-6)  # jaw width / cheek width

    # Jawline sharpness: angle between vectors (mid-jaw→chin) vs (mid-jaw→forehead)
    mid_jaw = (pts[L_JAW, :2] + pts[R_JAW, :2]) / 2.0
    v_chin = pts[CHIN, :2] - mid_jaw
    v_up   = pts[FOREHEAD, :2] - mid_jaw
    def angle_deg(v1, v2):
        a = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-6)
        a = np.clip(a, -1.0, 1.0)
        return float(np.degrees(np.arccos(a)))
    jaw_angle = angle_deg(v_chin, v_up)  # smaller → sharper jaw

    # Simple heuristic rules (tune thresholds on your photos)
    if ratio_wh < 0.80 and ratio_jaw_cheek < 0.80 and jaw_angle < 55:
        shape = 'Heart'
    elif 0.80 <= ratio_wh <= 0.95 and ratio_jaw_cheek < 0.85:
        shape = 'Oval'
    elif 0.95 < ratio_wh < 1.10 and ratio_jaw_cheek >= 0.90:
        shape = 'Round'
    elif ratio_wh >= 1.10 and ratio_jaw_cheek >= 0.90:
        shape = 'Square'
    else:
        shape = 'Oblong'

    return shape, {
        'face_w': round(face_w, 3),
        'face_h': round(face_h, 3),
        'ratio_wh': round(ratio_wh, 4),
        'jaw_w': round(jaw_w, 3),
        'ratio_jaw_cheek': round(ratio_jaw_cheek, 4),
        'jaw_angle_deg': round(jaw_angle, 2)
    }

if __name__ == '__main__':
    # Update the example path as needed
    lm_json = '/Users/yeji_kim/PycharmProjects/sunglasses_fit/out_landmarks/wonpil_landmarks.json'
    pts, meta = load_pts(lm_json)
    shape, metrics = face_shape_rules(pts)
    out = {'face_shape': shape, 'metrics': metrics}
    out_path = Path(lm_json).with_suffix('.faceshape.json')
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Face Shape:', shape)
    print('Metrics:', metrics)
    print('Saved to:', str(out_path))

