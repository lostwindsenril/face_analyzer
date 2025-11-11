#!/usr/bin/env python3
"""
Visualization utilities.
"""

import cv2
import numpy as np
from math import cos, sin

# Visualization config (hardcoded)
VISUALIZATION_COLORS = {
    'bbox': (0, 255, 0),           # Green bounding box
    'left_eye': (255, 0, 0),       # Blue left eye
    'right_eye': (255, 0, 0),      # Blue right eye
    'nose': (0, 255, 255),         # Yellow nose
    'left_mouth': (0, 0, 255),     # Red left mouth corner
    'right_mouth': (0, 0, 255),    # Red right mouth corner
    'quadrilateral': (0, 255, 255), # Cyan quadrilateral
    'border': (255, 255, 255),     # White border
    'axis_x': (0, 0, 255),         # Red X axis
    'axis_y': (0, 255, 0),         # Green Y axis
    'axis_z': (255, 0, 0),         # Blue Z axis
    'pose_text': (255, 255, 0),    # Yellow pose text
    'time_text': (0, 0, 255),      # Red time text
    'score_text': (255, 255, 0),   # Yellow score text
    'deviation_text': (0, 255, 255), # Cyan deviation text
}

# Font config (hardcoded)
FONT_CONFIG = {
    'font': cv2.FONT_HERSHEY_SIMPLEX,
    'scale': 0.8,
    'thickness': 2,
    'large_scale': 2,
    'large_thickness': 3,
    'huge_scale': 3,
    'huge_thickness': 4
}


def draw_face_bbox(frame, bbox, confidence, face_id, color=None):
    """
    Draw face bounding box.

    Args:
        frame: Image frame
        bbox: Bounding box [x1, y1, x2, y2]
        confidence: Confidence score
        face_id: Face ID
        color: Color; defaults to configured color
    """
    color = color or VISUALIZATION_COLORS['bbox']
    x1, y1, x2, y2 = map(int, bbox)

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    # Show confidence and ID
    label = f"Face {face_id}: {confidence:.3f}"
    text_y = max(y1 - 10, 20)
    cv2.putText(
        frame, label, (x1, text_y),
        FONT_CONFIG['font'], 0.7, color, 2
    )


def draw_face_landmarks(frame, landmarks, x_offset=0, y_offset=0):
    """
    Draw facial landmarks.

    Args:
        frame: Image frame
        landmarks: Landmarks
        x_offset: X offset
        y_offset: Y offset
    """
    if landmarks is None or len(landmarks) < 10:
        return

    # Parse landmarks
    left_eye = (int(landmarks[0]) + x_offset, int(landmarks[1]) + y_offset)
    right_eye = (int(landmarks[2]) + x_offset, int(landmarks[3]) + y_offset)
    nose = (int(landmarks[4]) + x_offset, int(landmarks[5]) + y_offset)
    left_mouth = (int(landmarks[6]) + x_offset, int(landmarks[7]) + y_offset)
    right_mouth = (int(landmarks[8]) + x_offset, int(landmarks[9]) + y_offset)

    # Draw landmarks
    cv2.circle(frame, left_eye, 4, VISUALIZATION_COLORS['left_eye'], -1)
    cv2.circle(frame, right_eye, 4, VISUALIZATION_COLORS['right_eye'], -1)
    cv2.circle(frame, nose, 4, VISUALIZATION_COLORS['nose'], -1)
    cv2.circle(frame, left_mouth, 4, VISUALIZATION_COLORS['left_mouth'], -1)
    cv2.circle(frame, right_mouth, 4, VISUALIZATION_COLORS['right_mouth'], -1)

    # Add white border
    cv2.circle(frame, left_eye, 5, VISUALIZATION_COLORS['border'], 1)
    cv2.circle(frame, right_eye, 5, VISUALIZATION_COLORS['border'], 1)
    cv2.circle(frame, nose, 5, VISUALIZATION_COLORS['border'], 1)
    cv2.circle(frame, left_mouth, 5, VISUALIZATION_COLORS['border'], 1)
    cv2.circle(frame, right_mouth, 5, VISUALIZATION_COLORS['border'], 1)


def draw_face_quadrilateral(frame, landmarks, x_offset=0, y_offset=0):
    """
    Draw face quadrilateral.

    Args:
        frame: Image frame
        landmarks: Landmarks
        x_offset: X offset
        y_offset: Y offset
    """
    if landmarks is None or len(landmarks) < 10:
        return

    # Parse landmarks
    left_eye = (int(landmarks[0]) + x_offset, int(landmarks[1]) + y_offset)
    right_eye = (int(landmarks[2]) + x_offset, int(landmarks[3]) + y_offset)
    left_mouth = (int(landmarks[6]) + x_offset, int(landmarks[7]) + y_offset)
    right_mouth = (int(landmarks[8]) + x_offset, int(landmarks[9]) + y_offset)

    # Draw quadrilateral
    quad_points = np.array([left_eye, right_eye, right_mouth, left_mouth], np.int32)
    cv2.polylines(frame, [quad_points], True, VISUALIZATION_COLORS['quadrilateral'], 2)

    # Draw landmarks
    draw_face_landmarks(frame, landmarks, x_offset, y_offset)


def draw_geometry_validation_result(frame, is_valid_quad, nose_inside):
    """
    Draw geometry validation results in the top-right.

    Args:
        frame: Image frame
        is_valid_quad: Whether quad is valid
        nose_inside: Whether nose is inside
    """
    _, w = frame.shape[:2]

    # Show result in top-right
    result_text_1 = f"Quad: {'T' if is_valid_quad else 'F'}"
    result_text_2 = f"Nose: {'T' if nose_inside else 'F'}"

    # Compute text position (top-right)
    font = FONT_CONFIG['font']
    font_scale = FONT_CONFIG['scale']
    thickness = FONT_CONFIG['thickness']

    # Get text size
    (text_w1, text_h1), _ = cv2.getTextSize(result_text_1, font, font_scale, thickness)
    (text_w2, text_h2), _ = cv2.getTextSize(result_text_2, font, font_scale, thickness)

    # Compute positions
    margin = 20
    x1 = w - max(text_w1, text_w2) - margin
    y1 = margin + text_h1
    y2 = y1 + text_h2 + 10

    # Draw background rectangle
    bg_x1 = x1 - 10
    bg_y1 = margin - 5
    bg_x2 = w - margin + 5
    bg_y2 = y2 + 10
    cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)  # Black background

    # Draw text
    color1 = (0, 255, 0) if is_valid_quad else (0, 0, 255)  # Green/red
    color2 = (0, 255, 0) if nose_inside else (0, 0, 255)    # Green/red

    cv2.putText(frame, result_text_1, (x1, y1), font, font_scale, color1, thickness)
    cv2.putText(frame, result_text_2, (x1, y2), font, font_scale, color2, thickness)


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=100):
    """
    Draw 3D axes.

    Args:
        img: Image
        yaw: Yaw angle
        pitch: Pitch angle
        roll: Roll angle
        tdx: Center X
        tdy: Center Y
        size: Axis length
    """
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx is not None and tdy is not None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width // 2
        tdy = height // 2

    # X axis to the right (red)
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y axis in green
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z axis (out of screen) in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), VISUALIZATION_COLORS['axis_x'], 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), VISUALIZATION_COLORS['axis_y'], 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), VISUALIZATION_COLORS['axis_z'], 2)

    return img


def draw_pose_info(frame, pose_angles, bbox):
    """
    Draw pose info.

    Args:
        frame: Image frame
        pose_angles: Pose angles dict
        bbox: Face bounding box
    """
    if not pose_angles:
        return

    x1, y1, x2, y2 = map(int, bbox)

    # Draw 3D axes
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    bbox_height = y2 - y1
    axis_size = bbox_height // 3

    draw_axis(frame, pose_angles['yaw'], pose_angles['pitch'], pose_angles['roll'],
             tdx=center_x, tdy=center_y, size=axis_size)

    # Show pose text
    pose_text = f"Y:{pose_angles['yaw']:.1f} P:{pose_angles['pitch']:.1f} R:{pose_angles['roll']:.1f}"
    cv2.putText(frame, pose_text, (x1, y1 - 50),
               FONT_CONFIG['font'], 0.6, VISUALIZATION_COLORS['pose_text'], 2)


def draw_processing_info(frame, inference_time, attempt_count, geometry_score=None, deviation=None):
    """
    Draw processing info.

    Args:
        frame: Image frame
        inference_time: Inference time (ms)
        attempt_count: Attempt count
        geometry_score: Geometry score
        deviation: Face deviation
    """
    # Show inference time and attempts in top-left
    time_label = f"{inference_time:.1f} ms (#{attempt_count})"
    cv2.putText(frame, time_label, (30, 70),
                FONT_CONFIG['font'], FONT_CONFIG['large_scale'],
                VISUALIZATION_COLORS['time_text'], FONT_CONFIG['large_thickness'])

    # Show geometry score
    if geometry_score is not None:
        score_label = f"Geo Score: {geometry_score}/2"
        cv2.putText(frame, score_label, (30, 140),
                    FONT_CONFIG['font'], FONT_CONFIG['large_scale'],
                    VISUALIZATION_COLORS['score_text'], FONT_CONFIG['large_thickness'])

    # Show face deviation
    if deviation is not None:
        deviation_text = f"{deviation:.1f} deg"
        # Use larger font and vivid color for deviation
        text_size = cv2.getTextSize(deviation_text, FONT_CONFIG['font'],
                                   FONT_CONFIG['huge_scale'], FONT_CONFIG['huge_thickness'])[0]
        cv2.rectangle(frame, (25, 200), (35 + text_size[0], 250), (0, 0, 0), -1)  # Black background
        cv2.putText(frame, deviation_text, (30, 240),
                   FONT_CONFIG['font'], FONT_CONFIG['huge_scale'],
                   VISUALIZATION_COLORS['deviation_text'], FONT_CONFIG['huge_thickness'])


# ============================================================================
# High-level composite functions (merged from VisualizationHandler)
# ============================================================================

def draw_face_detection_result(frame, face_info, x_offset: int, y_offset: int):
    """
    Draw face detection results.

    Args:
        frame: Image frame
        face_info (dict): Face info
        x_offset (int): X offset
        y_offset (int): Y offset

    Returns:
        tuple: Adjusted bounding box
    """
    bbox = face_info['bbox']
    confidence = face_info['confidence']
    landmarks = face_info['landmarks']
    face_id = face_info['id']

    # Adjust coordinates to original frame
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1, x2, y2 = x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset
    adjusted_bbox = (x1, y1, x2, y2)

    # Draw face bounding box
    draw_face_bbox(frame, adjusted_bbox, confidence, face_id)

    # Draw face quadrilateral and landmarks
    draw_face_quadrilateral(frame, landmarks, x_offset, y_offset)

    return adjusted_bbox


def draw_complete_result(frame, face_info, x_offset: int, y_offset: int,
                        is_valid_quad: bool, nose_inside: bool, inference_time: float,
                        attempt_count: int, geometry_score: int, pose_angles=None, deviation=None):
    """
    Draw complete detection and validation results.

    Args:
        frame: Image frame
        face_info (dict): Face info
        x_offset (int): X offset
        y_offset (int): Y offset
        is_valid_quad (bool): Whether quad is valid
        nose_inside (bool): Whether nose is inside
        inference_time (float): Inference time
        attempt_count (int): Attempt count
        geometry_score (int): Geometry score
        pose_angles (dict): Pose angles dict
        deviation (float): Face deviation

    Returns:
        tuple: Adjusted bounding box
    """
    # Draw face detection results
    adjusted_bbox = draw_face_detection_result(frame, face_info, x_offset, y_offset)

    # Draw geometry validation results
    draw_geometry_validation_result(frame, is_valid_quad, nose_inside)

    # Draw processing info
    draw_processing_info(frame, inference_time, attempt_count, geometry_score, deviation)

    # Draw pose info (if available)
    if pose_angles is not None:
        draw_pose_info(frame, pose_angles, adjusted_bbox)

    return adjusted_bbox
