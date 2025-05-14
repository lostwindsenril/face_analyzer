#!/usr/bin/env python3
"""
RetinaFace detection result data class.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np


class RetinaFaceResult:
    """RetinaFace single-face detection result class."""

    def __init__(self, face_id: int, confidence: float, bbox: List[float],
                 landmarks: Optional[np.ndarray] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a RetinaFace detection result.

        Args:
            face_id (int): Face ID (index within the current frame)
            confidence (float): Detection confidence
            bbox (List[float]): Bounding box info [x1, y1, x2, y2]
            landmarks (Optional[np.ndarray]): Landmark info (5 points, each with x, y)
            metadata (Optional[Dict[str, Any]]): Extra metadata
        """
        self.face_id = face_id
        self.confidence = confidence
        self.bbox = bbox
        self.landmarks = landmarks
        self.metadata = metadata

        # Compute and store commonly used attributes.
        self.x1 = self.bbox[0]
        self.y1 = self.bbox[1]
        self.x2 = self.bbox[2]
        self.y2 = self.bbox[3]
        self.width = self.x2 - self.x1
        self.height = self.y2 - self.y1
        self.area = self.width * self.height
        self.center = ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def get_landmark_points(self) -> Optional[Dict[str, Tuple[float, float]]]:
        """
        Get landmark points as a dict.

        Returns:
            Dict[str, Tuple[float, float]]: Landmark dict with left_eye, right_eye,
                nose, left_mouth, right_mouth
        """
        if self.landmarks is None or len(self.landmarks) < 10:
            return None
        
        # RetinaFace returns 5 keypoints: left/right eye, nose, left/right mouth corner.
        return {
            'left_eye': (float(self.landmarks[0]), float(self.landmarks[1])),
            'right_eye': (float(self.landmarks[2]), float(self.landmarks[3])),
            'nose': (float(self.landmarks[4]), float(self.landmarks[5])),
            'left_mouth': (float(self.landmarks[6]), float(self.landmarks[7])),
            'right_mouth': (float(self.landmarks[8]), float(self.landmarks[9]))
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dict format.

        Returns:
            Dict[str, Any]: Dict representation
        """
        result = {
            'face_id': self.face_id,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'width': self.width,
            'height': self.height,
            'area': self.area,
            'center': self.center
        }
        
        if self.landmarks is not None:
            result['landmarks'] = self.landmarks.tolist()
            result['landmark_points'] = self.get_landmark_points()
        
        if self.metadata is not None:
            result['metadata'] = self.metadata
            
        return result


def create_retinaface_list(detection_result):
    """
    Convert detection results to a list of RetinaFaceResult instances.

    Args:
        detection_result: RetinaFace detection result

    Returns:
        List[RetinaFaceResult]: List of RetinaFaceResult instances
    """
    face_results = []

    if detection_result is not None and len(detection_result) == 2:
        dets, landms = detection_result

        if dets is not None and len(dets) > 0:
            for i in range(len(dets)):
                det = dets[i]
                if len(det) < 5:
                    continue

                x1, y1, x2, y2, confidence = det[:5]

                # Fetch corresponding landmarks.
                landmarks = None
                if landms is not None and i < len(landms):
                    landmarks = landms[i]

                if landmarks is not None:
                    # Create a RetinaFaceResult instance via the standard __init__.
                    face_result = RetinaFaceResult(
                        face_id=i + 1,
                        confidence=float(confidence),
                        bbox=[float(x1), float(y1), float(x2), float(y2)],
                        landmarks=landmarks
                    )
                    face_results.append(face_result)

    return face_results
