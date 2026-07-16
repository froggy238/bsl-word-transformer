"""Landmark layout constants for the 105-point BSL skeleton.

MediaPipe Holistic emits 33 pose landmarks, 21 landmarks per hand and a
468-point face mesh. This project keeps a fixed 105-landmark subset per
frame, stored as a single (105, 3) array (x, y, z) in this block order:

    [0:33]    pose        (POSE_SLICE)      -- MediaPipe pose indices 0-32
    [33:54]   left hand   (LEFT_HAND_SLICE) -- MediaPipe left-hand 0-20
    [54:75]   right hand  (RIGHT_HAND_SLICE)-- MediaPipe right-hand 0-20
    [75:105]  mouth       (MOUTH_SLICE)     -- 30 face-mesh points listed
                                               in MOUTH_FACE_INDICES

Mapping from Holistic outputs to the subset: pose and hand blocks copy the
corresponding solution landmark lists verbatim; the mouth block gathers the
face-mesh points ``face_landmarks[i] for i in MOUTH_FACE_INDICES`` in order
(20 outer-lip points followed by 10 inner-lip points).

Only the mouth region of the face mesh is retained because mouth patterns
(mouthings and mouth gestures) carry discriminative information in BSL,
while including the full 468-point mesh would roughly triple the feature
dimensionality with mostly static, identity-related points.

With x, y, z per landmark a frame flattens to 315 features
(FEATURES_PER_FRAME).
"""

POSE_N = 33
HAND_N = 21
MOUTH_N = 30
N_LANDMARKS = POSE_N + 2 * HAND_N + MOUTH_N  # 105
N_COORDS = 3
FEATURES_PER_FRAME = N_LANDMARKS * N_COORDS  # 315

POSE_SLICE = slice(0, 33)
LEFT_HAND_SLICE = slice(33, 54)
RIGHT_HAND_SLICE = slice(54, 75)
MOUTH_SLICE = slice(75, 105)

# Indices into the 105-landmark array (pose block comes first, so these
# coincide with MediaPipe's pose indices for the shoulders).
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

# Fixed indices into MediaPipe's 468-point face mesh:
# 20 outer-lip points followed by 10 inner-lip points.
MOUTH_FACE_INDICES: list[int] = [
    # Outer lip contour (20 points).
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
    # Inner lip contour (10 points).
    78, 88, 87, 317, 318, 308, 310, 312, 82, 80,
]

assert len(MOUTH_FACE_INDICES) == MOUTH_N
