# DT2119 Project - Swedish Sign Language Recognition Pipeline

---

## 1. Data pre-processing

## 2. Feature Extraction

**Objective:** Extract normalized 3D hand and face features from images or videos using MediaPipe.

### 2.1 Landmark Detection

We use MediaPipe to extract 3D landmarks:

- **Hands:** `MediaPipe Hands` detects up to two hands (left/right), with 21 landmarks per hand.
- **Face:** `MediaPipe FaceMesh` detects 468 facial landmarks.

Each landmark is represented by its 3D position `(x, y, z)` in image-relative coordinates.

### 2.2 Normalization

To ensure robustness against scale and translation differences, landmarks are normalized independently for hands and face:

#### Hands

For each detected hand:

- Identify the `wrist` and the base of 4 fingers (index, middle, ring, pinky).
- Compute the `hand size` as the average `distance` between the wrist and these 4 finger bases.
- Translate landmarks to set the wrist as the origin.
- Scale coordinates by dividing by the hand size.
- Compute and append the gravity `center` of the hand as a 22nd landmark.
- If both hands are detected:
  - Compute the `distance` between hand centers.
  - Normalize this distance by the average hand size.

If a hand is not detected, its features are replaced by zeros.

#### Face

- Compute the `center` of all landmarks (gravity center).
- Subtract the center to re-center the face.
- Normalize by the standard deviation of all landmark coordinates.
- Append the center as a 469th landmark.

If the face is not detected, all values are set to zero.

### 2.3 Feature Vector

The final feature vector per frame consists of:

- 2 hands × 22 landmarks × 3 dimensions = **132 values**
- 1 inter-hand distance = **1 value**
- 469 facial landmarks × 3 dimensions = **1407 values**

**Total per frame:** **1540 features**

### Supported Input

- Single image (`data_type='image'`)
- Video (`data_type='video'`): returns a matrix of shape `(1540, nb_frames)`, where each column is a frame’s feature vector.

## 3. Sequential model
