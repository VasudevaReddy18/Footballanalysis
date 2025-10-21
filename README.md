# ⚽ Football Match Analysis using YOLO and Object Tracking

This project is a **computer vision-based football match analytics system** that automatically detects and tracks players, referees, and the ball in a football match video.  
It uses **YOLO (You Only Look Once)** for object detection and **ByteTrack** for tracking players across frames.  
The system visualizes **player movements, ball possession percentages**, and team color identification — creating analytics similar to professional sports broadcasting tools.

---

## 🚀 Features

- **Player, referee, and ball detection** using YOLOv8  
- **Multi-object tracking** using ByteTrack  
- **Team color identification** via K-Means clustering  
- **Ball possession estimation** (which team has the ball)  
- **Video annotation** with overlays and stats  
- **Output video generation** with all analytics displayed  

---

## 🧠 Project Architecture

1. **Frame Extraction:**  
   The input video is split into frames using OpenCV.  

2. **Object Detection:**  
   YOLOv8 detects players, referees, and the ball in each frame.  

3. **Object Tracking:**  
   ByteTrack assigns consistent IDs to players across frames to maintain continuity.  

4. **Team Assignment:**  
   Players are grouped into teams using K-Means clustering on their jersey colors.  

5. **Ball Possession Logic:**  
   The distance between each player and the ball is computed to estimate which team has control.  

6. **Visualization:**  
   Each frame is annotated with player markers, track IDs, possession indicators, and team stats.  

7. **Output:**  
   The processed frames are recompiled into a video showing all annotations.  

---

## 🛠️ Tech Stack

- **Language:** Python  
- **Deep Learning Framework:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)  
- **Tracking Library:** [Supervision + ByteTrack](https://github.com/tryolabs/supervision)  
- **Other Libraries:**  
  - OpenCV (frame processing and visualization)  
  - NumPy & Pandas (data manipulation)  
  - scikit-learn (K-Means clustering)  
  - pickle (track caching)

---

## 📂 Project Structure

