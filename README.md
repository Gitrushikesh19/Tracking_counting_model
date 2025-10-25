# Tracking, counting and Data Logging System
It is a real-time intelligent vehicle tracking and counting system designed to analyze traffic movement using computer vision. It leverages the YOLO object detection model, Kalman Filter–based motion tracking, and (DIoU) matching for precise object association across frames. This project is highly useful for smart city analytics, traffic density estimation, and vehicle flow management applications.

# Project implementation
The system tracks each vehicle uniquely and also counts how many cross a predefined line (representing a road section or lane boundary). Every crossing event is stored in a MongoDB database, capturing details like:
- Vehicle ID
- Lane number crossed
- Time of crossing
- Cumulative vehicle count

This project combines deep detection + robust state estimation + vehicle traffic data documentation, ensuring stable, continuous tracking even in complex traffic scenes.

# Project pipeline

<img width="243" height="577" alt="Screenshot 2025-10-26 001726" src="https://github.com/user-attachments/assets/3af8830e-3b94-40a6-99cd-070d5d2cb6e2" />
