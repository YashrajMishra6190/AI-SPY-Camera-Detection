# AI-SPY-Camera-Detection
Detects the Spy camera using Object detection and pre-trained Yolo.
----

Start
  ↓
Launch Streamlit App
  ↓
Set Streamlit Page Config
  ↓
Load YOLOv5 Pretrained Model
  ↓
User Chooses Input Method
  ┌──────────────┴──────────────┐
  │                             │
[Webcam Detection]       [Upload Image]
  ↓                             ↓
Capture Frame             Load Uploaded Image
  ↓                             ↓
Run YOLOv5 Detection       Run YOLOv5 Detection
  ↓                             ↓
Filter for Camera-like Objects
  ↓                             ↓
Draw Bounding Boxes & Labels
  ↓                             ↓
Display Detection Results
  ↓
Wait for Next Input or Stop Webcam
  ↓
End

