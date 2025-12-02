# Crazyflie-AIDeck-for-Plant-Detection

## DEMO
[![Demo](https://img.youtube.com/vi/h3_bopc3xCc/hqdefault.jpg)]

## 📋 Deskripsi Projek
Projek ini mengimplementasikan sistem deteksi tanaman menggunakan drone Crazyflie yang dilengkapi dengan AI-Deck. Sistem ini mampu melakukan penerbangan otomatis atau manual sambil melakukan deteksi objek tanaman secara real-time menggunakan model YOLO.

## 🛠️ Tech Stack

### Hardware
- **Crazyflie 2.X** - Nano quadcopter drone
- **AI-Deck** - ESP32-based WiFi streaming camera module
- **Crazyradio PA** - 2.4GHz radio USB dongle untuk komunikasi dengan drone
- **Lighthouse Positioning System** (opsional) - Sistem positioning untuk navigasi presisi

### Software & Libraries

#### Python Libraries
- **cflib (Crazyflie Python Library)** - Library utama untuk kontrol drone Crazyflie
  - `cflib.crtp` - Komunikasi radio dengan drone
  - `cflib.crazyflie.syncCrazyflie` - Sinkronisasi operasi drone
  - `cflib.positioning. motion_commander` - Kontrol gerakan high-level
  - `cflib.crazyflie.swarm` - Kontrol multi-drone (swarm)

- **Ultralytics YOLO** - Framework deep learning untuk object detection
  - Model: YOLOv8/YOLOv5 (custom trained untuk plant detection)

- **OpenCV (cv2)** - Computer vision untuk image processing
  - Decoding Bayer pattern dari AI-Deck camera
  - Image enhancement dan preprocessing

- **NumPy** - Array processing dan manipulasi data gambar

- **Socket & Struct** - Komunikasi TCP/IP dengan AI-Deck via WiFi

- **Threading** - Multi-threading untuk operasi paralel (flight control + camera)

- **Pynput** - Keyboard input handling untuk mode interaktif

#### AI/ML
- **YOLO Model** (YOLOv8) - Custom trained model untuk deteksi tanaman
- **Confidence Threshold**: 0.5

## 🔄 Flow Kerja Sistem

### 1. Inisialisasi
```
Hardware Setup → Radio Connection → AI-Deck WiFi Connection
```
- Menginisialisasi Crazyradio PA driver
- Koneksi ke drone via radio (URI: `radio://0/80/2M/E7E7E7E7XX`)
- Koneksi ke AI-Deck camera via WiFi (`192.168.4.1:5000`)

### 2.  Threading Architecture
```
Main Thread
├── Thread 1: Flight Control
│   ├── Arming drone
│   ├── Takeoff
│   ├── Autonomous/Manual navigation
│   └── Landing
│
└── Thread 2: Camera & Detection
    ├── Stream receiving dari AI-Deck
    ├── Image decoding (Bayer → BGR)
    ├── YOLO inference
    └── Bounding box visualization
```

### 3. Camera Streaming Flow
```
AI-Deck (ESP32) → WiFi Socket → Packet Reception → Image Reconstruction → YOLO Detection → Display
```

**Detail Proses:**
1. **Packet Reception**: Menerima header packet (4 bytes)
2. **Image Header Parsing**: Extract width, height, format (Bayer pattern)
3. **Chunk Assembly**: Mengumpulkan chunk data hingga image lengkap
4. **Bayer Decoding**: Convert Bayer pattern → BGR color image
5. **Green Channel Boost**: Enhancement untuk deteksi tanaman (factor: 0.8)
6. **YOLO Inference**: Deteksi objek tanaman
7. **Visualization**: Bounding boxes + confidence score

### 4. Flight Control Modes

#### Mode 1: Auto Sequence
- Pola terbang otomatis pre-programmed
- Sequence: Takeoff → Right movements → Circle maneuver → Landing
- Koordinasi gerakan dengan kecepatan dan jarak tertentu

#### Mode 2: Interactive Control
- Kontrol manual via terminal/keyboard
- Commands: W/A/S/D (movement), U/DW (vertical), L/R (rotation)
- Real-time velocity adjustment

#### Mode 3: Multi-Drone Synchronized
- Kontrol beberapa drone secara bersamaan
- Barrier synchronization untuk koordinasi
- Independent flight sequences per drone

#### Mode 4: Lighthouse Swarm
- High-precision positioning menggunakan Lighthouse system
- Synchronized choreography untuk multiple drones
- Absolute positioning (goto x, y, z)

### 5. Detection & Output
```
Camera Frame → YOLO Model → Detections → Filter (conf > 0.5) → Draw Boxes → Display Window
```

## 📁 Struktur Projek

```
Crazyflie-AIDeck-for-Plant-Detection/
├── STEP1/              # Installation instruction
├── STEP2/              # Update Firmware
├── STEP3/              # Flash Wifi Example
├── STEP4/              # LETS FLY!
│   ├── script/
│   │   ├── AutoSequence.py          # Auto flight pattern
│   │   ├── interactive. py           # Manual control (basic)
│   │   ├── interactive2.py          # Manual control (advanced velocity)
│   │   ├── multidrone.py            # Multi-drone basic sync
│   │   └── multidrone2.py           # Multi-drone advanced
│   ├── lighthouse flight/
│   │   └── synchronizedSequence.py  # Swarm choreography
│   └── model/
│       └── best. pt                  # Trained YOLO model
└── image/              # Assets & documentation images
```

## 🚀 Mode Penggunaan

### Prerequisites
```bash
pip install cflib opencv-python numpy ultralytics pynput
```

### Mode Auto Sequence
```bash
python STEP4/script/AutoSequence.py
```

### Mode Interactive
```bash
python STEP4/script/interactive.py
# atau dengan velocity control:
python STEP4/script/interactive2.py
```

### Mode Multi-Drone
```bash
python STEP4/script/multidrone.py
```

### Mode Lighthouse Swarm
```bash
python "STEP4/lighthouse flight/synchronizedSequence.py"
```

## ⚙️ Konfigurasi

Setiap script memiliki section CONFIG yang dapat disesuaikan:

```python
# Crazyflie URI
URI = 'radio://0/80/2M/E7E7E7E709'

# AI-Deck Settings
AI_DECK_IP = "192.168.4.1"
AI_DECK_PORT = 5000

# YOLO Settings
YOLO_MODEL_PATH = ". /../model/best.pt"
CONFIDENCE_THRESHOLD = 0. 5

# Flight Parameters
DIS = 0.25  # Distance in meters
VEL = 0.3   # Velocity in m/s
```

## 📊 Workflow Diagram

```
┌─────────────────┐         ┌──────────────────┐
│  Crazyradio PA  │◄───────►│  Crazyflie Drone │
└─────────────────┘         └──────────────────┘
        ▲                            │
        │                            │
        │                            ▼
┌───────┴────────┐          ┌──────────────────┐
│ Python Script  │          │     AI-Deck      │
│ (Flight Ctrl)  │          │   (ESP32 Cam)    │
└────────────────┘          └──────────────────┘
        │                            │
        │                            │ WiFi
        │                            ▼
        │                   ┌──────────────────┐
        │                   │  Camera Stream   │
        │                   │   + Detection    │
        │                   └──────────────────┘
        │                            │
        └────────────┬───────────────┘
                     ▼
            ┌─────────────────┐
            │  Real-time View │
            │  + Bounding Box │
            └─────────────────┘
```

## 🎯 Fitur Utama

✅ Real-time plant detection menggunakan YOLO  
✅ Multiple flight modes (Auto, Manual, Swarm)  
✅ Multi-drone synchronization  
✅ WiFi camera streaming dari AI-Deck  
✅ Interactive keyboard/terminal control  
✅ Lighthouse positioning support  
✅ Custom velocity & distance control  
✅ Green channel enhancement untuk plant detection  

## 👥 Our Team
* Ahmad Nur Rohim
* Dicky Anugraha 
* Habib Fabri A.
* Khansa Fitri Z. 
* Salsabila Azzahra

## Step 1
[Installation instruction](./STEP1/)
## Step 2
[Update Firmware](./STEP2/)
## Step 3
[Flash Wifi Example](./STEP3/)
## Step 4
[LETS FLY!](./STEP4/)

