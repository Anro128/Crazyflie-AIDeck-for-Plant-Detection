import logging
import time
import threading
import socket
import struct
import numpy as np
import cv2
from ultralytics import YOLO
import cflib.crtp
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# =========================== CONFIG ===========================
CRAZYFLIE_URIS = [
    'radio://0/80/2M/E7E7E7E708',
    'radio://0/80/2M/E7E7E7E709'
]

AI_DECK_IP = "192.168.4.1"
AI_DECK_PORT = 5000
SAVE_IMAGE = False
YOLO_MODEL_PATH = "./model/best.pt"
CONFIDENCE_THRESHOLD = 0.5
# =============================================================

def flight_sequence(uri):
    """Flight control logic for Crazyflie - CUSTOMIZED SEQUENCE"""
    print(f"[{uri}] Starting flight sequence")
    with SyncCrazyflie(uri) as scf:
        scf.cf.platform.send_arming_request(True)
        time.sleep(1.0)
        with MotionCommander(scf) as mc:
            print(f'[{uri}] Take off and rise 0.35m')
            mc.up(0.05)
            time.sleep(1)

            print(f'[{uri}] Rotate 180')
            mc.turn_right(180)
            time.sleep(1)

            print(f'[{uri}] Move left 1.2m')
            mc.left(1.2)
            time.sleep(1)

            print(f'[{uri}] Move forward 0.6m')
            mc.forward(0.6)
            time.sleep(2)

            print(f'[{uri}] Move right 1.0m')
            mc.right(1)
            time.sleep(2)

            print(f'[{uri}] Move back 1.3m')
            mc.back(1.3)
            time.sleep(1)

            print(f'[{uri}] Move right again 1.4m')
            mc.right(1.4)
            time.sleep(2)

            print(f'[{uri}] Descend 0.35m')
            mc.down(0.35)
            time.sleep(1)

            print(f'[{uri}] Landing complete')

def rx_bytes(sock, size):
    """Receive bytes from socket"""
    data = bytearray()
    while len(data) < size:
        data.extend(sock.recv(size - len(data)))
    return data

def camera_and_detection():
    """AI-deck camera and YOLO detection logic"""
    model = YOLO(YOLO_MODEL_PATH)
    print(f"[CAMERA] Connecting to {AI_DECK_IP}:{AI_DECK_PORT}")
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((AI_DECK_IP, AI_DECK_PORT))
    print("[CAMERA] Connected.")

    try:
        while True:
            packet_info = rx_bytes(client_socket, 4)
            length, routing, function = struct.unpack('<HBB', packet_info)

            img_header = rx_bytes(client_socket, length - 2)
            magic, width, height, depth, format, size = struct.unpack('<BHHBBI', img_header)

            if magic != 0xBC:
                continue

            img_data = bytearray()
            while len(img_data) < size:
                chunk_info = rx_bytes(client_socket, 4)
                chunk_len, dst, src = struct.unpack('<HBB', chunk_info)
                chunk = rx_bytes(client_socket, chunk_len - 2)
                img_data.extend(chunk)

            if format == 0:
                img_array = np.frombuffer(img_data, dtype=np.uint8)
                img_array.shape = (244, 324)
                img = cv2.cvtColor(img_array, cv2.COLOR_BayerBG2BGR)

                green_boost_factor = 0.8
                img[:, :, 1] = np.clip(img[:, :, 1] * green_boost_factor, 0, 255).astype(np.uint8)
                uk = 2
                resized = cv2.resize(img, (img.shape[1]*uk, img.shape[0]*uk), interpolation=cv2.INTER_CUBIC)

                if img is not None:
                    results = model.predict(source=resized, show=False, conf=CONFIDENCE_THRESHOLD, verbose=False)
                    for r in results:
                        annotated_frame = r.plot()
                    cv2.imshow("Camera", annotated_frame)

                    if SAVE_IMAGE:
                        cv2.imwrite(f"detected_{time.time():.0f}.jpg", img)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print("[CAMERA] Failed to decode image")
    except Exception as e:
        print(f"[CAMERA] Error: {e}")
    finally:
        client_socket.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    cflib.crtp.init_drivers()

    # Thread untuk kamera AI-deck
    camera_thread = threading.Thread(target=camera_and_detection)
    camera_thread.start()

    # Thread untuk masing-masing Crazyflie
    flight_threads = []
    for uri in CRAZYFLIE_URIS:
        t = threading.Thread(target=flight_sequence, args=(uri,))
        flight_threads.append(t)
        t.start()

    # Tunggu semua thread selesai
    for t in flight_threads:
        t.join()
    camera_thread.join()
