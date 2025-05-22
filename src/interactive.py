import logging
import time
import threading
import socket
import struct
import numpy as np
import cv2
from ultralytics import YOLO
from pynput.keyboard import Listener, Key
import cflib.crtp
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
import warnings
 
warnings.filterwarnings("ignore", category=DeprecationWarning)
 
# =========================== CONFIG ===========================
 
# Crazyflie URI
URI = 'radio://0/80/2M/E7E7E7E708'
 
# AI-deck camera config
AI_DECK_IP = "192.168.4.1"
AI_DECK_PORT = 5000
SAVE_IMAGE = False
YOLO_MODEL_PATH = "./model/best.pt"
CONFIDENCE_THRESHOLD = 0.5
 
# =============================================================

VELOCITY = 0.25

# Flag to keep looping
running = True

def terminal_command_control(mc):
    global running
    print("Masukkan perintah (w/a/s/d/u/d/l/r/x untuk keluar):")
    while running:
        try:
            cmd = input(">> ").strip().lower()
            if cmd == 'w':
                print("Forward")
                mc.forward(VELOCITY)
            elif cmd == 's':
                print("Backward")
                mc.back(VELOCITY)
            elif cmd == 'a':
                print("Left")
                mc.left(VELOCITY)
            elif cmd == 'd':
                print("Right")
                mc.right(VELOCITY)
            elif cmd == 'u':
                print("Up")
                mc.up(0.2)
            elif cmd == 'd':
                print("Down")
                mc.down(0.2)
            elif cmd == 'l':
                print("Turn left")
                mc.turn_left(45)
            elif cmd == 'r':
                print("Turn right")
                mc.turn_right(45)
            elif cmd == 'x':
                print("Exit command received.")
                running = False
                break
            else:
                print("Perintah tidak dikenali. Gunakan: w/a/s/d/u/d/l/r/x")
        except KeyboardInterrupt:
            running = False
            break


def interactive_flight():
    cflib.crtp.init_drivers()
    with SyncCrazyflie(URI) as scf:
        scf.cf.platform.send_arming_request(True)
        time.sleep(1.0)
        with MotionCommander(scf, default_height=0.3) as mc:
            print("Drone siap dikendalikan melalui terminal.")
            terminal_command_control(mc)
            print("Landing...")
            mc.land()
            time.sleep(1)
 
 
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
 
    # Start both tasks in parallel using threading
    flight_thread = threading.Thread(target=interactive_flight)
    camera_thread = threading.Thread(target=camera_and_detection)
 
    flight_thread.start()
    camera_thread.start()
 
    # Wait for both threads to complete
    flight_thread.join()
    camera_thread.join()