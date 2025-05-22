# Modified from: https://github.com/bitcraze/crazyflie-lib-python/blob/master/examples/swarm/synchronizedSequence.py
# -*- coding: utf-8 -*-
#
#     ||          ____  _ __
#  +------+      / __ )(_) /_______________ _____  ___
#  | 0xBC |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  +------+    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#   ||  ||    /_____/_/\__/\___/_/   \__,_/ /___/\___/
#
#  Copyright (C) 2019 Bitcraze AB
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA  02110-1301, USA.
"""
Simple example of a synchronized swarm choreography using the High level
commander.

The swarm takes off and flies a synchronous choreography before landing.
The take-of is relative to the start position but the Goto are absolute.
The sequence contains a list of commands to be executed at each step.

This example is intended to work with any absolute positioning system.
It aims at documenting how to use the High Level Commander together with
the Swarm class to achieve synchronous sequences.
"""
import logging
import time
import threading
import socket
import struct
import numpy as np
import cv2
from ultralytics import YOLO
import cflib.crtp
from collections import namedtuple
from queue import Queue
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.swarm import Swarm
from cflib.crazyflie.syncLogger import SyncLogger
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# =========================== CONFIG ===========================
# Time for one step in seconds
STEP_TIME = 1

# Define drone URIs
uris = [
    'radio://0/80/2M/E7E7E7E708',  # Drone 1
    'radio://0/80/2M/E7E7E7E709',  # Drone 2
]

# AI-deck camera config
AI_DECK_IP = "192.168.4.1"
AI_DECK_PORT = 5000
SAVE_IMAGE = False
YOLO_MODEL_PATH = "./model/best.pt"
CONFIDENCE_THRESHOLD = 0.5
# =============================================================

# Swarm command definitions
Takeoff = namedtuple('Takeoff', ['height', 'time'])
Land = namedtuple('Land', ['time'])
Goto = namedtuple('Goto', ['x', 'y', 'z', 'time'])
Ring = namedtuple('Ring', ['r', 'g', 'b', 'intensity', 'time'])
Quit = namedtuple('Quit', [])

# Flight sequence - same path for both drones but with a delay
# This resembles the original flight_sequence pattern but with absolute positioning
sequence = [
    # Step, Drone_id, action
    # Drone 1 takes off first
    (0,    0,      Takeoff(0.5, 2)),
    
    # Drone 2 takes off with a delay
    (2,    1,      Takeoff(0.5, 2)),
    
    # Path point 1 - Rotate and move left
    (4,    0,      Goto(-1.2, 0.0, 0.5, 2)),
    (6,    1,      Goto(-1.2, 0.0, 0.5, 2)),
    
    # Path point 2 - Move forward
    (8,    0,      Goto(-1.2, 0.6, 0.5, 2)),
    (10,   1,      Goto(-1.2, 0.6, 0.5, 2)),
    
    # Path point 3 - Move right
    (12,   0,      Goto(-0.2, 0.6, 0.5, 2)),
    (14,   1,      Goto(-0.2, 0.6, 0.5, 2)),
    
    # Path point 4 - Move back
    (16,   0,      Goto(-0.2, -0.7, 0.5, 2)),
    (18,   1,      Goto(-0.2, -0.7, 0.5, 2)),
    
    # Path point 5 - Move right again
    (20,   0,      Goto(1.2, -0.7, 0.5, 2)),
    (22,   1,      Goto(1.2, -0.7, 0.5, 2)),
    
    # Land both drones
    (24,   0,      Land(2)),
    (26,   1,      Land(2)),
    
    # Turn off rings if any
    (28,   0,      Ring(0, 0, 0, 0, 0)),
    (28,   1,      Ring(0, 0, 0, 0, 0)),
]

# Create control queues for each drone
controlQueues = []

def wait_for_position_estimator(scf):
    """Wait for the estimator to get good values"""
    print(f"[{scf.cf.link_uri}] Waiting for estimator to find position...")

    log_config = LogConfig(name='Kalman Variance', period_in_ms=500)
    log_config.add_variable('kalman.varPX', 'float')
    log_config.add_variable('kalman.varPY', 'float')
    log_config.add_variable('kalman.varPZ', 'float')

    var_y_history = [1000] * 10
    var_x_history = [1000] * 10
    var_z_history = [1000] * 10

    threshold = 0.001

    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            data = log_entry[1]

            var_x_history.append(data['kalman.varPX'])
            var_x_history.pop(0)
            var_y_history.append(data['kalman.varPY'])
            var_y_history.pop(0)
            var_z_history.append(data['kalman.varPZ'])
            var_z_history.pop(0)

            min_x = min(var_x_history)
            max_x = max(var_x_history)
            min_y = min(var_y_history)
            max_y = max(var_y_history)
            min_z = min(var_z_history)
            max_z = max(var_z_history)

            if (max_x - min_x) < threshold and (
                    max_y - min_y) < threshold and (
                    max_z - min_z) < threshold:
                break

def reset_estimator(scf):
    """Reset estimator"""
    cf = scf.cf
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    wait_for_position_estimator(scf)

def activate_high_level_commander(scf):
    """Activate high level commander"""
    scf.cf.param.set_value('commander.enHighLevel', '1')

def activate_mellinger_controller(scf):
    """Activate mellinger controller for smoother flight"""
    controller = 2  # Use mellinger
    scf.cf.param.set_value('stabilizer.controller', str(controller))

def set_ring_color(cf, r, g, b, intensity, time):
    """Set LED ring color with fade"""
    cf.param.set_value('ring.fadeTime', str(time))
    r *= intensity
    g *= intensity
    b *= intensity
    color = (int(r) << 16) | (int(g) << 8) | int(b)
    cf.param.set_value('ring.fadeColor', str(color))

def crazyflie_control(scf):
    """Control loop for a crazyflie"""
    cf = scf.cf
    uri = cf.link_uri
    control = controlQueues[uris.index(uri)]
    
    print(f"[{uri}] Starting control loop")
    
    # Set up controller and parameters
    activate_mellinger_controller(scf)
    commander = cf.high_level_commander
    
    # Set LED ring to off
    if cf.param.has_parameter('ring.effect'):
        set_ring_color(cf, 0, 0, 0, 0, 0)
        cf.param.set_value('ring.effect', '14')  # Fade to color effect
    
    # Process commands from the queue
    while True:
        command = control.get()
        if type(command) is Quit:
            print(f"[{uri}] Quitting control loop")
            return
        elif type(command) is Takeoff:
            print(f"[{uri}] Taking off to {command.height}m in {command.time}s")
            commander.takeoff(command.height, command.time)
        elif type(command) is Land:
            print(f"[{uri}] Landing in {command.time}s")
            commander.land(0.0, command.time)
        elif type(command) is Goto:
            print(f"[{uri}] Going to ({command.x}, {command.y}, {command.z}) in {command.time}s")
            commander.go_to(command.x, command.y, command.z, 0, command.time)
        elif type(command) is Ring and cf.param.has_parameter('ring.effect'):
            set_ring_color(cf, command.r, command.g, command.b, 
                           command.intensity, command.time)
        else:
            print(f"[{uri}] Warning! Unknown command: {command}")

def control_thread():
    """Thread that sends commands to drones according to the sequence"""
    pointer = 0
    step = 0
    stop = False

    while not stop:
        print(f'Step {step}:')
        while pointer < len(sequence) and sequence[pointer][0] <= step:
            cf_id = sequence[pointer][1]
            command = sequence[pointer][2]

            print(f' - Running: {command} on drone {cf_id}')
            controlQueues[cf_id].put(command)
            pointer += 1

            if pointer >= len(sequence):
                print('Reaching the end of the sequence, stopping!')
                stop = True
                break

        step += 1
        time.sleep(STEP_TIME)

    for ctrl in controlQueues:
        ctrl.put(Quit())

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
    
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((AI_DECK_IP, AI_DECK_PORT))
        print("[CAMERA] Connected.")

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
        if 'client_socket' in locals():
            client_socket.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    
    # Initialize control queues
    controlQueues = [Queue() for _ in range(len(uris))]
    
    # Initialize CRTP drivers
    cflib.crtp.init_drivers(enable_debug_driver=False)
    
    # Start camera thread for AI-deck
    camera_thread = threading.Thread(target=camera_and_detection)
    camera_thread.daemon = True  # Make it a daemon so it closes with the main program
    camera_thread.start()
    
    # Create and start the Crazyflie swarm
    factory = CachedCfFactory(rw_cache='./cache')
    
    # Using Swarm class for coordinated flight
    with Swarm(uris, factory=factory) as swarm:
        print("Swarm created!")
        
        # Initialize each Crazyflie
        swarm.parallel_safe(activate_high_level_commander)
        swarm.parallel_safe(reset_estimator)
        
        print('Starting synchronized sequence!')
        
        # Start the control thread that will execute the sequence
        control_thread = threading.Thread(target=control_thread)
        control_thread.daemon = True
        control_thread.start()
        
        # Start the crazyflie control loop for each drone
        swarm.parallel_safe(crazyflie_control)
        
        # Wait for the control thread to finish
        control_thread.join()
        print("Flight sequence complete!")