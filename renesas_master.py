from datetime import datetime
import socket, time

# Network Configuration
DEVICE_IP = "192.168.1.111"  # Replace with your device's static LAN IP
DEVICE_PORT = 20108         # Replace with your device's port number

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
    client_socket.connect((DEVICE_IP, DEVICE_PORT))
    print("Connected successfully! Starting stream...")

    while True:

        time_str = datetime.now().strftime("%d:%m:%Y:%H:%M:%S") #%S%M%H%d%m%y
        # print(f"raw time : {time_str}")

        # ascii_bytes = time_str.encode('ascii')
        # print(f"ascii : {ascii_bytes}")

        payload_str = f"ARN-SGPS>{time_str}\r"
        # print(f"payload : {payload_str}")

        client_socket.sendall(bytes(payload_str, "ascii"))
        print(f"Sent: {payload_str}", end="", flush=True)#, end="\r"

        time.sleep(1)
