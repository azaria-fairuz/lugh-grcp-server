import time
import socket
import random
from datetime import datetime

WITS_IP = "127.0.0.1"
WITS_PORT = 8504
WITS_CONNECTION = True

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((WITS_IP, WITS_PORT))

while WITS_CONNECTION:
    rpm = random.randint(0, 100)
    hkld = random.randint(0, 80)
    depth = random.randint(0, 2000)

    now = datetime.utcnow() 

    date_str = now.strftime("%y%m%d")
    time_str = now.strftime("%H%M%S")

    wits_message = (
        "&&\r\n"
        f"0101 {date_str}\r\n"
        f"0102 {time_str}\r\n"
        f"0208 {hkld}\r\n"
        f"0210 {rpm}\r\n"
        f"0221 {depth}\r\n"
        "!!\r\n"
    )

    sock.sendall(wits_message.encode("ascii"))
    time.sleep(1)
