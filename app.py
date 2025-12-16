import grpc
import json
import base64
import asyncio
import websockets
import frame_pb2
import frame_pb2_grpc

import time
import socket
import random
from datetime import datetime

WITS_IP   = "127.0.0.1"
WITS_PORT = 8504

class FrameServiceServicer(frame_pb2_grpc.FrameServiceServicer):
    def __init__(self, ws_uri):
        self.ws_uri = ws_uri
        self.websocket = None
        self.connected = False
        self.reconnecting = False

    async def connect_ws(self):
        if self.connected and self.websocket is not None:
            try:
                await self.websocket.ping()
                return
            except:
                self.websocket = None
                self.connected = False

        try:
            print(f"[WS] Reconnecting to {self.ws_uri}")
            self.reconnecting = True
            self.websocket = await websockets.connect(self.ws_uri)
            
            self.connected = True
            self.reconnecting = False
            print(f"[WS] Connected to {self.ws_uri}")

        except Exception as e:
            self.connected = False
            self.reconnecting = False
            print(f"[WS] Connection failed: {e}")

    async def SendFrame(self, request, context):
        if not self.reconnecting and not self.connected or self.websocket is None:
            await self.connect_ws()

        if not self.connected or self.websocket is None:
            print("[WS] Not connected, dropping frame")
            return frame_pb2.Empty()

        msg = json.dumps({
            "width" : request.width,
            "height" : request.height,
            "camera_id": request.camera_id,
            "data": base64.b64encode(request.data).decode('utf-8')
        })

        try:
            await self.websocket.send(msg)

        except Exception as e:
            print(f"[WS] Send failed: {e}")

            self.websocket = None
            self.connected = False

        return frame_pb2.Empty()

    def send_wits_data(self):
        c1 = random.randint(0, 10)
        c2 = random.randint(0, 80)
        c3 = random.randint(0, 200)
        c4 = random.randint(0, 400)

        now  = datetime.utcnow()
        date = now.strftime("%y%m%d")
        time = now.strftime("%H%M%S")

        wits_message = (
            "&&\r\n"
            f"0101 {date}\r\n"
            f"0102 {time}\r\n"
            f"0101 {c1}\r\n"
            f"0102 {c3}\r\n"
            f"0201 {c2}\r\n"
            f"0202 {c4}\r\n"
            "!!\r\n"
        )

        self.wits_sock.sendall(wits_message.encode("ascii"))
        print(f"[WITS0] {date}-{time} Data sent after frame")

async def serve():
    ws_uri = "ws://localhost:8502/frames?type=sender"
    server = grpc.aio.server(
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ]
    )

    frame_pb2_grpc.add_FrameServiceServicer_to_server(
        FrameServiceServicer(ws_uri),
        server
    )
    server.add_insecure_port('[::]:8501')
    await server.start()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((WITS_IP, WITS_PORT))
    print("[WITS0] sending data to WITS-0 port 8504")

    print("[GRPC] Listening on port 8501")
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())
