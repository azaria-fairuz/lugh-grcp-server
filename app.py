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

WITS_IP   = "host.docker.internal"
WITS_PORT = 8504
GRPC_PORT = 8501
WEBSOCKER_PORT = 8502

class FrameServiceServicer(frame_pb2_grpc.FrameServiceServicer):
    def __init__(self, ws_uri):
        self.ws_uri = ws_uri
        self.websocket = None
        self.connected = False
        self.reconnecting = False

        self.wits_writer = None
        self.wits_connecting = False
        self.last_wits_second = None

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
            await self.send_wits_data()

        except Exception as e:
            print(f"[WS] Send failed: {e}")

            self.websocket = None
            self.connected = False

        return frame_pb2.Empty()

    async def connect_wits0(self):
        while True:
            if self.wits_writer is not None:
                await asyncio.sleep(1)
                continue

            if self.wits_connecting:
                await asyncio.sleep(1)
                continue

            self.wits_connecting = True
            try:
                print(f"[WITS0] Connecting to {WITS_IP}:{WITS_PORT}...")
                _, writer = await asyncio.open_connection(WITS_IP, WITS_PORT)
                self.wits_writer = writer
                print(f"[WITS0] Connected")
            except Exception as e:
                print(f"[WITS0] Connection failed: {e}")
                await asyncio.sleep(3)
            finally:
                self.wits_connecting = False

    async def send_wits_data(self):
        if not self.wits_writer:
            return

        now = datetime.utcnow()
        current_second = now.replace(microsecond=0)

        if self.last_wits_second == current_second:
            return

        self.last_wits_second = current_second

        try:
            c1 = random.randint(0, 10)
            c2 = random.randint(0, 80)
            c3 = random.randint(0, 200)
            c4 = random.randint(0, 400)

            date = now.strftime("%y%m%d")
            time_ = now.strftime("%H%M%S")

            wits_message = (
                "&&\r\n"
                f"0101 {date}\r\n"
                f"0102 {time_}\r\n"
                f"0103 {c1}\r\n"
                f"0104 {c3}\r\n"
                f"0201 {c2}\r\n"
                f"0202 {c4}\r\n"
                "!!\r\n"
            )

            self.wits_writer.write(wits_message.encode("ascii"))
            await self.wits_writer.drain()

            print(f"[WITS0] {date}-{time_} Data sent")

        except Exception as e:
            print(f"[WITS0] Send failed: {e}")
            try:
                self.wits_writer.close()
                await self.wits_writer.wait_closed()
            except:
                pass
            self.wits_writer = None

async def serve():
    ws_uri = f"ws://localhost:{WEBSOCKER_PORT}/frames?type=sender"

    server = grpc.aio.server(
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024),
        ]
    )

    servicer = FrameServiceServicer(ws_uri)
    frame_pb2_grpc.add_FrameServiceServicer_to_server(servicer, server)

    server.add_insecure_port(f'[::]:{GRPC_PORT}')
    await server.start()

    print(f"[GRPC] Listening on port {GRPC_PORT}")

    asyncio.create_task(servicer.connect_wits0())
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())
