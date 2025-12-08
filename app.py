import grpc
import json
import base64
import asyncio
import websockets
import frame_pb2
import frame_pb2_grpc

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

async def serve():
    ws_uri = "ws://localhost:8503/frames"

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

    print("[START] Listening on port 8501")
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())
