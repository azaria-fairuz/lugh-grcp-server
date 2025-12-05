import cv2
import grpc
import asyncio
import frame_pb2
import traceback
import aiohttp_cors
import frame_pb2_grpc

import numpy as np

from aiohttp import web
from av import VideoFrame
from aiortc import RTCPeerConnection, VideoStreamTrack, RTCSessionDescription


camera_frames = {}
pcs = set()

#-- Async gRPC Server
class FrameServiceServicer(frame_pb2_grpc.FrameServiceServicer):
    async def SendFrame(self, request, context):
        camera_frames[request.camera_id] = request.data
        return frame_pb2.Empty()

async def serve_grpc():
    server = grpc.aio.server()
    frame_pb2_grpc.add_FrameServiceServicer_to_server(FrameServiceServicer(), server)
    
    server.add_insecure_port('[::]:8501')
    await server.start()

    print("Async gRPC server running on :8501")
    await server.wait_for_termination()

#-- WebRTC Track
class CameraTrack(VideoStreamTrack):
    def __init__(self, camera_id):
        super().__init__()
        self.camera_id = camera_id

    async def recv(self):
        pts, time_base = await self.next_timestamp()

        frame_bytes = camera_frames.get(self.camera_id)
        if not frame_bytes:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            nparr = np.frombuffer(frame_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros((480, 640, 3), dtype=np.uint8)

        frame = VideoFrame.from_ndarray(img, format="bgr24")
        frame.pts = pts
        frame.time_base = time_base
        return frame

#-- WebRTC Server
async def start_webrtc():
    app  = web.Application()
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        ),
        # "http://127.0.0.1:8500": aiohttp_cors.ResourceOptions(
        #     allow_credentials=True,
        #     expose_headers="*",
        #     allow_headers="*",
        # ),
    })
    
    async def offer(request):
        try:
            params = await request.json()
            camera_id = params.get("camera_id")
            
            if not camera_id or camera_id not in camera_frames:
                return web.json_response(
                    {"error": "The selected cctv is not available"}, 
                    status = 400
                )

            pc = RTCPeerConnection()
            pcs.add(pc)
            pc.addTrack(CameraTrack(camera_id))

            await pc.setRemoteDescription(
                RTCSessionDescription(params['sdp'], params['type'])
            )

            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            return web.json_response({"sdp": answer.sdp, "type": answer.type})

        except Exception as e:
            traceback.print_exc()
            return web.json_response({"error": str(e)}, status=500)


    resource = cors.add(app.router.add_post("/offer", offer))
    
    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, "0.0.0.0", 8502)
    await site.start()
    
    print("WebRTC server running on :8502")
    await asyncio.Event().wait()

async def main():
    await asyncio.gather(
        serve_grpc(),
        start_webrtc()
    )

if __name__ == "__main__":
    asyncio.run(main())
