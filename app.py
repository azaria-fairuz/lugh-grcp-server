import grpc
import frame_pb2
import frame_pb2_grpc

import numpy as np

from concurrent import futures

class FrameServiceServicer(frame_pb2_grpc.FrameServiceServicer):
    def SendFrame(self, request, context):
        print("[NEW] request has been accpeted")
        width = request.width
        height = request.height
        frame_bytes = request.data
        size_in_bytes = len(frame_bytes)

        print(f"[RECEIVED] Frame: {width}x{height}, {size_in_bytes / 1024:.2f} KB")

        return frame_pb2.Empty()
    
def serve():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 50 * 1024 * 1024),
            ('grpc.max_receive_message_length', 50 * 1024 * 1024)
        ]
    )

    frame_pb2_grpc.add_FrameServiceServicer_to_server(FrameServiceServicer(), server)
    server.add_insecure_port('[::]:8500')
    server.start()

    print("[START] Listening on port 8500")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
