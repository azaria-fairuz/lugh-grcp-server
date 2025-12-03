import grpc
import frame_pb2
import frame_pb2_grpc

import numpy as np

from concurrent import futures

class FrameServiceServicer(frame_pb2_grpc.FrameServiceServicer):
    def SendFrame(self, request, context):
        width = request.width
        height = request.height
        frame_bytes = request.data

        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(height, width, 3)
        print(f"recieved frame with width {width} and height {height}")

        # add sending logic in here to detection
    
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    frame_pb2_grpc.add_FrameServiceServicer_to_server(FrameServiceServicer(), server)
    server.add_insecure_port('[::]:8500')

    server.start()
    print("listening on port 8500")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
