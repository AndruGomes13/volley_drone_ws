import os, uuid
from pathlib import Path
import subprocess
from typing import Tuple
import numpy as np
import rospy
import zmq

SOCKET_ADDRESS_TEMPLATE = "ipc:///tmp/jax_ipc_{}.sock"
SYNC_ADDRESS_TEMPLATE = "ipc:///tmp/jax_sync_{}.sock"
INFERENCE_SERVER_PATH = os.path.join(os.path.dirname(__file__), "inference_server.py")

class PolicyServerInterface:
    def __init__(self, checkpoint_path: Path, observation_shape: Tuple[int, ...]):
        self.jax_policy_path = checkpoint_path
        self.observation_shape = observation_shape

        self.socket_address = SOCKET_ADDRESS_TEMPLATE.format(uuid.uuid4())
        self.sync_address = SYNC_ADDRESS_TEMPLATE.format(uuid.uuid4())

        self.setup_server()

    def setup_server(self):
        # --- Setup server with death pipe ---        
        death_pipe_read_fd, self.death_pipe_write_fd = os.pipe()
        
        # Start the inference server with the specified checkpoint directory and observation shape
        python_file = INFERENCE_SERVER_PATH
        self.server = subprocess.Popen(
            ["python3.11", "-u", python_file,
             "--checkpoint-dir", self.jax_policy_path.absolute(),
             "--observation-shape", f"{str(self.observation_shape)}",
             "--socket-address", self.socket_address, "--sync-socket-address", self.sync_address,
             "--death-pipe-fd", str(death_pipe_read_fd)],
            bufsize=1,  # line buffered
            pass_fds=[death_pipe_read_fd]
        )
        
        os.close(death_pipe_read_fd)
        
        # --- Setup ZMQ sockets ---
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)  # REP for "Reply"
        self.socket.connect(self.socket_address)

        sync_socket = self.context.socket(zmq.SUB)
        sync_socket.connect(self.sync_address)
        sync_socket.setsockopt_string(zmq.SUBSCRIBE, "READY") # Subscribe to the "READY" message

        print("Subprocess Id:", self.server.pid)
        rospy.loginfo(f"Inference server started with checkpoint directory: {self.jax_policy_path}")
        print("Client is waiting for server READY signal...")
        ready_signal = sync_socket.recv_string()
        sync_socket.close()
        print(f"'{ready_signal}' signal received. Server is up!")

    def run_inference(self, observation: np.ndarray) -> np.ndarray:
        assert observation.shape == self.observation_shape
        self.socket.send_json(observation.tolist())
        response = np.array(self.socket.recv_json())
        if isinstance(response, np.ndarray):
            return response
        elif isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Inference server returned error: {response['error']}")
        else:
            raise ValueError(f"Unexpected response type: {type(response)}")
        
