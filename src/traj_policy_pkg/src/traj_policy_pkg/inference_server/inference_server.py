#! /usr/bin/env python3.11
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import functools
from pathlib import Path
import sys
import time
from typing import Callable
import numpy as np
import jax.numpy as jp
import jax
from brax.training.types import Policy
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import checkpoint
import argparse
import zmq

def printl(*args, **kwargs):
    """Custom print function to handle output."""
    print("[Inf SERVER]",*args, **kwargs)


def _load_policy(checkpoint_dir: Path) -> Policy:
    """Restore a policy from `checkpoint_dir`."""
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} does not exist.")
    
    make_network = functools.partial(ppo_networks.make_ppo_networks, policy_obs_key = "actor", value_obs_key = "critic")
    return checkpoint.load_policy(str(checkpoint_dir), deterministic=True, network_factory=make_network)
       
def create_inference_function(checkpoint_dir: Path, observation_shape: tuple) -> Callable:
    """Create an inference server instance."""
    inference_fn_ = _load_policy(checkpoint_dir)
    
    @jax.jit
    def inference_fn(observations: jp.ndarray):
        """Inference function that takes observations and returns actions."""
        key_sample = jax.random.PRNGKey(0)
        return inference_fn_({"actor":observations}, key_sample)[0]
    
    # Warm-up the inference function
    warmup_observation = jp.zeros(observation_shape, dtype=jp.float32)
    for _ in range(10):
        try:
            inference_fn(warmup_observation)
        except Exception as e:
            printl(f"Error during warmup: {e}")
    
    return inference_fn           
                          

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference server for JAX policies.")
    parser.add_argument("--checkpoint-dir", type=Path, help="Path to the checkpoint directory.", required=True)
    parser.add_argument("--observation-shape", type=str, help="Shape of the observation as a tuple.", required=True)
    parser.add_argument("--death-pipe-fd", type=int, help="FD of pipe to monitor for parent death")
    parser.add_argument("--socket-address", type=str, default="ipc:///tmp/jax_ipc.sock", help="ZMQ socket address to bind to.")
    parser.add_argument("--sync-socket-address", type=str, default="ipc:///tmp/jax_sync.sock", help="ZMQ socket address to bind to.")
    
    args = parser.parse_args()
    
    # Parse the observation shape
    elements = list(filter(None, args.observation_shape.strip("()").split(",")))
    observation_shape = tuple(map(int, elements))
    socket_address = args.socket_address
    sync_address = args.sync_socket_address
    death_pipe_fd = args.death_pipe_fd
        
    # --- Setup ZQM sockets ---
    # --- Reply socket
    context = zmq.Context()
    socket = context.socket(zmq.REP)  # REP for "Reply"
    socket.bind(args.socket_address)
    
    # --- Sync Socket
    sync_socket = context.socket(zmq.PUB)
    sync_socket.bind(sync_address)
    
    # --- Poller
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    
    # --- Death pipe monitoring
    if death_pipe_fd is not None:
        # We can poll a raw file descriptor by wrapping it
        poller.register(open(death_pipe_fd), zmq.POLLIN)
        # printl(f"Monitoring parent process via pipe fd {death_pipe_fd}")


    # --- Create the inference function and send sync message ---
    try:
        inference_fn = create_inference_function(args.checkpoint_dir, observation_shape)
        # inference_fn = lambda obs: jp.zeros((10,))
        # inference_fn = jax.jit(inference_fn)
        # for _ in range(10):
        #     inference_fn(jp.zeros(observation_shape, dtype=jp.float32))
    except Exception as e:
        printl(f"Error creating inference function: {e}")
        sys.exit(1)
    
    # printl(f"Created inference function with observation shape {observation_shape}")
    time.sleep(0.5)
    sync_socket.send_string("READY")
    sync_socket.close()  # No longer needed


    # --- Run the inference server
    while True:
        try:
            # Wait for an event on either the ZMQ socket or the death pipe
            events = dict(poller.poll())
        except KeyboardInterrupt:
            break

        # Check if the death pipe fired (meaning the parent died)
        if death_pipe_fd in events:
            printl("Parent process appears to have exited. Shutting down.")
            break # Exit the loop and terminate

        # Check if the ZMQ socket has a message
        if socket in events:
            # Handle the inference request as before
            message = socket.recv_json()
            message = np.array(message)
            assert message is not None, "Received None message from ZMQ socket"
            assert isinstance(message, np.ndarray), "Expected a NumPy array from ZMQ socket"
            assert message.shape == observation_shape, f"Expected shape {observation_shape}, got {message.shape}"
            # Convert the NumPy array to a JAX array
            message = jp.array(message, dtype=jp.float32)
            
            result_jp = inference_fn(message)

            # Convert JAX array to a standard NumPy array for sending
            result_np = np.array(result_jp)
            
            # Send the result object back to the clients
            socket.send_json(result_np.tolist())

