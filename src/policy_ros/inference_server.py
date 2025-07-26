#! /usr/bin/env python3.11
# TODO: Need testing

import functools
import os
from pathlib import Path
import select
import sys
import json
from typing import Callable
import jax.numpy as jp
import jax
import traceback
from protocol.messages import InferenceMsg, InferenceReply, parse_message
from brax.training.types import Policy
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import checkpoint
import argparse


def _load_policy(checkpoint_dir: Path) -> Policy:
    """Restore a policy from `checkpoint_dir`."""
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} does not exist.")
    
    make_network = functools.partial(ppo_networks.make_ppo_networks, policy_obs_key = "actor", value_obs_key = "critic")
    return checkpoint.load_policy(str(checkpoint_dir), deterministic=True, network_factory=make_network)
            
def send(msg: dict):
    """Send a message to stdout."""
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()
        
class InferenceServer:
    """A simple inference server that reads from stdin and writes to stdout."""
    def __init__(self, inference_fn: Policy, observation_shape: tuple):
        self.observation_shape = observation_shape
        self.inference_fn = inference_fn
        
        # Register the stdin file descriptor for polling
        self.stdin_fd = sys.stdin.fileno()
        self.poller    = select.poll()
        POLLIN_IN  = select.POLLIN | select.POLLHUP
        self.poller.register(self.stdin_fd, POLLIN_IN)

    def inference(self, obs: jp.ndarray) -> jp.ndarray:
        """Run inference on the provided observation."""
        if obs.shape != self.observation_shape:
            raise ValueError(f"Observation shape mismatch: expected {self.observation_shape}, got {obs.shape}")
        
        # Run the inference function
        result = self.inference_fn(obs)
        return result
        
    def run(self):
        """Main loop to read from stdin and write to stdout."""
        # MAIN LOOP
        while True:
            events = self.poller.poll()
            for fd, flag in events:
                if fd == self.stdin_fd and flag & select.POLLIN:
                    line = sys.stdin.readline()
                    if line == "":        # <- parent closed the pipe
                        sys.stderr.write("Parent died. Exiting.\n")
                        sys.exit(0)
                    
                    try:
                        request_msg = parse_message(line.strip())
                        if isinstance(request_msg, InferenceMsg):
                            obs = jp.array(request_msg.obs, dtype=jp.float32)
                            result: jp.ndarray = inference_fn(obs)
                            inference_reply = InferenceReply(status="ok", result=result.tolist())
                            send(inference_reply.to_json())
                        else:
                            inference_reply = InferenceReply(status="error", error="Unknown command")
                            send(inference_reply.to_json())
                            
                    except Exception as e:
                        inference_reply = InferenceReply(status="error", error=str(e), traceback=traceback.format_exc())
                        send(inference_reply.to_json())
    
def create_inference_function(checkpoint_dir: Path, observation_shape: tuple) -> Callable:
    """Create an inference server instance."""
    inference_fn = _load_policy(checkpoint_dir)
    
    # JIT compile the inference function
    inference_fn = jax.jit(inference_fn)
    
    # Warm-up the inference function
    warmup_observation = jp.zeros(observation_shape, dtype=jp.float32)
    for _ in range(10):
        try:
            inference_fn(warmup_observation)
        except Exception as e:
            inference_reply = InferenceReply(status="error", error=str(e), traceback=traceback.format_exc())
            send(inference_reply.to_json())
            sys.exit(1)
    
    return inference_fn           
                          

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference server for JAX policies.")
    parser.add_argument("--checkpoint-dir", type=Path, help="Path to the checkpoint directory.", required=True)
    parser.add_argument("--observation-shape", type=str, help="Shape of the observation as a tuple.", required=True)
    
    args = parser.parse_args()
    
    # Parse the observation shape
    elements = list(filter(None, args.observation_shape.strip("()").split(",")))
    observation_shape = tuple(map(int, elements))

    # Create the inference server
    try:
        # inference_fn = create_inference_function(args.checkpoint_dir, observation_shape)
        inference_fn = lambda obs: jp.zeros((10,))
    except Exception as e:
        inference_reply = InferenceReply(status="error", error=str(e), traceback=traceback.format_exc())
        sys.exit(1)
    
    # Run the inference server
    inference_server = InferenceServer(inference_fn, observation_shape)
    inference_server.run()
