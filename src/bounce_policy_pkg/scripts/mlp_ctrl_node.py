#!/usr/bin/env python

from pathlib import Path

import psutil
from policy_package.mlp_ctrl import MLPPilot
import rospy
import subprocess
import signal, sys

child_processes = []
def shutdown_handler(sig, frame):
    """
    Gracefully shut down all child processes and exit.
    This function is triggered by Ctrl+C.
    """
    print("\nCtrl+C detected! Shutting down all inference servers...")
    
    for process in child_processes:
        print(f"Terminating child process PID: {process.pid}...")
        try:
            # Send the termination signal
            process.terminate()
            # Wait for the process to exit
            process.wait(timeout=5)
            print(f"Process {process.pid} terminated.")
        except psutil.NoSuchProcess:
            print(f"Process {process.pid} already terminated.")
        except subprocess.TimeoutExpired:
            print(f"Process {process.pid} did not terminate in time. Killing.")
            process.kill()

    print("All child processes shut down. Exiting.")
    sys.exit(0)


def main():    
    signal.signal(signal.SIGINT, shutdown_handler)
    
    rospy.init_node("mlp_ctrl_node", anonymous=True)
    policy_path = str(rospy.get_param("~policy_path"))
    drone_only = bool(rospy.get_param("~drone_only"))
    quad_name = str(rospy.get_param("~quad_name"))
    
    if not policy_path:
        rospy.logerr("Policy path not provided. Please set the ~policy_path parameter.")
        return
    if not drone_only:
        rospy.logerr("Drone-only mode not set. Please set the ~drone_only parameter to True.")
        return
    
    
    pilot = MLPPilot(
        quad_name=quad_name,
        policy_sampling_frequency=100,  # Hz
        jax_policy_path=Path(policy_path),
        drone_only=drone_only  # Set to True for drone-only mode
    )
    
    rate = rospy.Rate(300)
    while not rospy.is_shutdown():
        pilot.tick()
        rate.sleep()
        
        
    child_processes.append(pilot.inference_server.server)
    rospy.loginfo("MLP Control Node initialized and running.")
    rospy.spin()
        
if __name__ == "__main__":
    main()
    