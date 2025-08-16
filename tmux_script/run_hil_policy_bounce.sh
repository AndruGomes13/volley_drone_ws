#!/usr/bin/env bash

SESSION="run_onboard"
WORKDIR="$HOME/catkin_ws"
ROS_SETUP="source $HOME/catkin_ws/devel/setup.sh"
INIT_SETUP="$ROS_SETUP && cd $WORKDIR  && set_ros_ip && clear"

# Create base session and pane 0.0
tmux new-session -d -s $SESSION -n main -c "$WORKDIR"

# Split layout: horizontal, then verticals to make 4 panes
tmux split-window -h -t $SESSION:0.0 -c "$WORKDIR"    # Pane 0.1
tmux split-window -v -t $SESSION:0.0 -c "$WORKDIR"    # Pane 0.2
tmux split-window -v -t $SESSION:0.1 -c "$WORKDIR"    # Pane 0.3
tmux split-window -v -t $SESSION:0.3 -c "$WORKDIR"    # Pane 0.4

# Now send commands
tmux send-keys -t $SESSION:0.0 "$INIT_SETUP" C-m
tmux send-keys -t $SESSION:0.0 "roscore"

tmux send-keys -t $SESSION:0.1 "$INIT_SETUP" C-m
tmux send-keys -t $SESSION:0.1 "roslaunch agiros volley_quadrotor_onboard_betaflight.launch quad_name:=volley_drone"

tmux send-keys -t $SESSION:0.2 "$INIT_SETUP" C-m
tmux send-keys -t $SESSION:0.2 "roslaunch bounce_policy_pkg policy_node.launch quad_name:=volley_drone recovery_policy_name:=recovery_policy bounce_policy_name:=bounce_policy" 

tmux send-keys -t $SESSION:0.3 "$INIT_SETUP" C-m
tmux send-keys -t $SESSION:0.3 "clear; printf '%b\n' '\n NOTE: In order to run the bounce policy, you need to have a separate process running the Mujoco Hardware in the Loop simulation.\n\nFurthermore, you also need to run the zqm bridge node on that same machine. This bridge node is responsible for sending the drone state to the policy node and receiving the ball state from the Mujoco simulation.\n'" C-m

tmux send-keys -t $SESSION:0.4 "$INIT_SETUP" C-m
tmux send-keys -t $SESSION:0.4 "roslaunch motion_capture_ros mc_drone_aim_no_pc.launch" # Adjust quad_name as needed

tmux select-pane -t $SESSION:0.0
tmux attach -t $SESSION