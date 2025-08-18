#!/usr/bin/env bash

SESSION="mac_setup"
ROS_SETUP="source ./devel/setup.sh"
INIT_SETUP="$ROS_SETUP && ros_connect_orin_aim && clear"

# Create base session and pane 0.0
tmux new-session -d -s $SESSION -n main -c "$WORKDIR"

# Split layout: horizontal, then verticals to make 4 panes
tmux split-window -h -t $SESSION:0.0 -c "$WORKDIR"    # Pane 0.1
tmux split-window -v -t $SESSION:0.0 -c "$WORKDIR"    # Pane 0.2
tmux split-window -v -t $SESSION:0.1 -c "$WORKDIR"    # Pane 0.3

# Now send commands
tmux send-keys -t $SESSION:0.0 "$INIT_SETUP" C-m
tmux send-keys -t $SESSION:0.0 "roslaunch agiros_gui volley_base_computer.launch quad_name:=volley_drone"

tmux send-keys -t $SESSION:0.1 "$INIT_SETUP" C-m
tmux send-keys -t $SESSION:0.1 "roslaunch util_package state_to_zqm_bridge.launch" # Adjust quad_name as needed

tmux send-keys -t $SESSION:0.2 "$INIT_SETUP" C-m
tmux send-keys -t $SESSION:0.2 "roslaunch motion_capture_ros mc_mac_aim.launch quad_name:=volley_drone" # Adjust quad_name as needed

tmux send-keys -t $SESSION:0.3 "$INIT_SETUP" C-m

tmux select-pane -t $SESSION:0.0
tmux attach -t $SESSION