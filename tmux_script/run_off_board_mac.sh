#!/usr/bin/env bash

SESSION="run_offboard_mac"
WORKDIR="$HOME/Repos/agilicious_ws_mac"
ROS_SETUP="source $WORKDIR/devel/setup.sh"
INIT_SETUP="$ROS_SETUP && cd $WORKDIR  && clear && ros_connect_orin"

# Create base session and pane 0.0
tmux new-session -d -s $SESSION -n main -c "$WORKDIR"

# Split layout: horizontal, then verticals to make 4 panes
tmux split-window -h -t $SESSION:0.0 -c "$WORKDIR"    # Pane 0.1
tmux split-window -v -t $SESSION:0.0 -c "$WORKDIR"    # Pane 0.2
tmux split-window -v -t $SESSION:0.1 -c "$WORKDIR"    # Pane 0.3

# Now send commands
tmux send-keys -t $SESSION:0.0 "$INIT_SETUP" C-m
tmux send-keys -t $SESSION:0.0 "roslaunch motion_capture_ros base_computer.launch quad_name:=volley_drone rviz:=true"

tmux send-keys -t $SESSION:0.1 "$INIT_SETUP" C-m
tmux send-keys -t $SESSION:0.1 "rosbag -a " # To figure out

tmux send-keys -t $SESSION:0.2 "$INIT_SETUP" C-m

tmux send-keys -t $SESSION:0.3 "$INIT_SETUP" C-m

tmux select-pane -t $SESSION:0.0
tmux attach -t $SESSION