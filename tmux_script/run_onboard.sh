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

# Now send commands
tmux send-keys -t $SESSION:0.0 "$INIT_SETUP" C-m
tmux send-keys -t $SESSION:0.0 "roscore"

tmux send-keys -t $SESSION:0.1 "$INIT_SETUP" C-m
tmux send-keys -t $SESSION:0.1 "roslaunch motion_capture_ros motion_capture_optitrack.launch" # To figure out

tmux send-keys -t $SESSION:0.2 "$INIT_SETUP" C-m
tmux send-keys -t $SESSION:0.2 "roslaunch agiros volley_quadrotor_onboard_betaflight.launch quad_name:=volley_drone" # Adjust quad_name as needed

tmux send-keys -t $SESSION:0.3 "$INIT_SETUP" C-m

tmux select-pane -t $SESSION:0.0
tmux attach -t $SESSION