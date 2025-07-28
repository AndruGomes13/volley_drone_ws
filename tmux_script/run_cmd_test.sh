#!/usr/bin/env bash

SESSION="run_onboard"
WORKDIR="$HOME/catkin_ws"
ROS_SETUP="source $HOME/catkin_ws/devel/setup.sh"
INIT_SETUP="$ROS_SETUP && cd $WORKDIR  && clear"

# Create base session and pane 0.0
tmux new-session -d -s $SESSION -n main -c "$WORKDIR"

# Split layout: horizontal, then verticals to make 4 panes
tmux split-window -h -t $SESSION:0.0 -c "$WORKDIR"    # Pane 0.1
tmux split-window -v -t $SESSION:0.0 -c "$WORKDIR"    # Pane 0.2
tmux split-window -v -t $SESSION:0.2 -c "$WORKDIR"    # Pane 0.3

# Now send commands
tmux send-keys -t $SESSION:0.0 "$INIT_SETUP" C-m
tmux send-keys -t $SESSION:0.0 "roslaunch agiros ros_to_sbus_volley.launch quad_name:=volley_drone"

tmux send-keys -t $SESSION:0.1 "$INIT_SETUP" C-m
tmux send-keys -t $SESSION:0.1 "roslaunch load_cell_ros_node interactive_cmd_publisher.launch" # To figure out

tmux send-keys -t $SESSION:0.2 "$INIT_SETUP" C-m
tmux send-keys -t $SESSION:0.2 "rostopic echo /command" # Adjust quad_name as needed

tmux send-keys -t $SESSION:0.3 "$INIT_SETUP" C-m
tmux send-keys -t $SESSION:0.3 "rostopic pub /arm -1 std_msgs/Bool "data: True"" # Adjust quad_name as needed




tmux select-pane -t $SESSION:0.0
tmux attach -t $SESSION