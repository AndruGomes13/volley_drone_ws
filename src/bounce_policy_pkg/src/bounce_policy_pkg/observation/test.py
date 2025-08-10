from observation_history import make_history_cls
from observation import DroneStateObs


lengths = {
    "drone_position": 4,
    "drone_orientation": 4,
    "drone_velocity": 4,
    "drone_body_rate": 4,
    "previous_action": 4,
}   
drone_state_hist = make_history_cls(DroneStateObs, lengths, 2)

ex = DroneStateObs.generate_random(1)
hist = drone_state_hist.init(ex)
print("Initial history:", hist.to_array())
