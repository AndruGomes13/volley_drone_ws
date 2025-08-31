from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from typing_extensions import Self
import scipy
import pandas as pd
import numpy as np


class CsvData(Protocol):
    @classmethod
    def from_csv(cls, path: Path) -> Self:
        ...
    
    @property
    def df(self) -> pd.DataFrame:
        ...
        
    @property
    def time(self) -> pd.Series:
        ...
    
    @property
    def sequence(self) -> pd.Series:
        ...
    
    @property
    def no_sequence_data(self) -> pd.Series:
        ...
        
    
def sanitize_order(data: CsvData):
    """
    Ensure the DataFrame is sorted by time.
    Everything is done in place.
    """
    sequence = data.sequence    
    timestamps = data.time
    
    # Drop duplicate sequence numbers
    to_keep = sequence.drop_duplicates(keep="first").index
    df = data.df.loc[to_keep].reset_index(drop=True)
    sequence = sequence.loc[to_keep].reset_index(drop=True)
    timestamps = timestamps.loc[to_keep].reset_index(drop=True)
    
    # Sort by sequence number
    sorted_ix = sequence.sort_values().index
    df = df.loc[sorted_ix].reset_index(drop=True)
    sequence = sequence.loc[sorted_ix].reset_index(drop=True)
    timestamps = timestamps.loc[sorted_ix].reset_index(drop=True)
    
    # Ensure timestamps are in ascending order
    oldest_time = timestamps.iloc[0]
    invalid_indices = []
    duplicates = []
    for i in range(1, len(timestamps)):
        if timestamps.iloc[i] < oldest_time:
            invalid_indices.append(i)
        if timestamps.iloc[i] == oldest_time:
            # Sometimes repeated values are sent (check if the rest of the df is equal)
            is_equal = data.no_sequence_data.iloc[i].equals(data.no_sequence_data.iloc[i-1])
            if not is_equal:
                print(f"Index {i} is repeated but data is not equal at index {i}.")
                invalid_indices.append(i)
            duplicates.append(i)
        else:
            oldest_time = timestamps.iloc[i]
            
    if invalid_indices:
        print(f"Found {len(invalid_indices)} timestamp violations in {data.__class__.__name__}.")
        print("Indices of violations:", invalid_indices)

        # Check if they are individual or consecutive
        consecutive = np.diff(invalid_indices) == 1
        if np.any(consecutive):
            print(f"Consecutive violations: {consecutive.sum()} out of {len(invalid_indices)}")
            print("If there are too many consecutive violations, consider checking the data source.")
        else:
            print("All violations are individual.")

        print("Removing rows with timestamp violations.")
        df = df.drop(index=invalid_indices).reset_index(drop=True)

    if duplicates:
        print(f"Found {len(duplicates)} duplicate timestamps in {data.__class__.__name__}.")
        print("Indices of duplicates:", duplicates)
        # Optionally, you can drop duplicates or handle them as needed
        df = df.drop_duplicates(subset=["header_seq"], keep="first").reset_index(drop=True)
        print("Duplicates removed.")

    if not invalid_indices and not duplicates:
        print(f"No timestamp violations found in {data.__class__.__name__}.")
        
    return df

    
    

@dataclass
class QuadStateEstimates:
    df: pd.DataFrame
    
    @classmethod
    def from_csv(cls, path: Path) -> Self:
        df = pd.read_csv(path)
        # df = pd.read_csv(path, sep=",", quotechar='"', engine="python")
        

        # Drop columns you don't want
        cols_to_drop = [
            "motors",
            "acc_bias_x", "acc_bias_y", "acc_bias_z",
            "gyr_bias_x", "gyr_bias_y", "gyr_bias_z",
        ]
        
        # Cols to rename for consistency
        cols_to_rename = {
            "pose_orientation_x": "quat_x",
            "pose_orientation_y": "quat_y",
            "pose_orientation_z": "quat_z",
            "pose_orientation_w": "quat_w",
            "pose_position_x": "pos_x",
            "pose_position_y": "pos_y",
            "pose_position_z": "pos_z",
            "velocity_linear_x": "vel_lin_x",
            "velocity_linear_y": "vel_lin_y",
            "velocity_linear_z": "vel_lin_z",
            "velocity_angular_x": "vel_ang_x",
            "velocity_angular_y": "vel_ang_y",
            "velocity_angular_z": "vel_ang_z",
            "acceleration_linear_x": "acc_lin_x",
            "acceleration_linear_y": "acc_lin_y",
            "acceleration_linear_z": "acc_lin_z",
            "acceleration_angular_x": "acc_ang_x",
            "acceleration_angular_y": "acc_ang_y",
            "acceleration_angular_z": "acc_ang_z",
            
        }
        
        df = df.drop(columns=cols_to_drop, errors="ignore")  # errors="ignore" skips if missing
        df = df.rename(columns=cols_to_rename)

        self = cls(df)
        self.df = sanitize_order(self)
        
        return self
    @property
    def time(self):
        return self.df["header_stamp"]

    @property
    def sequence(self):
        return self.df["header_seq"]

    @property
    def position(self):
        return self.df[["pos_x", "pos_y", "pos_z"]]

    @property
    def orientation(self):
        return self.df[["quat_w", "quat_x", "quat_y", "quat_z"]]
    
    @property
    def orientation_xyzw(self):
        return self.df[["quat_x", "quat_y", "quat_z", "quat_w"]]

    @property
    def velocity_linear(self):
        return self.df[["vel_lin_x", "vel_lin_y", "vel_lin_z"]]

    @property
    def velocity_angular(self):
        return self.df[["vel_ang_x", "vel_ang_y", "vel_ang_z"]]
    
    @property
    def acceleration_linear(self):
        return self.df[["acc_lin_x", "acc_lin_y", "acc_lin_z"]]
    
    @property
    def acceleration_angular(self):
        return self.df[["acc_ang_x", "acc_ang_y", "acc_ang_z"]]
    
    @property
    def no_sequence_data(self):
        # Returns all but the sequence data
        return self.df.loc[:, self.df.columns != "header_seq"]
    
@dataclass
class ViconMeasurements:
    df: pd.DataFrame

    @classmethod
    def from_csv(cls, path: Path):
        df = pd.read_csv(path)
        self = cls(df)
        self.df = sanitize_order(self)
        return self
    
    @property
    def position(self):
        return self.df[["pos_x", "pos_y", "pos_z"]]

    @property
    def orientation(self):
        """ [w, x, y, z] quaternion format """
        return self.df[["quat_w", "quat_x", "quat_y", "quat_z"]]
    
    @property
    def orientation_xyzw(self):
        return self.df[["quat_x", "quat_y", "quat_z", "quat_w"]]
    
    @property
    def time(self):
        return self.df["header_stamp"]  
    
    @property
    def sequence(self):
        return self.df["header_seq"]
    
    @property
    def no_sequence_data(self):
        # Returns all but the sequence data
        return self.df.loc[:, self.df.columns != "header_seq"]

@dataclass
class Command:
    df: pd.DataFrame

    @classmethod
    def from_csv(cls, path: Path):
        df = pd.read_csv(path)
        self = cls(df)
        self.df = sanitize_order(self)
        return self

    @property
    def time(self):
        return self.df["t"]
    
    @property
    def sequence(self):
        return self.df["header_seq"]

    @property
    def collective_thrust(self):
        return self.df["collective_thrust"]

    @property
    def body_rates(self):
        return self.df[["bodyrates_x", "bodyrates_y", "bodyrates_z"]]

    @property
    def is_single_rotor_thrust(self):
        return self.df["is_single_rotor_thrust"]
    
    @property
    def no_sequence_data(self):
        # Returns all but the sequence data
        return self.df.loc[:, self.df.columns != "header_seq"]

    @property
    def thrusts(self):
        return self.df[["thrusts_1", "thrusts_2", "thrusts_3", "thrusts_4"]]


if __name__ == "__main__":
    # Example usage
    state_path = Path("/home/agilicious/catkin_ws/ros_logs/state_log.csv")
    estimates = QuadStateEstimates.from_csv(state_path)
    
    vicon_path = Path("/home/agilicious/catkin_ws/ros_logs/mocap_log.csv")
    vicon = ViconMeasurements.from_csv(vicon_path)

    print("Position:\n", estimates.position.head(n=10))
    print("Position Vicon:\n", vicon.position.head(n=10))
