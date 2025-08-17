from dataclasses import dataclass
from pathlib import Path
from typing import Self
import scipy
import pandas as pd
import numpy as np


@dataclass
class QuadStateEstimates:
    df: pd.DataFrame
    
    @classmethod
    def from_csv(cls, path: Path) -> Self:
        df = pd.read_csv(path)
        df = cls._sanitize_order(df)

        # Drop columns you don't want
        cols_to_drop = [
            "motors",
            "acc_bias_x", "acc_bias_y", "acc_bias_z",
            "gyr_bias_x", "gyr_bias_y", "gyr_bias_z",
        ]
        
        # Cols to rename for consistency
        cols_to_rename = {
            "pose_quat_x": "quat_x",
            "pose_quat_y": "quat_y",
            "pose_quat_z": "quat_z",
            "pose_quat_w": "quat_w",
            "pose_pos_x": "pos_x",
            "pose_pos_y": "pos_y",
            "pose_pos_z": "pos_z",
        }
        
        df = df.drop(columns=cols_to_drop, errors="ignore")  # errors="ignore" skips if missing
        df = df.rename(columns=cols_to_rename)

        return cls(df)
    @property
    def time(self):
        return self.df["t"]

    @property
    def position(self):
        return self.df[["pos_x", "pos_y", "pos_z"]]

    @property
    def orientation(self):
        return self.df[["quat_w", "quat_x", "quat_y", "quat_z"]]

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
    
    @staticmethod
    def _sanitize_order(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure the DataFrame is sorted by time.
        """
        df = df.sort_values(by="header_seq").reset_index(drop=True)
        df = df.drop_duplicates(subset=["header_seq"], keep="first").reset_index(drop=True)

        # Loop over timestamps to check for timestamp violations
        time_col = df["header_stamp"].values
        oldest_time = time_col[0]
        invalid_indices = []
        for i in range(1, len(time_col)):
            if time_col[i] <= oldest_time:
                invalid_indices.append(i)
            else:
                oldest_time = time_col[i]
                
        if invalid_indices:
            print(f"Found {len(invalid_indices)} timestamp violations in state estimates.")
            print("Indices of violations:", invalid_indices)
            df = df.drop(index=invalid_indices).reset_index(drop=True)
        else:
            print("No timestamp violations found in state estimates.")   
            
        return df


@dataclass
class ViconMeasurements:
    df: pd.DataFrame

    @classmethod
    def from_csv(cls, path: Path):
        df = pd.read_csv(path)
        df = cls._sanitize_order(df)
        return cls(df)
    
    @property
    def position(self):
        return self.df[["pos_x", "pos_y", "pos_z"]]

    @property
    def orientation(self):
        return self.df[["quat_w", "quat_x", "quat_y", "quat_z"]]
    
    @property
    def time(self):
        return self.df["header_stamp"]  
    
    @staticmethod
    def _sanitize_order(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure the DataFrame is sorted by time.
        """
        df = df.drop_duplicates(subset=["header_seq"], keep="first").reset_index(drop=True)
        df = df.sort_values(by="header_seq").reset_index(drop=True)

        # Loop over timestamps to check for timestamp violations
        time_col = df["header_stamp"].values
        oldest_time = time_col[0]
        invalid_indices = []
        for i in range(1, len(time_col)):
            if time_col[i] <= oldest_time:
                invalid_indices.append(i)
            else:
                oldest_time = time_col[i]
                
        if invalid_indices:
            print(f"Found {len(invalid_indices)} timestamp violations in Vicon data.")
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

        else:
            print("No timestamp violations found in Vicon data.")   
            
        return df

if __name__ == "__main__":
    # Example usage
    state_path = Path("/home/agilicious/catkin_ws/ros_logs/state_log.csv")
    estimates = QuadStateEstimates.from_csv(state_path)
    
    vicon_path = Path("/home/agilicious/catkin_ws/ros_logs/mocap_log.csv")
    vicon = ViconMeasurements.from_csv(vicon_path)

    print("Position:\n", estimates.position.head(n=10))
    print("Position Vicon:\n", vicon.position.head(n=10))
    # print("Orientation:\n", estimates.orientation.head())
    # print("Linear Velocity:\n", estimates.velocity_linear.head())
    # print("Angular Velocity:\n", estimates.velocity_angular.head())