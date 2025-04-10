from sumo_rl.environment.observations import ObservationFunction
import numpy as np
from gymnasium import spaces

from sumo_rl.environment.traffic_signal import TrafficSignal

class CustomObservationFunction(ObservationFunction):
    """Custom observation function for traffic signals."""

    def __init__(self, ts: TrafficSignal):
        """Initialize custom observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        """Return the custom observation."""
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        pressure = self.ts.get_pressure() 
        observation = np.array(phase_id + min_green + density + queue + [pressure], dtype=np.float32)
        return observation

    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        return spaces.Box(
            low=np.zeros(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes) + 1, dtype=np.float32),
            high=np.ones(self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes) + 1, dtype=np.float32),
        )
