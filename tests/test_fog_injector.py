import sys
import unittest
from pathlib import Path

import numpy as np


PIPELINE_SRC = (
    Path(__file__).resolve().parents[1]
    / "lidar-fault-localization_weather_faults"
    / "src"
)
sys.path.insert(0, str(PIPELINE_SRC))

from lfi.fault_models.FOG_Injector import FogSimulator  # noqa: E402


class FogSimulatorTest(unittest.TestCase):
    def test_seeded_injection_preserves_point_cloud_contract(self):
        ranges = np.linspace(1.0, 100.0, 2_000)
        points = np.column_stack((ranges, np.zeros_like(ranges), np.zeros_like(ranges)))

        first = FogSimulator(V=50, seed=7)
        second = FogSimulator(V=50, seed=7)
        first_output = first.apply_noise_fog(points.copy())
        second_output = second.apply_noise_fog(points.copy())

        np.testing.assert_allclose(first_output, second_output)
        self.assertEqual(first_output.shape[1], 3)
        self.assertGreater(first_output.shape[0], 0)
        self.assertLess(first_output.shape[0], points.shape[0])
        self.assertEqual(first.stats["total"], points.shape[0])
        self.assertEqual(
            first.stats["deleted"], points.shape[0] - first_output.shape[0]
        )

    def test_rejects_non_xyz_input(self):
        simulator = FogSimulator(V=50, seed=7)
        with self.assertRaisesRegex(ValueError, r"\(N,3\)"):
            simulator.apply_noise_fog(np.zeros((10, 4)))


if __name__ == "__main__":
    unittest.main()
