import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sim_types import MetanetParams, MetanetState
from traffic_sim import (
    METANET_Simulator,
    calculate_V,
    density_dynamics,
    origin_flow_dynamics_MN,
    queue_dynamics,
    velocity_dynamics_MN,
)


class TestMetanetEquations(unittest.TestCase):
    T, length = 10 / 3600, 0.4

    def test_queue_dynamics(self):
        result = queue_dynamics(current=20.0, demand=3000.0, flow_origin=9600.0, T=self.T)
        expected = 20.0 + self.T * (3000.0 - 9600.0)
        self.assertAlmostEqual(result, expected)

    def test_density_dynamics_with_ramps(self):
        result = density_dynamics(
            current=20.0,
            inflow=3000.0,
            outflow=2400.0,
            lanes=4.0,
            T=self.T,
            l=self.length,
            beta=0.1,
            r=200.0,
        )
        expected = max(1e-4, 20.0 + self.T / (self.length * 4.0)* (3000.0 - 2400.0 / (1.0 - 0.1) + 200.0))
        self.assertAlmostEqual(result, expected)

    def test_calculate_V_when_vsl_binds(self):
        result = calculate_V(rho=20.0, v_ctrl=50.0, a=1.4, p_crit=37.45, v_free=120.0)
        expected_free_flow = 120.0 * np.exp(-(20.0 / 37.45) ** 1.4 / 1.4)
        self.assertAlmostEqual(result, min(expected_free_flow, 50.0))
        self.assertAlmostEqual(result, 50.0)

    def test_calculate_V_when_density_binds(self):
        result = calculate_V(rho=50.0, v_ctrl=150.0, a=1.4, p_crit=37.45, v_free=120.0)
        expected = 120.0 * np.exp(-(50.0 / 37.45) ** 1.4 / 1.4)
        self.assertAlmostEqual(result, expected)

    def test_velocity_dynamics(self):
        result = velocity_dynamics_MN(
            current=70.0, prev_state=80.0, density=35.0, next_density=42.0,
            v_ctrl=65.0, T=self.T, l=self.length, eta_high=30.0,
            K=40.0, tau=18 / 3600, a=1.4, p_crit=37.45, v_free=120.0,
        )

        desired_speed = min(120.0 * np.exp(-(35.0 / 37.45) ** 1.4 / 1.4), 65.0)

        expected_raw = (
            70.0 + self.T / (18 / 3600) * (desired_speed - 70.0) + self.T / self.length * 70.0 * (80.0 - 70.0)
            - (30.0 * self.T) / ((18 / 3600) * self.length) * (42.0 - 35.0) / (35.0 + 40.0)
        )

        self.assertAlmostEqual(result, max(1e-4, expected_raw))

    def test_origin_flow(self):
        result = origin_flow_dynamics_MN(
            demand=3000.0, density_first=20.0, queue=20.0, lanes=4.0,
            T=self.T, p_max=180.0, p_crit=37.45, q_capacity=2400.0,
        )

        expected = min(
            3000.0 + 20.0 / self.T,
            4.0 * 2400.0 * (180.0 - 20.0) / (180.0 - 37.45),
            4.0 * 2400.0,
        )

        self.assertAlmostEqual(result, expected)

class TestSimulatorRegression(unittest.TestCase):
    def setUp(self):
        self.params: MetanetParams = {
            "tau": np.array([18, 20, 22], dtype=float) / 3600,
            "K": np.array([40, 42, 44], dtype=float),
            "eta_high": np.array([30, 35, 40], dtype=float),
            "p_crit": np.array([37.45, 40, 42], dtype=float),
            "v_free": np.array([120, 115, 110], dtype=float),
            "a": np.array([1.4, 1.6, 1.8], dtype=float),
            "q_capacity": np.array([2400, 2300, 2200], dtype=float),
            "r": np.array([0, 150, 0], dtype=float),
            "beta": np.array([0, 0.05, 0], dtype=float),
            "gamma": np.ones(3),
        }

        self.lanes = {0: 4.0, 1: 3.0, 2: 2.0}

        self.state = MetanetState(
            density=np.array([20.0, 35.0, 50.0]),
            velocity=np.array([100.0, 70.0, 40.0]),
            demand=3000.0, queue=20.0,
        )

    def make_simulator(self):
        return METANET_Simulator(
            T=10 / 3600, l=0.4,
            params=self.params,
            lanes=self.lanes,
            real_data=False,
        )

    def test_one_step_matches_current_behavior(self):
        demand = np.array([3000.0, 3200.0])
        downstream_density = np.array([45.0])
        vsl = np.array([[105.0, 80.0, 55.0]])

        final_state, _ = self.make_simulator().run(
            demand, downstream_density, self.state, vsl,
        )

        np.testing.assert_allclose(
            final_state.density,
            [
                22.77777777777778,
                35.956384015594544,
                61.63194444444444,
            ],
            rtol=1e-11, atol=1e-11,
        )

        np.testing.assert_allclose(
            final_state.velocity,
            [
                83.57336058516826,
                75.77018447929916,
                55.94398917099457,
            ],
            rtol=1e-11, atol=1e-11,
        )

        self.assertAlmostEqual(final_state.queue, 1.666666666666664)
        self.assertAlmostEqual(final_state.demand, 3799.999999999999)

    def test_history_methods_preserve_states_and_travel_time(self):
        for real_data in (False, True):
            for steps in (0, 1, 3):
                for controlled in (False, True):
                    with self.subTest(real_data=real_data, steps=steps, controlled=controlled):
                        demand = np.linspace(3000., 3300., steps + 1)
                        downstream = np.full(steps, 45.)
                        vsl = np.tile([105., 80., 55.], (steps, 1)) if controlled else None

                        def simulator():
                            sim = self.make_simulator()
                            sim.real_data = real_data
                            return sim

                        # Record the existing initialization/step contract independently
                        # of either public history collector.
                        reference = simulator()
                        reference._initialize(demand, downstream, self.state, vsl)
                        states = [reference.cur_state]
                        for t in range(steps):
                            states.append(reference._step(t, states[-1]))
                        expected_density = np.stack([s.density for s in states])
                        expected_velocity = np.stack([s.velocity for s in states])
                        expected_queue = np.array([[s.queue] for s in states])
                        expected_flow = np.array([[s.demand] for s in states])

                        final, streaming_ttt = simulator().run(demand, downstream, self.state, vsl)
                        history_sim = simulator()
                        density, velocity, queue, ttt = history_sim.run_with_history(
                            demand, downstream, self.state, vsl
                        )
                        opt_sim = simulator()
                        opt_density, opt_velocity, opt_queue, flow, fd, opt_ttt = opt_sim.run_with_opt(
                            demand, downstream, self.state, vsl
                        )

                        for actual, expected in (
                            (density, expected_density), (opt_density, expected_density),
                            (velocity, expected_velocity), (opt_velocity, expected_velocity),
                            (queue, expected_queue), (opt_queue, expected_queue),
                            (flow, expected_flow),
                        ):
                            np.testing.assert_array_equal(actual, expected)
                        for sim in (history_sim, opt_sim):
                            np.testing.assert_array_equal(sim.cur_state.density, final.density)
                            np.testing.assert_array_equal(sim.cur_state.velocity, final.velocity)
                            self.assertEqual(sim.cur_state.queue, final.queue)
                            self.assertEqual(sim.cur_state.demand, final.demand)
                        self.assertAlmostEqual(ttt, streaming_ttt)
                        self.assertAlmostEqual(opt_ttt, streaming_ttt)
                        expected_fd = np.empty((steps, 3))
                        for t in range(steps):
                            for i in range(3):
                                expected_fd[t, i] = calculate_V(
                                    expected_density[t, i], vsl[t, i] if vsl is not None else 1000,
                                    self.params["a"][i], self.params["p_crit"][i],
                                    self.params["v_free"][i],
                                )
                        np.testing.assert_allclose(fd, expected_fd, rtol=1e-12, atol=1e-12)
