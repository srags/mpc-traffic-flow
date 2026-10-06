"""Two critical failure checks: retry rejected solver output; never return a failed run."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import traffic_flow.solvers.mpc as mpc
from traffic_flow.model.parameters import load_metanet_params
from traffic_flow.types import MetanetState, MPCSolveResult


@pytest.fixture
def solver_loading_case(monkeypatch):
    """Build real Pyomo models, but never invoke an external solver."""
    result = SimpleNamespace(
        solver=SimpleNamespace(status=mpc.pyo.SolverStatus.error,
                               termination_condition=mpc.pyo.TerminationCondition.internalSolverError,
                               message="simulated solver failure"),
        solution=[object()],
    )
    loader = Mock(side_effect=AssertionError("Failed results must not be loaded"))
    solver = Mock()
    solver.options = {}

    def solve(model, **kwargs):
        # Before the production fix, automatic loading would throw ValueError
        # here, bypassing the outer loop's MPCSolveError retry handler.
        if kwargs.get("load_solutions", True):
            raise ValueError("Cannot load a SolverResults object with bad status: error")
        monkeypatch.setattr(model.solutions, "load_from", loader)
        return result

    solver.solve.side_effect = solve
    monkeypatch.setattr(mpc.pyo, "SolverFactory", Mock(return_value=solver))
    arguments = dict(
        T=10 / 3600, l=0.4, num_segments=2,
        traffic_demand=np.full(4, 1000.), downstream_density=np.full(4, 20.),
        horizon_p=3, horizon_c=2,
        starting_traffic_vars=MetanetState(np.array([10., 20.]), np.array([90., 80.]), 1000., 0.),
        lanes={0: 4., 1: 3.}, hold_len=1, params=load_metanet_params(num_segments=2),
    )
    return SimpleNamespace(result=result, loader=loader, solver=solver, arguments=arguments)


def test_reported_solver_error_reaches_next_initialization(solver_loading_case, monkeypatch):
    case = solver_loading_case
    original_opt = mpc.mpc_opt
    attempts = []
    policy = np.full((3, 2), 100.)

    def attempt(*args, **kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            return original_opt(*args, **kwargs)
        return MPCSolveResult(0, 0., policy, policy.copy(), 0., 0.)

    monkeypatch.setattr(mpc, "mpc_opt", attempt)
    monkeypatch.setattr(mpc, "tqdm", Mock(return_value=Mock()))
    actual = mpc.mpc_find_vsl(
        3, case.arguments["traffic_demand"], case.arguments["downstream_density"],
        case.arguments["lanes"], params=case.arguments["params"],
        T=10 / 3600, l=0.4, num_segments=2, pred_horizon=3, control_horizon=3,
        init_state=case.arguments["starting_traffic_vars"],
        initialize_vsl=np.full((4, 2), 100.), init_fixed=150.,
    )
    assert len(attempts) == 2
    assert attempts[0]["initialize"] is not None
    assert attempts[0]["init_fixed"] is None
    assert attempts[1]["initialize"] is None
    assert attempts[1]["init_fixed"] == 150.
    case.loader.assert_not_called()
    np.testing.assert_array_equal(actual, policy)


@pytest.fixture
def retry_case(monkeypatch):
    state = MetanetState(np.array([10., 20.]), np.array([90., 80.]), 1000., 0.)
    control = np.full((3, 2), 85.)
    result = MPCSolveResult(2, 0.01, control, control.copy(), 0., 0.)
    fitter = create_autospec(mpc.mpc_opt, return_value=result)
    monkeypatch.setattr(mpc, "mpc_opt", fitter)
    simulator = Mock()
    simulator.run.return_value = (state, 0.)
    monkeypatch.setattr(mpc, "METANET_Simulator", Mock(return_value=simulator))
    progress = Mock()
    monkeypatch.setattr(mpc, "tqdm", Mock(return_value=progress))

    def unexpected_solver(*args, **kwargs):
        pytest.fail("Contract tests must never invoke a solver")

    monkeypatch.setattr(mpc.pyo, "SolverFactory", unexpected_solver)
    initialization = np.full((7, 2), 100.)

    def run(**overrides):
        arguments = dict(
            total_time_steps=3, traffic_demand=np.full(7, 1000.),
            downstream_density=np.full(7, 20.), lanes={0: 4., 1: 3.},
            params=load_metanet_params(num_segments=2), num_segments=2,
            pred_horizon=3, control_horizon=3, init_state=state,
            initialize_vsl=initialization, init_fixed=[80., 90.],
        )
        arguments.update(overrides)
        return mpc.mpc_find_vsl(**arguments)

    return SimpleNamespace(run=run, fitter=fitter, simulator=simulator,
                           progress=progress, initialization=initialization,
                           result=result)


def test_exhausted_retries_raise_with_last_failure_as_cause(retry_case):
    case = retry_case
    failures = [mpc.MPCSolveError(f"attempt {i} failed") for i in range(4)]
    case.fitter.side_effect = failures
    with pytest.raises(mpc.MPCSolveError, match="t=0.*exhausting") as caught:
        case.run()
    assert caught.value.__cause__ is failures[-1]
    assert case.fitter.call_count == 4
    case.simulator.run.assert_not_called()
    case.progress.close.assert_called_once()
