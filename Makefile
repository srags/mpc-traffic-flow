.PHONY: test test-detailed

test:
	pytest tests --color=yes \
		--deselect=tests/test_solver_baseline.py::test_real_calibration_and_mpc_match_baseline

test-detailed:
	MPC_RUN_SOLVER_BASELINE=1 pytest tests --color=yes
