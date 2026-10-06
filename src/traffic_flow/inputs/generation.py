"""Synthetic demand and perturbed input generation."""
import numpy as np

def generate_demand(sim_time, time_step, peak_start=0.25, peak_end=0.75, flow_standard=2400, flow_peak=3000):
    """
    Generate a traffic demand profile for a given simulation time.

    Parameters:
    - sim_time: Total simulation time in hours.
    - time_step: Time step for the simulation in hours.
    - peak_start, peak_end: Peak interval in hours.
    - flow_standard, flow_peak: Demand outside/inside the peak (veh/hr).

    Returns:
    - Demand array, including the terminal sample (historical convention).
    """
    total_time_steps=int(sim_time/time_step)
    traffic_demand=np.full(total_time_steps+1,flow_standard)
    peak_start=int(peak_start/time_step)
    peak_end=int(peak_end/time_step)
    traffic_demand[peak_start:peak_end]=flow_peak
    return traffic_demand

def generate_perturbations(signal, percent_noise=0.1, seed=0, num=100):
    """
    Generate perturbations for a given signal.
    signal + normal(0, percent_noise * std(signal) / 100)
    """
    rng = np.random.default_rng(seed)
    sigma = percent_noise * np.std(signal) / 100
    noise = rng.normal(loc=0.0, scale=sigma, size=(num, *signal.shape))
    return signal[None, ...] + noise