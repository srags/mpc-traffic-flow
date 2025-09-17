
import numpy as np
import pandas as pd

class I24_30seg_200m_indivFD_Params:
    def __init__(self, num_segments=30, joint=False):
        
            # "tau": np.load('/Users/shreyaar/Desktop/PhD/research/MPC/src/sim_params/200m_10_sec/tau.npy').reshape(-1).tolist()[::-1],
            # "K": np.load('/Users/shreyaar/Desktop/PhD/research/MPC/src/sim_params/200m_10_sec/K.npy').reshape(-1).tolist()[::-1],
            # "eta_high": np.load('/Users/shreyaar/Desktop/PhD/research/MPC/src/sim_params/200m_10_sec/eta_high.npy').reshape(-1).tolist()[::-1],
            # "p_crit": np.load('/Users/shreyaar/Desktop/PhD/research/MPC/src/sim_params/200m_10_sec/rho_crit.npy').reshape(-1).tolist()[::-1],
            # "v_free": np.load('/Users/shreyaar/Desktop/PhD/research/MPC/src/sim_params/200m_10_sec/v_free.npy').reshape(-1).tolist()[::-1],
            # "a": np.load('/Users/shreyaar/Desktop/PhD/research/MPC/src/sim_params/200m_10_sec/a.npy').reshape(-1).tolist()[::-1],
            if joint:
                self.params = {
                    "tau": np.load('/Users/shreyaar/Desktop/PhD/research/MPC/src/sim_params/200_m_10_s_joint/tau.npy').reshape(-1).tolist(),
                    "K": np.load('/Users/shreyaar/Desktop/PhD/research/MPC/src/sim_params/200_m_10_s_joint/K.npy').reshape(-1).tolist(),
                    "eta_high": np.load('/Users/shreyaar/Desktop/PhD/research/MPC/src/sim_params/200_m_10_s_joint/eta_high.npy').reshape(-1).tolist(),
                    "p_crit": np.load('/Users/shreyaar/Desktop/PhD/research/MPC/src/sim_params/200_m_10_s_joint/rho_crit.npy').reshape(-1).tolist(),
                    "v_free": np.load('/Users/shreyaar/Desktop/PhD/research/MPC/src/sim_params/200_m_10_s_joint/v_free.npy').reshape(-1).tolist(),
                    "a": np.load('/Users/shreyaar/Desktop/PhD/research/MPC/src/sim_params/200_m_10_s_joint/a.npy').reshape(-1).tolist(),
                    'q_capacity':[2200 for i in range(num_segments)]
                }
            else:
                self.params = {
                    "tau": np.load('/Users/shreyaar/Desktop/PhD/research/MPC/src/sim_params/200m_10_sec/tau.npy').reshape(-1).tolist(),
                    "K": np.load('/Users/shreyaar/Desktop/PhD/research/MPC/src/sim_params/200m_10_sec/K.npy').reshape(-1).tolist(),
                    "eta_high": np.load('/Users/shreyaar/Desktop/PhD/research/MPC/src/sim_params/200m_10_sec/eta_high.npy').reshape(-1).tolist(),
                    "p_crit": np.load('/Users/shreyaar/Desktop/PhD/research/MPC/src/sim_params/200m_10_sec/rho_crit.npy').reshape(-1).tolist(),
                    "v_free": np.load('/Users/shreyaar/Desktop/PhD/research/MPC/src/sim_params/200m_10_sec/v_free.npy').reshape(-1).tolist(),
                    "a": np.load('/Users/shreyaar/Desktop/PhD/research/MPC/src/sim_params/200m_10_sec/a.npy').reshape(-1).tolist(),
                    'q_capacity':[2200 for i in range(num_segments)]
                }

    def get_params(self):
        return self.params

    def get_param(self, key):
        return self.params.get(key, None)