
class I24_400m_6seg_Params:
    def __init__(self, num_segments=6):
        self.tau = [
            0.02409958399431391,
            0.08694688873472797,
            0.014723330431847438,
            0.026617154394912794,
            0.012370746918461935,
            0.012766217659745394
        ]

        self.K = [
            -9.999868712736364e-09,
            -9.999795346938163e-09,
            -9.997551703130613e-09,
            -9.998592688908707e-09,
            16.110865667566273,
            -9.999876411758227e-09
        ]

        self.eta_high = [
            100.00000099999798,
            100.00000099999797,
            60.86948502082235,
            100.00000099997224,
            77.65643573760117,
            22.4239595015379
        ]



        self.params = {'v_free': [132.740 for i in range(num_segments)],
                'a': [1.445 for i in range(num_segments)],
                'p_crit':  [14.270 for i in range(num_segments)], #14.270,
                'q_capacity':[1533.7565549999936 for i in range(num_segments)],
                'K': self.K,
                'eta_high': self.eta_high,
                'tau':self.tau}
    
    def get_params(self):
        return self.params