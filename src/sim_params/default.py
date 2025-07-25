class DefaultParams:
    def __init__(self, num_segments):
        self.tau = [
            18/3600 for i in range(0,num_segments)
        ]

        self.K = [
            40 for i in range(0,num_segments)
        ]

        self.eta_high = [
            30 for i in range(0,num_segments)
        ]


        self.params = {'v_free':[120 for i in range(num_segments)], #88.654,
                'a': [1.4 for i in range(num_segments)],
                'p_crit':[37.45 for i in range(num_segments)],
                'q_capacity':[2200 for i in range(num_segments)],
                'K': self.K,
                'eta_high': self.eta_high,
                'tau':self.tau}
    
    def get_params(self):
        return self.params