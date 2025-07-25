
class I24_400mParams:
    def __init__(self, num_segments):
        self.tau = [
            0.024377753102834267 for i in range(0,num_segments)
        ]

        self.K = [
            3.042965938847628e-09 for i in range(0,num_segments)
        ]

        self.eta_high = [
            31.790742185850878 for i in range(0,num_segments)
        ]


        self.params = {'v_free':135.732, #88.654,
                'a':1.524,
                'p_crit':13.419,
                'q_capacity':1284.4218239999934,
                'K': self.K,
                'eta_high': self.eta_high,
                'tau':self.tau}
    
    def get_params(self):
        return self.params