class LinearCyclicBetaScheduler(object):
    def __init__(self, total_steps, n_cycles, min_beta, max_beta):
        self.steps_per_cycle = total_steps // n_cycles
        self.plateau_steps_per_cycle = self.steps_per_cycle // 2

        self.min_beta = min_beta
        self.max_beta = max_beta

        self.linear_coef = (max_beta - min_beta) / self.plateau_steps_per_cycle

    
    def __getitem__(self, step):
        cycle_step = step % self.steps_per_cycle

        if cycle_step >= self.plateau_steps_per_cycle:
            return self.max_beta

        else:
            return self.linear_coef * cycle_step + self.min_beta
