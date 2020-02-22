# -*- coding: utf-8 -*-

class Early_Stopping():
    def __init__(self, warmup=0, patience=10, tolerance=1e-3, is_minimize=True):
        self.warmup = warmup
        self.patience = patience
        self.tolerance = tolerance
        self.is_minimize = is_minimize

        self.step = 0
        self.best_step = 0
        self.best_metric = 0

        if not self.is_minimize:
            self.factor = -1
        else:
            self.factor = 1

    def __call__(self, metric):
        if self.step == 0:
            self.best_step = 1
            self.best_metric = metric
            self.step += 1
            return False

        self.step += 1

        if self.factor*metric<self.factor*self.best_metric-self.tolerance:
            self.best_metric = metric
            self.best_step = self.step
        elif self.step - self.best_step>self.patience:
            if self.step < self.warmup:
                return False
            else:
                print('Best Epoch: %d. Best Metric: %f.'%(self.best_step, self.best_metric))
                return True
