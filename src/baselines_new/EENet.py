import numpy as np

from .EENetClass import Decision_maker, Exploitation, Exploration


class EE_Net:
    def __init__(
        self,
        dim,
        n_arm,
        pool_step_size,
        lr_1=0.01,
        lr_2=0.01,
        lr_3=0.01,
        hidden=100,
        neural_decision_maker=False,
        kernel_size=100,
    ):
        self.f_1 = Exploitation(dim, n_arm, pool_step_size, lr_1, hidden)
        f_2_input_dim = self.f_1.total_param // pool_step_size + 1
        self.f_2 = Exploration(f_2_input_dim, lr_2, kernel_size=kernel_size)
        self.f_3 = Decision_maker(2, 20, lr_3)

        self.arm_select = 0
        self.exploit_scores = []
        self.explore_scores = []
        self.ee_scores = []
        self.grad_list = []
        self.contexts = []
        self.rewards = []
        self.decision_maker = neural_decision_maker

    def predict(self, context, t):
        self.exploit_scores, self.grad_list = self.f_1.output_and_gradient(context)
        self.explore_scores = self.f_2.output(self.grad_list)
        self.ee_scores = np.concatenate((self.exploit_scores, self.explore_scores), axis=1)

        if self.decision_maker and t > 500:
            self.arm_select = self.f_3.select(self.ee_scores)
        else:
            f_2_weight = 0.1
            if t > 1000:
                f_2_weight = 0.01
            scores = self.exploit_scores + f_2_weight * (self.explore_scores - 1.0)
            self.arm_select = int(np.argmax(scores))
        return self.arm_select

    def update(self, context, r_1, t):
        self.f_1.update(context[self.arm_select], r_1)
        self.contexts.append(context[self.arm_select])
        self.rewards.append(float(r_1))

        f_1_predict = self.exploit_scores[self.arm_select][0]
        r_2 = (r_1 - f_1_predict) + 1.0
        self.f_2.update(self.grad_list[self.arm_select], r_2)

        if t < 1000 and r_1 == 0:
            for idx, grad in enumerate(self.grad_list):
                if idx != self.arm_select:
                    self.f_2.update(grad, 1.2)

        self.f_3.update(self.ee_scores[self.arm_select], float(r_1))

    def train(self, t=None):
        loss_1 = self.f_1.train()
        loss_2 = self.f_2.train()
        loss_3 = self.f_3.train() if self.decision_maker else 0.0
        return loss_1, loss_2, loss_3
