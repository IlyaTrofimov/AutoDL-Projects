import torch
import torch.nn.functional as F

class KDLoss(torch.nn.Module):
    def __init__(self, T = 1, alpha = 0):
        super(KDLoss, self).__init__()
        self.T = T
        self.alpha = alpha

    def forward(self, y_s, target, y_t):
        loss = (1 - self.alpha) * F.cross_entropy(y_s, target)

        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss += self.alpha * F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss
