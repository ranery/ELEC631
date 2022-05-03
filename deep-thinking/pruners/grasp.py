import torch
import numpy as np
from tqdm import tqdm
from random import randrange

from .pruner import Pruner


def get_output_for_prog_loss(inputs, max_iters, net):
    # get features from n iterations to use as input
    n = randrange(0, max_iters)

    # do k iterations using intermediate features as input
    k = randrange(1, max_iters - n + 1)

    if n > 0:
        _, interim_thought = net(inputs, iters_to_do=n)
        interim_thought = interim_thought.detach()
    else:
        interim_thought = None

    outputs, _ = net(inputs, iters_elapsed=n, iters_to_do=k, interim_thought=interim_thought)
    return outputs, k

def cal_loss(model, data, targets, scale, mask=None):
    # prefix_sum
    max_iters = 30
    alpha = 0.01
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    outputs_max_iters, _ = model(data, iters_to_do=max_iters)
    outputs_max_iters = outputs_max_iters.view(outputs_max_iters.size(0),
                                               outputs_max_iters.size(1), -1)

    outputs = outputs_max_iters / scale

    loss_max_iters = criterion(outputs_max_iters, targets)

    outputs, k = get_output_for_prog_loss(data, max_iters, model)
    outputs = outputs.view(outputs.size(0), outputs.size(1), -1)

    outputs = outputs / scale

    loss_progressive = criterion(outputs, targets)

    # if problem == "mazes":
    if mask is not None:
        loss_max_iters = (loss_max_iters * mask)
        loss_max_iters = loss_max_iters[mask > 0]
        loss_progressive = (loss_progressive * mask)
        loss_progressive = loss_progressive[mask > 0]

    loss_max_iters_mean = loss_max_iters.mean()
    loss_progressive_mean = loss_progressive.mean()

    loss = (1 - alpha) * loss_max_iters_mean + alpha * loss_progressive_mean

    return loss

# Based on https://github.com/alecwangcq/GraSP/blob/master/pruner/GraSP.py#L49
class GraSP(Pruner):
    def __init__(self, masked_parameters):
        super(GraSP, self).__init__(masked_parameters)
        self.temp = 200
        self.eps = 1e-10

    def score(self, model, loss, dataloader, device):

        model.train()

        # first gradient vector without computational graph
        stopped_grads = 0
        for batch_idx, (data, target) in enumerate(tqdm(dataloader, leave=False)):
            data, target = data.to(device), target.to(device).long()
            target = target.view(target.size(0), -1)
            # output = model(data) / self.temp
            # L = loss(output, target)

            # for maze
            mask = data.view(data.size(0), data.size(1), -1).max(dim=1)[0] > 0

            L = cal_loss(model, data, target, scale=self.temp, mask=mask)

            grads = torch.autograd.grad(L, [p for (_, p) in self.masked_parameters], create_graph=False)
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])
            stopped_grads += flatten_grads

            # if batch_idx > 5:
            #     break

        # second gradient vector with computational graph
        for batch_idx, (data, target) in enumerate(tqdm(dataloader, leave=False)):
            data, target = data.to(device), target.to(device).long()
            target = target.view(target.size(0), -1)
            # output = model(data) / self.temp
            # L = loss(output, target)

            L = cal_loss(model, data, target, scale=self.temp)

            grads = torch.autograd.grad(L, [p for (_, p) in self.masked_parameters], create_graph=True)
            flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])

            gnorm = (stopped_grads * flatten_grads).sum()
            gnorm.backward()

            # if batch_idx > 5:
            #     break

        # calculate score Hg * theta (negate to remove top percent)
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p.data).detach()
            p.grad.data.zero_()

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.abs(torch.sum(all_scores)) + self.eps
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)
