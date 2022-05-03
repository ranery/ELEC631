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

# Based on https://github.com/mi-lad/snip/blob/master/snip.py#L18
class SNIP(Pruner):
    def __init__(self, masked_parameters):
        super(SNIP, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):

        # allow masks to have gradient
        for m, _ in self.masked_parameters:
            m.requires_grad = True

        criterion = torch.nn.CrossEntropyLoss(reduction="none")

        # compute gradient
        ## easy to hard
        model.train()

        for batch_idx, (data, targets) in enumerate(tqdm(dataloader, leave=False)):
            data, targets = data.to(device), targets.to(device).long()
            targets = targets.view(targets.size(0), -1)
            # if problem == "mazes":
            # mask = data.view(data.size(0), data.size(1), -1).max(dim=1)[0] > 0

            # prefix_sum
            max_iters = 30
            alpha = 1
            # alpha = 0.01

            outputs_max_iters, _ = model(data, iters_to_do=max_iters)
            outputs_max_iters = outputs_max_iters.view(outputs_max_iters.size(0),
                                                       outputs_max_iters.size(1), -1)
            loss_max_iters = criterion(outputs_max_iters, targets)

            outputs, k = get_output_for_prog_loss(data, max_iters, model)
            outputs = outputs.view(outputs.size(0), outputs.size(1), -1)

            loss_progressive = criterion(outputs, targets)

            # if problem == "mazes":
            # loss_max_iters = (loss_max_iters * mask)
            # loss_max_iters = loss_max_iters[mask > 0]
            # loss_progressive = (loss_progressive * mask)
            # loss_progressive = loss_progressive[mask > 0]

            loss_max_iters_mean = loss_max_iters.mean()
            loss_progressive_mean = loss_progressive.mean()

            loss = (1 - alpha) * loss_max_iters_mean + alpha * loss_progressive_mean
            loss.backward()

        ## keypoint coco
        # data_iterator = iter(dataloader)
        # for batch_idx, data in enumerate(data_iterator):
        #     input, target, target_weight, meta = data
        #     input, target, target_weight = input.to(device), target.to(device), target_weight.to(device)
        #     outputs = model(input)
        #     if isinstance(outputs, list):
        #         _loss = loss(outputs[0], target, target_weight)
        #         for output in outputs[1:]:
        #             _loss += loss(output, target, target_weight)
        #     else:
        #         output = outputs
        #         _loss = loss(output, target, target_weight)
        #     _loss.backward()

        ## ade20k / cityscapes
        # data_iterator = iter(dataloader)
        # for batch_idx, data in enumerate(data_iterator):
        #     if batch_idx > 100:
        #         break
        #     input, target = data
        #     input = input.to(device)
        #     target = target.cuda(non_blocking=True)
        #     output = model(input)
        #     _loss, acc = loss(output, target)
        #     _loss.backward()

        # calculate score |g * theta|
        for m, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(m.grad).detach().abs_()
            p.grad.data.zero_()
            m.grad.data.zero_()
            m.requires_grad = False

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.sum(all_scores)
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)