# pytorch-cw2


## Introduction

This is a _rich-documented_ [PyTorch](https://pytorch.org/) implementation of [Carlini-Wanger's L2 attack](https://arxiv.org/abs/1608.04644).  The main reason to develop this respository is to make it easier to do research using the attach technique.  Another implementation in PyTorch is [rwightman/pytorch-nips2017-attack-example](https://github.com/rwightman/pytorch-nips2017-attack-example.git).  However, the author failed to reproduce the result presented in the original paper.

`cw.py` has been tested under `python 2.7.12` and `torch-0.3.1`.

## References:

- [carlini/nn\_robust\_attacks](https://github.com/carlini/nn_robust_attacks.git)
- [rwightman/pytorch-nips2017-attack-example](https://github.com/rwightman/pytorch-nips2017-attack-example.git)

## Usage of this library module

First of all, make sure the `import runutils` statement in `cw.py` (line 19) is a valid import statement in your development environment.

In the following code sample, we assume that `net` is a pretrained network, such that `outputs = net(torch.autograd.Variable(inputs))` returns a `torch.autograd.Variable` of dimension `(batch_size, num_classes)` if `inputs` is of dimension `(batch_size, num_channels, height, width)`.  Assume also that when doing normalization to the inputs, the normalization transformation is presented something like:

```python
normalization = torchvision.transforms.Normalize(mean, std)
```

where `mean` and `std` are both 3-tuples of floats.

One more thing to notice is that when producing adversarial examples from inputs, `cw.py` prints debugging information.  To suppress such behavior, use

```bash
sed -i '/FIXME$/d' cw.py
```

to delete all printing statements.


### Code example

To make the following code snippet executable, these variables need to be assigned:

- `dataloader`: the dataloader (of type `torch.utils.data.DataLoader`)
- `mean`: the mean used in inputs normalization
- `std`: the standard deviation used in inputs normalization

```python
import torch
import cw

inputs_box = (min((0 - m) / s for m, s in zip(mean, std)),
              max((1 - m) / s for m, s in zip(mean, std)))
# an untargeted adversary
adversary = cw.L2Adversary(targeted=False,
                           confidence=0.0,
                           search_steps=10,
                           box=inputs_box,
                           optimizer_lr=5e-4)

inputs, targets = next(iter(dataloader))
adversarial_examples = adversary(net, inputs, targets, to_numpy=False)
assert isinstance(adversarial_examples, torch.FloatTensor)
assert adversarial_examples.size() == inputs.size()

# a targeted adversary
adversary = cw.L2Adversary(targeted=True,
                           confidence=0.0,
                           search_steps=10,
                           box=inputs_box,
                           optimizer_lr=5e-4)

inputs, _ = next(iter(dataloader))
# a batch of any attack targets
attack_targets = torch.ones(inputs.size(0)) * 3
adversarial_examples = adversary(net, inputs, attack_targets, to_numpy=False)
assert isinstance(adversarial_examples, torch.FloatTensor)
assert adversarial_examples.size() == inputs.size()
```

What's `to_numpy` parameter?  In the above examples, if it were `True`, `adversarial_examples` would be of type `numpy.ndarray`.  This behavior might be desirable if one would like to store the adversarial examples in compressed `npz` format using `numpy`.
