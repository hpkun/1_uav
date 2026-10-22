"""Read-only network statistics and detached PPO telemetry; no preprocessing."""
from contextlib import contextmanager
import random

import numpy as np
import torch
from torch.nn import functional as F


@contextmanager
def isolated_rng():
    """Restore Python, NumPy, CPU and initialized CUDA generators on all exits."""
    py, np_state = random.getstate(), np.random.get_state()
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


@torch.no_grad()
def statistics(value):
    x = torch.as_tensor(value).detach().to(device='cpu', dtype=torch.float64).flatten()
    finite = x[torch.isfinite(x)]
    result = dict(count=x.numel(), finite_count=finite.numel(), nonfinite_count=x.numel()-finite.numel())
    if not finite.numel():
        return dict(result, min=None, max=None, mean=None, std=None, quantiles={})
    q = (0., .01, .05, .25, .5, .75, .95, .99, 1.)
    return dict(result, min=finite.min().item(), max=finite.max().item(), mean=finite.mean().item(),
                std=finite.std(unbiased=False).item(),
                quantiles={str(p): v for p, v in zip(q, torch.quantile(finite, torch.tensor(q, dtype=torch.float64)).tolist())})


@torch.no_grad()
def _network_diagnostic(model, inputs, source, actor):
    if source not in ('actual_inspection_environment', 'synthetic_physical_inputs'):
        raise ValueError('explicit actual-inspection or synthetic-physical source required')
    width = model.network[0].in_features
    if not actor and width != (90 if source == 'actual_inspection_environment' else 60):
        raise ValueError('diagnostic source and critic protocol disagree')
    parameter = next(model.parameters())
    x = torch.as_tensor(inputs, device=parameter.device, dtype=parameter.dtype)
    if x.ndim != 2 or x.shape[1] != width or not len(x):
        raise ValueError('diagnostics require nonempty [samples,input_dimension]')
    report = dict(source=source, samples=len(x), input_dimension=width, dtype=str(x.dtype),
                  preprocessing='none; dtype/device conversion only',
                  input=[statistics(x[:, i]) for i in range(width)], layers=[])
    if not torch.isfinite(x).all():
        return dict(report, status='nonfinite_input_forward_skipped')
    captured, handles = {}, []

    def capture(name):
        def hook(module, args, output):
            captured[name] = output.detach()
        return hook

    try:
        for i in range(4):
            handles.append(model.network[i].register_forward_hook(capture(str(i))))
        with isolated_rng():
            output = model.distribution(x) if actor else model(x)
        for linear, tanh in ((0, 1), (2, 3)):
            h = captured[str(tanh)]
            report['layers'].append(dict(
                preactivation=statistics(captured[str(linear)]), activation=statistics(h),
                saturation_definition='abs(tanh_output) >= 0.99',
                saturation_fraction=(h.abs() >= .99).double().mean().item(),
                derivative=statistics(1-h.square())))
        if actor:
            report['beta_alpha'] = statistics(output.concentration1)
            report['beta_beta'] = statistics(output.concentration0)
        else:
            report['values'] = statistics(output)
        return dict(report, status='ok')
    finally:
        for handle in handles:
            handle.remove()


def diagnose_actor(actor, observations, *, source):
    return _network_diagnostic(actor, observations, source, True)


def diagnose_critic(critic, states, *, source):
    return _network_diagnostic(critic, states, source, False)


@torch.no_grad()
def advantage_statistics(advantages, slots, n):
    return [statistics(advantages[slots == i]) for i in range(n)]


class UpdateObserver:
    """Observe EXISTING forwards/grad norms; no extra network pass or RNG draw."""
    def __init__(self, actor, critic, n, clip):
        self.actor, self.critic, self.n, self.clip = actor, critic, n, clip
        self.handles, self.activations, self.beta = [], {}, []
        self.actor_norms, self.critic_norms = [], []
        self.actor_clips, self.critic_clips = [], []
        self.participation = [0]*n

    @torch.no_grad()
    def _tanh(self, name, output):
        x = output.detach()
        counts = torch.stack(((x.abs() >= .99).sum(), torch.isfinite(x).sum(), x.new_tensor(x.numel(), dtype=torch.int64)))
        self.activations[name] = self.activations.get(name, torch.zeros_like(counts)) + counts

    @torch.no_grad()
    def _beta(self, output):
        parameters = F.softplus(output.detach()) + 1
        a, b = parameters[..., :3], parameters[..., 3:]
        self.beta.append(torch.stack((a.min(), a.max(), b.min(), b.max(), torch.isfinite(parameters).all().float())))

    def __enter__(self):
        for name, model in (('actor', self.actor), ('critic', self.critic)):
            network = getattr(model, 'network', None)
            if network is not None:
                for index in (1, 3):
                    self.handles.append(network[index].register_forward_hook(
                        lambda module, args, output, key=f'{name}_tanh{(index+1)//2}': self._tanh(key, output)))
        from .networks import Actor
        if isinstance(self.actor, Actor):
            self.handles.append(self.actor.network[-1].register_forward_hook(lambda m, a, o: self._beta(o)))
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()

    @torch.no_grad()
    def minibatch(self, slots, actor_norm, critic_norm):
        for slot in slots.unique().tolist():
            self.participation[slot] += 1
        self.actor_norms.append(actor_norm.item())
        self.critic_norms.append(critic_norm.item())
        # Same dtype and epsilon as the installed PyTorch clip routine.
        self.actor_clips.append(bool(self.clip/(actor_norm+1e-6) < 1))
        self.critic_clips.append(bool(self.clip/(critic_norm+1e-6) < 1))

    @torch.no_grad()
    def report(self):
        activations = {}
        for name, counts in self.activations.items():
            saturated, finite, total = counts.tolist()
            activations[name] = dict(saturation_fraction=saturated/total, finite=finite == total,
                                     evaluated_cells=total, threshold=.99)
        beta = dict(available=bool(self.beta))
        if self.beta:
            b = torch.stack(self.beta)
            beta.update(alpha_min=b[:,0].min().item(), alpha_max=b[:,1].max().item(),
                        beta_min=b[:,2].min().item(), beta_max=b[:,3].max().item(), finite=bool(b[:,4].all()))
        return dict(per_uav_minibatches=self.participation, actor_gradient_norm=statistics(self.actor_norms),
                    critic_gradient_norm=statistics(self.critic_norms),
                    actor_gradient_clip_fraction=sum(self.actor_clips)/max(1,len(self.actor_clips)),
                    critic_gradient_clip_fraction=sum(self.critic_clips)/max(1,len(self.critic_clips)),
                    gradient_norm_definition='pre-clip L2; actual PyTorch coefficient max_grad_norm/(norm+1e-6) < 1 triggers clipping',
                    tanh=activations, beta_parameters=beta)
