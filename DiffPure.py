import torch 
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
import diffusers
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from torch_common import cpu, gpu, LOW_VRAM


# simple ddpm loop used to purify as described in 
# https://arxiv.org/abs/2205.07460
class DiffPure(nn.Module):
    def __init__(self, ddpm, classifier, 
                 steps = 10, noise_add = 100,
                 classifier_norm_function = None, 
                 sched_conf_path=None):
        super().__init__()

        # transforms to ensure data is in the correct format 
        # for ddpm and classifier models
        classifier_norm_function = (lambda x : x) \
            if classifier_norm_function is None else classifier_norm_function 
        self.norm_f = lambda x : x * 2 - 1
        self.norm_f_inv = lambda x : classifier_norm_function((x + 1.) / 2.)

        self.ddpm=ddpm
        self.classifier = classifier

        self.sched = DDIMScheduler.from_config(sched_conf_path) \
                     if sched_conf_path is not None else        \
                     DDIMScheduler(1000)

        self.noise_add = noise_add
        self.step =steps

    # only get x(t-1) given x(t), slightly denoise x
    def _get_prev_checkpoint(self, x, t, device):
        ts = t.expand(x.shape[0]).to(device)
        noise_pred = self.ddpm(x, ts).sample
        out = self.sched.step(noise_pred, t, x)
        return out.prev_sample

    # checkpointing to allow large ddpm chains to be held in memory
    def _get_prev(self, x, t, device):
        return self._get_prev_checkpoint(x, t, device) \
               if (not LOW_VRAM) else \
               checkpoint(self._get_prev_checkpoint, x, t, device,
                          use_reentrant=False)


    def forward(self, x):
        device = x.device
        x = self.norm_f(x).clone()

        # add noise at the start to attempt drowning out peturbations
        if self.do_noise_add_:
            noise = torch.randn_like(x)
            prev = self.sched.add_noise(
                    x, noise, 
                    Tensor([self.noise_add])
                        .long()
                            .to(device))
        else:
            prev = x

        # perform ddpm loop
        for t in self.sched.timesteps[-self.steps:]:
            prev = self._get_prev(prev, t, device)

        # scale and classify 
        x = self.norm_f_inv(prev)
        out = self.classifier(x)
        return out