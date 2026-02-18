import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from torch_common import cpu, gpu, LOW_VRAM, en
from DiffPure import DiffPure


class DiffPure_x0v(DiffPure):
    def __init__(self, model_ddpm, model_cl,
                 steps = 10, noise_add = 100,
                 T = 1000):
        super().__init__(model_ddpm, model_cl, 
                         steps, T, noise_add)

    # given output logits of the current x0 prediction, 
    # scale it such that later predictions have higher weight
    def _scale_vote(self, logits, i, n):
        # for all of these, the first term scales logits
        # the second term normalizes such that sum(scales)=1
        K = (i/n) * 2./(n+1.)
        #K = ((i/n)**2) * (6.*n / (2*n*n + 3*n + 1))
        #K = ((i/n)**3) * (4*n) / ((n+1.)**2)
        out = torch.exp(logits) * K
        return out

    # given an adversarial datapoint, predict x0 (non adversarial datapoint)
    # save scaled logits, and return slightly less noisey image for next iteration
    def _get_vote_checkpoint(self, x, t, i, device, return_x0_pred):

        # regular ddpm loop process
        ts = t.expand(x.shape[0]).to(device)
        noise_pred = self.ddpm(x, ts).sample
        out = self.sched.step(noise_pred, t, x)
        prev = out.prev_sample
        
        # gotten for free using ddim sampler
        orig_pred = out.pred_original_sample

        orig_pred = self.norm_f_inv(orig_pred)
        logits = self.classifier(orig_pred)
        vote = self._scale_vote(logits, i, self.steps)

        return (prev, vote) if (not return_x0_pred) else ((prev, orig_pred), vote)

    # low vram call of the above method
    def _get_vote(self, x, t, i, device=gpu, return_x0_pred=False):
        return self._get_vote_checkpoint(x,t,i,device, return_x0_pred) \
               if (not LOW_VRAM) else \
               checkpoint(self._get_vote_checkpoint, x,t,i,device,return_x0_pred,
                          use_reentrant=False)

    def forward(self, x):
        
        # same as diffpure
        device=x.device
        x = self.norm_f(x).clone()
        if self.do_noise_add_:
            noise = torch.randn_like(x)
            prev = self.sched.add_noise(
                    x, noise, 
                    Tensor([self.noise_add])
                        .long()
                            .to(device))
        else:
            prev = x

        out = torch.zeros_like(self.cl(torch.zeros_like(x)))

        for i, t in en(self.sched.timesteps[-self.steps:]):
            # voting
            prev, vote = self._get_vote(prev, t, i, device)
            out = out + vote

        return out