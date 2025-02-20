import torch
from typing import List, Optional, Tuple, Union

from diffusers.utils import (
    USE_PEFT_BACKEND,
    BaseOutput,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor

class EulerAncestralDiscreteSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor
    pred_original_sample: Optional[torch.FloatTensor] = None


def eul_step(
    self,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor, 
    fusion_latent,
    pipe,
    generator: Optional[torch.Generator] = None,
    return_dict: bool = True,
) -> Union[EulerAncestralDiscreteSchedulerOutput, Tuple]:

    if (
        isinstance(timestep, int)
        or isinstance(timestep, torch.IntTensor)
        or isinstance(timestep, torch.LongTensor)
    ):
        raise ValueError(
            (
                "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                " one of the `scheduler.timesteps` as a timestep."
            ),
        )

    if self.step_index is None:
        self._init_step_index(timestep)

    sigma = self.sigmas[self.step_index]

    # Upcast to avoid precision issues when computing prev_sample
    sample = sample.to(torch.float32)

    # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
    if self.config.prediction_type == "epsilon": ## True, 计算x_0
        pred_original_sample = sample - sigma * model_output
    elif self.config.prediction_type == "v_prediction":
        # * c_out + input * c_skip
        pred_original_sample = model_output * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))
    elif self.config.prediction_type == "sample":
        raise NotImplementedError("prediction_type not implemented yet: sample")
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
        )
        
    ## fusion latent
    pred_original_sample = fusion_latent
    
    sigma_from = self.sigmas[self.step_index]
    sigma_to = self.sigmas[self.step_index + 1]
    sigma_up = (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5

    # 2. Convert to an ODE derivative
    derivative = (sample - pred_original_sample) / sigma
    dt = sigma_down - sigma

    prev_sample = sample + derivative * dt

    device = model_output.device
    noise = randn_tensor(model_output.shape, dtype=model_output.dtype, device=device, generator=generator)

    prev_sample = prev_sample + noise * sigma_up

    # Cast sample back to model compatible dtype
    prev_sample = prev_sample.to(model_output.dtype)

    # upon completion increase step index by one
    self._step_index += 1

    if not return_dict:
        return (prev_sample,)

    return EulerAncestralDiscreteSchedulerOutput(
        prev_sample=prev_sample, pred_original_sample=pred_original_sample
    )