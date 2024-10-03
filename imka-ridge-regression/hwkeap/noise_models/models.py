from copy import deepcopy

import numpy as np
import torch
from aihwkit.inference import BaseNoiseModel
from aihwkit.inference.converter.conductance import SinglePairConductanceConverter
from torch import abs as torch_abs


class StandardGaussNoise(BaseNoiseModel):
    def __init__(self, g_max=None, g_converter=None, mean=0, std_dev=0.067, drift_scale=1.0):
        g_converter = deepcopy(g_converter) or SinglePairConductanceConverter(g_max=g_max)
        super().__init__(g_converter)
        self.g_max = g_max
        self.mean = mean
        self.std_dev = std_dev
        self.drift_scale = drift_scale

    @torch.no_grad()
    def apply_programming_noise_to_conductance(self, g_target: torch.Tensor) -> torch.Tensor:

        g_prog = g_target + self.mean + self.std_dev * torch.randn_like(g_target)
        g_prog.clamp_(min=0.0)  # no negative conductances allowed

        return g_prog

    @torch.no_grad()
    def generate_drift_coefficients(self, g_target: torch.Tensor) -> torch.Tensor:
        """Return drift coefficients ``nu`` based on PCM measurements."""
        g_relative = torch.clamp(torch_abs(g_target / self.g_max), min=1e-7)

        # gt should be normalized wrt g_max
        mu_drift = (-0.0155 * torch.log(g_relative) + 0.0244).clamp(min=0.049, max=0.1)
        sig_drift = (-0.0125 * torch.log(g_relative) - 0.0059).clamp(
            min=0.008, max=0.045
        )
        nu_drift = torch_abs(mu_drift + sig_drift * torch.randn_like(g_relative)).clamp(
            min=0.0
        )

        return nu_drift * self.drift_scale

    @torch.no_grad()
    def apply_drift_noise_to_conductance(
        self, g_prog: torch.Tensor, nu_drift: torch.Tensor, t_inference: float
    ) -> torch.Tensor:
        """Apply the noise and drift up to the assumed inference time
        point based on PCM measurements."""
        t = t_inference + self.t_0

        # drift
        if t > self.t_0: g_drift = g_prog * ((t / self.t_0) ** (-nu_drift))
        else:g_drift = g_prog

        # expected accumulated 1/f noise since start of programming at t=0
        if t > 0:
            q_s = (0.0088 / ((torch_abs(g_prog) / self.g_max) ** 0.65).clamp(min=1e-3)).clamp(max=0.2)
            sig_noise = q_s * torch.sqrt(torch.numpy_log((t + self.t_read) / (2 * self.t_read)))
            g_final = g_drift + torch_abs(g_drift) * self.read_noise_scale * sig_noise * torch.randn_like(g_prog)
        else: g_final = g_prog

        return g_final.clamp(min=0.0)


class NoNoise(BaseNoiseModel):
    def __init__(self, g_max=None, g_converter=None, drift_scale=1.0):
        g_converter = deepcopy(g_converter) or SinglePairConductanceConverter(g_max=g_max)
        super().__init__(g_converter)
        self.g_max = g_max
        self.drift_scale = drift_scale

    @torch.no_grad()
    def apply_programming_noise_to_conductance(self, g_target: torch.Tensor) -> torch.Tensor:

        return g_target

    @torch.no_grad()
    def generate_drift_coefficients(self, g_target: torch.Tensor) -> torch.Tensor:
        """Return drift coefficients ``nu`` based on PCM measurements."""
        g_relative = torch.clamp(torch_abs(g_target / self.g_max), min=1e-7)

        # gt should be normalized wrt g_max
        mu_drift = (-0.0155 * torch.log(g_relative) + 0.0244).clamp(min=0.049, max=0.1)
        sig_drift = (-0.0125 * torch.log(g_relative) - 0.0059).clamp(min=0.008, max=0.045)
        nu_drift = torch_abs(mu_drift + sig_drift * torch.randn_like(g_relative)).clamp(min=0.0)

        return nu_drift * self.drift_scale

    @torch.no_grad()
    def apply_drift_noise_to_conductance(
        self, g_prog: torch.Tensor, nu_drift: torch.Tensor, t_inference: float
    ) -> torch.Tensor:
        """Apply the noise and drift up to the assumed inference time
        point based on PCM measurements."""
        t = t_inference + self.t_0

        # drift
        if t > self.t_0: g_drift = g_prog * ((t / self.t_0) ** (-nu_drift))
        else: g_drift = g_prog

        # expected accumulated 1/f noise since start of programming at t=0
        if t > 0:
            q_s = (0.0088 / ((torch_abs(g_prog) / self.g_max) ** 0.65).clamp(min=1e-3)).clamp(max=0.2)
            sig_noise = q_s * torch.sqrt(torch.numpy_log((t + self.t_read) / (2 * self.t_read)))
            g_final = g_drift + torch_abs(g_drift) * self.read_noise_scale * sig_noise * torch.randn_like(g_prog)
        else:g_final = g_prog

        return g_final.clamp(min=0.0)


class UniformNoise(BaseNoiseModel):
    def __init__(self, g_max=None, g_converter=None, a=-1, b=1, drift_scale=1.0):
        g_converter = deepcopy(g_converter) or SinglePairConductanceConverter(g_max=g_max)
        super().__init__(g_converter)
        self.g_max = g_max
        self.a = a
        self.b = b
        self.drift_scale = drift_scale

    @torch.no_grad()
    def apply_programming_noise_to_conductance(self, g_target: torch.Tensor) -> torch.Tensor:

        g_prog = g_target + self.a + (self.b - self.a) * torch.rand_like(g_target)
        g_prog.clamp_(min=0.0)  # no negative conductances allowed

        return g_prog

    @torch.no_grad()
    def generate_drift_coefficients(self, g_target: torch.Tensor) -> torch.Tensor:
        """Return drift coefficients ``nu`` based on PCM measurements."""
        g_relative = torch.clamp(torch_abs(g_target / self.g_max), min=1e-7)

        # gt should be normalized wrt g_max
        mu_drift = (-0.0155 * torch.log(g_relative) + 0.0244).clamp(min=0.049, max=0.1)
        sig_drift = (-0.0125 * torch.log(g_relative) - 0.0059).clamp(min=0.008, max=0.045)
        nu_drift = torch_abs(mu_drift + sig_drift * torch.randn_like(g_relative)).clamp(min=0.0)

        return nu_drift * self.drift_scale

    @torch.no_grad()
    def apply_drift_noise_to_conductance(self, g_prog: torch.Tensor, nu_drift: torch.Tensor, t_inference: float) -> torch.Tensor:
        """Apply the noise and drift up to the assumed inference time
        point based on PCM measurements."""
        t = t_inference + self.t_0

        # drift
        if t > self.t_0: g_drift = g_prog * ((t / self.t_0) ** (-nu_drift))
        else: g_drift = g_prog

        # expected accumulated 1/f noise since start of programming at t=0
        if t > 0:
            q_s = (0.0088 / ((torch_abs(g_prog) / self.g_max) ** 0.65).clamp(min=1e-3)).clamp(max=0.2)
            sig_noise = q_s * torch.sqrt(torch.numpy_log((t + self.t_read) / (2 * self.t_read)))
            g_final = g_drift + torch_abs(g_drift) * self.read_noise_scale * sig_noise * torch.randn_like(g_prog)
        else:g_final = g_prog

        return g_final.clamp(min=0.0)


class QuantizationNoise(BaseNoiseModel):
    def __init__(self, g_max=None, g_converter=None, qbits=8, drift_scale=1.0):
        g_converter = deepcopy(g_converter) or SinglePairConductanceConverter(g_max=g_max)
        super().__init__(g_converter)
        self.g_max = g_max
        self.qbits = qbits
        self.drift_scale = drift_scale

    @torch.no_grad()
    def apply_programming_noise_to_conductance(
        self, g_target: torch.Tensor
    ) -> torch.Tensor:
        quantile = np.quantile(g_target, 0.999)
        g_prog = torch.round(torch.clamp(g_target, min=-quantile, max=quantile)/ quantile* (2 ** (self.qbits - 1) - 1)) * quantile / (2 ** (self.qbits - 1) - 1)
        g_prog.clamp_(min=0.0)  # no negative conductances allowed
        return g_prog

    @torch.no_grad()
    def generate_drift_coefficients(self, g_target: torch.Tensor) -> torch.Tensor:
        """Return drift coefficients ``nu`` based on PCM measurements."""
        g_relative = torch.clamp(torch_abs(g_target / self.g_max), min=1e-7)

        # gt should be normalized wrt g_max
        mu_drift = (-0.0155 * torch.log(g_relative) + 0.0244).clamp(min=0.049, max=0.1)
        sig_drift = (-0.0125 * torch.log(g_relative) - 0.0059).clamp(min=0.008, max=0.045)
        nu_drift = torch_abs(mu_drift + sig_drift * torch.randn_like(g_relative)).clamp(min=0.0)

        return nu_drift * self.drift_scale

    @torch.no_grad()
    def apply_drift_noise_to_conductance(self, g_prog: torch.Tensor, nu_drift: torch.Tensor, t_inference: float) -> torch.Tensor:
        """Apply the noise and drift up to the assumed inference time
        point based on PCM measurements."""
        t = t_inference + self.t_0

        # drift
        if t > self.t_0: g_drift = g_prog * ((t / self.t_0) ** (-nu_drift))
        else: g_drift = g_prog

        # expected accumulated 1/f noise since start of programming at t=0
        if t > 0:
            q_s = (0.0088 / ((torch_abs(g_prog) / self.g_max) ** 0.65).clamp(min=1e-3)).clamp(max=0.2)
            sig_noise = q_s * torch.sqrt(torch.numpy_log((t + self.t_read) / (2 * self.t_read)))
            g_final = g_drift + torch_abs(g_drift) * self.read_noise_scale * sig_noise * torch.randn_like(g_prog)
        else: g_final = g_prog

        return g_final.clamp(min=0.0)
