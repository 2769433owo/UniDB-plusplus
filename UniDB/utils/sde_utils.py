import math
import torch
import abc
from tqdm import tqdm
import torchvision.utils as tvutils
import os
from scipy import integrate


class SDE(abc.ABC):
    def __init__(self, T, device=None):
        self.T = T
        self.dt = 1 / T
        self.device = device

    @abc.abstractmethod
    def drift(self, x, t):
        pass

    @abc.abstractmethod
    def dispersion(self, x, t):
        pass

    @abc.abstractmethod
    def sde_reverse_drift(self, x, score, t):
        pass

    @abc.abstractmethod
    def score_fn(self, x, t):
        pass

    ################################################################################

    def forward_step(self, x, t):
        return x + self.drift(x, t) + self.dispersion(x, t)

    def reverse_sde_step_mean(self, x, score, t):  # train process
        return x - self.sde_reverse_drift(x, score, t)

    def forward(self, x0, T=-1):
        T = self.T if T < 0 else T
        x = x0.clone()
        for t in tqdm(range(1, T + 1)):
            x = self.forward_step(x, t)

        return x

    def reverse_sde(self, xt, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t)
            x = self.reverse_sde_step(x, score, t)

        return x

    def reverse_ode(self, xt, T=-1):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            score = self.score_fn(x, t)
            x = self.reverse_ode_step(x, score, t)
        return x


#############################################################################


class UniDB(SDE):
    '''
    Let timestep t start from 1 to T, state t=0 is never used
    '''

    def __init__(self, lambda_square, gamma, T=100, solver = "unidb-gou", solver_step=100, method="euler-sde", solver_type = "sde", schedule='cosine', eps=0.01, device=None):
        super().__init__(T, device)
        self.lambda_square = lambda_square / 255 if lambda_square >= 1 else lambda_square
        self.gamma = gamma
        self.solver = solver
        self.solver_step = solver_step
        self.method = method
        self.solver_type = solver_type
        self._initialize(self.lambda_square, T, schedule, eps)
        self.x_buffer = []
        self.noise_buffer = []
        self.data_buffer = []

    def _initialize(self, lambda_square, T, schedule, eps=0.01):

        def constant_theta_schedule(timesteps, v=1.):
            """
            constant schedule
            """
            print('constant schedule')
            timesteps = timesteps + 1  # T from 1 to 100
            return torch.ones(timesteps, dtype=torch.float32)

        def linear_theta_schedule(timesteps):
            """
            linear schedule
            """
            print('linear schedule')
            timesteps = timesteps + 1  # T from 1 to 100
            scale = 1000 / timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

        def cosine_theta_schedule(timesteps, s=0.008):
            """
            cosine schedule
            """
            print('cosine schedule')
            timesteps = timesteps + 2  # for truncating from 1 to -1
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:-1]
            return betas

        def get_thetas_cumsum(thetas):
            return torch.cumsum(thetas, dim=0)

        def get_sigmas(thetas):
            return torch.sqrt(lambda_square ** 2 * 2 * thetas)

        def get_sigma_bars(thetas_cumsum):
            return torch.sqrt(lambda_square ** 2 * (1 - torch.exp(-2 * thetas_cumsum * self.dt)))

        if schedule == 'cosine':
            thetas = cosine_theta_schedule(T)
        elif schedule == 'linear':
            thetas = linear_theta_schedule(T)
        elif schedule == 'constant':
            thetas = constant_theta_schedule(T)
        else:
            print('Not implemented such schedule yet!!!')

        sigmas = get_sigmas(thetas)
        thetas_cumsum = get_thetas_cumsum(thetas) - thetas[0]  # for that thetas[0] is not 0
        self.dt = -1 / thetas_cumsum[-1] * math.log(eps)
        sigma_bars = get_sigma_bars(thetas_cumsum)

        self.thetas = thetas.to(self.device)
        self.sigmas = sigmas.to(self.device)
        self.thetas_cumsum = thetas_cumsum.to(self.device)

        self.sigma_bars = sigma_bars.to(self.device)

        self.sigma_t_T = torch.sqrt(
            self.lambda_square ** 2 * (1 - torch.exp(-2 * (self.thetas_cumsum[-1] - self.thetas_cumsum) * self.dt)))

        self.f_sigmas = self.sigma_bars * self.sigma_t_T / self.sigma_bars[-1]

        self.mu = 0.
        self.model = None

    #####################################

    def update_list_1(self, list, element):
        if len(list) == 0:
            list.append(element)
        else:
            list[0] = element

    # set mu for different cases
    def set_mu(self, mu):
        self.mu = mu

    # set score model for reverse process
    def set_model(self, model):
        self.model = model

    def scaled_reverse_sde_step_mean(self, x, noise, t):
        tmp = torch.exp((self.thetas_cumsum[t] - self.thetas_cumsum[-1]) * self.dt)  # e^{-\bar\theta_{t:T}}}
        drift_h = - self.sigmas[t] ** 2 * tmp ** 2 / self.sigma_t_T[t] ** 2 * (x - self.mu)
        mask = (t == 100)
        mask_expanded = mask.expand_as(drift_h)
        drift_h[mask_expanded] = 0
        return self.f_sigma(t) * x - (
                    self.f_sigma(t) * self.thetas[t] * (self.mu - x) + self.f_sigma(t) * drift_h + self.sigmas[
                t] ** 2 * noise) * self.dt

    def scaled_reverse_optimum_step(self, xt_1_optimum, t):
        return self.f_sigma(t) * xt_1_optimum

    def reverse_sde_step(self, x, score, t):  # val process
        return x - self.sde_reverse_drift_1(x, score, t) - self.dispersion(x, t)

    def reverse_mean_ode_step(self, x, score, t):  # val process
        return x - self.sde_reverse_drift_1(x, score, t)
    
    def reverse_pf_ode_step(self, x, score, t):  # val process
        return x - self.sde_reverse_drift_2(x, score, t)

    #####################################

    # # TODO
    # def m(self, t):  # cofficient of x0 in marginal forward process
    #     return torch.exp(-self.thetas_cumsum[t] * self.dt) * self.sigma_t_T[t] ** 2 / self.sigma_bars[-1] ** 2

    # # TODO
    # def n(self, t):  # cofficient of xT in marginal forward process
    #     return ((1 - torch.exp(-self.thetas_cumsum[t] * self.dt)) * self.sigma_t_T[t] ** 2 + torch.exp(
    #         -2 * (self.thetas_cumsum[-1] - self.thetas_cumsum[t]) * self.dt) * self.sigma_bars[t] ** 2) / \
    #            self.sigma_bars[-1] ** 2

    # TODO
    def m(self, t):  # cofficient of x0 in marginal forward process
        return torch.exp(-self.thetas_cumsum[t] * self.dt) * (1 + self.gamma * self.sigma_t_T[t] ** 2) / (1 + self.gamma * self.sigma_bars[-1] ** 2)

    # TODO
    def n(self, t):  # cofficient of xT in marginal forward process
        return 1 - self.m(t)

    def f_m(self, t):  # cofficient of x_{t-1} in forward process
        return self.m(t) / self.m(t - 1)

    def f_n(self, t):  # cofficient of x_T in forward process
        return self.n(t) - self.n(t - 1) * self.m(t) / self.m(t - 1)

    def f_sigma_1(self, t):  # forward sigma with t : t-1 to t
        return torch.sqrt(self.f_sigma(t) ** 2 - self.f_sigma(t - 1) ** 2 * self.f_m(t) ** 2)

    def f_mean_1(self, xt_1, t):  # forward mean with t : t-1 to t
        return self.f_m(t) * xt_1 + self.f_n(t) * self.mu

    def r_sigma_1(self, t):  # reverse sigma with t : t to t-1
        return self.f_sigma_1(t) * self.f_sigma(t - 1) / self.f_sigma(t)

    def r_mean_1(self, xt, x0, t):  # reverse mean with t : t to t-1
        return (self.f_sigma(t - 1) ** 2 * self.f_m(t) * (xt - self.f_n(t) * self.mu) +
                self.f_sigma_1(t) ** 2 * self.f_mean(x0, t - 1)) / self.f_sigma(t) ** 2

    #####################################

    def mu_bar(self, x0, t):
        return self.mu + (x0 - self.mu) * torch.exp(-self.thetas_cumsum[t] * self.dt)

    def f_mean(self, x0, t):  # forward mean with t
        mean = self.m(t) * x0 + self.n(t) * self.mu
        return mean

    def sigma_bar(self, t):
        return self.sigma_bars[t]

    def f_sigma(self, t):  # marginal forward sigma with t
        return self.f_sigmas[t]

    # def drift(self, x, t):
    #     if t == 100:
    #         return (self.thetas[t] * (self.mu - x)) * self.dt
    #     # add h-transform term
    #     tmp = torch.exp(2 * (self.thetas_cumsum[t] - self.thetas_cumsum[-1]) * self.dt)  # e^{-2\bar\theta_{t:T}}}
    #     drift_h = - self.sigmas[t] ** 2 * tmp / self.sigma_t_T[t] ** 2 * (x - self.mu)
    #     return (self.thetas[t] * (self.mu - x) + drift_h) * self.dt

    # TODO
    def drift(self, x, t):
        if t == 100:
            return (self.thetas[t] * (self.mu - x)) * self.dt
        # add h-transform term
        tmp = torch.exp(2 * (self.thetas_cumsum[t] - self.thetas_cumsum[-1]) * self.dt)  # e^{-2\bar\theta_{t:T}}}
        drift_h = - (self.gamma * self.sigmas[t] ** 2 * tmp) / (1 + self.gamma * self.sigma_t_T[t] ** 2) * (x - self.mu)
        return (self.thetas[t] * (self.mu - x) + drift_h) * self.dt

    # def sde_reverse_drift_1(self, x, score, t):
    #     # add h-transform term
    #     if t == 100:
    #         return (self.thetas[t] * (self.mu - x) - self.sigmas[t] ** 2 * score) * self.dt  # drift_h=0
    #     tmp = torch.exp(2 * (self.thetas_cumsum[t] - self.thetas_cumsum[-1]) * self.dt)  # e^{-2\bar\theta_{t:T}}}
    #     drift_h = - self.sigmas[t] ** 2 * tmp / self.sigma_t_T[t] ** 2 * (x - self.mu)
    #     return (self.thetas[t] * (self.mu - x) + drift_h - self.sigmas[t] ** 2 * score) * self.dt

    # TODO
    def sde_reverse_drift_1(self, x, score, t):
        # add h-transform term
        if t == 100:
            return (self.thetas[t] * (self.mu - x) - self.sigmas[t] ** 2 * score) * self.dt  # drift_h=0
        tmp = torch.exp(2 * (self.thetas_cumsum[t] - self.thetas_cumsum[-1]) * self.dt)  # e^{-2\bar\theta_{t:T}}}
        drift_h = - (self.gamma * self.sigmas[t] ** 2 * tmp) / (1 + self.gamma * self.sigma_t_T[t] ** 2) * (x - self.mu)
        return (self.thetas[t] * (self.mu - x) + drift_h - self.sigmas[t] ** 2 * score) * self.dt
    
    # pf-ode
    def sde_reverse_drift_2(self, x, score, t):
        # add h-transform term
        if t == 100:
            return (self.thetas[t] * (self.mu - x) - 0.5 * self.sigmas[t] ** 2 * score) * self.dt  # drift_h=0
        tmp = torch.exp(2 * (self.thetas_cumsum[t] - self.thetas_cumsum[-1]) * self.dt)  # e^{-2\bar\theta_{t:T}}}
        drift_h = - (self.gamma * self.sigmas[t] ** 2 * tmp) / (1 + self.gamma * self.sigma_t_T[t] ** 2) * (x - self.mu)
        return (self.thetas[t] * (self.mu - x) + drift_h - 0.5 * self.sigmas[t] ** 2 * score) * self.dt

    def sde_reverse_drift(self, x, score, t):
        # add h-transform term
        tmp = torch.exp(2 * (self.thetas_cumsum[t] - self.thetas_cumsum[-1]) * self.dt)  # e^{-2\bar\theta_{t:T}}}
        drift_h = - self.sigmas[t] ** 2 * tmp / self.sigma_t_T[t] ** 2 * (x - self.mu)
        mask = (t == 100)
        mask_expanded = mask.expand_as(drift_h)
        drift_h[mask_expanded] = 0
        return (self.thetas[t] * (self.mu - x) + drift_h - self.sigmas[t] ** 2 * score) * self.dt

    def dispersion(self, x, t):
        return self.sigmas[t] * (torch.randn_like(x) * math.sqrt(self.dt)).to(self.device)

    def get_score_from_noise(self, noise, t):
        score = - noise / self.f_sigma(t)
        return score

    def score_fn(self, x, t, **kwargs):
        # need to pre-set mu and score_model
        return self.model(x, self.mu, t, **kwargs)

    def noise_fn(self, x, t, **kwargs):
        # need to pre-set mu and score_model
        return self.model(x, self.mu, t, **kwargs)

    def scaled_score(self, score, t):
        return self.sigmas[t] * score

    def get_real_score(self, xt, x0, t):
        real_score = -(xt - self.f_mean(x0, t)) / self.f_sigma(t) ** 2
        return real_score

    # optimum x_{t-1}
    def reverse_optimum_step(self, xt, x0, t):
        mean = self.r_mean_1(xt, x0, t)
        return mean

    def sigma(self, t):
        return self.sigmas[t]

    def theta(self, t):
        return self.thetas[t]

    def get_real_noise(self, xt, x0, t):
        real_noise = (xt - self.f_mean(x0, t)) / self.f_sigma(t)
        mask = (t == 100)
        mask_expanded = mask.expand_as(real_noise)
        real_noise[mask_expanded] = 0
        return real_noise

    # forward process to get x(T) from x(0)
    def forward(self, x0, T=-1, save_dir='forward_state'):
        T = self.T if T < 0 else T
        x = x0.clone()
        for t in tqdm(range(1, T + 1)):
            x = self.forward_step(x, t)

            os.makedirs(save_dir, exist_ok=True)
            tvutils.save_image(x.data, f'{save_dir}/state_{t}.png', normalize=False)
        return x

    def reverse_sde(self, xt, T=-1, save_states=False, save_dir='sde_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            noise = self.model(x, self.mu, t, **kwargs)
            score = - noise / self.f_sigma(t) if t != 100 else 0
            x = self.reverse_sde_step(x, score, t)

            if save_states:  # only consider to save 100 images
                interval = self.T // 100
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)

        return x

    def reverse_mean_ode(self, xt, T=-1, save_states=False, save_dir='ode_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            noise = self.model(x, self.mu, t, **kwargs)
            score = - noise / self.f_sigma(t) if t != 100 else 0
            x = self.reverse_mean_ode_step(x, score, t)

            if save_states:  # only consider to save 100 images
                interval = self.T // 100
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)

        return x
    
    def reverse_pf_ode(self, xt, T=-1, save_states=False, save_dir='ode_state', **kwargs):
        T = self.T if T < 0 else T
        x = xt.clone()
        for t in tqdm(reversed(range(1, T + 1))):
            noise = self.model(x, self.mu, t, **kwargs)
            score = - noise / self.f_sigma(t) if t != 100 else 0
            x = self.reverse_pf_ode_step(x, score, t)

            if save_states:  # only consider to save 100 images
                interval = self.T // 100
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)

        return x
    

    # ----------------------------------------------------------------------------------------------------------------

    def ode_alpha_t(self, t):
        tmp = torch.exp((self.thetas_cumsum[t] - self.thetas_cumsum[-1]) * self.dt) # e^{-\bar\theta_{t:T}}}
        return 1 / tmp - tmp
    
    def ode_alpha_t_gamma(self, t):
        tmp = torch.exp((self.thetas_cumsum[t] - self.thetas_cumsum[-1]) * self.dt) # e^{-\bar\theta_{t:T}}}
        return (1 + 1 / (self.gamma * self.lambda_square * self.lambda_square)) / tmp - tmp

    def ode_sigma_t(self, t):
        tmp = torch.exp(-self.thetas_cumsum[t] * self.dt) # e^{-\bar\theta_{t}}}
        return 1 / tmp - tmp
    
    def exp_theta_bar_s_t(self, s, t):
        return torch.exp(self.thetas_cumsum[t] * self.dt) / torch.exp(self.thetas_cumsum[s] * self.dt)

    def exp_a_bar_s_t(self, s, t):
        return self.ode_alpha_t(s) / self.ode_alpha_t(t)
    
    def exp_a_bar_s_t_gamma(self, s, t):
        return self.ode_alpha_t_gamma(s) / self.ode_alpha_t_gamma(t)

    def exp_minus_beta_t(self, t):
        return self.ode_sigma_t(t) / self.ode_alpha_t(t)
    
    def exp_minus_beta_t_gamma(self, t):
        return self.ode_sigma_t(t) / self.ode_alpha_t_gamma(t)
    
    def beta_t(self, t):
        return torch.log(self.ode_alpha_t(t) / self.ode_sigma_t(t))
    
    def beta_t_gamma(self, t):
        return torch.log(self.ode_alpha_t_gamma(t) / self.ode_sigma_t(t))
    
    def beta_inv(self, beta):
        real = 0.5 * torch.log((torch.exp(2 * self.thetas_cumsum[-1] * self.dt) + torch.exp(self.thetas_cumsum[-1] * self.dt) * torch.exp(beta)) / (1 + torch.exp(self.thetas_cumsum[-1] * self.dt) * torch.exp(beta)))
        diff = torch.abs(self.thetas_cumsum * self.dt - real)
        return torch.argmin(diff)
    
    def beta_inv_gamma(self, beta):
        real = 0.5 * torch.log((torch.exp(2 * self.thetas_cumsum[-1] * self.dt) * (1 + 1 / (self.gamma * self.lambda_square * self.lambda_square)) + torch.exp(self.thetas_cumsum[-1] * self.dt) * torch.exp(beta)) / (1 + torch.exp(self.thetas_cumsum[-1] * self.dt) * torch.exp(beta)))
        diff = torch.abs(self.thetas_cumsum * self.dt - real)
        return torch.argmin(diff)
    
    def delta_noise(self, s, t):
        temp = 1 / (self.gamma * self.lambda_square * self.lambda_square)
        res = self.lambda_square * self.ode_alpha_t_gamma(t) * torch.sqrt(1 / (temp + 1 - torch.exp((self.thetas_cumsum[s] - self.thetas_cumsum[-1]) * self.dt) ** 2) - 1 / (temp + 1 - torch.exp((self.thetas_cumsum[t] - self.thetas_cumsum[-1]) * self.dt) ** 2))
        return res
    
    def delta_data(self, s, t):
        res = self.lambda_square * self.ode_sigma_t(t) * torch.sqrt(1 / (torch.exp(self.thetas_cumsum[t] * self.dt) ** 2 - 1) - 1 / (torch.exp(self.thetas_cumsum[s] * self.dt) ** 2 - 1))
        if t == 0:
            res = 0
        return res
    

    def unidb_noise_mean_ode_solver(self, xt, T=-1, save_states=False, save_dir='ode_state', **kwargs):
        # T = self.solver_step if T < 0 else T
        step_size = self.T // self.solver_step
        x = xt.clone()
        for t in tqdm(range(self.T, 0, -step_size)):
            noise = self.model(x, self.mu, t, **kwargs) if t != self.T else 0
            t_i1 = t - step_size
            if t == self.T:
                x = self.exp_theta_bar_s_t(t_i1, t) * x + (1 - self.exp_theta_bar_s_t(t_i1, t)) * self.mu
            else:
                x = self.exp_a_bar_s_t_gamma(t_i1, t) * x + (1 - self.exp_a_bar_s_t_gamma(t_i1, t)) * self.mu - 2 * self.lambda_square * self.ode_alpha_t_gamma(t_i1) / torch.sqrt(self.ode_sigma_t(self.T)) * (torch.sqrt(self.exp_minus_beta_t_gamma(t)) - torch.sqrt(self.exp_minus_beta_t_gamma(t_i1))) * noise
                

            if save_states:  # only consider to save 100 images
                interval = self.T // self.solver_step
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)

        return x
    
    # with approximation
    def unidb_noise_mean_ode_solver_2(self, xt, T=-1, save_states=False, save_dir='ode_state', **kwargs):
        # T = self.solver_step if T < 0 else T
        step_size = self.T // self.solver_step
        x = xt.clone()
        for t in tqdm(range(self.T, 0, -step_size)):
            noise = self.model(x, self.mu, t, **kwargs) if t != self.T else 0
            # score = - noise / self.f_sigma(t) if t != 100 else 0
            # x = self.reverse_mean_ode_step(x, score, t)
            t_i1 = t - step_size
            betat_i1 = self.beta_t_gamma(t_i1)
            betat = self.beta_t_gamma(t)
            idx = self.beta_inv_gamma(0.5 * (betat_i1 + betat)).item()

            if t == self.T:
                x = self.exp_theta_bar_s_t(t_i1, t) * x + (1 - self.exp_theta_bar_s_t(t_i1, t)) * self.mu

            else:
                coeff = 2 * self.lambda_square * self.ode_alpha_t_gamma(t_i1) / torch.sqrt(self.ode_sigma_t(self.T)) * (torch.sqrt(self.exp_minus_beta_t_gamma(t)) - torch.sqrt(self.exp_minus_beta_t_gamma(t_i1)))
                mid = self.exp_a_bar_s_t_gamma(idx, t) * x + (1 - self.exp_a_bar_s_t_gamma(idx, t)) * self.mu - 2 * self.lambda_square * self.ode_alpha_t_gamma(idx) / torch.sqrt(self.ode_sigma_t(self.T)) * (torch.sqrt(self.exp_minus_beta_t_gamma(t)) - torch.sqrt(self.exp_minus_beta_t_gamma(idx))) * noise
                new_noise = self.model(mid, self.mu, idx, **kwargs)
                # x = self.exp_a_bar_s_t_gamma(t_i1, t) * x + (1 - self.exp_a_bar_s_t_gamma(t_i1, t)) * self.mu - coeff * noise - coeff2 * (new_noise - noise)
                x = self.exp_a_bar_s_t_gamma(t_i1, t) * x + (1 - self.exp_a_bar_s_t_gamma(t_i1, t)) * self.mu - coeff * new_noise

            if save_states:  # only consider to save 100 images
                interval = self.T // self.solver_step
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)

        return x
    
    def unidb_noise_sde_solver(self, xt, T=-1, save_states=False, save_dir='sde_state', **kwargs):
        # T = self.solver_step if T < 0 else T
        step_size = self.T // self.solver_step
        x = xt.clone()
        for t in tqdm(range(self.T, 0, -step_size)):
            noise = self.model(x, self.mu, t, **kwargs) if t != self.T else 0
            # score = - noise / self.f_sigma(t) if t != 100 else 0
            # x = self.reverse_mean_ode_step(x, score, t)
            t_i1 = t - step_size
            z = torch.randn_like(x)
            if t == self.T:
                # print("asdasd")
                x = self.exp_theta_bar_s_t(t_i1, t) * x + (1 - self.exp_theta_bar_s_t(t_i1, t)) * self.mu
            else:
                x = self.exp_a_bar_s_t(t_i1, t) * x + (1 - self.exp_a_bar_s_t(t_i1, t)) * self.mu - 2 * self.lambda_square * self.ode_alpha_t(t_i1) / torch.sqrt(self.ode_sigma_t(self.T)) * (torch.sqrt(self.exp_minus_beta_t(t)) - torch.sqrt(self.exp_minus_beta_t(t_i1))) * noise + self.lambda_square * self.ode_alpha_t(t) * torch.sqrt(1 / (1 / (self.gamma * self.lambda_square * self.lambda_square) + 1 - torch.exp((self.thetas_cumsum[t] - self.thetas_cumsum[-1]) * self.dt) ** 2) - 1 / (1 / (self.gamma * self.lambda_square * self.lambda_square) + 1 - torch.exp((self.thetas_cumsum[t_i1] - self.thetas_cumsum[-1]) * self.dt) ** 2)) * z
                
            if save_states:  # only consider to save 100 images
                interval = self.T // self.solver_step
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)

        return x
    
    # no approximation
    def unidb_noise_sde_solver_2(self, xt, T=-1, save_states=False, save_dir='sde_state', **kwargs):
        # T = self.solver_step if T < 0 else T
        step_size = self.T // self.solver_step
        x = xt.clone()

        r = 0.5 # super-parameter
        print("r", r)
        ty = "single_step" # decide gradient approximation
        assert ty in ["single_step", "multi_step"]
        
        # type_solver = "ode" # decide ode solver or sde solver
        # assert type_solver in ["sde", "ode"]

        for t in tqdm(range(self.T, 0, -step_size)):
            if ty == "single_step":
                assert r > 0 and r < 1
                noise = self.model(x, self.mu, t, **kwargs) if t != self.T else 0
                z1 = torch.randn_like(x)
                z2 = torch.randn_like(x)
                next = t - step_size # t_i1: left; t: right
                beta_next = self.beta_t_gamma(next)
                beta_now = self.beta_t_gamma(t)
                hi = beta_next - beta_now
                idx = self.beta_inv_gamma(beta_now + r * hi).item()
                del_noise_idx = self.delta_noise(t, idx) if self.solver_type == "sde" else 0
                del_noise = self.delta_noise(t, next) if self.solver_type == "sde" else 0

                if t == self.T:
                    # y = self.ode_alpha_t_gamma(idx) / self.ode_alpha_t_gamma(t) * x + (1 - self.ode_alpha_t_gamma(idx) / self.ode_alpha_t_gamma(t)) * self.mu
                    x = self.ode_alpha_t_gamma(next) / self.ode_alpha_t_gamma(t) * x + (1 - self.ode_alpha_t_gamma(next) / self.ode_alpha_t_gamma(t)) * self.mu
                elif next == 0:
                    x = self.ode_alpha_t_gamma(next) / self.ode_alpha_t_gamma(t) * x + (1 - self.ode_alpha_t_gamma(next) / self.ode_alpha_t_gamma(t)) * self.mu \
                        - 2 * self.lambda_square * self.ode_alpha_t_gamma(next) / torch.sqrt(self.ode_sigma_t(self.T)) * (torch.sqrt(self.exp_minus_beta_t(t)) - torch.sqrt(self.exp_minus_beta_t(next))) * noise + del_noise * z2
                else:
                    y = self.ode_alpha_t_gamma(idx) / self.ode_alpha_t_gamma(t) * x + (1 - self.ode_alpha_t_gamma(idx) / self.ode_alpha_t_gamma(t)) * self.mu \
                        - 2 * self.lambda_square * self.ode_alpha_t_gamma(idx) / torch.sqrt(self.ode_sigma_t(self.T)) * (torch.sqrt(self.exp_minus_beta_t(t)) - torch.sqrt(self.exp_minus_beta_t(idx))) * noise + del_noise_idx * z1
                    new_noise = self.model(y, self.mu, idx, **kwargs)
                    x = self.ode_alpha_t_gamma(next) / self.ode_alpha_t_gamma(t) * x + (1 - self.ode_alpha_t_gamma(next) / self.ode_alpha_t_gamma(t)) * self.mu \
                        - 4 * self.lambda_square * self.ode_alpha_t_gamma(next) / torch.sqrt(self.ode_sigma_t(self.T)) * (torch.sqrt(self.exp_minus_beta_t(t)) - 0.5 * hi * torch.sqrt(self.exp_minus_beta_t(next)) - torch.sqrt(self.exp_minus_beta_t(next))) * (new_noise - noise) / (r * hi) \
                        - 2 * self.lambda_square * self.ode_alpha_t_gamma(next) / torch.sqrt(self.ode_sigma_t(self.T)) * (torch.sqrt(self.exp_minus_beta_t(t)) - torch.sqrt(self.exp_minus_beta_t(next))) * noise + del_noise * z2
            else: # "multi_step"
                # TODO
                assert r == 1
                noise = self.model(x, self.mu, t, **kwargs) if t != self.T else 0
                z1 = torch.randn_like(x)
                z2 = torch.randn_like(x)
                next = t - step_size 
                
                beta_next = self.beta_t_gamma(next)
                beta_now = self.beta_t_gamma(t)
                
                hi = beta_next - beta_now
                
                del_noise = self.delta_noise(t, next) if self.solver_type == "sde" else 0

                if t < self.T:
                    last = t + step_size 
                    beta_last = self.beta_t_gamma(last)
                    hi_minus_1 = beta_now - beta_last
                    idx = self.beta_inv_gamma(beta_now - r * hi_minus_1).item()
                    del_noise_last_idx = self.delta_noise(last, idx) if self.solver_type == "sde" else 0

                if t == self.T:
                    x = self.ode_alpha_t_gamma(next) / self.ode_alpha_t_gamma(t) * x + (1 - self.ode_alpha_t_gamma(next) / self.ode_alpha_t_gamma(t)) * self.mu
                    y = self.mu
                    # update buffer
                    self.update_list_1(self.x_buffer, self.mu)
                    self.update_list_1(self.noise_buffer, noise)
                elif next == 0:
                    x = self.ode_alpha_t_gamma(next) / self.ode_alpha_t_gamma(t) * x + (1 - self.ode_alpha_t_gamma(next) / self.ode_alpha_t_gamma(t)) * self.mu \
                        - 2 * self.lambda_square * self.ode_alpha_t_gamma(next) / torch.sqrt(self.ode_sigma_t(self.T)) * (torch.sqrt(self.exp_minus_beta_t(t)) - torch.sqrt(self.exp_minus_beta_t(next))) * noise + del_noise * z2
                    y = self.mu
                elif last == self.T:
                    last_x = self.x_buffer[0]
                    last_noise = self.noise_buffer[0]
                    
                    # calculate y_i and epsilon(y_i, s_i)
                    y = self.ode_alpha_t_gamma(idx) / self.ode_alpha_t_gamma(last) * last_x + (1 - self.ode_alpha_t_gamma(idx) / self.ode_alpha_t_gamma(last)) * self.mu \
                        + del_noise_last_idx * z1 if r != 1 else last_x
                    new_noise = self.model(y, self.mu, idx, **kwargs) if r != 1 else last_noise

                    x = self.ode_alpha_t_gamma(next) / self.ode_alpha_t_gamma(t) * x + (1 - self.ode_alpha_t_gamma(next) / self.ode_alpha_t_gamma(t)) * self.mu \
                        - 4 * self.lambda_square * self.ode_alpha_t_gamma(next) / torch.sqrt(self.ode_sigma_t(self.T)) * (torch.sqrt(self.exp_minus_beta_t(t)) - 0.5 * hi * torch.sqrt(self.exp_minus_beta_t(next)) - torch.sqrt(self.exp_minus_beta_t(next))) * (noise - new_noise) / (r * hi_minus_1) \
                        - 2 * self.lambda_square * self.ode_alpha_t_gamma(next) / torch.sqrt(self.ode_sigma_t(self.T)) * (torch.sqrt(self.exp_minus_beta_t(t)) - torch.sqrt(self.exp_minus_beta_t(next))) * noise + del_noise * z2
                    
                    # update buffer
                    self.update_list_1(self.x_buffer, x)
                    self.update_list_1(self.noise_buffer, noise)
                else:
                    last_x = self.x_buffer[0]
                    last_noise = self.noise_buffer[0]
                    
                    # calculate y_i and epsilon(y_i, s_i)
                    y = self.ode_alpha_t_gamma(idx) / self.ode_alpha_t_gamma(last) * last_x + (1 - self.ode_alpha_t_gamma(idx) / self.ode_alpha_t_gamma(last)) * self.mu \
                        - 2 * self.lambda_square * self.ode_alpha_t_gamma(idx) / torch.sqrt(self.ode_sigma_t(self.T)) * (torch.sqrt(self.exp_minus_beta_t(last)) - torch.sqrt(self.exp_minus_beta_t(idx))) * last_noise \
                        + del_noise_last_idx * z1 if r != 1 else last_x
                    new_noise = self.model(y, self.mu, idx, **kwargs) if r != 1 else last_noise

                    x = self.ode_alpha_t_gamma(next) / self.ode_alpha_t_gamma(t) * x + (1 - self.ode_alpha_t_gamma(next) / self.ode_alpha_t_gamma(t)) * self.mu \
                        - 4 * self.lambda_square * self.ode_alpha_t_gamma(next) / torch.sqrt(self.ode_sigma_t(self.T)) * (torch.sqrt(self.exp_minus_beta_t(t)) - 0.5 * hi * torch.sqrt(self.exp_minus_beta_t(next)) - torch.sqrt(self.exp_minus_beta_t(next))) * (noise - new_noise) / (r * hi_minus_1) \
                        - 2 * self.lambda_square * self.ode_alpha_t_gamma(next) / torch.sqrt(self.ode_sigma_t(self.T)) * (torch.sqrt(self.exp_minus_beta_t(t)) - torch.sqrt(self.exp_minus_beta_t(next))) * noise + del_noise * z2
                    
                    # update buffer
                    self.update_list_1(self.x_buffer, x)
                    self.update_list_1(self.noise_buffer, noise)

            if save_states:  # only consider to save 100 images
                interval = self.T // self.solver_step
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)
                    # tvutils.save_image(y.data, f'{save_dir}/state_{idx}_y.png', normalize=False)
                    # tvutils.save_image(self.x_buffer[0].data, f'{save_dir}/state_{idx}_x.png', normalize=False)

        return x
    
    def unidb_mean_ode_solver_data_prediction(self, xt, T=-1, save_states=False, save_dir='ode_state', **kwargs):
        # T = self.solver_step if T < 0 else T
        step_size = self.T // self.solver_step
        x = xt.clone()
        for t in tqdm(range(self.T, 0, -step_size)):
            noise = self.model(x, self.mu, t, **kwargs) if t != self.T else 0
            # score = - noise / self.f_sigma(t) if t != 100 else 0
            # x = self.reverse_mean_ode_step(x, score, t)
            predicted_x0 = self.predict_x0_through_score(x, t, noise)
            t_i1 = t - step_size
            
            x = self.ode_sigma_t(t_i1) / self.ode_sigma_t(t) * x + (1 - self.ode_sigma_t(t_i1) / self.ode_sigma_t(t) + self.ode_sigma_t(t_i1) / self.ode_sigma_t(self.T) / self.exp_minus_beta_t_gamma(t) - self.ode_alpha_t_gamma(t_i1) / self.ode_sigma_t(self.T)) * self.mu + (self.ode_alpha_t_gamma(t_i1) / self.ode_sigma_t(self.T) - self.ode_sigma_t(t_i1) / self.ode_sigma_t(self.T) / self.exp_minus_beta_t_gamma(t)) * predicted_x0
            if save_states:  # only consider to save 100 images
                interval = self.T // self.solver_step
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)
                    tvutils.save_image(predicted_x0.data, f'{save_dir}/x0_{idx}.png', normalize=False)

        return x
    

    def unidb_sde_solver_data_prediction(self, xt, T=-1, save_states=False, save_dir='sde_state', **kwargs):
        # T = self.solver_step if T < 0 else T
        step_size = self.T // self.solver_step
        x = xt.clone()
        for t in tqdm(range(self.T, 0, -step_size)):
            noise = self.model(x, self.mu, t, **kwargs) if t != self.T else 0
            # score = - noise / self.f_sigma(t) if t != 100 else 0
            predicted_x0 = self.predict_x0_through_score(x, t, noise)
            # print(predicted_x0)
            t_i1 = t - step_size
            z = torch.randn_like(x)
            delta = self.lambda_square * self.ode_sigma_t(t_i1) * torch.sqrt(1 / (torch.exp((self.thetas_cumsum[t_i1]) * self.dt) ** 2 - 1) - 1 / (torch.exp((self.thetas_cumsum[t]) * self.dt) ** 2 - 1))
            if t == step_size:
                delta = 0
            x = self.ode_sigma_t(t_i1) / self.ode_sigma_t(t) * x + (1 - self.ode_sigma_t(t_i1) / self.ode_sigma_t(t) + self.ode_sigma_t(t_i1) / self.ode_sigma_t(self.T) / self.exp_minus_beta_t_gamma(t) - self.ode_alpha_t_gamma(t_i1) / self.ode_sigma_t(self.T)) * self.mu + (self.ode_alpha_t_gamma(t_i1) / self.ode_sigma_t(self.T) - self.ode_sigma_t(t_i1) / self.ode_sigma_t(self.T) / self.exp_minus_beta_t_gamma(t)) * predicted_x0 + delta * z
    
            if save_states:  # only consider to save 100 images
                interval = self.T // self.solver_step
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)
                    tvutils.save_image(predicted_x0.data, f'{save_dir}/x0_{idx}.png', normalize=False)

        return x
    
    def unidb_sde_solver_data_prediction_update(self, xt, T=-1, save_states=False, save_dir='sde_state', **kwargs):
        # T = self.solver_step if T < 0 else T
        step_size = self.T // self.solver_step
        x = xt.clone()
        for t in tqdm(range(self.T, 0, -step_size)):
            noise = self.model(x, self.mu, t, **kwargs) if t != self.T else 0
            # score = - noise / self.f_sigma(t) if t != 100 else 0
            predicted_x0 = self.predict_x0_through_score(x, t, noise)
            # print(predicted_x0)
            t_next = t - step_size
            z = torch.randn_like(x)
            delta = self.lambda_square * self.ode_sigma_t(t_next) * torch.sqrt(1 / (torch.exp((self.thetas_cumsum[t_next]) * self.dt) ** 2 - 1) - 1 / (torch.exp((self.thetas_cumsum[t]) * self.dt) ** 2 - 1))
            if t == step_size:
                delta = 0

            coeff1 = self.ode_alpha_t_gamma(t_next) * self.ode_alpha_t(t) * self.ode_sigma_t(t_next) / (self.ode_alpha_t_gamma(t) * self.ode_alpha_t(t_next) * self.ode_sigma_t(t))
            coeff3 = self.ode_alpha_t_gamma(t_next) / self.ode_alpha_t_gamma(0) - self.ode_alpha_t_gamma(t_next) * self.ode_sigma_t(t_next) / (self.ode_alpha_t(t_next) * self.ode_alpha_t_gamma(0) * self.exp_minus_beta_t_gamma(t))
            coeff2 = 1 - coeff1 - coeff3
            x = coeff1 * x + coeff2 * self.mu + coeff3 * predicted_x0 + delta * z
            # x = self.ode_sigma_t(t_next) / self.ode_sigma_t(t) * x + (1 - self.ode_sigma_t(t_next) / self.ode_sigma_t(t) + self.ode_sigma_t(t_next) / self.ode_sigma_t(self.T) / self.exp_minus_beta_t_gamma(t) - self.ode_alpha_t_gamma(t_next) / self.ode_sigma_t(self.T)) * self.mu + (self.ode_alpha_t_gamma(t_next) / self.ode_sigma_t(self.T) - self.ode_sigma_t(t_next) / self.ode_sigma_t(self.T) / self.exp_minus_beta_t_gamma(t)) * predicted_x0 + delta * z
            
            if save_states:  # only consider to save 100 images
                interval = self.T // self.solver_step
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)
                    tvutils.save_image(predicted_x0.data, f'{save_dir}/x0_{idx}.png', normalize=False)

        return x
    
    def goub_sde_solver_data_prediction(self, xt, T=-1, save_states=False, save_dir='sde_state', **kwargs):
        # T = self.solver_step if T < 0 else T
        step_size = self.T // self.solver_step
        x = xt.clone()
        for t in tqdm(range(self.T, 0, -step_size)):
            noise = self.model(x, self.mu, t, **kwargs) if t != self.T else 0
            # score = - noise / self.f_sigma(t) if t != 100 else 0
            # x = self.reverse_mean_ode_step(x, score, t)
            predicted_x0 = self.predict_x0_through_score(x, t, noise)
            # print(predicted_x0)
            t_i1 = t - step_size
            z = torch.randn_like(x)
            delta = self.lambda_square * self.ode_sigma_t(t_i1) * torch.sqrt(1 / (torch.exp((self.thetas_cumsum[t_i1]) * self.dt) ** 2 - 1) - 1 / (torch.exp((self.thetas_cumsum[t]) * self.dt) ** 2 - 1))
            if t == step_size:
                delta = 0
            x = self.ode_sigma_t(t_i1) / self.ode_sigma_t(t) * x + (1 - self.ode_sigma_t(t_i1) / self.ode_sigma_t(t) + self.ode_sigma_t(t_i1) / self.ode_sigma_t(self.T) / self.exp_minus_beta_t(t) - self.ode_alpha_t(t_i1) / self.ode_sigma_t(self.T)) * self.mu + (self.ode_alpha_t(t_i1) / self.ode_sigma_t(self.T) - self.ode_sigma_t(t_i1) / self.ode_sigma_t(self.T) / self.exp_minus_beta_t(t)) * predicted_x0 + delta * z
    
            if save_states:  # only consider to save 100 images
                interval = self.T // self.solver_step
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)
                    tvutils.save_image(predicted_x0.data, f'{save_dir}/x0_{idx}.png', normalize=False)

        return x
    
    # no approximation
    def unidb_reverse_sde_solver_2_data_prediction(self, xt, T=-1, save_states=False, save_dir='sde_state', **kwargs):
        # T = self.solver_step if T < 0 else T
        step_size = self.T // self.solver_step
        x = xt.clone()

        r = 1 # super-parameter
        print("r", r)
        ty = "multi_step" # decide gradient approximation
        assert ty in ["single_step", "multi_step"]

        for t in tqdm(range(self.T, 0, -step_size)):
            if ty == "single_step":
                assert r > 0 and r < 1
                noise = self.model(x, self.mu, t, **kwargs) # if t != self.T else 0
                predicted_x0 = self.predict_x0_through_score(x, t, noise)

                z1 = torch.randn_like(x)
                z2 = torch.randn_like(x)

                next = t - step_size
                beta_next = self.beta_t_gamma(next)
                beta_now = self.beta_t_gamma(t)
                hi = beta_next - beta_now
                idx = self.beta_inv_gamma(beta_now + r * hi).item()
                # print(idx)
                # print(beta_next)

                del_data_idx = self.delta_data(t, idx) if self.solver_type == "sde" else 0
                del_data = self.delta_data(t, next) if self.solver_type == "sde" else 0

                coeff = self.ode_alpha_t_gamma(next) / self.ode_sigma_t(self.T) - self.ode_sigma_t(next) / self.ode_sigma_t(self.T) / self.exp_minus_beta_t_gamma(t)

                y = self.ode_sigma_t(idx) / self.ode_sigma_t(t) * x \
                    + (1 - self.ode_sigma_t(idx) / self.ode_sigma_t(t) + self.ode_sigma_t(idx) / self.ode_sigma_t(self.T) / self.exp_minus_beta_t_gamma(t) - self.ode_alpha_t_gamma(idx) / self.ode_sigma_t(self.T)) * self.mu \
                    + (self.ode_alpha_t_gamma(idx) / self.ode_sigma_t(self.T) - self.ode_sigma_t(idx) / self.ode_sigma_t(self.T) / self.exp_minus_beta_t_gamma(t)) * predicted_x0 + del_data_idx * z1
                new_noise = self.model(y, self.mu, idx, **kwargs)
                new_predicted_x0 = self.predict_x0_through_score(y, idx, new_noise)

                coeff2 = (self.ode_sigma_t(next) / self.ode_sigma_t(self.T) / self.exp_minus_beta_t_gamma(t) + self.ode_alpha_t_gamma(t) / self.ode_sigma_t(self.T) * (hi - 1)) / (r * hi)
                
                if t == step_size:
                    coeff2 = self.ode_alpha_t_gamma(t) / self.ode_sigma_t(self.T) / r
                
                x = self.ode_sigma_t(next) / self.ode_sigma_t(t) * x \
                    + (1 - self.ode_sigma_t(next) / self.ode_sigma_t(t) + self.ode_sigma_t(next) / self.ode_sigma_t(self.T) / self.exp_minus_beta_t_gamma(t) - self.ode_alpha_t_gamma(next) / self.ode_sigma_t(self.T)) * self.mu \
                    + coeff * predicted_x0 + coeff2 * (new_predicted_x0 - predicted_x0) + del_data * z2

            else: # "multi_step"
                # TODO
                assert r == 1
                noise = self.model(x, self.mu, t, **kwargs) # if t != self.T else 0
                predicted_x0 = self.predict_x0_through_score(x, t, noise)

                z1 = torch.randn_like(x)
                z2 = torch.randn_like(x)
                next = t - step_size 
                
                beta_next = self.beta_t_gamma(next)
                beta_now = self.beta_t_gamma(t)
                
                
                hi = beta_next - beta_now
                
                del_data = self.delta_data(t, next) if self.solver_type == "sde" else 0

                if t < self.T:
                    last = t + step_size 
                    beta_last = self.beta_t_gamma(last)
                    hi_minus_1 = beta_now - beta_last
                    idx = self.beta_inv_gamma(beta_now - r * hi_minus_1).item()
                    del_data_last_idx = self.delta_data(last, idx) if self.solver_type == "sde" else 0

                    coeff2 = (self.ode_sigma_t(next) / self.ode_sigma_t(self.T) / self.exp_minus_beta_t_gamma(t) + self.ode_alpha_t_gamma(t) / self.ode_sigma_t(self.T) * (hi - 1)) / (r * hi_minus_1) if t != step_size else 0

                coeff = self.ode_alpha_t_gamma(next) / self.ode_sigma_t(self.T) - self.ode_sigma_t(next) / self.ode_sigma_t(self.T) / self.exp_minus_beta_t_gamma(t)
                # print(coeff)
                
                if t == self.T:
                    # coeff = 0
                    x = self.ode_sigma_t(next) / self.ode_sigma_t(t) * x \
                        + (1 - self.ode_sigma_t(next) / self.ode_sigma_t(t) + self.ode_sigma_t(next) / self.ode_sigma_t(self.T) / self.exp_minus_beta_t_gamma(t) - self.ode_alpha_t_gamma(next) / self.ode_sigma_t(self.T)) * self.mu \
                        + coeff * predicted_x0 + del_data * z2
                    y = self.mu
                    new_data_prediction = self.mu
                    # update buffer
                    self.update_list_1(self.x_buffer, x)
                    self.update_list_1(self.data_buffer, predicted_x0)
                    self.update_list_1(self.noise_buffer, noise)
                else:
                    
                    last_x = self.x_buffer[0]
                    last_data_prediction = self.data_buffer[0]
                    last_noise = self.noise_buffer[0]
                    
                    # calculate y_i and epsilon(y_i, s_i)
                    y = self.ode_sigma_t(idx) / self.ode_sigma_t(last) * last_x \
                        + (1 - self.ode_sigma_t(idx) / self.ode_sigma_t(last) + self.ode_sigma_t(idx) / self.ode_sigma_t(self.T) / self.exp_minus_beta_t_gamma(last) - self.ode_alpha_t_gamma(idx) / self.ode_sigma_t(self.T)) * self.mu \
                        + (self.ode_alpha_t_gamma(idx) / self.ode_sigma_t(self.T) - self.ode_sigma_t(idx) / self.ode_sigma_t(self.T) / self.exp_minus_beta_t_gamma(last)) * last_data_prediction \
                        + del_data_last_idx * z1 if r != 1 else last_x
                    
                    new_noise = self.model(y, self.mu, idx, **kwargs) if r != 1 else last_noise
                    new_data_prediction = self.predict_x0_through_score(y, idx, new_noise) if r != 1 else last_data_prediction

                    x = self.ode_sigma_t(next) / self.ode_sigma_t(t) * x \
                        + (1 - self.ode_sigma_t(next) / self.ode_sigma_t(t) + self.ode_sigma_t(next) / self.ode_sigma_t(self.T) / self.exp_minus_beta_t_gamma(t) - self.ode_alpha_t_gamma(next) / self.ode_sigma_t(self.T)) * self.mu \
                        + coeff * predicted_x0 + coeff2 * (predicted_x0 - new_data_prediction) + del_data * z2

                    # update buffer
                    self.update_list_1(self.x_buffer, x)
                    self.update_list_1(self.data_buffer, predicted_x0)
                    self.update_list_1(self.noise_buffer, noise)

            if save_states:  # only consider to save 100 images
                interval = self.T // self.solver_step
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)
                    tvutils.save_image(y.data, f'{save_dir}/state_{idx}_y.png', normalize=False)
                    tvutils.save_image(predicted_x0.data, f'{save_dir}/state_{idx}_data.png', normalize=False)
                    # tvutils.save_image(new_data_prediction.data, f'{save_dir}/state_{idx}_mid_data.png', normalize=False)

        return x

    def unidb_pf_ode_solver_data_prediction(self, xt, T=-1, save_states=False, save_dir='ode_state', **kwargs):
        # T = self.ode_T if T < 0 else T
        step_size = self.T // self.ode_T
        x = xt.clone()
        for t in tqdm(range(self.T, 0, -step_size)):
            noise = self.model(x, self.mu, t, **kwargs) if t != self.T else 0
            # score = - noise / self.f_sigma(t) if t != 100 else 0
            predicted_x0 = self.predict_x0_through_score(x, t, noise)
            # print(predicted_x0)
            t_i1 = t - step_size
            c1 = torch.sqrt(self.ode_alpha_t_gamma(t_i1) * self.ode_sigma_t(t_i1)) / torch.sqrt(self.ode_alpha_t_gamma(t) * self.ode_sigma_t(t))
            c3 = torch.sqrt(self.ode_alpha_t_gamma(t_i1) * self.ode_alpha_t_gamma(t) * self.ode_sigma_t(t_i1)) / (self.ode_sigma_t(self.T) * torch.sqrt(self.ode_sigma_t(t))) \
                - self.ode_alpha_t_gamma(t_i1) / self.ode_sigma_t(self.T)
            c2 = 1 - c1 + c3
            # print("c1: ", c1)
            # print("c2: ", c2)
            # print("c3: ", c3)
            x = c1 * x + c2 * self.mu - c3 * predicted_x0
    
            if save_states:  # only consider to save 100 images
                interval = self.T // self.ode_T
                if t % interval == 0:
                    idx = t // interval
                    os.makedirs(save_dir, exist_ok=True)
                    tvutils.save_image(x.data, f'{save_dir}/state_{idx}.png', normalize=False)
                    tvutils.save_image(predicted_x0.data, f'{save_dir}/x0_{idx}.png', normalize=False)

        return x

    
    # ----------------------------------------------------------------------------------------------------------------

    # solver with data predictor

    def predict_x0_through_score(self, xt, t, noise):
        if t == self.T:
            return self.mu
        return (xt - self.n(t) * self.mu - self.f_sigma(t) * noise) / self.m(t)


    # ----------------------------------------------------------------------------------------------------------------


    ################################################################

    def weights(self, t):
        return torch.exp(-self.thetas_cumsum[t] * self.dt)

    # sample states for training
    def generate_random_states(self, x0, mu):
        x0 = x0.to(self.device)
        mu = mu.to(self.device)
        
        self.set_mu(mu)
        batch = x0.shape[0]
        timesteps = torch.randint(1, self.T, (batch, 1, 1, 1)).long()

        state_mean = self.f_mean(x0, timesteps)
        noises = torch.randn_like(state_mean)
        noise_level = self.f_sigma(timesteps)
        noisy_states = noises * noise_level + state_mean
        return timesteps, noisy_states.to(torch.float32)
