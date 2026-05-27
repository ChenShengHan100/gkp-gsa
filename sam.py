import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups: #  only one group
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class GSA(torch.optim.Optimizer):
   
    def __init__(self, params, base_optimizer, 
                 rho_scale_C=1.0,
                 rho_min=2e-3,     
                 rho_max=3e-3,      
                 sigma=1, lmbda=0.9, adaptive=True, **kwargs):
        defaults = dict(adaptive=adaptive, **kwargs) 
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        super().__init__(self.param_groups, defaults)
        self.defaults.update(self.base_optimizer.defaults)
        self.rho_scale_C = rho_scale_C 
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.sigma = sigma
        self.lmbda = lmbda
        self.d_total = self._compute_total_dimensionality()
        self.ascent_vectors = {} 
        self.total_batch_weight = 0.0
        self._cached_rho_k_star = {}
        self._cached_theta_norm = None 

    def _compute_total_dimensionality(self):
        d = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    d += p.numel()
        return torch.tensor(d, dtype=torch.float)

    def _compute_theta_norm(self):
        all_params_norm = []
        device = None
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    all_params_norm.append(p.norm(p=2))
                    if device is None: 
                        device = p.device
        if not all_params_norm:
             if device is None:
                 return torch.tensor(0.0)
             return torch.tensor(0.0, device=device)
        return torch.norm(torch.stack(all_params_norm), p=2).to(device)


    @torch.no_grad()
    def compute_group_ascent_vector(self, group_index, group_weight, n_k):
        if self._cached_theta_norm is None:
            self._cached_theta_norm = self._compute_theta_norm()
        theta_norm = self._cached_theta_norm
        
        group_grads_for_norm = []
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                
                grad = p.grad.clone() 
                momentum_key = f"momentum_{group_index}"
                if not momentum_key in self.state[p]:
                    self.state[p][momentum_key] = grad
                else:
                    p.grad -= self.state[p][momentum_key] * self.sigma
                    self.state[p][momentum_key] = self.state[p][momentum_key] * self.lmbda + grad * (1 - self.lmbda)
                
                norm_p = ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2)
                group_grads_for_norm.append(norm_p)

        if not group_grads_for_norm: return 
        grad_norm_k = torch.norm(torch.stack(group_grads_for_norm), p=2) + 1e-12
        device = grad_norm_k.device
        d_term = self.d_total.to(device).pow(-0.25)
        n_k_safe = max(n_k, 2) 
        n_k_term = torch.tensor(n_k_safe - 1.0, device=device).pow(-0.25)
        inner_sqrt_term = torch.sqrt(theta_norm / (2.0 * grad_norm_k))
        rho_k_star_theory = (inner_sqrt_term * d_term * n_k_term)
        rho_k_star_scaled = self.rho_scale_C * rho_k_star_theory
        #print(f"  [Group {group_index}] Rho_Theory: {rho_k_star_theory.item():.6f}, "
              # f"Rho_Scaled (C={self.rho_scale_C}): {rho_k_star_scaled.item():.6f}")
        rho_k_star_final = rho_k_star_scaled.clamp(min=self.rho_min, max=self.rho_max)
        #rho_k_star_final=rho_k_star_scaled
        self._cached_rho_k_star[group_index] = rho_k_star_final
        magnitude_M_k = self.d_total.to(device).pow(0.5) * rho_k_star_final
        scale_k = magnitude_M_k / grad_norm_k
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                e_w_k = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale_k.to(p)
                weighted_e_w_k = e_w_k * group_weight
                if p not in self.ascent_vectors:
                    self.ascent_vectors[p] = weighted_e_w_k
                else:
                    self.ascent_vectors[p].add_(weighted_e_w_k)
        
        self.total_batch_weight += group_weight


   
    
    @torch.no_grad()
    def get_regularization_term(self, group_index, n_k):
        if group_index not in self._cached_rho_k_star or self._cached_theta_norm is None:
            default_device = self.param_groups[0]["params"][0].device
            return torch.tensor(0.0, device=default_device), \
                   torch.tensor(0.0, device=default_device)
        rho_k_star_final = self._cached_rho_k_star[group_index]
        theta_norm = self._cached_theta_norm
        n_k_safe = max(n_k, 2)
        n_k_minus_1 = torch.tensor(n_k_safe - 1.0, device=theta_norm.device)
        reg_term = (1.0 / (2.0 * torch.sqrt(n_k_minus_1))) * (theta_norm / rho_k_star_final)
        return reg_term, rho_k_star_final
    
    @torch.no_grad()
    def apply_global_ascent(self, zero_grad=False):
        if not self.ascent_vectors:
            return
        for group in self.param_groups:
            for p in group["params"]:
                if p in self.ascent_vectors:
                    e_w_total = self.ascent_vectors[p]
                    if self.total_batch_weight > 0:
                        e_w_total = e_w_total / self.total_batch_weight
                    self.state[p]["old_p"] = p.data.clone()
                    p.add_(e_w_total)
        self.ascent_vectors.clear()
        self.total_batch_weight = 0.0
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                if "old_p" in self.state[p]:
                    p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        self._cached_rho_k_star.clear()
        self._cached_theta_norm = None
        if zero_grad: self.zero_grad()
        
    def _grad_norm(self): pass
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


