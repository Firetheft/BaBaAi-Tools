import torch
from tqdm.auto import trange
import comfy.samplers

def append_dims(x, target_dims):
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    expanded = x[(...,) + (None,) * dims_to_append]

    return expanded.detach().clone() if expanded.device.type == 'mps' else expanded

def to_d(x, sigma, denoised):
    return (x - denoised) / append_dims(sigma, x.ndim)

@torch.no_grad()
def sample_sa_ode_stable(model, x, sigmas, extra_args=None, callback=None, disable=None, 
                         solver_order=3, use_adaptive_order=True, use_velocity_smoothing=True, 
                         convergence_threshold=0.15, smoothing_factor=0.8):

    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    velocity_buffer = []
    smoothed_velocity = None
    step_count = 0
    
    def get_adaptive_order(sigma, num_inference_steps):
        if not use_adaptive_order:
            return solver_order
        
        if num_inference_steps <= 8:
            return min(2, solver_order)

        if sigma > 0.7:
            return min(2, solver_order)

        elif sigma > convergence_threshold:
            return solver_order

        else:
            return max(1, solver_order - 1)
    
    def compute_multistep_velocity(order):
        if not velocity_buffer:
            raise RuntimeError("velocity_buffer is empty")
        
        if len(velocity_buffer) < order:
            order = len(velocity_buffer)
        
        if order >= 3 and len(velocity_buffer) >= 3:
            v = (
                (23/12) * velocity_buffer[-1] -
                (16/12) * velocity_buffer[-2] +
                (5/12) * velocity_buffer[-3]
            )
        elif order >= 2 and len(velocity_buffer) >= 2:
            v = 1.5 * velocity_buffer[-1] - 0.5 * velocity_buffer[-2]
        elif len(velocity_buffer) >= 1:
            v = velocity_buffer[-1]
        else:
            raise RuntimeError("No velocity data available")
        
        return v
    
    def apply_velocity_smoothing(velocity, sigma):
        if not use_velocity_smoothing:
            return velocity
        
        num_inference_steps = len(sigmas)
        if num_inference_steps <= 8:
            return velocity
        
        if sigma < convergence_threshold:
            nonlocal smoothed_velocity
            if smoothed_velocity is None:
                smoothed_velocity = velocity
            else:
                alpha = smoothing_factor
                smoothed_velocity = alpha * smoothed_velocity + (1 - alpha) * velocity
            return smoothed_velocity
        else:
            smoothed_velocity = velocity
            return velocity
    
    num_inference_steps = len(sigmas)
    
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        
        denoised = model(x, sigma * s_in, **extra_args)
        
        d = to_d(x, sigma, denoised)
        
        velocity_buffer.append(d)
        while len(velocity_buffer) > solver_order + 1:
            velocity_buffer.pop(0)
        
        current_order = get_adaptive_order(sigma.item(), num_inference_steps)
        
        if len(velocity_buffer) >= 2:
            velocity = compute_multistep_velocity(current_order)
        else:
            velocity = d
        
        velocity = apply_velocity_smoothing(velocity, sigma.item())
        
        dt = sigma_next - sigma

        if num_inference_steps > 8 and sigma.item() < convergence_threshold:
            damping = 0.5 + 0.5 * (sigma.item() / convergence_threshold)
            dt = dt * damping
        
        if sigma_next == 0:
            x = denoised
        else:
            x = x + velocity * dt
        
        if num_inference_steps > 8 and sigma.item() < 0.05 and len(velocity_buffer) >= 3:
            avg_velocity = sum(velocity_buffer[-3:]) / 3
            stabilized = x + avg_velocity * dt
            blend_factor = sigma.item() / 0.05
            x = blend_factor * x + (1 - blend_factor) * stabilized
        
        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': denoised})

        step_count += 1
    
    return x

extra_samplers = {'sa_solver_ode_stable': sample_sa_ode_stable}

def add_samplers():
    from comfy.samplers import KSampler, k_diffusion_sampling
    added = 0
    for sampler in extra_samplers:
        if sampler not in KSampler.SAMPLERS:
            try:
                idx = KSampler.SAMPLERS.index("sa_solver")
                KSampler.SAMPLERS.insert(idx+1, sampler)
                setattr(k_diffusion_sampling, "sample_{}".format(sampler), extra_samplers[sampler])
                added += 1
            except ValueError as err:
                try:
                    KSampler.SAMPLERS.append(sampler)
                    setattr(k_diffusion_sampling, "sample_{}".format(sampler), extra_samplers[sampler])
                    added += 1
                except Exception as e:
                    print(f"[SA-ODE-Sampler error] Failed to add sampler {sampler}: {e}")

    if added > 0:
        import importlib
        importlib.reload(k_diffusion_sampling)
        print("[INFO] SA ODE Stable sampler added successfully!")

add_samplers()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}