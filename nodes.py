import torch
from tqdm.auto import trange

import comfy.model_management
from comfy_api.latest import IO, io
from comfy.samplers import process_conds, sampling_function

from comfy_extras.nodes_trellis2 import _dino_encode_batch


class Trellis2MultiViewSamplerSelect(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Trellis2MultiViewSamplerSelect",
            display_name="Trellis2 Multi-View Sampler Select",
            category="model/conditioning/trellis",
            inputs=[
                io.Combo.Input("sampler_name", options=["euler", "heun", "rk4", "rk5"]),
            ],
            outputs=[io.Sampler.Output()],
        )

    @classmethod
    def execute(cls, sampler_name):
        # SAMPLERS: each step receives (get_blended_v, x, t, dt) and returns (x_next, k1)
        # Based on FlowEulerMultiViewSampler.sample_once()
        def _step_euler(get_blended_v, x, t, dt, t_next):
            k1 = get_blended_v(latent=x, sigma=t)

            pred_x_prev = x - (t - t_next) * k1
            return pred_x_prev, k1

        # Based on FlowHeunMultiViewSampler.sample_once()
        def _step_heun(get_blended_v, x, t, dt, t_next):
            # Heun's Method (RK2) Evaluations
            # Step 1: Predictor (Euler step)
            k1 = get_blended_v(latent=x, sigma=t)
            x_temp = x + k1 * dt

            # Step 2: Corrector
            k2 = get_blended_v(latent=x_temp, sigma=t + dt)

            # Combine
            pred_x_prev = x + 0.5 * dt * (k1 + k2)
            return pred_x_prev, k1

        # Based on FlowRK4MultiViewSampler.sample_once()
        def _step_rk4(get_blended_v, x, t, dt, t_next):
            k1 = get_blended_v(x, t)
            k2 = get_blended_v(x + k1 * (0.5 * dt), t + 0.5 * dt)
            k3 = get_blended_v(x + k2 * (0.5 * dt), t + 0.5 * dt)
            k4 = get_blended_v(x + k3 * dt, t + dt)

            pred_x_prev = x + (k1 + k2 * 2 + k3 * 2 + k4) * (dt / 6.0)
            return pred_x_prev, k1

        # Based on FlowRK5MultiViewSampler.sample_once()
        def _step_rk5(get_blended_v, x, t, dt, t_next):
            # Butcher Tableau Intermediate steps
            c2, c3, c4, c5, c6 = 1/4, 1/4, 1/2, 3/4, 1.0

            k1 = get_blended_v(x, t)
            k2 = get_blended_v(x + k1 * (dt * 1/4), t + dt * c2)
            k3 = get_blended_v(x + (k1 * 1/8 + k2 * 1/8) * dt, t + dt * c3)
            k4 = get_blended_v(x + (k2 * -1/2 + k3 * 1.0) * dt, t + dt * c4)
            k5 = get_blended_v(x + (k1 * 3/16 + k4 * 9/16) * dt, t + dt * c5)
            k6 = get_blended_v(x + (k1 * -3/7 + k2 * 2/7 + k3 * 12/7 + k4 * -12/7 + k5 * 8/7) * dt, t + dt * c6)
            
            pred_x_prev = x + (k1 * 7/90 + k3 * 32/90 + k4 * 12/90 + k5 * 32/90 + k6 * 7/90) * dt
            return pred_x_prev, k1

        if sampler_name == "heun":
            sampler_function = _step_heun
        elif sampler_name == "rk4":
            sampler_function = _step_rk4
        elif sampler_name == "rk5":
            sampler_function = _step_rk5
        else:
            sampler_function = _step_euler
    
        @torch.no_grad()
        def sampler(model, x, sigmas, extra_args=None, callback=None, disable=None, **kw):
            ts = sigmas.tolist()

            def get_v(latent, sigma):
                t_tensor = torch.full((latent.shape[0],), sigma, device=latent.device, dtype=torch.float32)
                return extra_args["get_blended_v"](latent, t_tensor)
    
            for i in trange(len(sigmas) - 1, disable=disable):
                t, t_next = ts[i], ts[i + 1]
                dt = t_next - t

                pred_x_prev, k1 = sampler_function(get_v, x, t, dt, t_next)

                if callback is not None:
                    # Estimate pred_x_0 based on k1 for tracking
                    pred_x_0 = x - t * k1
                    callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], 
                            "denoised": pred_x_0})
                
                x = pred_x_prev
            return x

        return io.NodeOutput(comfy.samplers.KSAMPLER(sampler))

class Trellis2MultiViewConditioning(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2MultiViewConditioning",
            display_name="Trellis2 Multi-View Conditioning",
            category="model/conditioning/trellis",
            inputs=[
                IO.ClipVision.Input("clip_vision_model"),
                IO.Image.Input("front", optional=True),
                IO.Image.Input("left", optional=True),
                IO.Image.Input("back", optional=True),
                IO.Image.Input("right", optional=True),
            ],
            outputs=[
                IO.Conditioning.Output(display_name="positive"),
                IO.Conditioning.Output(display_name="negative"),
            ]
        )

    @classmethod
    def execute(cls, clip_vision_model, front=None, left=None, back=None, right=None) -> IO.NodeOutput:
        out_device = comfy.model_management.intermediate_device()
        
        # Collect views
        views = {}
        if front is not None: views['front'] = front
        if left is not None: views['left'] = left
        if back is not None: views['back'] = back
        if right is not None: views['right'] = right
        if front is None and left is None and right is None and back is None:
            raise ValueError("[Trellis2 Multi-View Conditioning]: At least an image is required!")
        
        views_list = list(views.keys())

        positive, positive_embeds = {}, {}
        negative, negative_embeds = {}, {}
        # Calculate conditioning per view
        for v, img in views.items():
            cond = _dino_encode_batch(clip_vision_model, img, out_device)
            positive[v] = cond["global_512"]
            positive_embeds[v] = cond["global_1024"]

            negative[v] = torch.zeros_like(cond["global_512"])
            negative_embeds[v] = torch.zeros_like(cond["global_1024"])
        
        pos_extras = {
            "embeds": positive_embeds[views_list[0]], "views_list": views_list,
            "true_positive": positive,
            "true_positive_embeds": positive_embeds,
        }
        neg_extras = {
            "embeds": negative_embeds[views_list[0]],
            "true_negative": negative,
            "true_negative_embeds": negative_embeds,
        }

        # We don't actually use the single tensors (positive[views_list[0]] and negative[views_list[0]),
        # but to use the default stage nodes this structure is required.
        # The "true_positive" etc are handled (cast to device etc) inside the guider
        return IO.NodeOutput(
            [[positive[views_list[0]], pos_extras]],
            [[negative[views_list[0]], neg_extras]]
        )


class MultiViewCFGGuider(comfy.samplers.CFGGuider):
    def set_trellis2_opt(self, views, blend_temperature, coords_resolution, extras):

        self.views = views
        self.blend_temperature = blend_temperature
        self.resolution = coords_resolution
        self.extras = extras
        self.sigma_min = 1e-5
        self.extra_model_options = None
        self.seed = None

    def inner_set_conds(self, conds):
        pos_extras = conds["positive"][0][1]
        neg_extras = conds["negative"][0][1]

        # positive: views dict at 512; positive_embeds: views dict at 1024
        # negative: views dict at 512; negative_embeds: views dict at 1024
        for v in self.views:
            conds[f"positive_{v}"] = [[
                pos_extras["true_positive"][v],
                {**self.extras, "embeds": pos_extras["true_positive_embeds"][v]}
            ]]
            conds[f"negative_{v}"] = [[
                neg_extras["true_negative"][v],
                {**self.extras, "embeds": neg_extras["true_negative_embeds"][v]}
            ]]

        super().inner_set_conds(conds)

    # Taken as is from https://github.com/visualbruno/ComfyUI-Trellis2/blob/main/trellis2/pipelines/samplers/flow_euler.py, in FlowEulerMultiViewSampler
    def _compute_view_weights_sparse(self, coords, views, blend_temperature=2.0, front_axis='z') -> torch.Tensor:
        """
        Compute blending weights for sparse voxels.
        """
        # Normalize coords to [-1, 1] range (roughly)
        z = (coords[:, 1].float() / self.resolution) * 2 - 1.0
        x = (coords[:, 3].float() / self.resolution) * 2 - 1.0

        if front_axis == 'z':
            # Front (+Z), Back (-Z), Right (+X), Left (-X)
            view_vectors = {
                'front': torch.stack([torch.zeros_like(z), z], dim=1), # (0, z)
                'back':  torch.stack([torch.zeros_like(z), -z], dim=1),
                'right': torch.stack([x, torch.zeros_like(x)], dim=1),
                'left':  torch.stack([-x, torch.zeros_like(x)], dim=1),
            }
        else: # front_axis == 'x' (swapped)
            # Front (+X), Back (-X), Right (+Z), Left (-Z)
             view_vectors = {
                'front': torch.stack([x, torch.zeros_like(x)], dim=1),
                'back':  torch.stack([-x, torch.zeros_like(x)], dim=1),
                'right': torch.stack([torch.zeros_like(z), z], dim=1),
                'left':  torch.stack([torch.zeros_like(z), -z], dim=1),
            }

        scores = []
        for view in views:
            if view in view_vectors:
                v_vec = view_vectors[view]
                score = v_vec.sum(dim=1)
                scores.append(score)
            else:
                scores.append(torch.full_like(z, -10.0))
        
        scores = torch.stack(scores, dim=1) # (N, num_views)
        weights = torch.softmax(scores * blend_temperature, dim=1)
        return weights
    
    # Taken as is from https://github.com/visualbruno/ComfyUI-Trellis2/blob/main/trellis2/pipelines/samplers/flow_euler.py, in FlowEulerMultiViewSampler
    def _compute_view_weights_dense(self, shape, device, views, blend_temperature=2.0, front_axis='z') -> torch.Tensor:
        """
        Compute blending weights for dense grid (B, C, D, H, W).
        Returns weights of shape (1, 1, D, H, W, NumViews) for easy broadcasting (actually we want (1, 1, D, H, W) per view)
        """
        # shape is (B, C, D, H, W)
        D, H, W = shape[2], shape[3], shape[4]
        
        # Create meshgrid in [-1, 1]
        # We assume D is Z axis, W is X axis (usually D, H, W = Z, Y, X in 3D tensors?)
        # Let's verify standard: (Batch, Channel, Depth, Height, Width) -> (B, C, Z, Y, X)
        
        dz = torch.linspace(-1, 1, D, device=device)
        dy = torch.linspace(-1, 1, H, device=device)
        dx = torch.linspace(-1, 1, W, device=device)
        
        # meshgrid 'ij' indexing: (D, H, W) order
        grid_z, grid_y, grid_x = torch.meshgrid(dz, dy, dx, indexing='ij') 
        
        # Flatten for vector calc? Or keep structural. Keep structural.
        
        if front_axis == 'z':
             # Front (+Z), Back (-Z), Right (+X), Left (-X)
             # Vectors are scalar fields here
             view_scores = {
                 'front': grid_z,
                 'back': -grid_z,
                 'right': grid_x,
                 'left': -grid_x,
             }
        else:
             view_scores = {
                 'front': grid_x,
                 'back': -grid_x,
                 'right': grid_z,
                 'left': -grid_z,
             }
             
        scores = []
        for view in views:
            if view in view_scores:
                scores.append(view_scores[view])
            else:
                scores.append(torch.full_like(grid_z, -10.0))
                
        # Stack: (NumViews, D, H, W)
        scores = torch.stack(scores, dim=0) 

        # Softmax over views dimension (0)
        weights = torch.softmax(scores * blend_temperature, dim=0)
        
        # Reshape for broadcasting: (NumViews, 1, 1, D, H, W) -> No wait, loop is over views.
        # We want to return something we can index like weights[i] -> (1, 1, D, H, W)
        
        # Current shape: (NumViews, D, H, W)
        return weights

    def sample_once(self, x):
        coords = self.extras.get("trellis2_coords", None)
        is_sparse = True if coords is not None else False

        if is_sparse:
            # Compute per-voxel weights based on current sparse coords
            weights = self._compute_view_weights_sparse(
                coords, self.views, self.blend_temperature)
            weights = weights.t().reshape(len(self.views), 1, 1, -1, 1) # (V,1,1,N,1)
        else:
            # Dense tensor
            weights = self._compute_view_weights_dense(
                x.shape, x.device, self.views, self.blend_temperature)

        return weights
        
    def predict_noise(self, x, timestep, model_options=None, seed=None):
        """
        Sample with multi-view blending.
        """
        model_options = model_options or self.extra_model_options
        seed = self.seed if seed is None else seed
        
        # Follow wrapper pipeline: skip cfg rescale when cfg is 1 or 0
        if self.cfg == 1 or self.cfg == 0:
            model_options = {k: v for k, v in model_options.items()
                            if k != "sampler_cfg_function"}

        # Avoids reduction dim 0 to have zero sizes
        # Required for Heun, Rk4, Rk5 samplers
        timestep = timestep.clamp(min=self.sigma_min) # never 0

        # Run model for each view and blend predictions
        pred_v_accum = x.clone()

        for i, view in enumerate(self.views):
            pred_v_view = sampling_function(
                self.inner_model, x, timestep,
                self.conds[f"negative_{view}"], self.conds[f"positive_{view}"], 
                self.cfg, model_options=model_options, seed=seed
            )

            # Weighted accumulation
            w = self.weights[i]
            pred_v_accum += (pred_v_view - x) * w

        # x0 -> v
        return (x.float() - pred_v_accum.float()) / timestep.flatten()[0]

    def inner_sample(self, noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=None):
        self.inner_model.latent_shapes = latent_shapes

        if latent_image is not None and torch.count_nonzero(latent_image) > 0: # Don't shift the empty latent image.
            latent_image = self.inner_model.process_latent_in(latent_image)

        self.conds = process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed, latent_shapes=latent_shapes)

        extra_model_options = comfy.model_patcher.create_model_options_clone(self.model_options)
        extra_model_options.setdefault("transformer_options", {})["sample_sigmas"] = sigmas
        extra_args = {"model_options": extra_model_options, "seed": seed}

        self.weights = self.sample_once(noise)
        extra_args["get_blended_v"] = self.__call__
        extra_args["sigma_min"] = self.sigma_min
        self.extra_model_options = extra_model_options
        self.seed = seed

        executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(
            sampler.sample,
            sampler,
            comfy.patcher_extension.get_all_wrappers(
                comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE, 
                extra_args["model_options"], 
                is_model_options=True
            )
        )
        with self.model_patcher.writable_models(self.conds, noise.shape, self.model_options):
            samples = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
        return self.inner_model.process_latent_out(samples.to(torch.float32))


class Trellis2MultiViewCFGGuider(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Trellis2MultiViewCFGGuider",
            display_name="Trellis2 Multi-View CFG Guider",
            category="model/sampling/guiders",
            description="A guider for multi-view reference generation (requires conditionings from \"Trellis2 Multi-View Conditioning\" and \"Trellis2 Multiview Sampler Select\").",
            inputs=[
                IO.Model.Input("model"),
                IO.Conditioning.Input("positive", tooltip="positive from \"Trellis2 Multi-View Conditioning\" node"),
                IO.Conditioning.Input("negative", tooltip="negative from \"Trellis2 Multi-View Conditioning\" node"),
                IO.Float.Input("cfg", default=7.50, min=0.00, max=99.99, step=0.01),
                IO.Float.Input("blend_temperature", default=1.0, min=0.1, max=10.0, step=0.1),
                IO.Int.Input("coords_resolution", default=32, min=32, max=2048, step=32, tooltip="Default at structure generation resolution. Other stage resolutions can be acquired with \"Trellis2 Get Mesh and Coords Resolution\" node"),
            ],
            outputs=[
                io.Guider.Output("guider"),
            ]
        )
    
    @classmethod
    def execute(cls, model, positive, negative, cfg, blend_temperature, coords_resolution):
        pos_extras = positive[0][1]
        views_list = pos_extras.get("views_list")
        if views_list is None:
            raise ValueError("views_list not found in the positive conditioning!")
        
        _CUSTOM_KEYS = {"views_list", "true_positive", "true_positive_embeds",
                "true_negative", "true_negative_embeds", "embeds", "neg_embeds"}
        extras = {k: v for k, v in pos_extras.items() if k not in _CUSTOM_KEYS}

        guider = MultiViewCFGGuider(model)
        guider.set_cfg(cfg)

        guider.set_trellis2_opt(
            views=views_list, 
            blend_temperature=blend_temperature,
            coords_resolution=coords_resolution, 
            extras=extras)
        
        guider.set_conds(positive, negative)

        return io.NodeOutput(guider)


NODE_CLASSES = {
    "Trellis2MultiViewConditioning": Trellis2MultiViewConditioning,
    "Trellis2MultiViewSamplerSelect": Trellis2MultiViewSamplerSelect,
    "Trellis2MultiViewCFGGuider": Trellis2MultiViewCFGGuider,
}

NODE_CLASS_NAMES = {
    "Trellis2MultiViewConditioning": "Trellis2 Multi-View Conditioning",
    "Trellis2MultiViewSamplerSelect": "Trellis2 Multi-View Sampler Select",
    "Trellis2MultiViewCFGGuider": "Trellis2 Multi-View CFG Guider",
}
