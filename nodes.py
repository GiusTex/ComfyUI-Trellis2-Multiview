import torch
from tqdm.auto import trange

import comfy.model_management
from comfy_api.latest import IO, io
from comfy.samplers import process_conds

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
        # Based on FlowEulerSampler
        def _v_to_xstart_eps(x, t, v, sigma_min):
            x_0 = (1 - sigma_min) * x - (sigma_min + (1 - sigma_min) * t) * v
            return x_0
        
        # SAMPLERS: each step receives (get_blended_v, x, t, dt) and returns (x_next, k1)
        # Based on FlowEulerMultiViewSampler.sample_once()
        def _step_euler(get_blended_v, x, t, dt, t_next):
            k1 = get_blended_v(x, t)

            pred_x_prev = x - (t - t_next) * k1
            return pred_x_prev, k1

        # Based on FlowHeunMultiViewSampler.sample_once()
        def _step_heun(get_blended_v, x, t, dt, t_next):
            # Heun's Method (RK2) Evaluations
            # Step 1: Predictor (Euler step)
            k1 = get_blended_v(x, t)
            x_temp = x + k1 * dt

            # Step 2: Corrector
            k2 = get_blended_v(x_temp, t + dt)

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
        if sampler_name == "rk4":
            sampler_function = _step_rk4
        if sampler_name == "rk5":
            sampler_function = _step_rk5
        else:
            sampler_function = _step_euler
    
        @torch.no_grad()
        def sampler(model, x, sigmas, extra_args=None, callback=None, disable=None, **kw):
            ts = sigmas.tolist()
            for i in trange(len(sigmas) - 1, disable=disable):
                t, t_next = ts[i], ts[i + 1]
                dt = t_next - t
                
                pred_x_prev, k1 = sampler_function(
                    extra_args["get_blended_v"], x, t, dt, t_next)
                
                if callback is not None:
                    # Estimate pred_x_0 based on k1 for tracking
                    pred_x_0 = _v_to_xstart_eps(x, t, k1, extra_args["sigma_min"])
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
        if front is not None:
            views['front'] = front
        if left is not None:
            views['left'] = left
        if back is not None:
            views['back'] = back
        if right is not None:
            views['right'] = right
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


class Trellis2CFGGuiderClass(comfy.samplers.CFGGuider):
    _dict = None

    def set_trellis2_opt(self, positive, negative, extras, views, cfg_interval, cfg_rescale, blend_temperature):

        self.positive = positive
        self.negative = negative
        self.extras = extras
        self.views = views
        self.cfg_interval = cfg_interval
        self.cfg_rescale = cfg_rescale
        self.blend_temperature = blend_temperature
        self._dict = None
        self.sigma_min = 1e-5

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

    # The wrapper uses _compute_view_weights_sparse() for the shape generation, 
    # but since we don't have access to the extra info in the latent (or in the 
    # pixal3d extras), we use a all-in-one function for the last three stages
    # Coded by Claude
    def _compute_view_weights_coords(self, coords, views, blend_temperature, width_sign=1.0, depth_sign=-1.0):
        def norm(c):
            c = c.float()
            lo, hi = torch.quantile(c, 0.02), torch.quantile(c, 0.98) # extra attention to small details like strands of hair and fingers
            return ((c - (lo + hi) / 2) / ((hi - lo) / 2).clamp(min=1.0)).clamp(-1, 1)

        w = norm(coords[:, 1]) * width_sign # width
        d = norm(coords[:, 2]) * depth_sign # depth
        table = {"front": d, "back": -d, "right": w, "left": -w}
        scores = [table[v] if v in table else torch.full_like(d, -10.0) for v in views]
        return torch.softmax(torch.stack(scores, dim=1) * blend_temperature, dim=1)

    def _build_dict(self, x): # Once for sampling
        def _model_dtype(self):
            m = self.inner_model
            # ComfyUI: with manual_cast inference dtype is different from weights one
            if hasattr(m, "get_dtype_inference"):
                return m.get_dtype_inference()
            return getattr(m, "manual_cast_dtype", None) or m.get_dtype()

        dtype = _model_dtype(self)
        device=x.device
        key = (tuple(x.shape), device, dtype)
        if self._dict is not None and self._dict["key"] == key:
            return self._dict

        def _to(v): # conds remain fp32, everything else in extras/kwargs (like "trellis2_shape_slat" passed in the texture stage) align with the model
            if not torch.is_tensor(v):
                return v
            if torch.is_floating_point(v):
                return v.to(device=device, dtype=dtype)
            return v.to(device=device)

        extras = {k: _to(v) for k, v in self.extras.items() if k not in {"is_padded", "embeds", "neg_embeds"}}
        positive = {v: _to(self.positive[v]) for v in self.views}
        negative = {v: _to(self.negative[v]) for v in self.views}
        embeds = {v: _to(self.extras["embeds"][v]) for v in self.views}
        neg_embeds = {v: _to(self.extras["neg_embeds"][v]) for v in self.views}

        if self.extras.get("is_padded", False):
            coords = self.extras["trellis2_coords"].to(x.device)
            counts = self.extras["trellis2_coord_counts"].to(x.device).long()            
            w = self._compute_view_weights_coords(coords, self.views, self.blend_temperature)
                    
            n_tokens = x.shape[2]
            valid = torch.arange(n_tokens, device=x.device)[None] < counts[:, None]
            weights = w.new_zeros(x.shape[0], n_tokens, w.shape[1])
            weights[valid] = w
        else:
            weights = self._compute_view_weights_dense(x.shape, x.device, self.views, self.blend_temperature)

        self._dict = dict(
            key=key, dtype=dtype, extras=extras,
            embeds=embeds, neg_embeds=neg_embeds,
            positive=positive, negative=negative,
            weights=weights,
        )
        return self._dict

    def _forward(self, xc, t_tensor, cond, kwargs):
        out = self.inner_model.diffusion_model(
            xc, t_tensor, cond,
            transformer_options=self.model_options.get("transformer_options", {}),
            **kwargs,
        )
        return out.float()

    def _pred_to_xstart(self, x_t, t, pred):
        return (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * pred

    def _xstart_to_pred(self, x_t, t, x_0):
        return ((1 - self.sigma_min) * x_t - x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def get_blended_v(self, x, t):
        d = self._build_dict(x) # Check if dict is already built
        padded = self.extras.get("is_padded", False)

        xc = x.to(d["dtype"])
        t_tensor = torch.tensor([1000 * t] * x.shape[0], device=x.device, dtype=torch.float32)

        cfg = self.cfg if self.cfg_interval[0] <= t <= self.cfg_interval[1] else 1.0
        
        v_accum = 0
        for i, view in enumerate(self.views):
            if cfg == 1:
                kwargs = {**d["extras"], "embeds": d["embeds"][view]}
                v = self._forward(xc, t_tensor, d["positive"][view], kwargs)
            elif cfg == 0:
                kwargs = {**d["extras"], "embeds": d["neg_embeds"][view]}
                v = self._forward(xc, t_tensor, d["negative"][view], kwargs)
            else:
                kwargs_pos = {**d["extras"], "embeds": d["embeds"][view]}
                v_pos = self._forward(xc, t_tensor, d["positive"][view], kwargs_pos)
                kwargs_neg = {**d["extras"], "embeds": d["neg_embeds"][view]}
                v_neg = self._forward(xc, t_tensor, d["negative"][view], kwargs_neg)
                v = v_neg + cfg * (v_pos - v_neg)

                if self.cfg_rescale > 0:
                    x_0_pos = self._pred_to_xstart(x, t, v_pos)
                    x_0_cfg = self._pred_to_xstart(x, t, v)
                    std_pos = x_0_pos.std(dim=list(range(1, x_0_pos.ndim)), keepdim=True)
                    std_cfg = x_0_cfg.std(dim=list(range(1, x_0_cfg.ndim)), keepdim=True)
                    x_0_rescaled = x_0_cfg * (std_pos / std_cfg)
                    x_0 = self.cfg_rescale * x_0_rescaled + (1 - self.cfg_rescale) * x_0_cfg
                    v = self._xstart_to_pred(x, t, x_0)

            if padded:
                w_i = d["weights"][:, :, i][:, None, :, None].to(v.dtype)
            else:
                w_i = d["weights"][i].unsqueeze(0).unsqueeze(0)

            v_accum += v * w_i

        return v_accum
    
    def inner_sample(self, noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=None):
        self.inner_model.latent_shapes = latent_shapes

        if latent_image is not None and torch.count_nonzero(latent_image) > 0: # Don't shift the empty latent image.
            latent_image = self.inner_model.process_latent_in(latent_image)

        self.conds = process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed, latent_shapes=latent_shapes)

        extra_model_options = comfy.model_patcher.create_model_options_clone(self.model_options)
        extra_model_options.setdefault("transformer_options", {})["sample_sigmas"] = sigmas
        extra_args = {"model_options": extra_model_options, "seed": seed,
                    "get_blended_v": self.get_blended_v, "sigma_min": self.sigma_min}

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

    def sample(self, *args, **kwargs):
        self._dict = None # reset cache
        try:
            return super().sample(*args, **kwargs)
        finally: # free vram
            self._dict = None


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
                IO.Float.Input("cfg_start", default=0.10, min=0.00, max=1.00, step=0.01, tooltip="Outside this interval cfg will be set to 1"),
                IO.Float.Input("cfg_end", default=1.00, min=0.00, max=1.00, step=0.01, tooltip="Outside this interval cfg will be set to 1"),
                IO.Float.Input("cfg_rescale", default=0.05, min=0.00, max=1.00, step=0.01, tooltip="Automatically skipped when cfg is 0 or 1. Suggested values to start from:\ncfg 7.5 & rescale 0.05;\ncfg 6.5 & rescale 0.20"),
                IO.Float.Input("blend_temperature", default=1.0, min=0.1, max=10.0, step=0.1, tooltip="Highers values create a mesh with a softer surface and edges less pronounced"),
            ],
            outputs=[
                io.Guider.Output("guider"),
            ]
        )
    
    @classmethod
    def execute(cls, model, positive, negative, cfg, blend_temperature, cfg_start, cfg_end, cfg_rescale):

        guider = Trellis2CFGGuiderClass(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)

        pos_extras = positive[0][1]
        views_list = pos_extras.get("views_list")
        if views_list is None:
            raise ValueError("views_list not found in the positive conditioning!")
        neg_extras = negative[0][1]
        
        # pos: views dict at 512; positive_embeds: views dict at 1024
        pos = {v: pos_extras["true_positive"][v] for v in views_list}
        positive_embeds = {v: pos_extras["true_positive_embeds"][v] for v in views_list}
        # negative: views dict at 512; negative_embeds: views dict at 1024
        negative = {v: neg_extras["true_negative"][v] for v in views_list}
        negative_embeds = {v: neg_extras["true_negative_embeds"][v] for v in views_list}

        _CUSTOM_KEYS = {"views_list", "true_positive", "true_positive_embeds",
                        "true_negative", "true_negative_embeds"}
        extras = {k: v for k, v in pos_extras.items() if k not in _CUSTOM_KEYS}
        extras["embeds"], extras["neg_embeds"] = positive_embeds, negative_embeds

        mode = pos_extras.get("trellis2_generation_mode", None)
        extras["is_padded"] = False
        if mode in ("shape_generation_512", "shape_generation", "texture_generation"):
            extras["is_padded"] = True

        guider.set_trellis2_opt(
            positive=pos, negative=negative,
            extras=extras, views=views_list,
            cfg_interval=[cfg_start, cfg_end],
            cfg_rescale=cfg_rescale,
            blend_temperature=blend_temperature,
        )  
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
