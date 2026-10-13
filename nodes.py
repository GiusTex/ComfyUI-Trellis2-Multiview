import torch

import comfy.model_management
from comfy_api.latest import IO, io

from comfy_extras.nodes_trellis2 import (infer_batched_coord_layout, _dino_encode_batch,
                                        flatten_batched_sparse_latent, shape_norm)

from .trellis2.pipelines import samplers
from comfy.ldm.trellis2.model import compute_stage_proj_feats

from .utils import tensor2pil


class Trellis2SamplerSelect(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2SamplerSelect",
            display_name="Trellis2 Sampler Select",
            category="model/conditioning/trellis",
            description="Trellis2/Pixal3d samplers for single/multi view referenced generation",
            inputs=[
                IO.Combo.Input("sampler_name", ["euler", "heun", "rk4", "rk5"], default="euler"),
                IO.Combo.Input("stage", ["sparse", "shape (res 512)", "shape (res 1024)", "upsample/texture"], default="sparse"),
            ],
            outputs=[
                IO.Sampler.Output(display_name="sampler"),
            ]
        )

    @classmethod
    def getSamplerName(cls, sampler):
        if sampler == 'euler':
            return 'Euler'
        elif sampler == 'heun':
            return 'Heun'
        elif sampler == 'rk4':
            return 'RK4'
        elif sampler == 'rk5':
            return 'RK5'
        else:
            return 'Euler'  
    
    @classmethod
    def execute(cls, sampler_name, stage):
        sampler_prefix = cls.getSamplerName(sampler_name)        
        flow_model_resolution = 64
        
        if stage == "sparse":
            sampler_class = getattr(samplers, f"Flow{sampler_prefix}MultiViewGuidanceIntervalSampler", samplers.FlowEulerMultiViewGuidanceIntervalSampler)
            flow_model_resolution = 16
        elif stage == "shape (res 512)":
            sampler_class = getattr(samplers, f"Flow{sampler_name}MultiViewGuidanceIntervalSampler", samplers.FlowEulerMultiViewGuidanceIntervalSampler)
            flow_model_resolution = 32
        elif stage == "shape (res 1024)":
            sampler_class = getattr(samplers, f"Flow{sampler_name}MultiViewGuidanceIntervalSampler", samplers.FlowEulerMultiViewGuidanceIntervalSampler)
        elif stage == "upsample/texture": # (1024 texture model)
            sampler_class = getattr(samplers, f"Flow{sampler_prefix}MultiViewGuidanceIntervalSampler", samplers.FlowEulerMultiViewGuidanceIntervalSampler)

        sampler = sampler_class(
            sigma_min=1e-5,
            resolution=flow_model_resolution
        )
        return IO.NodeOutput(sampler)


class Trellis2MultiViewConditioning(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2MultiViewConditioning",
            display_name="Trellis2 Multi-View Conditioning",
            category="model/conditioning/trellis",
            inputs=[
                IO.ClipVision.Input("clip_vision_model"),
                IO.Image.Input("front", tooltip="Preprocessed image from ImageCropToMask (pad_factor=1.0 for TRELLIS.2)."),
                IO.Image.Input("left", optional=True, tooltip="Preprocessed image from ImageCropToMask (pad_factor=1.0 for TRELLIS.2)."),
                IO.Image.Input("back", optional=True, tooltip="Preprocessed image from ImageCropToMask (pad_factor=1.0 for TRELLIS.2)."),
                IO.Image.Input("right", optional=True, tooltip="Preprocessed image from ImageCropToMask (pad_factor=1.0 for TRELLIS.2)."),
            ],
            outputs=[
                IO.Conditioning.Output(display_name="positive"),
                IO.Conditioning.Output(display_name="negative"),
            ]
        )

    @classmethod
    def execute(cls, clip_vision_model, front, left=None, back=None, right=None) -> IO.NodeOutput:
        out_device = comfy.model_management.intermediate_device()
        
        # Collect views
        tensor_views = {'front': front}
        views = {'front': tensor2pil(front)}
        if left is not None:
            tensor_views['left'] = left
            views['left'] = tensor2pil(left)
        if back is not None:
            tensor_views['back'] = back
            views['back'] = tensor2pil(back)
        if right is not None:
            tensor_views['right'] = right
            views['right'] = tensor2pil(right)
        views_list = list(views.keys())

        positive_512 = {}
        positive_1024 = {}
        negative_512 = {}
        negative_1024 = {}
        # Calculate conditioning per view
        for v, img in tensor_views.items():
            cond = _dino_encode_batch(clip_vision_model, img, out_device)
            cond_512_batched = cond["global_512"]
            cond_1024_batched = cond["global_1024"]
            neg_cond_batched = torch.zeros_like(cond_512_batched)
            neg_embeds_batched = torch.zeros_like(cond_1024_batched)
            
            positive_512[v] = {"cond": cond_512_batched}
            positive_1024[v] = {"cond": cond_1024_batched}
            negative_512[v] = {"neg_cond": neg_cond_batched}
            negative_1024[v] = {"neg_cond": neg_embeds_batched}
        
        positive_512["extras"] = {"embeds": positive_1024}
        negative_512["extras"] = {"embeds": negative_1024}

        positive_512["extras"]["views_list"] = views_list

        del tensor_views
        return IO.NodeOutput(positive_512, negative_512)


class Trellis2CFGGuiderClass(comfy.samplers.CFGGuider):
    def set_trellis2_opt(self,
        cond, neg_cond, extras,
        views, front_axis, blend_temperature, verbose, 
        tqdm_desc, sampler_params,
        dino_lock, dino_substeps, dino_foundation_cap):

        self.cond = cond
        self.neg_cond = neg_cond
        self.extras = extras
        self.views = views
        self.front_axis = front_axis
        self.blend_temperature = blend_temperature
        self.verbose = verbose
        self.tqdm_desc = tqdm_desc
        self.sampler_params = sampler_params
        self.dino_lock = dino_lock
        self.dino_substeps = dino_substeps
        self.dino_foundation_cap = dino_foundation_cap
    
    def cast_dtype(self, x_t, t, cond, **kw):
        dtype = self.inner_model.get_dtype()
        dev = x_t.device
        xc = x_t.to(dtype)
        
        def _mv(v):
            if not torch.is_tensor(v):
                return v
            if torch.is_floating_point(v):
                return v.to(device=dev, dtype=dtype)
            return v.to(device=dev)

        ctx = _mv(cond)
        kw_c = {k: _mv(v) for k, v in kw.items()}
        t = t.to(device=dev) if torch.is_tensor(t) else t
        out = self.inner_model.diffusion_model(
            xc, t, ctx, 
            transformer_options=self.model_options.get("transformer_options", {}), 
            **kw_c
        )
        return out.float()

    def inner_sample(self, noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes=None):
        self.inner_model.latent_shapes = latent_shapes

        #if latent_image is not None and torch.count_nonzero(latent_image) > 0: # Don't shift the empty latent image.
        #    latent_image = self.inner_model.process_latent_in(latent_image)

        #device = device=self.model_patcher.offload_device
        #self.conds = process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed, latent_shapes=latent_shapes)

        extra_model_options = comfy.model_patcher.create_model_options_clone(self.model_options)
        extra_model_options.setdefault("transformer_options", {})["sample_sigmas"] = sigmas
        extra_args = {"model_options": extra_model_options, "seed": seed}

        executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(
            sampler.sample,
            sampler,
            comfy.patcher_extension.get_all_wrappers(
                comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE, 
                extra_args["model_options"], 
                is_model_options=True
            )
        )

        #samples = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
        samples = executor.execute(
            self.cast_dtype,
            noise,
            cond=self.cond, neg_cond=self.neg_cond,
            **self.sampler_params,
            views=self.views,
            front_axis=self.front_axis,
            steps=len(sigmas) - 1,
            blend_temperature=self.blend_temperature,
            concat_cond=self.extras.get("trellis2_shape_slat", None), # if self.extras is not None else None,
            verbose=self.verbose,
            dino_lock=self.dino_lock,
            dino_substeps=self.dino_substeps,
            dino_foundation_cap=self.dino_foundation_cap,
            tqdm_desc=self.tqdm_desc,
            **self.extras,
        ).samples
        
        return self.inner_model.process_latent_out(samples.to(torch.float32))


class Trellis2MultiViewCFGGuider(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Trellis2MultiViewCFGGuider",
            display_name="Trellis2 MultiView CFG Guider",
            category="model/sampling/guiders",
            description="A guider for multi-view reference generation (requires conditionings from \"Trellis2 Multi-View Conditioning\" and \"Trellis2 Multiview Sampler Select\").",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                IO.Float.Input("guidance_strength", default=6.50, min=0.00, max=99.99, step=0.01),
                IO.Float.Input("guidance_rescale", default=0.05,  min=0.00, max=1.00, step=0.01),
                IO.Float.Input("rescale_t", default=4.00, min=0.00, max=9.99, step=0.01),
                IO.Float.Input("guidance_interval_start", default=0.10, min=0.00, max=1.00, step=0.01),
                IO.Float.Input("guidance_interval_end", default=1.00, min=0.00, max=1.00, step=0.01),
                IO.Float.Input("blend_temperature", default=1.0, min=0.1, max=10.0, step=0.1),
                IO.Boolean.Input("verbose", default=False),
            ],
            outputs=[
                io.Guider.Output("guider"),
            ]
        )
    
    @classmethod
    def execute(cls, model, 
        positive, negative,
        guidance_strength, guidance_rescale, 
        rescale_t, guidance_interval_start, 
        guidance_interval_end, blend_temperature, 
        verbose):

        guider = Trellis2CFGGuiderClass(model)
        device = comfy.model_management.get_torch_device()

        guidance_interval=[guidance_interval_start,guidance_interval_end]        
        sampler_params = {
            "guidance_strength": guidance_strength,
            "guidance_rescale": guidance_rescale,
            "guidance_interval": guidance_interval,
            "rescale_t": rescale_t
        }

        extras = dict(positive.get("extras") or {})
        neg_extras = negative.get("extras") or {}
        stage = positive.get("stage") or "sparse_generation"

        proj_pack = positive["extras"].get("proj_feat_pack", None)
        views_list = positive["extras"].get("views_list", None)
        if proj_pack is not None:
            # We don't want the global cond etc
            raise ValueError("Pixal3d settings found in positive/negative. Encode them for Trellis2.\nPixal3d, also, requires \Trellis2 CFG Guider\" node.")
        if views_list is None:
            # We need views_list for the visualbruno multiview samplers
            raise ValueError("views_list not found in the positive conditioning!")
        
        cond = {v: positive[v]["cond"].to(device) for v in views_list}
        neg_cond = {v: negative[v]["neg_cond"].to(device) for v in views_list}
        # embeds 1024 per view: from nested dict to {view: tensor}
        extras["embeds"] = {v: extras["embeds"][v]["cond"].to(device) for v in views_list}
        extras["neg_embeds"] = {v: neg_extras["embeds"][v]["neg_cond"].to(device) for v in views_list}

        extras["is_padded_sparse"] = False
        extras["is_padded_dense"] = False
        if stage == "shape_generation":
            tqdm_desc = "Sampling shape SLat (MultiView)"
            extras["is_padded_sparse"] = True
        elif stage == "shape_generation_cascade":
            tqdm_desc = "Sampling shape SLat (MultiView HR)"
            extras["is_padded_dense"] = True
        elif stage == "texture_generation":
            tqdm_desc="Sampling texture SLat (MultiView)"
            extras["is_padded_dense"] = True
        else:
            tqdm_desc = "Sampling sparse structure (MultiView)"
            # is_dense
        
        guider.set_trellis2_opt(
            cond=cond,
            neg_cond=neg_cond,
            extras=extras,
            views=views_list,
            front_axis="x", # Trellis2
            blend_temperature=blend_temperature,            
            verbose=verbose,
            tqdm_desc=tqdm_desc,
            sampler_params = sampler_params,
            dino_lock = 0.0,
            dino_substeps = 4,
            dino_foundation_cap = 1.0,
        )
        
        return io.NodeOutput(guider)


class Trellis2ShapeStageAlt(IO.ComfyNode):
    """Sets up the first shape-stage sampling pass: extracts sparse coords from
    the dense structure voxel produced by VaeDecodeStructureTrellis2, builds an
    empty sparse latent, and attaches per-stage metadata to the conditioning so
    the model reads it via extra_conds at sample time. For the second shape pass
    (post-upsample), use Trellis2UpsampleStage instead — it combines the cascade
    and the second-pass stage setup."""
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2ShapeStageAlt",
            category="model/conditioning/trellis2",
            display_name="Trellis2 Shape Stage Alt",
            inputs=[
                IO.Conditioning.Input("positive"),
                IO.Voxel.Input("voxel", tooltip="Dense structure voxel from VaeDecodeStructureTrellis2."),
                IO.Combo.Input("model_res", ["512", "1024"], default="1024"),
            ],
            outputs=[
                IO.Conditioning.Output(display_name="shape positive"),
                IO.Latent.Output(display_name="shape latent"),
            ]
        )

    @classmethod
    def execute(cls, positive, voxel, model_res):
        decoded = voxel.data.unsqueeze(1)
        coords = torch.argwhere(decoded.bool())[:, [0, 2, 3, 4]].int()
        coord_resolution = int(decoded.shape[-1])

        # Dispatch based on the upstream voxel resolution, mirroring upstream's
        # pipeline_type → ss_res table:
        #   coord_res == 32 → first cascade shape pass OR pure-512 pipeline
        #                     (img2shape_512 + shape_512 proj stage, 512 DINO).
        #   coord_res >  32 → pure-1024 non-cascade pipeline
        #                     (img2shape + shape_1024 proj stage, 1024 DINO).
        #if coord_resolution <= 32:
        if model_res == "512":
            mode = "shape_generation_512"
            stage = "shape_512"
        else:
            mode = "shape_generation"
            stage = "shape_1024"

        batch_size, counts, max_tokens = infer_batched_coord_layout(coords)
        latent = torch.zeros(batch_size, 32, max_tokens, 1)

        prev_extras = positive["extras"]
        #proj_pack = _proj_pack_from_conditioning(positive)
        proj_pack = prev_extras.get("proj_feat_pack", None)
        extras = {
            #"trellis2_generation_mode": "shape_generation_512" if model_res == "512" else "shape_generation",
            **prev_extras,
            "trellis2_generation_mode": mode,
            "trellis2_coords": coords,
            "trellis2_coord_counts": counts,
            "coord_resolution": coord_resolution,
        }
        if proj_pack is not None:
            extras["trellis2_proj_feats"] = compute_stage_proj_feats(
                proj_pack, stage, coords=coords, coord_resolution=coord_resolution,
            )

        positive_out = dict(positive)
        positive_out["extras"] = extras
        positive_out["stage"] = "shape_generation"

        out_latent = {
            "samples": latent, 
            "coords": coords, "coord_counts": counts,"coord_resolution": coord_resolution, 
            "type": "trellis2",
            "model_frame": "y_up" if proj_pack is not None else "z_up"
        }
        return IO.NodeOutput(positive_out, out_latent)


class Trellis2UpsampleStageAlt(IO.ComfyNode):
    """Cascade-upsamples a 512-resolution shape latent into high-resolution
    sparse coords and sets up the second shape-stage sampling pass at the
    target resolution, attaching per-stage metadata to the conditioning for
    the model to consume via extra_conds."""
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2UpsampleStageAlt",
            category="model/conditioning/trellis",
            display_name="Trellis2 Upsample Stage Alt",
            inputs=[
                IO.Conditioning.Input("shape_positive"),
                IO.Latent.Input("shape_latent", tooltip="The 512-resolution shape latent output from the first shape-stage KSampler."),
                IO.Vae.Input("vae"),
                IO.Combo.Input("target_resolution", [1024,1536], default=1536),
                IO.Int.Input("max_num_tokens", default=12000, min=0, max=999999, step=1),
            ],
            outputs=[
                IO.Conditioning.Output(display_name="positive"),
                IO.Latent.Output(),
            ]
        )

    @classmethod
    def execute(cls, shape_positive, shape_latent, vae, 
                target_resolution, 
                max_num_tokens):
        device = comfy.model_management.get_torch_device()
        vae.prepare_decode(shape_latent["samples"].shape)

        prev_extras = shape_positive["extras"]
        sparse_structure_resolution = prev_extras["coord_resolution"]
        if sparse_structure_resolution == 32:
            from_resolution = 512
        else:
            from_resolution = 1024
        positive_only = {c: shape_positive[c] for c in shape_positive if c not in ["extras", "stage"]}
        
        coord_counts = shape_latent.get("coord_counts")
        shape_vae = vae.first_stage_model
        feats, coords_512 = flatten_batched_sparse_latent(
            shape_latent["samples"], shape_latent["coords"], coord_counts,
        )
        slat = shape_norm(feats.to(device), coords_512.to(device))
        hr_coords = shape_vae.upsample_shape(slat.to(vae.vae_dtype), upsample_times=4)
        
        lr_resolution = from_resolution
        hr_resolution = target_resolution
        ratio = (sparse_structure_resolution / 32)
        
        while True:
            quant_coords = torch.cat([
                hr_coords[:, :1],
                ((hr_coords[:, 1:] + 0.5) / (lr_resolution * ratio) * (hr_resolution // 16)).int(),
            ], dim=1)
            coords = quant_coords.unique(dim=0)
            num_tokens = coords.shape[0]
            if num_tokens < max_num_tokens:
                if hr_resolution != target_resolution:
                    print(f"Due to the limited number of tokens, the resolution is reduced to {hr_resolution}.")
                print(f"Num Tokens: {num_tokens}")
                break
            hr_resolution -= 128
            if hr_resolution < 1024 and target_resolution >= 1024:
                print(f"Num Tokens: {num_tokens}")
                hr_resolution = 1024
                break
            if hr_resolution < 512:
                print(f"Num Tokens: {num_tokens}")
                hr_resolution = 512
                break


        coords = coords.clone()
        coords[:, 0] = 0 # un solo campione
        counts = torch.tensor([coords.shape[0]], dtype=torch.int64)
        coord_resolution = hr_resolution // 16
        print("Coords resolution: ", coord_resolution)

        batch_size, _, max_tokens_out = infer_batched_coord_layout(coords)
        in_channels = 32
        latent = torch.zeros(batch_size, in_channels, max_tokens_out, 1)

        proj_pack = prev_extras.get("proj_feat_pack", None)
        extras = {
            **prev_extras,
            "trellis2_generation_mode": "shape_generation",
            "trellis2_coords": coords,
            "trellis2_coord_counts": counts,
            "coord_resolution": coord_resolution,
        }
        if proj_pack is not None:
            extras["trellis2_proj_feats"] = compute_stage_proj_feats(
                proj_pack, "shape_1024", coords=coords, coord_resolution=coord_resolution,
            )
        positive_out = dict(positive_only)
        positive_out["extras"] = extras
        positive_out["stage"] = "shape_generation_cascade"
        
        out_latent = {
            "samples": latent, 
            "coords": coords, "coord_counts": counts, "coord_resolution": coord_resolution, 
            "type": "trellis2",
            "model_frame": shape_latent.get("model_frame",
                                            "y_up" if proj_pack is not None else "z_up")
        }
        
        return IO.NodeOutput(positive_out, out_latent)


class Trellis2TextureStageAlt(IO.ComfyNode):
    """Sets up the texture-stage sampling pass. Reads coords / coord_counts /
    coord_resolution and the shape_slat (the per-voxel shape latent) from the
    incoming shape_latent dict — set there by Trellis2ShapeStage or
    Trellis2UpsampleStage. Builds an empty sparse latent at the same coord
    layout and attaches per-stage metadata to the conditioning."""
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2TextureStageAlt",
            display_name="Trellis2 Texture Stage Alt",
            category="model/conditioning/trellis",
            inputs=[
                IO.Conditioning.Input("positive"),
                IO.Latent.Input("shape_latent"),
            ],
            outputs=[
                IO.Conditioning.Output(display_name="positive"),
                IO.Latent.Output(),
            ]
        )
    
    @classmethod
    def execute(cls, positive, shape_latent):
        channels = 32
        coords = shape_latent["coords"]
        coord_resolution = shape_latent.get("coord_resolution")

        batch_size, counts, max_tokens = infer_batched_coord_layout(coords)

        shape_slat = shape_latent["samples"]
        if shape_slat.ndim == 4:
            shape_slat, _ = flatten_batched_sparse_latent(shape_slat, coords, counts)

        latent = torch.zeros(batch_size, channels, max_tokens, 1)
        proj_pack = positive.get("proj_feat_pack", None)
        model_frame = shape_latent.get("model_frame",
                                       "y_up" if proj_pack is not None else "z_up")

        extras = {
            **positive["extras"],
            "trellis2_generation_mode": "texture_generation",
            "trellis2_coords": coords,
            "trellis2_coord_counts": counts,
            "trellis2_shape_slat": shape_slat,
            "trellis2_model_frame": model_frame,
            "coord_resolution": coord_resolution,
        }
        if proj_pack is not None and coord_resolution is not None:
            extras["trellis2_proj_feats"] = compute_stage_proj_feats(
                proj_pack, "tex_1024", coords=coords, coord_resolution=coord_resolution,
            )
        positive_out = dict(positive)
        positive_out["extras"] = extras
        positive_out["stage"] = "texture_generation"

        out_latent = {
            "samples": latent, 
            "type": "trellis2", 
            "coords": coords, "coord_counts": counts,
            "model_frame": shape_latent.get("model_frame",
                                            "y_up" if proj_pack is not None else "z_up")
        }
        
        if coord_resolution is not None:
            out_latent["coord_resolution"] = coord_resolution
        return IO.NodeOutput(positive_out, out_latent)


NODE_CLASSES = {
    "Trellis2MultiViewConditioning": Trellis2MultiViewConditioning,
    "Trellis2SamplerSelect": Trellis2SamplerSelect,
    "Trellis2MultiViewCFGGuider": Trellis2MultiViewCFGGuider,
    "Trellis2ShapeStageAlt": Trellis2ShapeStageAlt,
    "Trellis2UpsampleStageAlt": Trellis2UpsampleStageAlt,
    "Trellis2TextureStageAlt": Trellis2TextureStageAlt,
}

NODE_CLASS_NAMES = {
    "Trellis2MultiViewConditioning": "Trellis2 Multi-View Conditioning",
    "Trellis2SamplerSelect": "Trellis2 Sampler Select",
    "Trellis2MultiViewCFGGuider": "Trellis2 MultiView CFG Guider",
    "Trellis2ShapeStageAlt": "Trellis2 Shape Stage Alt",
    "Trellis2UpsampleStageAlt": "Trellis2 Upsample Stage Alt",
    "Trellis2TextureStageAlt": "Trellis2 Texture Stage Alt",
}