import gc
import copy
import torch
from typing import *

import folder_paths
import comfy.utils
import comfy.model_management as mm
import comfy.model_patcher

from comfy_api.latest import io
from comfy.ldm.trellis2.vae import SparseTensor
from comfy_extras.nodes_trellis2 import shape_slat_format

from .encoder_class import FlexiDualGridVaeEncoder

from .encoder_utils import set_conv_backend
from .flexible_dual_grid import mesh_to_flexible_dual_grid

_BYTES_PER_VOXEL = 1700 # last check: voxel=4159094 peak=5868 MiB -> 1479 B/voxel


class Trellis2LoadVaeEncoder(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Trellis2LoadVaeEncoder",
            display_name="Trellis2 Load Vae Encoder",
            category="Trellis2",
            inputs=[
                io.Combo.Input("vae_name", options=folder_paths.get_filename_list("vae")),
                io.Combo.Input("backend", ['none', 'spconv', 'torchsparse', 'flex_gemm'], default="none", tooltip="flex_gemm requires flex_gemm library installed")
            ],
            outputs=[
                io.Vae.Output(display_name="trellis2 encoder"),
            ],
        )

    @classmethod
    def execute(cls, vae_name, backend):
        set_conv_backend(backend)

        # Config of shape_enc_next_dc_f16c32_fp16.json
        model = FlexiDualGridVaeEncoder(
            model_channels=[64, 128, 256, 512, 1024],
            latent_channels=32,
            num_blocks=[0, 4, 8, 16, 4],
            block_type=["SparseConvNeXtBlock3d"] * 5,
            down_block_type=["SparseResBlockS2C3d"] * 4,
            block_args=[{}, {}, {}, {}, {}],
        )
        path = folder_paths.get_full_path_or_raise("vae", vae_name)
        sd = comfy.utils.load_torch_file(path)
        # from_pretrained in wrapper uses strict=False
        # "missing" keys = parameters remained with casual init (xavier) -> silent wrong results
        result = model.load_state_dict(sd, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"[Trellis2 Load Vae Encoder]: missing={result.missing_keys[:5]} unexpected={result.unexpected_keys[:5]}")
        del sd
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        model.dtype=torch.float32 # forward uses self.dtype for blocks activations
        model.float()
        
        patcher = comfy.model_patcher.ModelPatcher(
            model,
            load_device=mm.get_torch_device(),
            offload_device=mm.vae_offload_device(),
        )
        return io.NodeOutput(patcher)


class Trellis2EncodeShapeLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Trellis2EncodeShapeLatent",
            display_name="Trellis2 Encode Shape Latent",
            category="Trellis2",
            description="Encode a mesh into its latent shape using torch and QEF; runs on cpu. No extra dependencies needed.",
            is_experimental=True,
            inputs=[
                io.Vae.Input("trellis2_encoder", display_name="trellis2 encoder"),
                io.Mesh.Input("mesh"),
                io.Int.Input("mesh_resolution", default=1536, min=512, max=2048, step=128),
                io.Boolean.Input("is_pixal3d", default=False),
            ],
            outputs=[
                io.Latent.Output("latent"),
            ],
        )

    @classmethod
    def preprocess(cls, mesh):
        mesh = copy.copy(mesh)
        vertices = mesh.vertices[0].clone()

        vertices_min = vertices.min(dim=0).values
        vertices_max = vertices.max(dim=0).values
        center = (vertices_min + vertices_max) / 2
        scale = 0.99999 / (vertices_max - vertices_min).max()
        vertices = (vertices - center) * scale

        tmp = vertices[:, 1].clone()
        vertices[:, 1] = -vertices[:, 2]
        vertices[:, 2] = tmp
        assert torch.all(vertices >= -0.5) and torch.all(vertices <= 0.5), 'vertices out of range'
        
        mesh.vertices = vertices        
        return mesh
    
    @classmethod
    def execute(cls, trellis2_encoder, mesh, mesh_resolution, is_pixal3d, unload=True):
        # Preprocess input mesh
        m = cls.preprocess(mesh)
        vertices = m.vertices.float()
        faces = m.faces[0].long()

        #--------------------------------------------------------------
        grid_size = torch.as_tensor(mesh_resolution, dtype=torch.int32, device="cpu").expand(3).clone()
        aabb = torch.as_tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype=torch.float32, device="cpu")
        voxel_size = (aabb[1] - aabb[0]) / grid_size

        vertices = vertices.cpu() - aabb[0].reshape(1, 3)
        grid_range = torch.stack([torch.zeros_like(grid_size), grid_size]).int()
        
        voxel_indices, dual_vertices, intersected = mesh_to_flexible_dual_grid(
            vertices, faces.cpu(), 
            voxel_size, 
            grid_range,
            face_weight=1.0, 
            boundary_weight=0.2, 
            regularization_weight=1e-2,
        )
        voxel_indices = voxel_indices.cpu()
        dual_vertices = dual_vertices.cpu()
        intersected = intersected.cpu()
        #--------------------------------------------------------------

        feats = dual_vertices * mesh_resolution - voxel_indices
        n_voxel = int(voxel_indices.shape[0])

        device = mm.get_torch_device()
        vertices_st = SparseTensor(
            feats=feats,
            coords=torch.cat([torch.zeros_like(voxel_indices[:, 0:1]), voxel_indices], dim=-1)
        ).to(device)
        intersected_st = vertices_st.replace(intersected).to(device)

        #--------------------------------------------------------------
        # Native memory management
        est_bytes = int(n_voxel * _BYTES_PER_VOXEL)
        try:
            mm.load_models_gpu([trellis2_encoder], memory_required=est_bytes, force_full_load=True)
        except TypeError: # ComfyUI versions without force_full_load
            mm.free_memory(trellis2_encoder.model_size() + est_bytes, device)
            mm.load_models_gpu([trellis2_encoder])

        bad = [n for n, p in trellis2_encoder.model.named_parameters() if p.device != device]
        if bad:
            print(f"[Trellis2 Encode Shape Latent] {len(bad)} parameters not on GPU (es. {bad[:3]}): model manually moved")
            trellis2_encoder.model.to(device)
        
        # Encode
        with torch.no_grad():
            slat = trellis2_encoder.model(vertices_st, intersected_st)
        #-----------------------------------------------------------------------
        
        z = slat.feats.float()
        zero = torch.zeros(1, z.shape[1], device=z.device)
        mean = shape_slat_format.process_out(zero)
        std = shape_slat_format.process_out(torch.ones_like(zero)) - mean
        z_norm = (z - mean) / std

        # Unload model
        if unload:
            trellis2_encoder.detach() # unload using patcher
            del vertices_st, intersected_st # input sparse tensors (~100 MB)
            gc.collect()
            mm.soft_empty_cache()

        coords = slat.coords # res/16 grid, one line for token
        counts = torch.tensor([coords.shape[0]], device=z.device)
        latent = z_norm.movedim(-1, 0)[None, :, :, None].to(mm.intermediate_device())

        latent_res = mesh_resolution // 16
        print(f"Setting {latent_res} as coord_resolution")
        out_latent = {
            "samples": latent, 
            "coords":coords, "coord_counts":counts, "coord_resolution": latent_res,
            "type": "trellis2",
            "model_frame": "y_up" if is_pixal3d else "z_up"
        }
        return io.NodeOutput(out_latent)


ENCODER_CLASSES = {
    "Trellis2LoadVaeEncoder": Trellis2LoadVaeEncoder,
    "Trellis2EncodeShapeLatent": Trellis2EncodeShapeLatent,
}

ENCODER_NAMES = {
    "Trellis2LoadVaeEncoder": "Trellis2 Load Vae Encoder",
    "Trellis2EncodeShapeLatent": "Trellis2 Encode Shape Latent",
}
