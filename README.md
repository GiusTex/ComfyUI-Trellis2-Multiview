# ComfyUI-Trellis2-Multiview
Partial porting of [visualbruno's wrapper for Trellis2](https://github.com/visualbruno/ComfyUI-Trellis2) in native ComfyUI; specifically, this extensions adds visualbruno's [multiview samplers](https://github.com/visualbruno/ComfyUI-Trellis2/blob/main/trellis2/pipelines/samplers/flow_euler.py) for Trellis2.

<img width="506" height="328" alt="image" src="https://github.com/user-attachments/assets/c18a2008-a832-4842-b362-febfbf38499c" />

#### Release notes
- v1.1, 31/10/2026:
  - Made the guider work with native OverrideCFG and RescaleCFG nodes;
  - Simplified the code, and made it rely on native guider functions
- v1.0, 25/10/2026:
  - Patched rk4 and rk5 samplers;
  - Made multi-view conditionings work with native stage nodes;
  - Added native encode node;
  - Updated workflow.
- v0.1, 11/10/2026: Upload code with two working samplers, and workflow

### To do list:
- [x] Upload working code and workflow
- ~~[ ] Make the custom conditionings work in the native ksampler~~
- [x] Make the custom conditionings work in the native stage nodes
- ~~[ ] Add the custom samplers in the native ksampler~~
- [x] Patch rk4 and rk5 samplers
- [x] Make the guider work with native OverrideCFG and RescaleCFG nodes

### Installation
Go to `comfyui/custom_nodes` folder, then `git clone https://github.com/GiusTex/ComfyUI-Trellis2-Multiview` inside it.

### Nodes and workflow
[Workflow](https://github.com/GiusTex/ComfyUI-Trellis2-Multiview/blob/main/Trellis2-MultiView-Mesh-Generation.json) to generate the upsampled mesh. You can also turn on the texture generation groups, but I got some small grey artifacts, so I suggest using the workflow only for the mesh generation. You can still turn on the last groups and check if you like the results.

- **Trellis2 Multi-View Conditionining**

  Accepts from 1 to 4 reference images.

- **Trellis2 Multi-View CFG Guider**

  Runs a prediction for each view and blends them based on the blend temperature. Uses visualbruno's `_compute_view_weights_sparse` and `_compute_view_weights_dense` from his wrapper.

  Blend temperature: smooth factor and side views importance. Helps decreasing mesh holes in the structure generation stage too.

  Coords resolution: required for the views weights normalization. You can get it with `Trellis2 Get Mesh and Coords Resolution`.

  Requires `Trellis2 Multi-View Conditionining` and `Trellis2 Multi-View Sampler Select`.

- **Trellis2 Multi-View Sampler Select**

  Uses visualbruno's samplers in his wrapper of Trellis2. Each sampler runs n. times the prediction of the views:
  - Euler: samples 1 time the views;
  - Heun: samples 2 times the views;
  - Rk4: samples 4 times the views;
  - Rk5: samples 6 times the views.

- **Trellis2 Load Vae Encoder**

  Experimental node. I copied the class to load the vae encoder from visualbruno's wrapper.

- **Trellis2 Encode Shape Latent**

  Experimental node. Uses a vibe coded python implementation made by Claude based on visualbruno's `flexible_dual_grid` made in C. Encodes a mesh into a latent, like the one you get from a ksampler.

  It doesn't require extra dependencies and it should work, but returns strange colors when used for texture generation. I suggest using it when still generating the mesh, if you really need it.

  The mesh resolution can be obtained with `Trellis2 Get Mesh and Coords Resolution` if you still have the latent, or by remembering/checking the resolution the mesh had when you saved it.

  You can find the mesh encoder model [here](https://huggingface.co/microsoft/TRELLIS.2-4B/blob/main/ckpts/shape_enc_next_dc_f16c32_fp16.safetensors). Put it in `models/vae`. In the workflow it has a different name because I renamed it for simplicity, but it's the linked file.

<img width="882" height="281" alt="Trellis2MultiView_NodesExtra_V1" src="https://github.com/user-attachments/assets/2c9a398d-dfb3-4fa6-b40b-c04e7ffeeb19" />

### Credits
- Claude for helping creating and debugging the code
- Microsoft for Trellis2
- Visualbruno for the wrapper and the custom samplers
