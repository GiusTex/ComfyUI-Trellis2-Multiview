# ComfyUI-Trellis2-Multiview
Partial porting of [visualbruno's wrapper for Trellis2](https://github.com/visualbruno/ComfyUI-Trellis2) in native ComfyUI; specifically, this extensions adds visualbruno's [multiview samplers](https://github.com/visualbruno/ComfyUI-Trellis2/blob/main/trellis2/pipelines/samplers/flow_euler.py) for Trellis2.

<img width="296" height="387" alt="Trellis2MultiView_Nodes_V1" src="https://github.com/user-attachments/assets/d281c2bb-fab8-4f58-b4e4-98d978d3d498" />

#### Release notes
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

### Details (WIP)
 
[Workflow](https://github.com/GiusTex/ComfyUI-Trellis2-Multiview/blob/main/Trellis2-MultiView-MeshGeneration.json).


<img width="882" height="281" alt="Trellis2MultiView_NodesExtra_V1" src="https://github.com/user-attachments/assets/2c9a398d-dfb3-4fa6-b40b-c04e7ffeeb19" />

Encoder (the [model](https://huggingface.co/microsoft/TRELLIS.2-4B/blob/main/ckpts/shape_enc_next_dc_f16c32_fp16.safetensors) goes in models/vae)


### Credits
- Claude for helping creating and debugging the code
- Microsoft for Trellis2
- Visualbruno for the wrapper and the custom samplers
