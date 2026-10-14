# ComfyUI-Trellis2-Multiview (WIP)
Partial porting of [visualbruno's wrapper for Trellis2](https://github.com/visualbruno/ComfyUI-Trellis2) in native ComfyUI; specifically, this extensions adds visualbruno's [multiview samplers](https://github.com/visualbruno/ComfyUI-Trellis2/blob/main/trellis2/pipelines/samplers/flow_euler.py) for Trellis2.

<img width="1004" height="415" alt="NodesImg" src="https://github.com/user-attachments/assets/f34cff87-4db8-4e10-b097-944f7df60213" />

#### Release notes
- v1.0: Upload code with two working samplers, and workflow

### To do list:
- [x] Upload working code and workflow
- [ ] Make the custom conditionings work in the native ksampler and stage nodes
- [ ] Add the custom samplers in the native ksampler
- [ ] Patch rk4 and rk5 samplers

### Details (WIP)
 
[Workflow](https://github.com/GiusTex/ComfyUI-Trellis2-Multiview/blob/main/Trellis2-MultiView_workflow.json).

### Credits
- Claude for helping creating and debugging the code
- Microsoft for Trellis2
- Visualbruno for the wrapper and the custom samplers
