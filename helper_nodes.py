from comfy_api.latest import IO, io


class Trellis2GetMeshAndCoordsResolution(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="Trellis2GetMeshAndCoordsResolution", 
            display_name="Trellis2 Get Mesh And Coords Resolution",
            category="model/latent/trellis", 
            description="Retrieves \"coord_resolution\" from a trellis2 type latent and outputs the int value",
            inputs=[
                io.Latent.Input("latent"),
            ],
            outputs=[
                io.Int.Output(display_name="mesh res", tooltip="Mesh resolution"),
                io.Int.Output(display_name="coords res", tooltip="Coords resolution"),
            ],
        )

    @classmethod
    def execute(cls, latent):
        if latent["type"] != "trellis2":
            raise ValueError("Latent type is not trellis2!")
        return IO.NodeOutput(latent["coord_resolution"]*16,latent["coord_resolution"])


HELPER_NODE_CLASSES = {
    "Trellis2GetMeshAndCoordsResolution": Trellis2GetMeshAndCoordsResolution,
}

HELPER_NODE_CLASS_NAMES = {
    "Trellis2GetMeshAndCoordsResolution": "Trellis2 Get Mesh And Coords Resolution",
}
