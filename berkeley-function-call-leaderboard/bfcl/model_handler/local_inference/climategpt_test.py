from bfcl.model_handler.local_inference.llama_3_1 import LlamaHandler_3_1

class ClimateGPT_Test_Handler(LlamaHandler_3_1):
    """
    Handler for ClimateGPT-8B series models in function calling mode.
    """
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self.model_name_huggingface = model_name.replace("-FC", "")