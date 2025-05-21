from bfcl.model_handler.local_inference.llama_3_1 import LlamaHandler_3_1

class LlamaHandler_3_1_TestModel(LlamaHandler_3_1):
    """
    Handler for Llama 3.1 series models in function calling mode.
    This class can be used to easily test local models without having to specify the model name as it will query the name of the model from the provided server and port assuming there is only one model available.
    """
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)
        self._model_name_huggingface_cache = None
    
    @property
    def model_name_huggingface(self):
        if self._model_name_huggingface_cache is None:
            self._model_name_huggingface_cache = self._query_model_name()
        return self._model_name_huggingface_cache
        
    def _query_model_name(self):
        import requests
        response = requests.get(f"http://{self.vllm_host}:{self.vllm_port}/v1/models")
        models = response.json()
        if len(models) == 0:
            raise ValueError("No models found")
        return models[0]
        