from bfcl.model_handler.local_inference.llama_3_1 import LlamaHandler_3_1
from overrides import override

class LlamaHandler_3_1_TestModel(LlamaHandler_3_1):
    """
    Handler for Llama 3.1 series models in function calling mode.
    This class can be used to easily test local models without having to specify the model name as it will query the name of the model from the provided server and port assuming there is only one model available.
    """
    def __init__(self, model_name, temperature) -> None:
        # First, initialize parent normally
        super().__init__(model_name, temperature)
        # Flag for lazy loading
        self._model_queried = False
        
    # Override the batch_inference method to ensure the model name is loaded
    @override
    def batch_inference(self, *args, **kwargs):
        # Lazy-load the model name before the first inference
        if not self._model_queried:
            try:
                model_name = self._query_model_name()
                # Only update if we got a valid result
                if model_name:
                    # Direct attribute access to avoid property issues
                    self.__dict__['model_name_huggingface'] = model_name
                self._model_queried = True
            except Exception as e:
                print(f"Error querying model name: {e}, using default value")
                self._model_queried = True
        
        # Continue with the original method
        return super().batch_inference(*args, **kwargs)
    
    def _query_model_name(self):
        import requests
        try:
            response = requests.get(f"http://{self.vllm_host}:{self.vllm_port}/v1/models")
            models = response.json()
            if len(models) == 0:
                print("No models found, using default model name")
                return None
            return models[0]
        except Exception as e:
            print(f"Error querying model name: {e}")
            return None
        