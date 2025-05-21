from bfcl.model_handler.local_inference.llama_3_1 import LlamaHandler_3_1

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
        
    # Use descriptor protocol to intercept model_name_huggingface access
    # This will be triggered whenever the batch_inference method or any other method
    # accesses the model_name_huggingface attribute
    
    # Override __getattribute__ to intercept attribute access
    def __getattribute__(self, name):
        # First get the attribute using the standard method
        attr = super().__getattribute__(name)
        
        # If we're accessing model_name_huggingface and haven't queried yet, do the lazy loading
        if name == 'model_name_huggingface' and not super().__getattribute__('_model_queried'):
            try:
                # Set the flag first to prevent infinite recursion
                object.__setattr__(self, '_model_queried', True)
                # Query the model name
                model_name = self._query_model_name()
                # Only update if we got a valid result
                if model_name:
                    # Update the attribute directly in __dict__ to bypass any property issues
                    object.__setattr__(self, 'model_name_huggingface', model_name)
                    # Get the updated value
                    return super().__getattribute__(name)
            except Exception as e:
                print(f"Error querying model name: {e}, using default value")
        
        # Return the attribute
        return attr
    
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
        