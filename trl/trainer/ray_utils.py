import os
import ray
from vllm import LLM, SamplingParams


@ray.remote
class vLLMActor:
    """
    A Ray remote actor class for managing and interacting with a vLLM instance.

    Attributes:
        llm (LLM): An instance of the LLM class initialized with the provided parameters.

    Methods:
        __init__(model: str, cuda_devices: str, tensor_parallel_size: int, gpu_memory_utilization: float):
            Initializes the vLLMActor with the specified model, CUDA devices, tensor parallel size, and GPU memory utilization.
        
        generate(prompts, sampling_params):
            Generates outputs from the LLM based on the provided prompts and sampling parameters.
        
        ping():
            Returns a readiness status message.
        
        load_weights(state_dict):
            Loads the model weights from the provided state dictionary.
    """
    def __init__(
        self,
        model: str,
        cuda_devices: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    def generate(self, prompts, sampling_params):
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        return outputs

    def ping(self):
        return "ready"

    def load_weights(self, state_dict):
        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        llm_model.load_weights(state_dict.items())
        print("Weights loaded successfully")
