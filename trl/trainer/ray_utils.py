import os
import ray
from vllm import LLM, SamplingParams


@ray.remote
class vLLMActor:
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

    def load_weights(self, state_dict):
        # Call load_weights on the model inside the actor.
        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        llm_model.load_weights(state_dict.items())
        print("Weights loaded successfully")
