# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import ray
import textwrap
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Union
from unittest.mock import patch

import time
import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather_object
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from ..data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from ..import_utils import is_vllm_available
from ..models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from .grpo_config import GRPOConfig
from .utils import compute_logps_with_prompt_cache, generate_model_card, get_comet_experiment_url, pad

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.utils import get_ip, get_open_port
    from vllm.worker.worker import Worker
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup
    from vllm.distributed.parallel_state import get_world_group

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


def stateless_init_process_group(master_address, master_port, rank, world_size, device):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes)
    and vLLM workers.
    """
    print("Creating process group")
    pg = StatelessProcessGroup.create(host=master_address, port=master_port, rank=rank, world_size=world_size)
    print("Process group created")

    print("Creating PyNcclCommunicator")
    pynccl = PyNcclCommunicator(pg, device=device)
    print("PyNcclCommunicator created")

    return pynccl


class MyWorker(Worker):
    """
    The `MyWorker` class inherits from `Worker` to provide custom functions.
    For simplicity, we define the `MyWorker` class in this self-contained
    script. Normally, we should define the `MyWorker` class in a separate
    file and pass the qualified name of the class to the `worker_cls`
    parameter.
    """

    def init_weight_update_group(self, master_address, master_port, rank_offset, world_size):
        rank = get_world_group().rank + rank_offset
        self.model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            self.device,
        )


    def update_weight(self, name, dtype, shape):
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        self.model_update_group.broadcast(weight, src=0, stream=torch.cuda.current_stream())
        self.model_runner.model.load_weights(weights=[(name, weight)])
        del weight


class GRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.use_vllm = args.use_vllm
        self.gradient_checkpointing = args.gradient_checkpointing

        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.accelerator.is_main_process:
                vllm_device = self.args.vllm_device
                use_single_vllm_device = self.args.vllm_tensor_parallel_size == 1

                # If the device is set to "auto", we automatically select the next available GPU
                if use_single_vllm_device:
                    if vllm_device == "auto":
                        vllm_device = f"cuda:{self.accelerator.num_processes}"

                    # Check that the requested device is available
                    if (
                        vllm_device.split(":")[0] == "cuda"
                        and int(vllm_device.split(":")[1]) >= torch.cuda.device_count()
                    ):
                        raise ValueError(
                            f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                            "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                            "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                            f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
                        )
                    # Check that the requested device is not also used for training
                    if vllm_device in {f"cuda:{idx}" for idx in range(self.accelerator.num_processes)}:
                        warnings.warn(
                            f"The requested device {vllm_device} is also used for training. This may lead to unexpected "
                            "behavior. It is recommended to use a dedicated device for vLLM."
                        )
                    # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
                    # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
                    # setting (profiling_patch).
                    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
                    profiling_patch = patch(
                        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
                        return_value=None,
                    )
                    with world_size_patch, profiling_patch:
                        self.llm = LLM(
                            model=model.name_or_path,
                            device=vllm_device,
                            gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                        )

                else:
                    # The device must be sth like "i,j,..,k" -> a list of cuda indices
                    # Make sure the vllm_tensor_parallel_size matches the number of GPUs
                    vllm_cuda_devices_len = len(vllm_device.split(","))
                    if vllm_cuda_devices_len != self.args.vllm_tensor_parallel_size:
                        raise ValueError(
                            f"The number of GPUs specified in `vllm_device` ({vllm_cuda_devices_len}) must "
                            f"match the `vllm_tensor_parallel_size` ({self.args.vllm_tensor_parallel_size})."
                        )

                    print(f"Using {vllm_cuda_devices_len} GPUs for vLLM on devices {vllm_device}")


                    os.environ["CUDA_VISIBLE_DEVICES"] = vllm_device
                    
                    ray.init()

                    @ray.remote(num_gpus=vllm_cuda_devices_len)
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
                            # os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

                            dist_keys = [
                                # "RANK",
                                # "LOCAL_RANK",
                                # "WORLD_SIZE",
                                # "LOCAL_WORLD_SIZE",
                                # "GROUP_RANK",
                                # "ROLE_RANK",
                                # "ROLE_NAME",
                                # "OMP_NUM_THREADS",
                                # "MASTER_ADDR",
                                # "MASTER_PORT",
                                # "TORCHELASTIC_USE_AGENT_STORE",
                                # "TORCHELASTIC_MAX_RESTARTS",
                                # "TORCHELASTIC_RUN_ID",
                                # "TORCH_NCCL_ASYNC_ERROR_HANDLING",
                                # "TORCHELASTIC_ERROR_FILE",
                            ]

                            for dist_key in dist_keys:
                                del os.environ[dist_key]

                            self.model_name = model
                            self.tensor_parallel_size = tensor_parallel_size
                            self.gpu_memory_utilization = gpu_memory_utilization
                            self.llm = None

                        def initialize(self):
                            self.llm = LLM(
                                model=self.model_name,
                                tensor_parallel_size=self.tensor_parallel_size,
                                gpu_memory_utilization=self.gpu_memory_utilization,
                                worker_cls=MyWorker,
                                distributed_executor_backend="mp",
                            )

                        def generate(self, prompts, sampling_params):
                            outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
                            return outputs
                        
                        def collective_rpc(self, func_name, args):
                            return getattr(self.llm, func_name)(*args)

                    self.vllm_actor = vLLMActor.remote(
                        model=model.name_or_path,
                        cuda_devices=vllm_device,
                        tensor_parallel_size=vllm_cuda_devices_len,
                        gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                    )
                    print("vLLM actor initializing")

                    ray.get(self.vllm_actor.initialize.remote())
                    print("vLLM actor initialized")

                self.sampling_params = SamplingParams(
                    n=self.num_generations,
                    temperature=args.temperature,
                    max_tokens=self.max_completion_length,
                )

            self._last_loaded_step = 0  # tag to avoid useless loading during grad checkpointing

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=args.temperature,
                num_return_sequences=self.num_generations,
                pad_token_id=processing_class.pad_token_id,
            )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        start_time = time.perf_counter()

        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                    state_dict = unwrapped_model.state_dict()

                if self.accelerator.is_main_process:
                    print("Updating vllm model weights...")
                    if self.args.vllm_tensor_parallel_size == 1:
                        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights(state_dict.items())
                    else:
                        # Only set up communication and broadcast from main process
                        master_address = get_ip()
                        master_port = get_open_port()
                        world_size = 1 + self.args.vllm_tensor_parallel_size

                        print(f"Master address: {master_address}")
                        print(f"Master port: {master_port}")
                        print(f"World size: {world_size}")
                        print(f"Device: {self.accelerator.device}")

                        # Set up vLLM side
                        handle = self.vllm_actor.collective_rpc.remote(
                            "init_weight_update_group",
                            args=(master_address, master_port, 1, 1 + self.args.vllm_tensor_parallel_size),
                        )
                        
                        print("vLLM weight update handle created")
                        
                        # Initialize process group only for rank 0 and vLLM processes
                        model_update_group = stateless_init_process_group(
                            master_address,
                            master_port,
                            rank=0,  # Just rank 0 for training
                            world_size=world_size,  # rank 0 + vLLM processes
                            device=self.accelerator.device,
                        )

                        print("Model update group initialized")

                        ray.get(handle)

                        # Broadcast weights from rank 0 to vLLM processes
                        for name, param in state_dict.items():
                            handle = self.vllm_actor.collective_rpc.remote(
                                "update_weight", args=(name, param.dtype, param.shape)
                            )
                            model_update_group.broadcast(param, src=0, stream=torch.cuda.current_stream())
                            ray.get(handle)

                    print("Weights updated successfully")

                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                print("Generating completions...")
                if self.args.vllm_tensor_parallel_size == 1:
                    outputs = self.llm.generate(all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False)
                else:
                    outputs = ray.get(
                        self.vllm_actor.generate.remote(all_prompts_text, sampling_params=self.sampling_params)
                    )
                completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
            else:
                completion_ids = [None] * len(all_prompts_text) * self.num_generations

            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts) * self.num_generations,
                (self.accelerator.process_index + 1) * len(prompts) * self.num_generations,
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_inputs_repeated = torch.repeat_interleave(prompt_inputs["input_ids"], self.num_generations, dim=0)
            prompt_completion_ids = torch.cat([prompt_inputs_repeated, completion_ids], dim=1)
        else:
            # Regular generation path
            with torch.no_grad(), unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    **prompt_inputs, generation_config=self.generation_config
                )

        end_time = time.perf_counter()
        if self.accelerator.is_main_process:
            print(f"Generation took {end_time - start_time:0.4f} seconds")

        # Compute prompt length and extract completion ids
        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Get the logit computation mini-batch size from the config
        mini_batch_size = self.args.logit_computation_mini_batch_size
        start_time = time.perf_counter()

        if not self.gradient_checkpointing:
            # Current policy logprobs (with grad)
            per_token_logps = compute_logps_with_prompt_cache(
                model=model,
                prompt_inputs=prompt_inputs,
                completion_ids=completion_ids,
                mini_batch_size=mini_batch_size,
                requires_grad_for_completion=True,
            )

            # Reference model logprobs (no grad)
            if self.ref_model is not None:
                ref_per_token_logps = compute_logps_with_prompt_cache(
                    model=self.ref_model,
                    prompt_inputs=prompt_inputs,
                    completion_ids=completion_ids,
                    mini_batch_size=mini_batch_size,
                    requires_grad_for_completion=False,
                )
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = compute_logps_with_prompt_cache(
                        model=model,
                        prompt_inputs=prompt_inputs,
                        completion_ids=completion_ids,
                        mini_batch_size=mini_batch_size,
                        requires_grad_for_completion=False,
                    )
        # If gradient checkpointing is used, fall back to original implementation
        else:
            # Concatenate prompt_mask with completion_mask for logit computation
            prompt_mask_repeated = prompt_inputs["attention_mask"].repeat_interleave(self.num_generations, dim=0)
            attention_mask = torch.cat([prompt_mask_repeated, completion_mask], dim=1)  # (B*G, P+C)

            # Get the per-token log probabilities for the completions for the model and the reference model
            def get_per_token_logps(model, input_ids, attention_mask, num_logits_to_keep, mini_batch_size):
                mini_batch_size = input_ids.size(0) if mini_batch_size == 0 else mini_batch_size
                per_token_logps = []
                for i in range(0, input_ids.size(0), mini_batch_size):
                    mini_batch_input_ids = input_ids[i : i + mini_batch_size, :]  # (B_mini, P+C)
                    mini_batch_attention_mask = attention_mask[i : i + mini_batch_size, :]  # (B_mini, P+C)
                    logits = model(
                        input_ids=mini_batch_input_ids,
                        attention_mask=mini_batch_attention_mask,
                        num_logits_to_keep=num_logits_to_keep + 1,
                    ).logits[:, -num_logits_to_keep - 1 : -1]  # (B_mini, P+C, Vocab_size)

                    token_index = mini_batch_input_ids[:, -num_logits_to_keep:].unsqueeze(-1)  # (B_mini, P+C, 1)
                    token_logits = torch.gather(logits, dim=-1, index=token_index).squeeze(-1)
                    logsumexp_values = torch.stack([torch.logsumexp(l, dim=-1) for l in logits])
                    del logits
                    token_log_prob = token_logits - logsumexp_values
                    per_token_logps.append(token_log_prob)
                return torch.cat(per_token_logps, dim=0)

            num_logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
            per_token_logps = get_per_token_logps(
                model=model,
                input_ids=prompt_completion_ids,
                attention_mask=attention_mask,
                num_logits_to_keep=num_logits_to_keep,
                mini_batch_size=mini_batch_size,
            )

            with torch.inference_mode():
                if self.ref_model is not None:
                    ref_per_token_logps = get_per_token_logps(
                        model=self.ref_model,
                        input_ids=prompt_completion_ids,
                        attention_mask=attention_mask,
                        num_logits_to_keep=num_logits_to_keep,
                        mini_batch_size=mini_batch_size,
                    )
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_per_token_logps = get_per_token_logps(
                            model=model,
                            input_ids=prompt_completion_ids,
                            attention_mask=attention_mask,
                            num_logits_to_keep=num_logits_to_keep,
                            mini_batch_size=mini_batch_size,
                        )
        end_time = time.perf_counter()
        if self.accelerator.is_main_process:
            print(f"Logits computation took {end_time - start_time:0.4f} seconds")

        # Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        torch.cuda.empty_cache()
        del mean_kl, prompt_inputs, per_token_loss, per_token_logps, ref_per_token_logps

        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
