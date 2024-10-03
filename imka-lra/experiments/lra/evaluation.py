import logging
import os
from tqdm import tqdm

import torch
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq.data import SubsampleDataset
from torch.utils.data import DataLoader

from imc_performer.src.evaluation import accuracy
from imc_performer.src.parsing import get_evaluation_args
from imc_performer.src.utils import (
    BATCH_SZ,
    LRA_TASKS,
    TASK_TO_DATA_MAP,
    fix_random,
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def main():
    logging.getLogger("fairseq").setLevel(logging.WARNING)
    args = get_evaluation_args()

    if args.seed == "all":
        seeds = list(range(10))
    else:
        seeds = [int(args.seed)]
    logging.info(f"Using seeds {seeds}")

    #NOTE remove listops from exclude_tasks
    exclude_tasks = ["listops", "cifar10_relu"]
    tasks = [t for t in LRA_TASKS if not t in exclude_tasks] if args.task == "all" else [args.task]
    res = {}

    for task in tasks:
        res[task] = []
        for seed in tqdm(seeds):
            fix_random(seed)
            if args.mode == "hw_aware":
                model_dir = os.path.join(f"resources/models/hw_aware/{task}")
            elif args.mode == "fp":
                model_dir = os.path.join(f"resources/models/fp/{task}")
            elif args.mode == "fp_attn_only":
                model_dir = os.path.join(f"resources/models/fp_attn_only/{task}")
            else:
                raise ValueError(f"Unknown mode {args.mode}")

            print(f"Using model {model_dir} for task {task}")
            
            load_folder = args.mode if args.mode != "hw_aware" else "fp"
            if args.mode == "hw_aware" and task == "listops":
                load_folder = "hw_aware"

            model, _, task_obj = load_model_ensemble_and_task(
                filenames=[os.path.join(
                        os.path.join(f"resources/models/{load_folder}/{task}"),
                        "checkpoint.th"
                    )
                ],
                new_data_dir=os.path.join(os.path.expanduser(args.data_dir), f"{TASK_TO_DATA_MAP[task]}"),
            )
            model = model[0]


            if args.mode == "hw_aware" and task != "listops":
                # load the digital model
                hw_aware_fp_sd = torch.load(os.path.join(model_dir, "inference_model.th"),map_location="cpu")
                model.load_state_dict(hw_aware_fp_sd)

                # load the input ranges
                input_range_dict = torch.load(os.path.join(model_dir, "input_range_dict.th"))

                # Conversion to analog model
                # create rpu_config
                from aihwkit.simulator.configs import TorchInferenceRPUConfig,NoiseManagementType,BoundManagementType
                from aihwkit.nn.conversion import convert_to_analog
                from aihwkit.nn import AnalogLinear
                from fairseq.utils import NoisyLinear

                rpu_config = TorchInferenceRPUConfig()
                rpu_config.forward.out_noise = 0.0
                rpu_config.forward.is_perfect = True
                rpu_config.pre_post.input_range.enable = True
                rpu_config.forward.noise_management = NoiseManagementType.NONE
                rpu_config.forward.bound_management = BoundManagementType.NONE
                rpu_config.forward.out_bound = -1
                rpu_config.forward.inp_bound = -1
                rpu_config.mapping.max_input_size = 0
                rpu_config.mapping.max_output_size = 0

                model = convert_to_analog(model,rpu_config=rpu_config,conversion_map={NoisyLinear: AnalogLinear,})

                # assign the input ranges
                num_matched = 0
                for module_key, module in model.named_analog_modules():
                    # ir_key = find_matching_key(module_key, input_range_dict)
                    ir_key = None
                    for k in input_range_dict:
                        if module_key != "" and module_key in k:
                            ir_key = k
                            break
                    if ir_key is None: continue
                    module.analog_module.input_range.data = torch.tensor([input_range_dict[ir_key]])
                    num_matched += 1

            model = model.eval()
            model = model.to(DEVICE)

            if DEVICE == "cuda":
                for i in range(len(model.encoder.encoder.layers)):
                    encoder_layer = model.encoder.encoder.layers[i]
                    if encoder_layer.self_attn.attention.sep_proj:
                        encoder_layer.self_attn.attention.feature_map_q.device = "cuda"
                        encoder_layer.self_attn.attention.feature_map_q.projection_layer.cuda()
                        encoder_layer.self_attn.attention.feature_map_k.device = "cuda"
                        encoder_layer.self_attn.attention.feature_map_k.projection_layer.cuda()
                    else:
                        encoder_layer.self_attn.attention.feature_map.device = "cuda"
                        encoder_layer.self_attn.attention.feature_map.projection_layer.cuda()
            
            
            task_obj.load_dataset(split="test",new_data_dir=os.path.join(os.path.expanduser(args.data_dir), f"{TASK_TO_DATA_MAP[task]}"),)
            dataset = task_obj.datasets["test"]
            if args.samples > 0 and len(dataset) > args.samples: dataset = SubsampleDataset(dataset=dataset, size_ratio=args.samples / len(dataset))
            loader = DataLoader(dataset=dataset,batch_size=BATCH_SZ[task],collate_fn=dataset.collater, drop_last=False, )

            f_get_normalized_probs = model.get_normalized_probs

            acc_before_conversion = accuracy(model=model,loader=loader,get_norm_p=f_get_normalized_probs,device=DEVICE,)
            print(f"Seed {seed} Raw (fairseq) model performance {acc_before_conversion:.4f}%")
            res[task].append(acc_before_conversion)

    # print the results
    import numpy as np
    for task in res:
        print(f"Task {task}: mean {np.mean(res[task]):.4f}%, std {np.std(res[task]):.4f}%")

if __name__ == "__main__":
    main()