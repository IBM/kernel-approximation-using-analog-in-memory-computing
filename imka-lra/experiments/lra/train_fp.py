import os

from imc_performer.src.exec import Run
from imc_performer.src.parsing import get_fp_train_args
from imc_performer.src.utils import LRA_TASKS, TASK_TO_DATA_MAP

if __name__ == "__main__":
    args = get_fp_train_args()
    tasks = [args.task] if args.task != "all" else LRA_TASKS
    if not os.path.isdir(args.base_dir): os.makedirs(args.base_dir)

    for task in tasks:
        if not os.path.isdir(os.path.join(args.base_dir, task)): os.makedirs(os.path.join(args.base_dir, task))
        baseline = Run(model_name=args.model_name + task, task_data=os.path.join(args.data_dir, TASK_TO_DATA_MAP[task]), task_name=task, seed=0, hour_q=6, bias=None if "attn_only" in args.config_name else True, no_save=False, config_path=args.config_name, base_store_dir=args.base_dir, finetune_from_model=False,)
        baseline.execute(force=False, mode=args.mode)
