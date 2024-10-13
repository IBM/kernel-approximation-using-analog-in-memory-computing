import argparse

from imc_performer.src.utils import LRA_TASKS, SUPPORTED_CMD_MODES


def get_fp_train_args():
    parser = argparse.ArgumentParser(description="Run network training")
    parser.add_argument("--task", help="LRA task name", default="all", choices=["all"] + LRA_TASKS,)
    parser.add_argument("--mode", default="jbsub", help="Execution mode", choices=SUPPORTED_CMD_MODES,)
    parser.add_argument("--data_dir", help="Base data directory", default="/dccstor/broccoli/LRA",)
    parser.add_argument("--base_dir", help="Model store directory", default="/dccstor/broccoli/performer",)
    parser.add_argument("--model_name", help="Base model name", default="transformer_lra_",)
    parser.add_argument("--config_name", help="Path to the config file", default=None,)
    args = parser.parse_args()
    return args


def get_evaluation_args():
    parser = argparse.ArgumentParser(description="Run network evaluation")
    parser.add_argument("--task", default="listops", help="LRA task name", choices=LRA_TASKS + ["all"])
    parser.add_argument("--mode", choices=["fp", "hw_aware", "fp_attn_only"], help="What to evaluate.")
    parser.add_argument("--data-dir", help="Base data directory", default="~/LRA/LRA",)
    parser.add_argument("--seed", default="all", help="Specific seed.")
    parser.add_argument("--samples", default=2000, type=int, help="Number of samples in the test set. 2000 or -1 for all.")
    args = parser.parse_args()
    return args
