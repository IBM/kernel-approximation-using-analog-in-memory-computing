import os
import sys
from imc_performer.src.parsing import get_fp_train_args
from imc_performer.src.utils import TASK_2_Q, TASK_TO_DATA_MAP
from imc_performer.src.exec import Run

TASK2LR = {"aan": 0.001,"listops": 0.001,"imdb": 0.0006,"pf32": 0.0001,"cifar10": 0.005,}

#NOTE: uncomment for attention-only
TASK_TO_MODEL = {"imdb": os.path.expanduser("~/fp-imc-performer/resources/models/fp_attn_only/imdb"),"cifar10": os.path.expanduser("~/fp-imc-performer/resources/models/fp_attn_only/cifar10"),"pf32": os.path.expanduser("~/fp-imc-performer/resources/models/fp_attn_only/pf32"),"aan": os.path.expanduser("~/fp-imc-performer/resources/models/fp_attn_only/aan"),"listops": os.path.expanduser("~/fp-imc-performer/resources/models/fp_attn_only/listops"),}
# TASK_TO_MODEL = {"imdb": os.path.expanduser("~/fp-imc-performer/resources/models/fp/imdb"),"cifar10": os.path.expanduser("~/fp-imc-performer/resources/models/fp/cifar10"),"pf32": os.path.expanduser("~/fp-imc-performer/resources/models/fp/pf32"),"aan": os.path.expanduser("~/fp-imc-performer/resources/models/fp/aan"),"listops": os.path.expanduser("~/fp-imc-performer/resources/models/fp/listops"),}

# where to store the models
BASE_STORE_DIR = f"/dccstor/broccoli/tmp-imc-performer/aihwkit"

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
base_path = os.getcwd()
sys.path.append(base_path)


def aihwkit_train(task: str,no_save: bool,mode: str,decay: float,l_output_noise: list,eta: float,forward_is_perf: bool,ir_enable: bool,):
    if not os.path.isdir(BASE_STORE_DIR): os.mkdir(BASE_STORE_DIR)
    if not os.path.isdir(os.path.join(BASE_STORE_DIR, task)): os.mkdir(os.path.join(BASE_STORE_DIR, task))
    args = get_fp_train_args()
    runs = [Run(model_name="analog_" + args.model_name + task,task_data=os.path.join(args.data_dir, TASK_TO_DATA_MAP[task]),base_store_dir=BASE_STORE_DIR,task_name=task,config_path=args.config_name,no_save=no_save,hour_q=TASK_2_Q[task],bias=None if "attn_only" in TASK_TO_MODEL[task] else True,checkpoint_restore_file=os.path.join(TASK_TO_MODEL[task], "checkpoint.th"),finetune_from_model=True,forward_bound_management="none",forward_inp_bound=1.0,forward_inp_res=2**8 - 2,forward_is_perfect=forward_is_perf, forward_noise_management="none",forward_out_bound=1e6,forward_out_noise=out_noise,forward_out_res=-1,input_range_decay=decay,input_range_enable=ir_enable,input_range_init_from_data=100,input_range_input_min_percentage=0.95,max_input_size=0,max_output_size=0,modifier_std_dev=eta,modifier_type="add_gauss",noise_per_sample=True,clip_sigma=2.0,clip_type="gaussian",seed=0,lr=[TASK2LR[task] / 10.0], ) for out_noise in l_output_noise]
    for run in runs:
        run.execute(force=False, mode=mode)


if __name__ == "__main__":
    # NOTE: For the attention-only models, we do:
        # forward_is_perf=True
        # ir_enable=False
    # The clip sigma of 2.0 is already correct
    aihwkit_train("imdb",False,"local",decay=0.5,l_output_noise=[0.1],eta=0.12,forward_is_perf=False,ir_enable=True)
    # aihwkit_train("cifar10",False,"local",decay=0.5,l_output_noise=[0.1],eta=0.12,forward_is_perf=False,ir_enable=True)
    # aihwkit_train("pf32",no_save=False,mode="local",decay=0.5,l_output_noise=[0.1],eta=0.12,forward_is_perf=False,ir_enable=True,)
    # aihwkit_train("listops",no_save=False,mode="local",decay=0.5,l_output_noise=[0.1],eta=0.12,forward_is_perf=False,ir_enable=True,)
    # aihwkit_train("aan",no_save=False,mode="local",decay=0.5,l_output_noise=[0.1],eta=0.12,forward_is_perf=False,ir_enable=True,)
