import os
import random
from collections import defaultdict
from enum import Enum
from functools import partial
from tqdm import tqdm

from fairseq.utils import move_to_cuda
import matplotlib.pyplot as plt
import numpy as np
import torch
from flatten_dict import flatten
from matplotlib import rcParams

from aihwkit.nn.conversion import AnalogWrapper
from aihwkit.simulator.tiles.base import AnalogTileStateNames

class Experiment(Enum):
    Attention_FP = 1
    Attention_HW = 2
    FullClipped = 3
    FullNoisy = 4


rcParams["font.size"] = 16
rcParams["axes.linewidth"] = 1.1
rcParams["axes.labelpad"] = 10.0
plot_color_cycle = plt.cycler("color", ["#9b59b6", "#3498db", "#95a5a6","#e74c3c","#34495e","#2ecc71","#1E2460","#B5B8B1","#734222","#A52019",],)
rcParams["axes.prop_cycle"] = plot_color_cycle
rcParams["axes.xmargin"] = 0
rcParams["axes.ymargin"] = 0
rcParams.update({"figure.figsize": (6.4, 4.8),"figure.subplot.left": 0.177,"figure.subplot.right": 0.946,"figure.subplot.bottom": 0.156,"figure.subplot.top": 0.965,"axes.autolimit_mode": "round_numbers","axes.grid": True,"xtick.major.size": 7,"xtick.minor.size": 3.5,"xtick.major.width": 1.1,"xtick.minor.width": 1.1,"xtick.major.pad": 5,"xtick.minor.visible": True,"ytick.major.size": 7,"ytick.minor.size": 3.5,"ytick.major.width": 1.1,"ytick.minor.width": 1.1,"ytick.major.pad": 5,"ytick.minor.visible": True,"lines.markersize": 10,"lines.markerfacecolor": "none","lines.markeredgewidth": 0.8,})
TASK_2_Q = {"aan": 12, "listops": 6, "imdb": 6, "pf32": 12, "cifar10": 6, "cifar10_relu": 6}
BATCH_SZ = {"aan": 1, "listops": 4, "imdb": 1, "pf32": 4, "cifar10": 1, "cifar10_relu": 1}
BATCH_SZ_CCC = {"aan": 12, "listops": 4, "imdb": 32, "pf32": 4, "cifar10": 64}
TASK_TO_CONFIG_DIR = {"aan": "config/lra","listops": "config/lra","imdb": "config/lra","pf32": "config/lra","cifar10": "config/lra","rxn": "config/rxn","PRET": "config/glue","MNLI": "config/glue","QNLI": "config/glue","STS-B": "config/glue","CoLA": "config/glue","SST-2": "config/glue","RTE": "config/glue","QQP": "config/glue","MRPC": "config/glue",}
LRA_TASKS = ["listops","aan","cifar10","cifar10_relu","imdb","pf32",]
TASK2TARGET = {"STS-B": 9,"RTE": 3,"MRPC": 0,"MNLI": -1,"SST-2": 1,"CoLA": 1,"QQP": -1,"QNLI": -1,}
TASK2COL = {"STS-B": (7, 8),"RTE": (1, 2),"MRPC": (3, 4),"MNLI": (8, 9),"SST-2": (0, None),"CoLA": (3, None),"QQP": (3, 4),"QNLI": (1, 2),}
GLUE_TASKS = ["PRET", "RTE","MRPC","MNLI","QNLI","QQP","SST-2","CoLA","STS-B",]
SUPPORTED_TASKS = LRA_TASKS + ["rxn"] + GLUE_TASKS
TASK_TO_DATA_MAP = {"aan": "aan","listops": "listops","imdb": "imdb-4000","pf32": "pathfinder","cifar10": "cifar10","cifar10_relu": "cifar10",}
SUPPORTED_CMD_MODES = ["local", "local-debug", "jbsub-interactive", "jbsub"]
IGNORE_KEYS = [("checkpoint", "save_dir"), ("common", "tensorboard_logdir")]
TASK_TO_MODEL = {"listops": "performer/listops/dutiful_funny","cifar10": "performer/cifar10/few_chance","aan": "performer/aan/kindhearted_series","pf32": "performer/pf32/lawful_kitchen","imdb": "performer/imdb/creepy_let",}

def disable_output_noise(model):
    if isinstance(model, AnalogWrapper):
        for tile in model.analog_tiles():
            rpu_config = tile.rpu_config
            rpu_config.forward.out_noise = 0.0
            tile_state = tile.__getstate__()
            tile_state[AnalogTileStateNames.RPU_CONFIG] = rpu_config
            tile.__setstate__(tile_state)

def find_matching_key(module_key: str, input_range_dict: dict):
    for ir_key in input_range_dict: 
        if module_key in ir_key: return ir_key
    raise Exception(f"Couldn't find key that matches/includes {module_key}")

def analog2model(model):
    return model.module if isinstance(model, AnalogWrapper) else model

def compare_yaml_dicts(a, b):
    not_same_keys = []
    f_a = flatten(a)
    f_b = flatten(b)
    for k in f_a:
        if k not in f_b or f_a[k] != f_b[k]: not_same_keys.append(k)
    return not_same_keys


def remove_by_val(_list: list, value):
    try: _list.remove(value)
    except ValueError: pass
    return _list


def remove_by_vals(_list: list, values: list):
    for v in values:
        _list = remove_by_val(_list, v)
    return _list


def sort_tuples_by_date(tuples_list):
    sorted_tuples = sorted(tuples_list, key=lambda t: t[1], reverse=True)
    return sorted_tuples


def fix_random(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def set_tracing(model, trace):
    for _, m in model.named_modules(): 
        if hasattr(m, "traceable"): m.traceable = trace


def print_results(res: defaultdict):
    print("\nEvaluation Results\n")
    for key, values in res.items():
        formatted_values = [f"{np.mean(val):.2f} ({np.std(val):.2f})" for val in values]
        value_row = "| " + key + " | " + " | ".join(formatted_values) + " |"
        print(value_row)


def save_results(path: str, res: defaultdict):
    out = "Evaluation Results\n"
    out += "| Task  | Pipelined | Input Quantization | Programming Noise | LDPU model | Output Noise |"
    for key, values in res.items():
        formatted_values = [f"{np.mean(val):.2f} ({np.std(val):.2f})" for val in values]
        value_row = "\n| " + key + " | " + " | ".join(formatted_values) + " |"
        out += value_row
    with open(path, "w") as f: f.write(out)

def save_results_pickle(path: str, res: dict):
    out = ""
    for key, values in res.items():
        value_row = f"{key}: {np.mean(values):.2f} ({np.std(values):.2f})\n"
        out += value_row
    with open(path, "w") as f: f.write(out)

def cache_hook(mod,input_args,cache_key: str,global_cache,max_samples: int = 1000,) -> None:
    scale = 127. / mod.input_range.item()
    x_input = input_args[0]
    x_input_pwm = torch.clamp(x_input * scale, -127, 127).round()
    cache = global_cache[cache_key]
    if hasattr(mod, "input_range") and mod.input_range is not None: cache["ir"] = mod.input_range.item()
    cache["pwm"] = torch.cat([cache["pwm"], x_input_pwm.reshape(-1, x_input_pwm.size(-1)).clone().detach().cpu()])
    cache["raw"] = torch.cat([cache["raw"], x_input.reshape(-1, x_input.size(-1)).clone().detach().cpu()])
    cache["max"] = max(cache["max"], x_input.abs().max().item())
    # Shuffle and limit the number
    cache["pwm"] = cache["pwm"][torch.randperm(cache["pwm"].size(0))[:max_samples]]
    cache["raw"] = cache["raw"][torch.randperm(cache["raw"].size(0))[:max_samples]]
    global_cache[cache_key] = cache

def plot_weights(model):
    for n,tile in model.named_analog_tiles():
        w, _ = tile.get_weights(False)
        plt.figure()
        plt.hist(w.flatten().numpy(), bins=50)
        plt.title(n)
        plt.tight_layout()
        plt.savefig(f"resources/debug/{n}_weight.png")

def analyse_model(model, loader):
    plot_weights(model)

    handles = []
    cache = {}
    for tile_name, tile in model.named_analog_tiles():
        cache[tile_name] = {"pwm": torch.tensor([]), "raw": torch.tensor([]), "max": -1.}
        hook = partial(cache_hook,cache_key=tile_name,global_cache=cache,max_samples=1000,)
        handles.append(tile.register_forward_pre_hook(hook, prepend=True))
    
    for i, x in tqdm(enumerate(loader)):
        x = move_to_cuda(x)
        model(x)
        if i > 500: break

    for handle in handles:
        handle.remove()

    for layer_name, inputs_dict in cache.items():
        max_raw = inputs_dict["max"]
        inputs_pwm = inputs_dict["pwm"]
        inputs_raw = inputs_dict["raw"]
        inputs_raw = inputs_raw[~(inputs_raw == 0).all(-1)]
        inputs_pwm = inputs_pwm[~(inputs_pwm == 0).all(-1)]
        
        plt.figure(figsize=(20,10))
        plt.title(layer_name)
        plt.subplot(1,3,1)
        plt.plot(inputs_pwm.abs().sum(-1).numpy())
        plt.subplot(1,3,2)
        plt.hist(inputs_pwm.flatten().numpy(), bins=50)
        plt.subplot(1,3,3)
        plt.hist(inputs_raw.flatten().numpy(), bins=50)
        plt.axvline(x=max_raw, color="r", linewidth=2.0)
        plt.axvline(x=-max_raw, label="max observed", color="r", linewidth=2.0)
        if "ir" in inputs_dict: ir = inputs_dict["ir"]
        plt.axvline(x=ir, label="IR", color="g", linewidth=2.0)
        plt.axvline(x=-ir, color="g", linewidth=2.0)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"resources/debug/input_{layer_name}.png")
