import os
import random
from datetime import datetime
import torch
import yaml
try:
    from fairseq.utils import assign_default_rpu_params
    RPU_AVAIL = True
except:
    RPU_AVAIL = False
    pass

from imc_performer.src.utils import IGNORE_KEYS, SUPPORTED_CMD_MODES, SUPPORTED_TASKS, TASK_TO_CONFIG_DIR, compare_yaml_dicts, remove_by_vals, sort_tuples_by_date

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Run:
    def __init__(self, model_name: str, task_data: str, task_name: str, no_save: bool, hour_q: int, base_store_dir: str, seed: int, bias: bool = True, world_size: int = 1, epochs: int = None, checkpoint_restore_file: str = "checkpoint_last.pt", config_path: str = None, finetune_from_model: bool = False, entropy_reg_params: dict = None, lr: list = None, **kwargs):
        # integrity checks
        assert task_name in SUPPORTED_TASKS, f"Invalid task name, supported: {SUPPORTED_TASKS}"
        assert world_size > 0, "World size must be > 0"
        assert hour_q in [1, 6, 12, 24, ], f"Unknown hour num. {hour_q} for queue. Must be in [1,6,12,24]"

        # run parameters
        self.bias = bias
        self.no_save = no_save
        self.model_name = model_name
        self.task_data = task_data
        self.task_name = task_name
        self.checkpoint_restore_file = checkpoint_restore_file
        self.base_store_dir = base_store_dir
        self.finetune_from_model = finetune_from_model
        self.hour_q = hour_q
        self.world_size = world_size
        self.epochs = epochs
        self.seed = seed
        self.entropy_reg_params = entropy_reg_params
        self.lr = lr
        self.config_path = config_path

        if RPU_AVAIL:
            # create RPU config with kwargs
            rpu_config = Namespace()
            assign_default_rpu_params(rpu_config)
            self.rpu_config = rpu_config.__dict__
            for key, val in kwargs.items():
                assert key in self.rpu_config, f"Key {key} not in possible keys for rpu_config"
                self.rpu_config[key] = val

            # will avoid splitting
            if not self.rpu_config["split_layers"]:
                self.rpu_config["max_input_size"] = 0
                self.rpu_config["max_output_size"] = 0
        else: self.rpu_config = {}

        self.yaml_dict = self.get_yaml_dict()

        # check if there already exists a run like that
        task_dir = os.path.join(base_store_dir, task_name)
        candidate_runs = get_candidate_set(task_dir, self.yaml_dict)

        if candidate_runs == []:
            self.run_id = self.get_random_id()
            self.run_save_path = os.path.join(task_dir, self.run_id)
            self.tensorboard_path = os.path.join(self.run_save_path, "logs")
            assert not os.path.isdir(self.run_save_path), "Run path already exists"
            assert os.path.isdir(self.base_store_dir), f"{base_store_dir} does not exist. Check your base_store_dir"
            if not os.path.isdir(task_dir): os.mkdir(task_dir)
            os.mkdir(self.run_save_path)
            # populate the save_dir field in the yaml
            self.yaml_dict["checkpoint"]["save_dir"] = self.run_save_path
            self.yaml_dict["common"]["tensorboard_logdir"] = self.tensorboard_path
            # write the yaml file to the folder
            self.write_yaml()

        else:
            # a run like this already exists, fetch its attributes
            self.run_id = candidate_runs[0][0].split("/")[-1]
            self.run_save_path = candidate_runs[0][0]
            self.tensorboard_path = os.path.join(self.run_save_path, "logs")
            with open(os.path.join(self.run_save_path, "run.yaml"), "r") as f: self.yaml_dict = yaml.safe_load(f)

    def get_random_id(self):
        """Generate random id for the run.

        Returns:
            id: string id.
        """
        with open("resources/id/nouns.txt") as f:
            nams = f.readlines()
            nams = [x.rstrip("\n") for x in nams]
        with open("resources/id/adjectives.txt") as f:
            adjs = f.readlines()
            adjs = [x.rstrip("\n") for x in adjs]
        return "".join(f"{random.choice(adjs)}_{random.choice(nams)}")

    def get_yaml_dict(self):
        """Generate run YAML config.

        Returns:
            yaml: yaml config dictionary.
        """

        # fetch base yaml from config
        if self.config_path is not None: path = self.config_path
        else: path = os.path.join(TASK_TO_CONFIG_DIR[self.task_name], f"{self.task_name}.yaml")
        with open(path, "r") as stream:
            try: yaml_dict = yaml.safe_load(stream)
            except yaml.YAMLError as exc: print(exc)

        # fill in the blanks that are run dependent
        yaml_dict["common"]["seed"] = self.seed
        yaml_dict["task"]["data"] = self.task_data
        yaml_dict["checkpoint"]["restore_file"] = self.checkpoint_restore_file
        yaml_dict["model"]["_name"] = self.model_name
        if self.bias is not None: yaml_dict["model"]["bias"] = self.bias
        yaml_dict["checkpoint"]["no_save"] = self.no_save
        yaml_dict["checkpoint"]["reset_lr_scheduler"] = self.finetune_from_model
        yaml_dict["checkpoint"]["reset_meters"] = self.finetune_from_model
        yaml_dict["checkpoint"]["reset_optimizer"] = self.finetune_from_model
        # yaml_dict["distributed_training"]["distributed_world_size"] = self.world_size

        if self.entropy_reg_params is not None:
            yaml_dict["criterion"]["_name"] = "rxn_entropy_reg"
            yaml_dict["criterion"]["beta"] = self.entropy_reg_params["beta"]
            yaml_dict["criterion"]["max_samples"] = self.entropy_reg_params["max_samples"]
            yaml_dict["criterion"]["sigma"] = self.entropy_reg_params["sigma"]
            yaml_dict["criterion"]["n_bins"] = self.entropy_reg_params["n_bins"]

        if self.epochs is not None: yaml_dict["optimization"]["max_epoch"] = self.epochs
        if self.lr is not None: yaml_dict["optimization"]["lr"] = self.lr
        if yaml_dict["optimization"]["lr"] == "???": raise Exception("Learning rate needs to be set.")

        # add RPU config params to the yaml config
        for key, val in self.rpu_config.items():
            if key == "eta_eval" and val == 0.0: continue
            yaml_dict["model"][key] = val
        return yaml_dict

    def write_yaml(self):
        """Write the yaml file to disk."""
        with open(os.path.join(self.run_save_path, "run.yaml"), "w") as f: yaml.dump(self.yaml_dict, f)

    def get_run_cmd(self, mode):
        """Return the execution command to start the run.

        Args:
            mode (str): execution mode.

        Returns:
            str: execution string.
        """
        assert mode in SUPPORTED_CMD_MODES, f"mode must be in {SUPPORTED_CMD_MODES}"
        args = f"--config-dir {self.run_save_path} --config-name run hydra.run.dir={os.path.join(self.run_save_path,'logs')}" 
        require_a100 = "" if self.world_size > 1 else "-require a100_80gb"
        if mode == "local": return f"fairseq-hydra-train {args}"
        elif mode == "local-debug":
            print(f"python -m debugpy --listen 0.0.0.0:1326 --wait-for-client ~/fairseq/fairseq_cli/hydra_train.py {args}")
            return ""
        elif mode == "jbsub-interactive": return f"jbsub -interactive -cores 1+{self.world_size} -mem 32g {require_a100} -q x86_{self.hour_q}h 'fairseq-hydra-train {args}'"
        elif mode == "jbsub": return f"jbsub -name {self.run_id} -cores 1+{self.world_size} -mem 32g -q x86_{self.hour_q}h {require_a100} 'fairseq-hydra-train {args}'"

    def execute(self, force: bool, mode: str):
        cmd = self.get_run_cmd(mode)
        if force: os.system(cmd)
        else:
            candidate_set = get_candidate_set(os.path.join(self.base_store_dir, self.task_name), self.yaml_dict)
            if candidate_set == []: os.system(cmd)
            else: return candidate_set[0]


def get_best_run(run_path: str, run_name: str):
    """Find the run with the highest validation score in a directory.

    Args:
        run_path (str): directory

    Returns:
        str: run name
    """
    if run_name == "best": runs = [f.path for f in os.scandir(run_path) if f.is_dir()]
    else: runs = [os.path.join(run_path, run_name)]
    best_run = None
    best_val_acc = 0
    for run in runs:
        if os.path.isfile(os.path.join(run, "checkpoint_best.pt")):
            ckpt = torch.load(os.path.join(run, "checkpoint_best.pt"))
            val_acc = ckpt["extra_state"]["best"]
            if val_acc > best_val_acc:
                best_run = run
                best_val_acc = val_acc
    return best_run


def get_candidate_set(base_dir: str, run_yaml_dict: dict):
    """Returns a candidate of runs that were performed with the same hyperparameters.

    Args:
        base_dir (str): models saving dir
        run_yaml_dict (dict): config to compare with

    Returns:
        candidate_set: list containing path and date of similar runs
    """
    run_paths = [f.path for f in os.scandir(base_dir) if f.is_dir()]
    candidate_set = []
    for run_path in run_paths:
        if has_run_finished(run_path):
            config_path = os.path.join(run_path, "run.yaml")
            if os.path.isfile(config_path):
                with open(config_path, "r") as f: loaded_yaml = yaml.safe_load(f)
                not_same_keys = compare_yaml_dicts(loaded_yaml, run_yaml_dict)
                not_same_keys = remove_by_vals(not_same_keys, IGNORE_KEYS)
                run_date = str(datetime.fromtimestamp(os.path.getmtime(config_path)))
                if len(not_same_keys) == 0: candidate_set.append((run_path, run_date))
    candidate_set = [] if candidate_set == [] else sort_tuples_by_date(candidate_set)
    return candidate_set


def has_run_finished(run_path: str):
    """Check whether a given run has finished or not.

    Args:
        run_path (str): run path

    Returns:
        bool: boolean flag to indicate whether the run has finished
    """
    assert os.path.isfile(os.path.join(run_path, "run.yaml")), f"This does not seem like a run {run_path}"
    done_file = os.path.join(run_path, "done")
    crashed_file = os.path.join(run_path, "crashed")
    if os.path.isfile(done_file): done = True
    else:
        log_path = os.path.join(run_path, "logs/hydra_train.log")
        if not os.path.isfile(log_path):
            done = False
            os.system(f"echo >> {crashed_file}")
        else:
            with open(log_path, "r") as f:
                try:
                    ll = f.readlines()[-1]
                    done = "done training" in ll
                except IndexError: done = False
            if done:
                if os.path.isfile(crashed_file): os.remove(crashed_file)
                os.system(f"echo >> {done_file}")
            else: os.system(f"echo >> {crashed_file}")
    return done


def delete_crashed_runs(base_dir: str):
    """Delete all the crashed runs in a directory."""
    candidate_paths = []
    run_paths = [f.path for f in os.scandir(base_dir) if f.is_dir()]
    for run_path in run_paths:
        if not has_run_finished(run_path): candidate_paths.append(run_path)
    is_enter = input(f"Have found {len(candidate_paths)} crashed runs. Hit enter to delete...")
    if is_enter == "":
        for run_path in candidate_paths:
            print(f"Removing {run_path}")
            os.system(f"rm -rf {run_path}")
