"""
===========================
Enhancing Multi-Fidelity Optimization using Proxy and Evolutionary Techniques
===========================
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Mapping, Optional
from functools import partial
import time
import joblib
import numpy as np
import torch
from torchsummary import summary
from earlystopping import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torchvision import transforms
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, UniformIntegerHyperparameter, CategoricalHyperparameter, Constant

from ConfigSpace import (
    Configuration,
    ConfigurationSpace,
    Float,
    Integer,
    Constant,
    InCondition,
    Categorical
)
from sklearn.model_selection import train_test_split
from ConfigSpace.read_and_write import json as cs_json
from ConfigSpace.read_and_write import pcs_new, pcs
from sklearn.model_selection import StratifiedKFold
from smac.facade.multi_fidelity_facade import MultiFidelityFacade as SMAC4MF
from smac.intensifier.hyperband import Hyperband
from smac.scenario import Scenario
from smac.initial_design import LatinHypercubeInitialDesign, FactorialInitialDesign
from torch.utils.data import DataLoader, Subset
from dask.distributed import get_worker
import hashlib
from cnn import Model
import pickle 
import json

from datasets import load_deep_woods, load_fashion_mnist
date_time = time.strftime("%Y%m%d-%H%M%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a custom logger
handler = logging.FileHandler(f'./logs/run_{date_time}.log', encoding='utf-8')
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)


# print("Logger: ", logging.getLogger(__name__))

CV_SPLIT_SEED = 42

val_loss_array = []
epochs_array = []
learning_rate_array = []



def configuration_space(
        device: str,
        dataset: str,
        cv_count: int = 3,
        budget_type: str = "epochs", #"img_size",
        datasetpath: str | Path = Path("."),
        cs_file: Optional[str | Path] = None
) -> ConfigurationSpace:
    """Build Configuration Space which defines all parameters and their ranges."""
    if cs_file is None:
        # This serves only as an example of how you can manually define a Configuration Space
        # To illustrate different parameter types;
        # we use continuous, integer and categorical parameters.
        cs = ConfigurationSpace(
            {
                "n_conv_layers": Integer("n_conv_layers", (1, 3), default=3),
                "use_BN": Categorical("use_BN", [True, False], default=True),
                "global_avg_pooling": Categorical("global_avg_pooling", [True, False], default=True),
                "n_channels_conv_0": Integer("n_channels_conv_0", (32, 512), default=512, log=True),
                "n_channels_conv_1": Integer("n_channels_conv_1", (16, 512), default=512, log=True),
                "n_channels_conv_2": Integer("n_channels_conv_2", (16, 512), default=512, log=True),
                "n_fc_layers": Integer("n_fc_layers", (1, 3), default=3),
                "n_channels_fc_0": Integer("n_channels_fc_0", (32, 512), default=512, log=True),
                "n_channels_fc_1": Integer("n_channels_fc_1", (16, 512), default=512, log=True),
                "n_channels_fc_2": Integer("n_channels_fc_2", (16, 512), default=512, log=True),
                "batch_size": Integer("batch_size", (1, 1000), default=200, log=True),
                "learning_rate_init": Float(
                    "learning_rate_init",
                    (1e-5, 1.0),
                    default=1e-3,
                    log=True,
                ),
                "kernel_size": Constant("kernel_size", 3),
                "dropout_rate": Constant("dropout_rate", 0.2),
                "device": Constant("device", device),
                "dataset": Constant("dataset", dataset),
                "datasetpath": Constant("datasetpath", str(datasetpath.absolute())),
            }
        )

        # Add conditions to restrict the hyperparameter space
        use_conv_layer_2 = InCondition(cs["n_channels_conv_2"], cs["n_conv_layers"], [3])
        use_conv_layer_1 = InCondition(cs["n_channels_conv_1"], cs["n_conv_layers"], [2, 3])

        use_fc_layer_2 = InCondition(cs["n_channels_fc_2"], cs["n_fc_layers"], [3])
        use_fc_layer_1 = InCondition(cs["n_channels_fc_1"], cs["n_fc_layers"], [2, 3])

        # Add multiple conditions on hyperparameters at once:
        cs.add_conditions([use_conv_layer_2, use_conv_layer_1, use_fc_layer_2, use_fc_layer_1])
    else:
        with open(cs_file, "r") as fh:
            cs_string = fh.read()
            if cs_file.suffix == ".json":
                cs = cs_json.read(cs_string)
            elif cs_file.suffix in [".pcs", ".pcs_new"]:
                cs = pcs_new.read(pcs_string=cs_string)
        # logging.info(f"Loaded configuration space from {cs_file}")
        logger.info(f"Loaded configuration space from {cs_file}")

        # print(f"Loaded configuration space from {cs_file}")

        if "device" not in cs:
            cs.add_hyperparameter(Constant("device", device))
        if "dataset" not in cs:
            cs.add_hyperparameter(Constant("dataset", dataset))
        if "cv_count" not in cs:
            cs.add_hyperparameter(Constant("cv_count", cv_count))
        if "budget_type" not in cs:
            cs.add_hyperparameter(Constant("budget_type", budget_type))
        if "datasetpath" not in cs:
            cs.add_hyperparameter(Constant("datasetpath", str(datasetpath.absolute())))
        # logging.debug(f"Configuration space:\n{cs}")
        logger.debug(f"Configuration space:\n{cs}")
        # print(f"Configuration space:\n{cs}")

    return cs


def get_optimizer_and_criterion(
        cfg: Mapping[str, Any]
) -> tuple[
    type[torch.optim.AdamW | torch.optim.Adam | torch.optim.SGD ],
    type[torch.nn.MSELoss | torch.nn.CrossEntropyLoss],
]:
    if cfg["optimizer"] == "AdamW":
        model_optimizer = torch.optim.AdamW
    elif cfg["optimizer"] == "SGD":
        model_optimizer = torch.optim.SGD
    else:
        model_optimizer = torch.optim.Adam

    if cfg["train_criterion"] == "mse":
        train_criterion = torch.nn.MSELoss
    else:
        train_criterion = torch.nn.CrossEntropyLoss

    return model_optimizer, train_criterion


# Target Algorithm
# The signature of the function determines what arguments are passed to it
# i.e., budget is passed to the target algorithm if it is present in the signature
# This is specific to SMAC
def cnn_from_cfg(
        cfg: Configuration,
        seed: int,
        budget: float,
        test: bool = False,
) -> float:
    """
    Creates an instance of the torch_model and fits the given data on it.
    This is the function-call we try to optimize. Chosen values are stored in
    the configuration (cfg).

    :param cfg: Configuration (basically a dictionary)
        configuration chosen by smac
    :param seed: int or RandomState
        used to initialize the rf's random generator
    :param budget: float
        used to set max iterations for the MLP
    Returns
    -------
    val_accuracy cross validation accuracy
    """
    try:
        worker_id = get_worker().name
    except ValueError:
        worker_id = 0

    # If data already existing on disk, set to False
    download = False

    lr = cfg["learning_rate_init"]
    dataset = cfg["dataset"]
    device = cfg["device"]
    batch_size = cfg["batch_size"]
    ds_path = cfg["datasetpath"]

    # unchangeable constants that need to be adhered to, the maximum fidelities
    #img_size = max(4, int(np.floor(budget)))  # example fidelity to use
    img_size = 32

    print('\n\n\ batch_size: ', batch_size)


    # Device configuration
    torch.manual_seed(seed)
    model_device = torch.device(device)

    if "fashion_mnist" in dataset:
        input_shape, train_val, test_data = load_fashion_mnist(datadir=Path(ds_path, "FashionMNIST"))
    elif "deepweedsx" in dataset:
        input_shape, train_val, test_data = load_deep_woods(
            datadir=Path(ds_path, "deepweedsx"),
            resize=(img_size, img_size),
            balanced="balanced" in dataset,
            download=download,
        )
    else:
        raise NotImplementedError

    mean = [0.3403, 0.3121, 0.3214]
    std = [0.2724, 0.2608, 0.2669]

    train_transform = transforms.Compose(
        [
            transforms.TrivialAugmentWide(),
            # transforms.RandomRotation(10),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,#[0.485, 0.456, 0.406], 
                std=std, #[0.229, 0.224, 0.225],
            ),
        ]
    )

    # returns the cross-validation accuracy
    # to make CV splits consistent
    cv = StratifiedKFold(n_splits=3, random_state=CV_SPLIT_SEED, shuffle=True)
    # train_val.transform = train_transform

    score = []
    cv_splits = cv.split(train_val, train_val.targets)
    for cv_index, (train_idx, valid_idx) in enumerate(cv_splits, start=1):
        #logging.info(f"Worker:{worker_id} ------------ CV {cv_index} -----------")
        logger.info(f"Worker:{worker_id} ------------ CV {cv_index} -----------")
        # print(f"Worker:{worker_id} ------------ CV {cv_index} -----------")
        train_data = Subset(train_val, list(train_idx))
        val_data = Subset(train_val, list(valid_idx))

        # train_data.transform = train_transform

        train_loader = DataLoader(
            dataset=train_data,
            batch_size=batch_size,
            shuffle=True,
        )

        val_loader = DataLoader(
            dataset=val_data,
            batch_size=batch_size,
            shuffle=False,
        )        
        model = Model(
            config=cfg,
            input_shape=input_shape,
            num_classes=len(train_val.classes),
        )
        # model = CustomMobileNetV2(
        #     config=cfg,
        #     num_classes=len(train_val.classes),
        # )
        model = model.to(model_device)

        summary(model, input_shape, device=device)
        logger.info(f"Summary : {str(model)}")

        model_optimizer, train_criterion = get_optimizer_and_criterion(cfg)
        print("\n\nOptimizer:",cfg["optimizer"])
        weight_decay = 3e-5

        if model_optimizer == torch.optim.SGD:
            momentum = 0.9
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        else:    
            optimizer = model_optimizer(model.parameters(), lr=lr)

        train_criterion = train_criterion().to(device)
        # scheduler = StepLR(optimizer, step_size=3, gamma=0.8, verbose=True)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2, eta_min=0.000001, verbose=True)
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        
        epochs = int(np.ceil(budget))
        # epochs = 20
        lr_list = []
        early_stopping = EarlyStopping(patience=6, verbose=True)
        epochs_array.append(epochs)
        for epoch in range(epochs): #range(20):  # 20 epochs
            # logging.info(f"Worker:{worker_id} " + "#" * 50)
            logger.info(f"Worker:{worker_id} " + "#" * 50)
            # logging.info(f"Worker:{worker_id} Epoch [{epoch + 1}/{20}]")
            logger.info(f"Worker:{worker_id} Epoch [{epoch + 1}/{20}]")
            # print(f"Worker:{worker_id} Epoch [{epoch + 1}/{20}]")
            train_score, train_loss = model.train_fn(
                optimizer=optimizer,
                criterion=train_criterion,
                loader=train_loader,
                device=model_device
            )
            # logging.info(f"Worker:{worker_id} => Train accuracy {train_score:.3f} | loss {train_loss}")
            print((f"Worker:{worker_id} => Train accuracy {train_score:.3f} | loss {train_loss}"))
            
            # Entry for early stopping
            val_loss = 1- model.eval_fn(val_loader, device)
            logger.info(f"Worker:{worker_id} => Train accuracy {train_score:.3f} | loss {train_loss} | Val Acc {1- val_loss} " )
            # scheduler.step(train_loss) # for ReduceLROnPlateau
            # scheduler.step() # for CosineAnnealingWarmRestarts
            # early_stopping(train_loss, model)

            current_lr = optimizer.param_groups[0]['lr']
            lr_list.append(current_lr)

            # if early_stopping.early_stop:
            #     print("Early stopping")
            #     logger.info(f"Worker:{worker_id} => Early stopping")
            #     break
            
        val_score = model.eval_fn(val_loader, device)
        # logging.info(f"Worker:{worker_id} => Val accuracy {val_score:.3f}")
        logger.info(f"Worker:{worker_id} => Val accuracy {val_score:.3f}")
        print(f"Worker:{worker_id} => Val accuracy {val_score:.3f}")
        score.append(val_score)

    val_error = 1 - np.mean(score)  # because minimize
    val_loss_array.append(val_error)
    results = val_error

    if test:

        test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=True,
        )

        results = model.eval_fn(test_loader, device)
        logger.info(f"Best incumbent => Test accuracy {results:.3f}")
        print(f"Best incumbent => Test accuracy {results:.3f}")

    # print(f"Results: {results}")
    learning_rate_array.append(lr_list)
    return results


def proxy_from_cfg(
        cfg: Configuration,
        seed: int,
        budget: float,
        optimize: bool = True,
) -> float:
    "Creates a proxy instance of the model and fits the data"
    try:
        worker_id = get_worker().name
    except ValueError:
        worker_id = 0
    
    # If data already existing on disk, set to False
    download = False

    logger.info(f"\n\n\nWorker:{worker_id} ------------ Proxy -----------")
    print("\n\nIn proxy function------------------")

    lr = cfg["learning_rate_init"]
    dataset = cfg["dataset"]
    device = cfg["device"]
    batch_size = cfg["batch_size"]
    ds_path = cfg["datasetpath"]

    img_size = 16
    # Device configuration
    torch.manual_seed(seed)
    model_device = torch.device(device)

    if "fashion_mnist" in dataset:
        input_shape, train_val, _ = load_fashion_mnist(datadir=Path(ds_path, "FashionMNIST"))
    elif "deepweedsx" in dataset:
        input_shape, train_val, _ = load_deep_woods(
            datadir=Path(ds_path, "deepweedsx"),
            resize=(img_size, img_size),
            balanced="balanced" in dataset,
            download=download,
        )
    else:
        raise NotImplementedError

    # Split the data into train and validation sets (30% each)
    train_indices, val_indices = train_test_split(
        range(len(train_val)),
        train_size=0.3,  # 30% of the data for training
        test_size=0.3,   # 30% of the data for validation
        stratify=train_val.targets,
        random_state=42
    )

    train_data = Subset(train_val, train_indices)
    val_data = Subset(train_val, val_indices)

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        dataset=val_data,
        batch_size=batch_size,
        shuffle=False,
    )

    config_str = str(cfg).encode()
    config_hash = hashlib.md5(config_str).hexdigest()

    print("\n\nhash: ", config_hash)

    model = Model(
        config=cfg,
        input_shape=input_shape,
        num_classes=len(train_val.classes),
    )

    model = model.to(model_device)
    summary(model, input_shape, device=device)
    logger.info(f"Config: {cfg}")
    logger.info(f"Summary : {str(model)}")

    # print("\n\nOptimizer:",cfg["optimizer"])
    weight_decay = 3e-5

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_criterion = torch.nn.CrossEntropyLoss().to(device)

    early_stopping = EarlyStopping(patience=3, verbose=True)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2, eta_min=0.000001, verbose=True)
    # scheduler = StepLR(optimizer, step_size=2, gamma=0.8, verbose=True)
    epochs = 8
    vl_loss = []
    for epoch in range(epochs):
        logger.info(f"Worker:{worker_id} " + "#" * 50)
        logger.info(f"Worker:{worker_id} Epoch [{epoch + 1}/{epochs}]")
        train_score, train_loss = model.train_fn(
            optimizer=optimizer,
            criterion=train_criterion,
            loader=train_loader,
            device=model_device
        )

        scheduler.step()
        logger.debug(f"Worker:{worker_id} => Train accuracy {train_score:.3f} | loss {train_loss}")
        # val_score = model.eval_fn(val_loader, device)
        # logger.info(f"Worker:{worker_id} => Val accuracy {val_score:.3f}")
        # early_stopping(val_score, model)
        if early_stopping.early_stop:
            print("-----------------Early stopping--------------------")
            logger.info(f"Worker:{worker_id} => Early stopping")
            break

    val_score = 1- model.eval_fn(val_loader, device)
    vl =[]
    vl.append(1-val_score)

    logger.debug(f"Worker:{worker_id} => Val accuracy {1- val_score:.3f}")
    print("\n\n\n --------------------------proxy ended----------------------------------")
    if optimize: 
        if (1- val_score) > 0.35:
            print("\n\n------------Real evaluation for config-------------")
            logger.info(f"\n\n------------Real evaluation for config-------------")
            val_score = cnn_from_cfg(cfg, seed, budget)
    
    vl.append(val_score)
    vl_loss.append(vl)
    val_loss_array.append(vl_loss)
    return val_score

def proxy_function(config_space, time_limit) -> list:
    print("\n\n\n inside proxy_function")

    warmup_configs = []
    start_time = time.time()

    while time.time() - start_time < time_limit:
        config_time_start = time.time()
        cfg = config_space.sample_configuration()

        try:
            val_score = 1- proxy_from_cfg(cfg, args.seed, args.max_budget, optimize=False)
            config_time_end = time.time()
            logger.info(f"Time taken for config: {config_time_end - config_time_start}")
            print(f"Time taken for config: {config_time_end - config_time_start}")
            warmup_configs.append((cfg, val_score))
        except Exception as e:
            continue       

    best_configs = [config[0] for config in warmup_configs if config[1] > 0.40]

    return best_configs

    
if __name__ == "__main__":
    """
    This is just an example of how to implement BOHB as an optimizer!
    Here we do not consider any of the forbidden clauses.
    """

    parser = argparse.ArgumentParser(description="MF example using BOHB.")
    parser.add_argument(
        "--dataset",
        choices=["deepweedsx", "deepweedsx_balanced", "fashion_mnist"],
        default="deepweedsx_balanced",
        help="dataset to use (task for the project: deepweedsx_balanced)",
    )
    parser.add_argument(
        "--working_dir",
        default=f"./tmp/{date_time}",
        type=str,
        help="directory where intermediate results are stored",
    )
    parser.add_argument(
        "--runtime",
        default= 60*60*6,
        type=float,
        help="Running time (seconds) allocated to run the algorithm",
    )
    parser.add_argument(
        "--max_budget",
        type=float,
        default=20,
        help="maximal budget epochs/ (image_size) to use with BOHB",
    )
    parser.add_argument(
        "--min_budget", type=float, default=10, help="Minimum budget (image_size) for BOHB"
    )
    parser.add_argument("--eta", type=int, default=3, help="eta for BOHB")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to run the models"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="num of workers to use with BOHB"
    )
    parser.add_argument(
        "--n_trials", type=int, default=1000, help="Number of iterations to run SMAC for"
    )
    parser.add_argument(
        "--cv_count",
        type=int,
        default=3,
        help="Number of cross validations splits to create. "
             "Will not have an effect if the budget type is cv_splits",
    )
    parser.add_argument(
        "--log_level",
        choices=[
            "NOTSET"
            "CRITICAL",
            "FATAL",
            "ERROR",
            "WARN",
            "WARNING",
            "INFO",
            "DEBUG",
        ],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument('--configspace', type=Path, default= "new_config.json",
                        help='Path to file containing the configuration space')
    parser.add_argument('--datasetpath', type=Path, default=Path('./data/'),
                        help='Path to directory containing the dataset')
    args = parser.parse_args()
    
    logging.basicConfig(level=args.log_level)

    configspace = configuration_space(
        device=args.device,
        dataset=args.dataset,
        cv_count=args.cv_count,
        datasetpath=args.datasetpath,
        cs_file=args.configspace
    )
    warmup_configs = []
    warm_time_start = time.time()
    # warmup_configs = proxy_function(configspace, 60)
    warm_time_end = time.time()
    # #warmup_configs = [x[0] for x in warmup_configs]    
    print(f"\n\nWarmup Configs: {len(warmup_configs)}")
    logger.info(f"\n\nWarmup Configs: {len(warmup_configs)}")
    print(f"\n\n Time taken for warmup: {warm_time_end - warm_time_start} seconds")
    logger.info(f"\n\n Time taken for warmup: {warm_time_end - warm_time_start} seconds")
    remaining_time = args.runtime - (warm_time_end - warm_time_start)

    # Setting up SMAC to run BOHB
    scenario = Scenario(
        name="ExampleMFRunWithBOHB",
        configspace=configspace,
        deterministic=True,
        output_directory=args.working_dir,
        seed=args.seed,
        n_trials=args.n_trials,
        max_budget=args.max_budget,
        min_budget=args.min_budget,
        n_workers=args.workers,
        walltime_limit= remaining_time#args.runtime
    )
        
    # You can mess with SMACs own hyperparameters here (checkout the documentation at https://automl.github.io/SMAC3)
    smac = SMAC4MF(
        target_function= cnn_from_cfg,#proxy_from_cfg, #cnn_from_cfg,
        scenario=scenario,
        # initial_design = warmup_configs,
        initial_design = SMAC4MF.get_initial_design(scenario=scenario, n_configs=3, additional_configs = warmup_configs),
        intensifier=Hyperband(
            scenario=scenario,
            incumbent_selection="highest_budget",
            eta=args.eta,
        ),
        overwrite=True,
        logging_level= 0 # args.log_level,  # https://automl.github.io/SMAC3/main/advanced_usage/8_logging.html
    )

    print("\n\nStarting SMAC run------------------")
    logger.info("\n\nStarting SMAC run------------------")
    # Start optimization
    incumbent = smac.optimize()
    print("\n\n\nOptimized Incumbent:\n", incumbent)
    logger.debug(f"\n\n\n\nOptimized Incumbent:\n {incumbent}")

    # evaluate the incumbent with the test_set_data
    test_score = cnn_from_cfg(incumbent, args.seed, args.max_budget, test=True)
    logger.info(f"Test accuracy of best found configuration: {test_score:.4f}")
    print(f"Test accuracy of best found configuration: {test_score:.4f}")


    with open(f"./val_loss_array.pkl", "wb") as fh:
        pickle.dump(val_loss_array, fh)
    
    with open(f"./epochs_array.pkl", "wb") as fh:
        pickle.dump(epochs_array, fh)
    
    with open(f"./learning_rate_array.pkl", "wb") as fh:
        pickle.dump(learning_rate_array, fh)
    
    # save the best found configuration to a file
    with open(f"./best_config.json", "w") as fh:
        json.dump(incumbent.get_dictionary(), fh)

    with open(f"./incumbent.pkl", "wb") as fh:
            pickle.dump(incumbent, fh)