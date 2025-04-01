import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data_providers import test_dataset,val_dataset,train_dataset
from arg_extractor import get_args
from experiment_builder import ExperimentBuilder
from models.test_model import TestModel
from models.unet_model import UNet

import os 
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

args = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
torch.manual_seed(seed=args.seed)  # sets pytorch's seed

train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

custom_conv_net = TestModel(input_channels = args.image_num_channels,
          n_filters = args.num_filters, 
          dropout_prob = 0, 
          n_classes = args.num_classes)


conv_experiment = ExperimentBuilder(network_model=custom_conv_net,
                                    experiment_name=args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    weight_decay_coefficient=args.weight_decay_coefficient,
                                    use_gpu=args.use_gpu,
                                    continue_from_epoch=args.continue_from_epoch,
                                    train_data=train_data_loader, val_data=val_data_loader,
                                    test_data=test_data_loader,
                                    lr = args.lr,
                                    )  # build an experiment object
experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics
