import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data_providers import test_dataset,val_dataset,train_dataset
from arg_extractor import get_args, available_models
from experiment_builder import ExperimentBuilder

args = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
torch.manual_seed(seed=args.seed)  # sets pytorch's seed

train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

# Define the model
assert args.model_name in available_models, f"Invalid model name. must be one of {available_models}"
if args.model_name == 'TestModel':
    from models.test_model import TestModel
    nn_model = TestModel(input_channels = args.image_num_channels,
          n_filters = args.num_filters, 
          dropout_prob = 0, 
          n_classes = args.num_classes)
elif args.model_name == 'UNet':
    from models.unet_model import UNet
    nn_model = UNet(input_channels = args.image_num_channels,
          n_filters = args.num_filters, 
          n_classes = args.num_classes)
elif args.model_name == 'CLIP':
    from models.clip_based_unet import CLIPResnetSegmentationModel
    device = torch.device("cuda" if (torch.cuda.is_available() and args.use_gpu) else "cpu")
    nn_model = CLIPResnetSegmentationModel(device, args.num_classes)


conv_experiment = ExperimentBuilder(network_model=nn_model,
                                    experiment_name=args.model_name + "_" + args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    weight_decay_coefficient=args.weight_decay_coefficient,
                                    use_gpu=args.use_gpu,
                                    continue_from_epoch=args.continue_from_epoch,
                                    train_data=train_data_loader, val_data=val_data_loader,
                                    test_data=test_data_loader,
                                    lr = args.lr,
                                    )  # build an experiment object
experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics
