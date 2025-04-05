import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from Datasets.Segmentation.data_providers import test_dataset,val_dataset,train_dataset
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
    print("TestModel ready")

elif args.model_name == 'UNet':
    from models.unet_model import UNet
    nn_model = UNet(input_channels = args.image_num_channels,
          n_filters = args.num_filters, 
          n_classes = args.num_classes)
    
    if args.load_encoder_weights:
        print("Loading pretrained Autoencoder weights")
        path_to_weights = "models/weights/autoencoder/autoencoder_weights_0.002858656363969203.pth"
        print(f"Loading pretrained Autoencoder weights from {path_to_weights}")
        autoencoder_weights = torch.load(path_to_weights, map_location=torch.device('cpu'))
        from models.unet_model import Autoencoder
        autoencoder = Autoencoder(input_channels=3, n_filters=64)
        autoencoder.load_state_dict(autoencoder_weights, strict=True)
        print("Moving encoder weights from Autoencoder to UNet")
        nn_model.cblock1.load_state_dict(autoencoder.cblock1.state_dict())
        nn_model.cblock2.load_state_dict(autoencoder.cblock2.state_dict())
        nn_model.cblock3.load_state_dict(autoencoder.cblock3.state_dict())
        nn_model.cblock4.load_state_dict(autoencoder.cblock4.state_dict())
        nn_model.cblock5.load_state_dict(autoencoder.cblock5.state_dict())
        print("Freezing encoder weights")
        for param in [nn_model.cblock1.parameters(), nn_model.cblock2.parameters(),
                    nn_model.cblock3.parameters(), nn_model.cblock4.parameters(),
                    nn_model.cblock5.parameters()]:
            for p in param:
                p.requires_grad = False
        print("UNet with autoencoder's encoder weights ready")
    else:
        print("UNet with random encoder weights ready")

elif args.model_name == 'CLIP':
    from models.clip_based_unet import CLIPResnetSegmentationModel
    device = torch.device("cuda" if (torch.cuda.is_available() and args.use_gpu) else "cpu")
    nn_model = CLIPResnetSegmentationModel(device, args.num_classes)
    print("Unet with CLIP's encoder weights ready")

elif args.model_name == 'PromptUNet':
    from models.prompt_based_unet import PromptUNet
    device = torch.device("cuda" if (torch.cuda.is_available() and args.use_gpu) else "cpu")
    nn_model = PromptUNet(device, args.num_classes)
    print("Promp Based Unet with CLIP's encoder weights ready")

    # For this model the data changes a bit, hence:
    from Datasets.Prompt_based_segmentation.data_providers import test_dataset,val_dataset,train_dataset
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)



conv_experiment = ExperimentBuilder(network_model=nn_model,
                                    experiment_name=args.model_name + "_" + args.experiment_name,
                                    num_epochs=args.num_epochs,
                                    weight_decay_coefficient=args.weight_decay_coefficient,
                                    use_gpu=args.use_gpu,
                                    continue_from_epoch=args.continue_from_epoch,
                                    train_data=train_data_loader, 
                                    val_data=val_data_loader,
                                    test_data=test_data_loader,
                                    lr = args.lr,
                                    model_name = args.model_name,
                                    )  # build an experiment object
experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics
