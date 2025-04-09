import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import os
import numpy as np
import time

from storage_utils import save_statistics
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from collections import Counter

matplotlib.rcParams.update({"font.size": 8})

def compute_class_weights(loader, num_classes=4):
    class_counts = Counter()
    for _, mask, _ in loader:
        unique_classes, counts = torch.unique(mask, return_counts=True)
        for cls, count in zip(unique_classes.tolist(), counts.tolist()):
            class_counts[cls] += count

    total_pixels = sum(class_counts.values())
    class_weights = []

    for i in range(num_classes):
        if i in class_counts:
            freq = class_counts[i] / total_pixels
            weight = 1.0 / (freq + 1e-6)
        else:
            weight = 0.0  # no aparece esa clase
        class_weights.append(weight)

    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    class_weights = class_weights * (num_classes / class_weights.sum())
    return class_weights

class ExperimentBuilder(nn.Module):
    def __init__(
        self,
        network_model,
        experiment_name,
        num_epochs,
        train_data,
        val_data,
        test_data,
        weight_decay_coefficient,
        use_gpu,
        continue_from_epoch=-1, 
        lr=1e-3,
        model_name = "TestModel"
    ):
        """
        Initializes an ExperimentBuilder object. Such an object takes care of running training and evaluation of a deep net
        on a given dataset. It also takes care of saving per epoch models and automatically inferring the best val model
        to be used for evaluating the test set metrics.
        :param network_model: A pytorch nn.Module which implements a network architecture.
        :param experiment_name: The name of the experiment. This is used mainly for keeping track of the experiment and creating and directory structure that will be used to save logs, model parameters and other.
        :param num_epochs: Total number of epochs to run the experiment
        :param train_data: An object of the DataProvider type. Contains the training set.
        :param val_data: An object of the DataProvider type. Contains the val set.
        :param test_data: An object of the DataProvider type. Contains the test set.
        :param weight_decay_coefficient: A float indicating the weight decay to use with the adam optimizer.
        :param use_gpu: A boolean indicating whether to use a GPU or not.
        :param continue_from_epoch: An int indicating whether we'll start from scrach (-1) or whether we'll reload a previously saved model of epoch 'continue_from_epoch' and continue training from there.
        :param lr: A float indicating the learning rate to use with the adam optimizer.
        :param model_name: A string indicating the name of the model to be used. Used to change the training loop for different models.
        """
        super(ExperimentBuilder, self).__init__()

        self.experiment_name = experiment_name
        self.model = network_model
        self.model_name = model_name

        if torch.cuda.device_count() >= 1 and use_gpu:
            self.device = torch.device("cuda")
            self.model.to(self.device)  # sends the model from the cpu to the gpu
        else:
            self.device = torch.device("cpu")  # sets the device to be CPU
        print("Using ",self.device)

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        num_conv_layers = 0
        num_linear_layers = 0
        total_num_parameters = 0
        for name, value in self.named_parameters():
            if all(item in name for item in ["conv", "weight"]):
                num_conv_layers += 1
            if all(item in name for item in ["linear", "weight"]):
                num_linear_layers += 1
            total_num_parameters += np.prod(value.shape)

        print("Total number of parameters", total_num_parameters)
        print("Total number of conv layers", num_conv_layers)
        print("Total number of linear layers", num_linear_layers)

        self.optimizer = optim.Adam(
            self.parameters(), amsgrad=False, weight_decay=weight_decay_coefficient, lr=lr
        )
        self.learning_rate_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=0.00002
        )

        # Loss function
        if self.model_name == "PromptUNet":
            print("No weight to classes in PromptUNet")
            self.loss_criterion = nn.CrossEntropyLoss().to(self.device) 
        else:
            class_weights = compute_class_weights(self.val_data, network_model.n_classes) # Classes weighted by their frequency in the dataset
            print("Class weights: ", class_weights)
            if use_gpu:
                class_weights = class_weights.to(self.device)
            self.loss_criterion = nn.CrossEntropyLoss(weight=class_weights).to(
                self.device
            ) 

        # Generate the directory names
        self.experiment_folder = os.path.abspath(experiment_name)
        self.experiment_logs = os.path.abspath(
            os.path.join(self.experiment_folder, "result_outputs")
        )
        self.experiment_saved_models = os.path.abspath(
            os.path.join(self.experiment_folder, "saved_models")
        )

        # Set best models to be at 0 since we are just starting
        self.best_val_model_idx = 0
        self.best_val_model_acc = 0.0

        if not os.path.exists(
            self.experiment_folder
        ):  # If experiment directory does not exist
            os.mkdir(self.experiment_folder)  # create the experiment directory
            os.mkdir(self.experiment_logs)  # create the experiment log directory
            os.mkdir(
                self.experiment_saved_models
            )  # create the experiment saved models directory

        self.num_epochs = num_epochs
        

        if (
            continue_from_epoch == -2
        ):  # if continue from epoch is -2 then continue from latest saved model
            self.state, self.best_val_model_idx, self.best_val_model_acc = (
                self.load_model(
                    model_save_dir=self.experiment_saved_models,
                    model_save_name="train_model",
                    model_idx="latest",
                )
            )  # reload existing model from epoch and return best val model index
            # and the best val acc of that model
            self.starting_epoch = int(self.state["model_epoch"])

        elif continue_from_epoch > -1:  # if continue from epoch is greater than -1 then
            self.state, self.best_val_model_idx, self.best_val_model_acc = (
                self.load_model(
                    model_save_dir=self.experiment_saved_models,
                    model_save_name="train_model",
                    model_idx=continue_from_epoch,
                )
            )  # reload existing model from epoch and return best val model index
            # and the best val acc of that model
            self.starting_epoch = continue_from_epoch
        else:
            self.state = dict()
            self.starting_epoch = 0

    def get_num_parameters(self):
        total_num_params = 0
        for param in self.parameters():
            total_num_params += np.prod(param.shape)

        return total_num_params

    def plot_func_def(self, all_grads, layers, epoch):
        """
        Plot function definition to plot the average gradient with respect to the number of layers in the given model
        :param all_grads: Gradients wrt weights for each layer in the model.
        :param layers: Layer names corresponding to the model parameters
        :return: plot for gradient flow
        """
        
        colormap = matplotlib.cm.get_cmap("viridis") 
        color = colormap(epoch / self.num_epochs) 
        plt.plot(all_grads, alpha=0.7, color=color)
        plt.hlines(0, 0, len(all_grads) + 1, linewidth=1, color="k")
        plt.xticks(range(0, len(all_grads), 1), layers, rotation="vertical")
        plt.xlim(xmin=0, xmax=len(all_grads))
        plt.xlabel("Layers")
        plt.ylabel("Average Gradient")
        plt.title("Gradient Flow")
        plt.grid(True)
        plt.tight_layout()

        return plt

    def plot_grad_flow(self, named_parameters):
        """
        The function is being called in Line 298 of this file.
        Receives the parameters of the model being trained. Returns plot of gradient flow for the given model parameters.

        This function takes as
        input the model parameters during training, accumulates the absolute mean of the gradients in all_grads and
        the layer names in layers. The matplotlib function plt plots gradient values for each layer and the function
        plot_grad_flows() returns this final plot

        """
        all_grads = []
        layers = []

        """
        Complete the code in the block below to collect absolute mean of the gradients for each layer in all_grads with the             
        layer names in layers.
        """
        for name, param in named_parameters:
            if param.requires_grad and param.grad is not None and "bias" not in name:
                # Compute the absolute mean of the gradient
                grad_mean = param.grad.abs().mean().item()
                all_grads.append(grad_mean)
                modified_name = (
                    name.replace("layer_dict.", "")
                    .replace(".weight", "")
                    .replace(".", "_")
                )
                layers.append(modified_name)
        epoch = self.current_epoch
        plt = self.plot_func_def(all_grads, layers,epoch)

        return plt

    def iou_score(self, y_pred, y_true):

        # Convert predictions to class indices
        y_pred = torch.argmax(y_pred, dim=1)

        # Flatten the tensors
        y_pred_flat = y_pred.view(-1)
        y_true_flat = y_true.view(-1)

        # Compute intersection and union
        intersection = torch.sum(y_pred_flat == y_true_flat).item()
        union = len(y_pred_flat) + len(y_true_flat) - intersection

        # Compute IoU
        iou = intersection / union if union != 0 else 0.0

        return iou

    def run_train_iter(self, x, y, prompt):

        self.train()  # sets model to training mode (in case batch normalization or other methods have different procedures for training and evaluation)
        x, y, prompt = x.to(device=self.device), y.to(device=self.device), prompt.to(device=self.device)   # send data to device as torch tensors
        out = self.model(x, prompt) if self.model_name == "PromptUNet" else self.model(x) # forward the data in the model


        
        loss = self.loss_criterion(out, y)  # compute loss

        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        loss.backward()  # backpropagate to compute gradients for current iter loss

        self.optimizer.step()  # update network parameters
        self.learning_rate_scheduler.step()  # update learning rate scheduler

        # Compute Intersection over Union
        iou = self.iou_score(out, y)  # get iou score for current iter

        return loss.item(), iou

    def run_evaluation_iter(self, x, y, prompt):

        self.eval()  # sets the system to validation mode
        x, y, prompt = x.to(device=self.device), y.to(device=self.device), prompt.to(device=self.device)   # send data to device as torch tensors
        out = self.model(x, prompt) if self.model_name == "PromptUNet" else self.model(x) # forward the data in the model

        loss = self.loss_criterion(out, y)  # compute loss

        # Compute Intersection over Union
        iou = self.iou_score(out, y)  # get iou score for current iter

        return loss.item(), iou

    def save_model(
        self,
        model_save_dir,
        model_save_name,
        model_idx,
        best_validation_model_idx,
        best_validation_model_acc,
    ):
        """
        Save the network parameter state and current best val epoch idx and best val accuracy.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :param best_validation_model_idx: The index of the best validation model to be stored for future use.
        :param best_validation_model_acc: The best validation accuracy to be stored for use at test time.
        :param model_save_dir: The directory to store the state at.
        :param state: The dictionary containing the system state.

        """
        self.state["network"] = (
            self.state_dict()
        )  # save network parameter and other variables.
        self.state["best_val_model_idx"] = (
            best_validation_model_idx  # save current best val idx
        )
        self.state["best_val_model_acc"] = (
            best_validation_model_acc  # save current best val acc
        )
        torch.save(
            self.state,
            f=os.path.join(
                model_save_dir, "{}_{}".format(model_save_name, str(model_idx))
            ),
        )  # save state at prespecified filepath

    def load_model(self, model_save_dir, model_save_name, model_idx):
        """
        Load the network parameter state and the best val model idx and best val acc to be compared with the future val accuracies, in order to choose the best val model
        :param model_save_dir: The directory to store the state at.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :return: best val idx and best val model acc, also it loads the network state into the system state without returning it
        """
        state = torch.load(
            f=os.path.join(
                model_save_dir, "{}_{}".format(model_save_name, str(model_idx))
            )
        )
        self.load_state_dict(state_dict=state["network"])
        return state, state["best_val_model_idx"], state["best_val_model_acc"]

    def run_experiment(self):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model and val model accuracy after each epoch
        :return: The summary current_epoch_losses from starting epoch to total_epochs.
        """
        ruonceflag=True
        total_losses = {
            "train_iou": [],
            "train_loss": [],
            "val_iou": [],
            "val_loss": [],
        }  # initialize a dict to keep the per-epoch metrics
        for i, epoch_idx in enumerate(range(self.starting_epoch, self.num_epochs)):
            epoch_start_time = time.time()
            current_epoch_losses = {
                "train_iou": [],
                "train_loss": [],
                "val_iou": [],
                "val_loss": [],
            }
            self.current_epoch = epoch_idx
            with tqdm.tqdm(
                total=len(self.train_data)
            ) as pbar_train:  # create a progress bar for training
                for idx, (images, masks, prompt) in enumerate(self.train_data):  # get data batches
                    loss, iou = self.run_train_iter(
                        x=images, y=masks, prompt=prompt
                    )  # take a training iter step
                    current_epoch_losses["train_loss"].append(
                        loss
                    )  # add current iter loss to the train loss list
                    current_epoch_losses["train_iou"].append(
                        iou
                    )  # add current iter acc to the train acc list
                    pbar_train.update(1)
                    pbar_train.set_description(
                        "loss: {:.4f}, iou: {:.4f}".format(loss, iou)
                    )

            with tqdm.tqdm(
                total=len(self.val_data)
            ) as pbar_val:  # create a progress bar for validation
                for x, y, prompt  in self.val_data:  # get data batches
                    loss, iou = self.run_evaluation_iter(
                        x=x, y=y, prompt=prompt
                    )  # run a validation iter
                    current_epoch_losses["val_loss"].append(
                        loss
                    )  # add current iter loss to val loss list.
                    current_epoch_losses["val_iou"].append(
                        iou
                    )  # add current iter acc to val acc lst.
                    pbar_val.update(1)  # add 1 step to the progress bar
                    pbar_val.set_description(
                        "loss: {:.4f}, iou: {:.4f}".format(loss, iou)
                    )
            val_mean_accuracy = np.mean(current_epoch_losses["val_iou"])
            if (
                val_mean_accuracy > self.best_val_model_acc
            ):  # if current epoch's mean val acc is greater than the saved best val acc then
                self.best_val_model_acc = val_mean_accuracy  # set the best val model acc to be current epoch's val accuracy
                self.best_val_model_idx = epoch_idx  # set the experiment-wise best val idx to be the current epoch's idx

            for key, value in current_epoch_losses.items():
                total_losses[key].append(
                    np.mean(value)
                )  # get mean of all metrics of current epoch metrics dict, to get them ready for storage and output on the terminal.

            save_statistics(
                experiment_log_dir=self.experiment_logs,
                filename="summary.csv",
                stats_dict=total_losses,
                current_epoch=i,
                continue_from_mode=(
                    True if (self.starting_epoch != 0 or i > 0) else False
                ),
            )  # save statistics to stats file.

            # load_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv') # How to load a csv file if you need to

            out_string = "_".join(
                [
                    "{}_{:.4f}".format(key, np.mean(value))
                    for key, value in current_epoch_losses.items()
                ]
            )
            # create a string to use to report our epoch metrics
            epoch_elapsed_time = (
                time.time() - epoch_start_time
            )  # calculate time taken for epoch
            epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
            print(
                "Epoch {}:".format(epoch_idx),
                out_string,
                "epoch time",
                epoch_elapsed_time,
                "seconds",
            )
            self.state["model_epoch"] = epoch_idx
            self.save_model(
                model_save_dir=self.experiment_saved_models,
                # save model and best val idx and best val acc, using the model dir, model name and model idx
                model_save_name="train_model",
                model_idx=epoch_idx,
                best_validation_model_idx=self.best_val_model_idx,
                best_validation_model_acc=self.best_val_model_acc,
            )
            self.save_model(
                model_save_dir=self.experiment_saved_models,
                # save model and best val idx and best val acc, using the model dir, model name and model idx
                model_save_name="train_model",
                model_idx="latest",
                best_validation_model_idx=self.best_val_model_idx,
                best_validation_model_acc=self.best_val_model_acc,
            )

            ################################################################
            ##### Plot Gradient Flow at each Epoch during Training  ######
            print("Generating Gradient Flow Plot at epoch {}".format(epoch_idx))
            plt = self.plot_grad_flow(self.model.named_parameters()) # ADD EPOCH LEGEND HERE

            ### Adding a colorbar to the plot to show the epochs
            if ruonceflag==True:
                numepochs = self.num_epochs
                colormap = matplotlib.cm.get_cmap("viridis") 
                norm = Normalize(vmin=1, vmax=numepochs)
                sm = ScalarMappable(cmap=colormap, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=plt.gca(), aspect=30, pad=0.02)
                cbar.set_label("Epochs")
                cbar.set_ticks([1, numepochs])
                cbar.set_ticklabels(["Epoch 1", f"Epoch {numepochs}"])
                ruonceflag=False
            ###


            if not os.path.exists(
                os.path.join(self.experiment_saved_models, "gradient_flow_plots")
            ):
                os.mkdir(
                    os.path.join(self.experiment_saved_models, "gradient_flow_plots")
                )
                # plt.legend(loc="best")
            print(
                "save_loc: ",
                os.path.join(
                    self.experiment_saved_models,
                    "gradient_flow_plots",
                    "epoch{}.pdf".format(str(epoch_idx)),
                ),
            )
            plt.savefig(
                os.path.join(
                    self.experiment_saved_models,
                    "gradient_flow_plots",
                    "epoch{}.pdf".format(str(epoch_idx)),
                )
            )
            ################################################################

        print("Generating test set evaluation metrics")
        self.load_model(
            model_save_dir=self.experiment_saved_models,
            model_idx=self.best_val_model_idx,
            # load best validation model
            model_save_name="train_model",
        )
        current_epoch_losses = {
            "test_iou": [],
            "test_loss": [],
        }  # initialize a statistics dict
        with tqdm.tqdm(total=len(self.test_data)) as pbar_test:  # ini a progress bar
            for x, y, prompt in self.test_data:  # sample batch
                loss, iou = self.run_evaluation_iter(
                    x=x, y=y, prompt=prompt
                )  # compute loss and iou by running an evaluation step
                current_epoch_losses["test_loss"].append(loss)  # save test loss
                current_epoch_losses["test_iou"].append(iou)  # save test iou
                pbar_test.update(1)  # update progress bar status
                pbar_test.set_description(
                    "loss: {:.4f}, iou: {:.4f}".format(loss, iou)
                )  # update progress bar string output

        test_losses = {
            key: [np.mean(value)] for key, value in current_epoch_losses.items()
        }  # save test set metrics in dict format
        save_statistics(
            experiment_log_dir=self.experiment_logs,
            filename="test_summary.csv",
            # save test set metrics on disk in .csv format
            stats_dict=test_losses,
            current_epoch=0,
            continue_from_mode=False,
        )

        return total_losses, test_losses
