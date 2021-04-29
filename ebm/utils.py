## Standard libraries
import os
import numpy as np 
from tqdm.notebook import tqdm
from IPython.display import clear_output
import shutil

## Imports for plotting
import matplotlib.pyplot as plt
from matplotlib import cm

## PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

# Torchvision
import torchvision
from torchvision.utils import make_grid

# Custom modules
from ebm.config import *
from ebm.models import CNNModel


def scheduler_stats(model, mcmc_steps_schedule, train_set_len, target_iters):
    batches_per_epoch = int(train_set_len / model.batch_size)
    print(f"batches_per_epoch: {batches_per_epoch}")
    target_iterations = target_iters
    epochs_2_target = int(np.ceil(target_iterations / batches_per_epoch))
    print(f"epochs_2_target: <= {epochs_2_target}")
    effective_tot_iters = epochs_2_target * batches_per_epoch
    print(f"effective_tot_iters: {effective_tot_iters}")
    area = 0
    for i in range(len(mcmc_steps_schedule)):
        if i == 0:
            prev = 0
        else:
            prev = mcmc_steps_schedule[i-1][0]
        area += (mcmc_steps_schedule[i][0] - prev) * mcmc_steps_schedule[i][1]
    area += (effective_tot_iters - mcmc_steps_schedule[i][0]) * mcmc_steps_schedule[i][1]
    avg_mcmc_steps_per_iter = area / effective_tot_iters
    print(f"avg_mcmc_steps_per_iter: {avg_mcmc_steps_per_iter:.1f}")
    return epochs_2_target


class DeepEnergyModel:
    """
    model_name (str) - Any name to visually recognize the model, like the #run.
    model_description (str) - Will be logged by tensorboard as "text"
    model_family (str) - When running multiple experiments, it may be useful to divide
        the models and their logged results in families (subdirs of checkpoint path).
        This param can have the form of a path/to/subfolder.
    overwrite (bool) - If the logs folder already exists, if True "overwrite" it (namely, 
        add also the new logs, without removing the onld ones).
    """
    def __init__(self,
                 img_shape,
                 batch_size,
                 alpha=1,
                 lr=1e-4,
                 weight_decay=1e-4,
                 mcmc_step_size=1e-5,
                 mcmc_steps=250,
                 model_name="unnamed",
                 model_description="",
                 model_family="Langevin_vanilla",
                 device="cuda:1",
                 overwrite=False,
                 **CNN_args):
        super().__init__()

        # Model
        self.img_shape = img_shape
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        print("Running on device:", self.device) 
        # Use CNNModel by default
        self.cnn = CNNModel(**CNN_args).to(self.device)

        # Optimizers
        self.lr = lr
        self.weight_decay = weight_decay

        # Reg loss weigth
        self.alpha = alpha

        # Dataset
        self.batch_size = batch_size

        # MCMC
        self.mcmc_step_size = mcmc_step_size
        self.mcmc_steps = mcmc_steps
        self.sigma_sq_noise = 1

        # Logging
        # General purpose: add new element each iteration (batch)
        self.log_dict = dict()
        # MCMC sampling: add element each MCMC iteration
        self.mcmc_evolution_logs = dict()
        # Final sample of generated images
        self.final_sampled_images = None

        # Tensorboard
        self.model_name = model_name
        self.model_description = model_description
        self.model_family = model_family
        self.overwrite = overwrite
        
        # The following global variables are employed in different functions (in
        # different ways) to compute the SummaryWriter global_step.
        self.epoch_n = 0
        self.tot_batches = 0
        self.iter_n = 0
        
        # Convert mcmc_steps to string if it's a list of tuples (veriable mcmc steps)
        if not isinstance(mcmc_steps, int):
            # Write in the hparams dict a brief string.
            conv_mcmc_steps = "schedule"
        else:
            conv_mcmc_steps = mcmc_steps
        # Hyperparams dict
        self.hparams_dict = {
            'mcmc_step_size': self.mcmc_step_size,
            'mcmc_steps': conv_mcmc_steps,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'optimizer': 'Adam',
            'alpha': self.alpha
        }
        
        # Setup flag: check the model has been properly set up before starting
        self.is_setup = False

    def setup(self):
        """Setup the optimizers, setup the Tensorboard SummaryWriter, process hyperparams dict."""
        # Optimizers
        self.configure_optimizers()
        
        # Tensorboard logs
        hparams_str = "__".join(["=".join(
            [str(el) for el in dict_entry]) for dict_entry in self.hparams_dict.items()])
        full_name = self.model_name + "__" + hparams_str
        self.ckpt_path = os.path.join(CHECKPOINT_PATH, self.model_family, full_name)
        if os.path.exists(self.ckpt_path) and not self.overwrite:
            print("Model path: " + self.ckpt_path)
            raise NameError("Model already exists! Set self.overwrite=True to overwrite it.")
        elif os.path.exists(self.ckpt_path) and self.overwrite:
            # Remove existsing folder
            shutil.rmtree(self.ckpt_path)
            print("Overwritten existsing logs")
        # Create writer
        self.tb_writer = SummaryWriter(self.ckpt_path)
        # Add some textual notes
        # 1. Add docstring to interpret logs
        self.tb_writer.add_text('logs_documentation', self.tb_logs_doc())
        # 2. Add mcmc steps scheduler, if present
        mcmc_steps_schedule = ""
        if not isinstance(self.mcmc_steps, int):
            mcmc_steps_schedule = "Schedule of MCMC steps:\n\n"
            mcmc_steps_schedule += "-".join([f"{it}:{st}" for it,st in self.mcmc_steps])
        descr = "Model description:\n" + self.model_description + "\n\n\n"
        self.tb_writer.add_text("model_description", descr + mcmc_steps_schedule, 100)
        
        # Set is_setup flag to True
        self.is_setup = True
        
    def clear(self):
        # Tensorboard writer
        self.tb_writer.close()
        
        # TO-DO: remove cumbersome data structures
        # ...
        
    def configure_optimizers(self):
        # Energy models can have issues with momentum as the loss surfaces changes with its parameters.
        # Hence, we set it to 0 by default.
        # Optimize only the layers that require grad
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.cnn.parameters()),
                                    lr=self.lr,
                                    weight_decay=self.weight_decay)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.97) # Exponential decay over epochs
        pass

    def prepare_data(self, train_set, test_set):
        # Prepare data
        self.train_loader = data.DataLoader(train_set,
                                            batch_size=self.batch_size,
                                            shuffle=True,
                                            drop_last=True,
                                            num_workers=2,
                                            pin_memory=True)
        self.test_loader = data.DataLoader(test_set,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           drop_last=False,
                                           num_workers=2)
        pass

    ######################################################
    ################ Training section ####################
    ######################################################

    def training_step(self, batch):

        # Train mode
        self.cnn.train()

        # We add minimal noise to the original images to prevent the model from focusing on purely "clean" inputs
        real_imgs, _ = batch
        real_imgs = real_imgs.to(self.device)

        #small_noise = torch.randn_like(real_imgs) * 0.005
        #real_imgs.add_(small_noise).clamp_(min=-1.0, max=1.0)

        # Obtain samples
        fake_imgs = self.generate_samples()

        # Predict energy score for all images
        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        real_out, fake_out = self.cnn(inp_imgs).chunk(2, dim=0)

        # Calculate losses
        cdiv_loss = real_out.mean() - fake_out.mean()
        if self.alpha > 0:
            reg_loss = self.alpha * (real_out**2 + fake_out**2).mean()
            loss = reg_loss + cdiv_loss
        else:
            reg_loss = torch.tensor(0)
            loss = cdiv_loss

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Logging
        self.log('loss', loss)
        self.log('loss_reg', reg_loss)
        self.log('loss_cdiv', cdiv_loss)
        self.log('energy_avg_real', real_out.mean())
        self.log('energy_avg_fake', fake_out.mean())

        # Log layers weigth / bias norms
        for layer_id, layer in enumerate(self.cnn.cnn_layers):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                self.log('layer%d_weight_norm' % layer_id,
                         torch.norm(
                             layer.weight).clone().detach().cpu().numpy(),
                         printable=False)
                self.log('layer%d_bias_norm' % layer_id,
                         torch.norm(layer.bias).clone().detach().cpu().numpy(),
                         printable=False)

        # Free memory
        del real_imgs, fake_imgs, inp_imgs

    def fit(self, n_epochs=None):
        
        assert self.is_setup, "Model is not properly setup. Call .setup() before running!"

        if self.train_loader is None:
            print("Train data not loaded")
            return

        # Epochs
        self.tot_batches = len(self.train_loader)
        for self.epoch_n in range(n_epochs):
            clear_output()
            print("Epoch #" + str(self.epoch_n + 1))

            # Iterations
            self.log_active = True
            for self.iter_n, batch in tqdm(enumerate(self.train_loader),
                                           total=self.tot_batches,
                                           position=0,
                                           leave=True):

                self.training_step(batch)

            ############## Tensorboard ###############
            # Evolution thoughout a mcmc simulation
            self.tb_mcmc_simulation()

            # Log a generaed sample of images
            self.tb_mcmc_images(batch_size=25, evaluation=True)

            # Force tensorboard to write to disk (to be sure)
            self.tb_writer.flush()

            ############# Other logs ################
            # Plot evolution of gradients and parameters norms
            #self.plot_epoch_evolution()

            # Print logged measures
            self.flush_logs()
            
            # Save model state dict (params)
            self.save_model()

        # TB: Log a final batch of images sampled form the model
        mcmc_iter = 1000
        self.final_sampled_images = self.tb_mcmc_images(
            batch_size=64, mcmc_steps=mcmc_iter, name="final_images_sample", evaluation=True)
        # Plot them
        print("Final sample after %d mcmc iterations:" % mcmc_iter)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.final_sampled_images.permute(1, 2, 0))
        plt.show()

    ######################################################
    ################ Langevin dynamics ###################
    ######################################################
    

    def generate_samples(self,
                         evaluation=False,
                         batch_size=None,
                         mcmc_steps=None):
        """
        Draw samples using Langevin dynamics
        evaluation: if True, avoids logging mcmc stats. It means we're sampling 
        from the model with arbitrary batchsize/mcmc_steps and it isn't related to training.
        noise_scale: Optional. float. If None, set to np.sqrt(step_size * 2)
        """
        batch_size = self.batch_size if batch_size is None else batch_size
        mcmc_steps = self.mcmc_steps if mcmc_steps is None else mcmc_steps
        if not isinstance(mcmc_steps, int):
            # mcmc steps not fixed, but using a scheduler of the form
            # [(up_to_iter_i, mcmc_steps), (up_to_iter_j, mcmc_steps), ...]
            global_iter = self.epoch_n * self.tot_batches + self.iter_n
            curr_steps = None
            for max_iter,steps in mcmc_steps:
                if max_iter > global_iter:
                    break
            mcmc_steps = steps
        
        is_training = self.cnn.training
        self.cnn.eval()

        # Init images with RND normal noise: x_i ~ N(0,1)
        x = torch.randn((batch_size, ) + self.img_shape, device=self.device)
        original_x = x.clone().detach()
        x.requires_grad = True
        
        noise_scale = np.sqrt(self.mcmc_step_size * 2)
        
        # Pre-allocate additive noise (for Langevin step)
        noise = torch.randn_like(x, device=self.device)

        
        def append_norm(in_tensor, array):
                return np.append(
                array,
                torch.norm(in_tensor,
                           dim=[2, 3]).mean().clone().detach().cpu().numpy())

        grad_norms = np.array([])
        data_norms = np.array([])

        # To study the evolution within an mcmc simulation
        distances = np.array([])
        prev_distances = np.array([])
        time_window = 50

        for _ in range(mcmc_steps):

            if self.iter_n < time_window:
                #Used to compute prev_distances items
                old_x = x.clone().detach()
            
            # x.data.clamp_(min=-1.0, max=1.0)

            # Re-init noise tensor
            noise.normal_(mean=0.0, std=noise_scale)
            out = self.cnn(x)
            grad = autograd.grad(out.sum(), x, only_inputs=True)[0]
            # grad is in "device" by default

            ##### Normalize gradient with Frobenius norm (~smaller tha the unit sphere)######
            #n_pixels = img_shape[0] * img_shape[1]
            #k = torch.tensor(torch.sqrt(n_pixels) / torch.norm(grad, dim=[2, 3]).mean())
            # grad.data.multiply_(k)

            # Avoid clamping
            # grad.data.clamp_(-0.03, 0.03) 

            dynamics = self.mcmc_step_size * grad + noise
            x = x - dynamics

            # Save stats            
            grad_norms = append_norm(grad, grad_norms)
            data_norms = append_norm(x, data_norms)

            if self.iter_n < time_window:
                prev_distances = append_norm(x - old_x, prev_distances)
                distances = append_norm(x - original_x, distances)

        self.cnn.train(is_training)

        ####### Evolution within Langevin dynamics ######
        # If at the beginning of an epoch, save the evolution of
        # grad and img norms, for a time window of width K.
        # These quantities will be logged within fit() function

        def append_mcmc_logs(prop_name, prop_array):
            full_name = "%s_epoch_%d" % (prop_name, self.epoch_n + 1)
            entry = self.mcmc_evolution_logs.get(full_name, None)
            if entry is None:
                self.mcmc_evolution_logs[full_name] = prop_array
            else:
                self.mcmc_evolution_logs[full_name] = np.vstack(
                    (entry, prop_array))
            return
        
        if not evaluation:
            # Beginning of epoch e
            # 'langevin_evolution_' metrics describe the evolution 
            # within a mcmc sampling process. Computed over a time_window of iterations.
            if self.iter_n < time_window:
                # Gradient norm
                append_mcmc_logs("langevin_evolution_grad_norm", grad_norms)

                # Data norm
                append_mcmc_logs("langevin_evolution_img_norm", data_norms)

                # Distance from previous point
                append_mcmc_logs("langevin_evolution_distance2prevstep", prev_distances)
                
                # Distance from starting point
                append_mcmc_logs("langevin_evolution_distance2start", distances)


            # Always log the avg
            # 'langevin_avg_' metrics describe the avg value of a measure
            # within a mcmc sampling process. Computed at each iteration.
            self.log('langevin_avg_grad_norm', np.mean(grad_norms))
            self.log('langevin_avg_img_norm', np.mean(data_norms))
            e2e_distances = torch.norm(
                x - original_x, dim=[2, 3]).mean().clone().detach().cpu().numpy()
            self.log('langevin_avg_distance_start2end', e2e_distances)

        return x.detach()
    
    def normalize_batch(self, batch):
        """normalizes a images batch of size BxCxWxH in [-1,1], as MNIST images"""
        img_area = self.img_shape[1] * self.img_shape[2]
        batch -= batch.view(-1,img_area).min(axis=1)[0].view(-1,1,1,1)
        batch /= batch.view(-1,img_area).max(axis=1)[0].view(-1,1,1,1)
        # From [0,1] to [-1,1]
        batch = batch * 2 - 1 
        return batch
    ######################################################
    #################### Utilities #######################
    ######################################################
    
    def save_model(self):
        """Saves the state dict of the model"""
        torch.save(self.cnn.state_dict(), self.ckpt_path + "/model_state_dict.pt")

    def tb_mcmc_simulation(self):
        """
        This function writes to tensorboard the evolution of a 
        measure duing MCMC simulation. We have an array of misurations,
        each one obtained at an iteration of the mcmc method.
        K measurments are collected and the resulting arrays are vertically
        stacked, to obtain a matrix. For this reason, the mean is obtained by 
        averaging on the 0 axis.
        """
        # In this dict there are only 2D arrays!
        for name, array in self.mcmc_evolution_logs.items():
            if array.ndim != 2:
                raise NameError("expected 2-dimensional array here!")
            array = array.mean(axis=0)
            for i in range(array.shape[0]):
                self.tb_writer.add_scalar(name, array[i], i)
        # Free
        del self.mcmc_evolution_logs
        self.mcmc_evolution_logs = dict()

    def tb_mcmc_images(self, name=None, batch_size=None, **MCMC_args):
        """
        Generate B images from the currently learned model and add them as 
        images grid to tensorboard.
        """
        img_name = "sample_images_epoch_%d" % (self.epoch_n +
                                               1) if name is None else name
        batch_size = self.batch_size if batch_size is None else batch_size
        fake_imgs = self.generate_samples(batch_size=batch_size, **MCMC_args)
        grid_img = make_grid(fake_imgs.clone().detach().cpu(),
                             nrow=int(np.sqrt(batch_size)),
                             normalize=True,
                             range=(0, 1))
        g_step = self.epoch_n * self.tot_batches + self.iter_n
        self.tb_writer.add_image(img_name, grid_img, g_step)
        return grid_img

    def downsample(self, l, downsampling_perc_=0.05):
        """Downsample a list of measurements"""
        l = np.array(l)
        step_ = int(np.ceil(l.shape[0] * downsampling_perc_))
        l = l[::step_]
        x = step_ * np.array(range(l.shape[0]))
        return x, l

    def plot_epoch_evolution(self):
        """
        Plots the charts of MCMC gradient and images norm, model parameters norms and losses
        """
        FIGSIZE = (14, 20)
        fig, ax = plt.subplots(figsize=FIGSIZE,
                               nrows=3,
                               ncols=1,
                               gridspec_kw={'hspace': 0.3})

        # MCMC stats
        grad_norms, _ = self.log_dict.get('langevin_avg_grad_norm', (None, None))
        data_norms, _ = self.log_dict.get('langevin_avg_img_norm', (None, None))
        if grad_norms is not None and data_norms is not None:
            x_g, grad_norms = self.downsample(grad_norms)
            x_d, data_norms = self.downsample(data_norms)

            color = 'tab:red'
            ax[0].set_xlabel('Iterations (batches)', fontsize=14)
            ax[0].set_ylabel('grad norm', color=color, fontsize=14)
            ax[0].plot(x_g, grad_norms, color=color)
            ax[0].tick_params(axis='y', labelcolor=color)

            ax2 = ax[0].twinx()

            color = 'tab:blue'
            ax2.set_ylabel('img norm', color=color, fontsize=14)
            ax2.plot(x_d, data_norms, color=color)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_title("Langevin dynamics", fontsize=16)

        # Weigths / biases stats
        for layer_id, layer in enumerate(self.cnn.cnn_layers):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                # Weights
                layer_name = 'layer%d_weight_norm' % layer_id
                layer_weights, _ = self.log_dict.get(layer_name, (None, None))
                if layer_weights is not None:
                    x_w, layer_weights = self.downsample(layer_weights)
                    ax[1].plot(x_w, layer_weights, label=layer_name)

                # Biases
                layer_name = 'layer%d_bias_norm' % layer_id
                layer_biases, _ = self.log_dict.get(layer_name, (None, None))
                if layer_weights is not None:
                    x_b, layer_biases = self.downsample(layer_biases)
                    ax[1].plot(x_b, layer_biases, label=layer_name)

        ax[1].set_title("Model parameters", fontsize=16)
        ax[1].set_xlabel('Iterations (batches)', fontsize=14)
        ax[1].set_ylabel('Norm', fontsize=14)
        ax[1].legend()

        # Losses
        losses_to_plot = [
            'loss', 'loss_cdiv', 'loss_reg', 'energy_avg_real',
            'energy_avg_fake'
        ]
        loss_markers = ['x', '-', '-', '-', '-']
        if self.alpha == 0:
            reg_id = losses_to_plot.index('loss_reg')
            del losses_to_plot[reg_id]
            del loss_markers[reg_id]

        for loss_name, marker in zip(losses_to_plot, loss_markers):
            l, _ = self.log_dict.get(loss_name, (None, None))
            if l is not None:
                x_l, l = self.downsample(l)
                ax[2].plot(x_l, l, marker, label=loss_name)
        ax[2].set_title("Losses & energies", fontsize=16)
        ax[2].set_xlabel('Iterations (batches)', fontsize=14)
        ax[2].legend()

        fig.tight_layout
        plt.show()

    def log(self, name, val, printable=True):
        """
        name: string name of the property to log
        val: value
        print: whether to print this quantity or not. If false the quantity is just for "intermediate" use by another function.
        """
        if not self.log_active:
            return

        # Parse the value to log
        if isinstance(val, torch.Tensor):
            if val.dim() == 0:
                # Single element tensor (e.g. loss)
                payload = val.item()
            else:
                # Mupliple dimensions tensor (e.g. vector)
                payload = val.numpy(
                )  # Fine also for 1 element tensors, instead of .item()
        else:
            payload = val

        # Add the value to the logs list
        if self.log_dict.get(name, None) is None:
            self.log_dict[name] = ([payload], printable)
        else:
            self.log_dict[name][0].append(payload)

        # Add to tensorboard
        global_step = self.epoch_n * self.tot_batches + self.iter_n
        self.tb_writer.add_scalar(name, payload, global_step=global_step)

    def flush_logs(self):
        """
        Called each epoch.
        Print the average of the current logged measure and remove it from the dict
        """
        for name, (measures_list, printable) in self.log_dict.items():
            if printable:
                print(f"{name}: {np.mean(np.array(measures_list)):.3f}")
        print()

        # Clean the active logs dictionary
        del self.log_dict
        self.log_dict = dict()
    
    def tb_logs_doc(self):
        return """
        Documentation of Tensorboard logs
        
        'langevin_evolution_' metrics describe the evolution within 
        a mcmc sampling process. 
        E.g. the norm of the generated images at each mcmc step: it's
        an array.
        Computed over a `time_window` of first K iterations of an epoch.
        
        'langevin_avg_' metrics describe the avg value of a measure
        within a mcmc sampling process. 
        E.g. the *avg* norm of the generated images at each mcmc step:
        it's a scalar. 
        Computed at each iteration.
        
        'energy_avg_': avg energy of real/fakes images at current iteration.
        
        'loss': can be `loss`, `loss_cdiv`, `loss_reg` (regularization loss, weigthed
        by alpha hparam).
        
        'layer_': norm of weights/biases of a given layer
        """

    ######################################################
    ################# Validation step ####################
    ######################################################

    def validation_step(self, batch, batch_idx):
        # For validating, we calculate the contrastive divergence between purely random images and unseen examples
        # Note that the validation/test step of energy-based models depends on what we are interested in the model
        self.cnn.eval()

        real_imgs, _ = batch
        real_imgs = real_imgs.to(self.device)

        #fake_imgs = torch.randn_like(real_imgs)
        fake_imgs = self.generate_samples()

        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        real_out, fake_out = self.cnn(inp_imgs).chunk(2, dim=0)

        cdiv = real_out.mean() - fake_out.mean()
        self.log('val_loss', cdiv)
        self.log('val_fake_out', fake_out.mean())
        self.log('val_real_out', real_out.mean())

    def validate(self, n_epochs=None):
        """To-Do: finish implementation of this"""
        if self.test_loader is None:
            print("Test data not loaded")
            return

        # Iterations
        self.log_active = True
        for batch in tqdm(self.test_loader,
                          total=len(self.test_loader),
                          position=0,
                          leave=True):
            self.training_step(batch)
            
            
            

            
            
            
            
class EBMLangVanilla(DeepEnergyModel):
    """"Vanilla Langevin Dynamics"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class EBMLang2Ord(DeepEnergyModel):
    """Second order Langevin Dynamics, with leapfrog"""
    def __init__(self, C=2, mass=1, **kwargs):
        super().__init__(**kwargs)
        self.C = C
        self.hparams_dict['C'] = C
        self.mass = mass
        self.hparams_dict['mass'] = mass
    
    def generate_samples(self,
                         evaluation=False,
                         batch_size=None,
                         mcmc_steps=None):
        """
        Draw samples using Langevin dynamics
        evaluation: if True, avoids logging mcmc stats. It means we're sampling 
        from the model with arbitrary batchsize/mcmc_steps and it isn't related to training.
        noise_scale: Optional. float. If None, set to np.sqrt(step_size * 2)
        """
        batch_size = self.batch_size if batch_size is None else batch_size
        mcmc_steps = self.mcmc_steps if mcmc_steps is None else mcmc_steps
        if not isinstance(mcmc_steps, int):
            # mcmc steps not fixed, but using a scheduler of the form
            # [(up_to_iter_i, mcmc_steps), (up_to_iter_j, mcmc_steps), ...]
            global_iter = self.epoch_n * self.tot_batches + self.iter_n
            curr_steps = None
            for max_iter,steps in mcmc_steps:
                if max_iter > global_iter:
                    break
            mcmc_steps = steps
        
        is_training = self.cnn.training
        self.cnn.eval()

        # Init images with RND normal noise: x_i ~ N(0,1)
        x = torch.randn((batch_size, ) + self.img_shape, device=self.device)
        original_x = x.clone().detach()
        x.requires_grad = True
        
        # Init momentum
        #momentum = torch.randn((batch_size, ) + self.img_shape, device=self.device)
        momentum = torch.zeros_like(x, device=self.device)
        noise_scale = np.sqrt(self.mcmc_step_size * 2 * self.C)
        
        # Pre-allocate additive noise (for Langevin step)
        noise = torch.randn_like(x, device=self.device)

        
        def append_norm(in_tensor, array):
                return np.append(
                array,
                torch.norm(in_tensor,
                           dim=[2, 3]).mean().clone().detach().cpu().numpy())

        grad_norms = np.array([])
        data_norms = np.array([])
        momentum_norms = np.array([])

        # To study the evolution within an mcmc simulation
        distances = np.array([])
        prev_distances = np.array([])
        time_window = 50

        for _ in range(mcmc_steps):

            if self.iter_n < time_window:
                #Used to compute prev_distances items
                old_x = x.clone().detach()
            
            # Re-init noise tensor
            noise.normal_(mean=0.0, std=noise_scale)
            out = self.cnn(x)
            grad = autograd.grad(out.sum(), x, only_inputs=True)[0]
            momentum = momentum - self.mass * momentum * self.mcmc_step_size * self.C - self.mcmc_step_size * grad + noise
            x = x + self.mcmc_step_size * self.mass * momentum 

            # Save stats            
            grad_norms = append_norm(grad, grad_norms)
            data_norms = append_norm(x, data_norms)
            momentum_norms = append_norm(momentum, momentum_norms)

            if self.iter_n < time_window:
                prev_distances = append_norm(x - old_x, prev_distances)
                distances = append_norm(x - original_x, distances)

        self.cnn.train(is_training)

        ####### Evolution within Langevin dynamics ######
        # If at the beginning of an epoch, save the evolution of
        # grad and img norms, for a time window of width K.
        # These quantities will be logged within fit() function

        def append_mcmc_logs(prop_name, prop_array):
            full_name = "%s_epoch_%d" % (prop_name, self.epoch_n + 1)
            entry = self.mcmc_evolution_logs.get(full_name, None)
            if entry is None:
                self.mcmc_evolution_logs[full_name] = prop_array
            else:
                self.mcmc_evolution_logs[full_name] = np.vstack(
                    (entry, prop_array))
            return
        
        if not evaluation:
            # Beginning of epoch e
            # 'langevin_evolution_' metrics describe the evolution 
            # within a mcmc sampling process. Computed over a time_window of iterations.
            if self.iter_n < time_window:
                # Gradient norm
                append_mcmc_logs("langevin_evolution_grad_norm", grad_norms)

                # Data norm
                append_mcmc_logs("langevin_evolution_img_norm", data_norms)
                
                # Momentum norm
                append_mcmc_logs("langevin_evolution_momentum_norm", momentum_norms)

                # Distance from previous point
                append_mcmc_logs("langevin_evolution_distance2prevstep", prev_distances)
                
                # Distance from starting point
                append_mcmc_logs("langevin_evolution_distance2start", distances)


            # Always log the avg
            # 'langevin_avg_' metrics describe the avg value of a measure
            # within a mcmc sampling process. Computed at each iteration.
            self.log('langevin_avg_grad_norm', np.mean(grad_norms))
            self.log('langevin_avg_img_norm', np.mean(data_norms))
            self.log('langevin_avg_momentum_norm', np.mean(momentum_norms))
            e2e_distances = torch.norm(
                x - original_x, dim=[2, 3]).mean().clone().detach().cpu().numpy()
            self.log('langevin_avg_distance_start2end', e2e_distances)

        return x.detach()
    
    
    
        