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
                 img_shape=(1, 28, 28), 
                 batch_size=64,
                 lr=1e-4,
                 weight_decay=1e-4,
                 mcmc_step_size=1e-5,
                 mcmc_steps=250,
                 model_name="unnamed",
                 model_description="",
                 model_family="Langevin_vanilla",
                 mcmc_init_type="gaussian",
                 device="cuda:1",
                 overwrite=False,
                 start_epoch=0,
                 reload_model=False,
                 **CNN_args):
        super().__init__()

        # Model
        self.img_shape = img_shape
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        print("Running on device:", self.device) 
        # Use CNNModel by default
        self.cnn = CNNModel(**CNN_args).to(self.device)
        self.reload_model = reload_model

        # Optimizers
        self.lr = lr
        self.weight_decay = weight_decay

        # Dataset
        self.batch_size = batch_size

        # MCMC
        self.mcmc_step_size = mcmc_step_size
        self.mcmc_steps = mcmc_steps
        self.mcmc_persistent_data = (2 * torch.rand((10000,) + img_shape, device=self.device) - 1)
        self.mcmc_init_type = mcmc_init_type


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
        self.start_epoch = start_epoch
        
        # Setup flag: check the model has been properly set up before starting
        self.is_setup = False

    def setup(self):
        """Setup the optimizers, setup the Tensorboard SummaryWriter, process hyperparams dict."""
        # Optimizers
        self.configure_optimizers()
        
        # Hyperparams dict
        self.hparams_dict = {
            'mcmc_step_size': self.mcmc_step_size,
            'mcmc_steps': self.mcmc_steps,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'batch_size': self.batch_size,
            'optimizer': self.optimizer.__class__.__name__
        }
        
        # Tensorboard logs
        hparams_str = "__".join(["=".join(
            [str(el) for el in dict_entry]) for dict_entry in self.hparams_dict.items()])
        full_name = self.model_name + "__" + hparams_str
        self.ckpt_path = os.path.join(CHECKPOINT_PATH, self.model_family, full_name)
        # Reload existing model?
        if os.path.exists(self.ckpt_path) and self.reload_model:
            path = os.path.join(self.ckpt_path, "model_state_dict.pt")
            self.cnn.load_state_dict(torch.load(path))
            print("Loaded pretrained existsing model")
        
        ## Ovrewrite existsing model ##
        # Unautorized overwrite
        elif os.path.exists(self.ckpt_path) and not self.overwrite:
            print("Model path: " + self.ckpt_path)
            raise NameError("Model already exists! Set self.overwrite=True to overwrite it.")
        # Autorized overwrite
        elif os.path.exists(self.ckpt_path) and self.overwrite:
            # Remove existsing folder
            shutil.rmtree(self.ckpt_path)
            print("Overwriting existing logs")
        
        # Create writer
        self.tb_writer = SummaryWriter(self.ckpt_path)
        # Add docstring to interpret logs
        self.tb_writer.add_text('logs_documentation', self.tb_logs_doc())
        descr = "Model description:\n" + self.model_description + "\n\n\n"
        self.tb_writer.add_text("model_description", descr, 100)
        
        # Set is_setup flag to True
        self.is_setup = True
        
    def clear(self):
        # Tensorboard writer
        self.tb_writer.close()

        
    def configure_optimizers(self):
        # Optimize only the layers that require grad
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.cnn.parameters()),
                                    lr=self.lr,
                                    weight_decay=self.weight_decay)

    def prepare_data(self, train_set, test_set):
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

    ######################################################
    ################ Training section ####################
    ######################################################

    def training_step(self, batch):
        # Train mode
        self.cnn.train()

        real_imgs, _ = batch
        real_imgs = real_imgs.to(self.device)
        
        # Obtain samples
        fake_imgs = self.generate_samples()

        # Predict energy score for all images
        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        real_out, fake_out = self.cnn(inp_imgs).chunk(2, dim=0)
        cdiv_loss = real_out.mean() - fake_out.mean() 

        # Free memory
        del real_imgs, fake_imgs, inp_imgs

        # Optimize
        self.optimizer.zero_grad()
        cdiv_loss.backward()
        self.optimizer.step()

        # Logging
        self.log('loss_cdiv', cdiv_loss)
        self.log('energy_avg_real', real_out.mean())
        self.log('energy_avg_fake', fake_out.mean())
        
        # Log layers weigth / bias norms
        mod = 1
        for m in self.cnn.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                if m.weight is not None:
                    self.log(f"{m.__class__.__name__} #{str(mod)}: weight norm",
                         torch.norm(
                                     m.weight).clone().detach().cpu().numpy(),
                         printable=False)
                if m.bias is not None:
                    self.log(f"{m.__class__.__name__} #{str(mod)}: bias norm",
                         torch.norm(
                                     m.bias).clone().detach().cpu().numpy(),
                         printable=False)
                    
                mod +=1
        

    def fit(self, n_epochs=None):
        
        assert self.is_setup, "Model is not properly setup. Call .setup() before running!"

        if self.train_loader is None:
            print("Train data not loaded")
            return

        # Epochs
        self.tot_batches = len(self.train_loader)
        self.tot_epochs = n_epochs - self.start_epoch
        epochs_bar = tqdm(range(self.start_epoch, n_epochs), total=self.tot_epochs, leave=True)
        epochs_bar.set_description("Epochs")
        for self.epoch_n in epochs_bar:
            # Iterations
            self.log_active = True
            iters_bar = tqdm(enumerate(self.train_loader),
                                           total=self.tot_batches,
                                           position=0,
                                           leave=False)
            iters_bar.set_description("Batches (iterations)")
            for self.iter_n, batch in iters_bar:

                self.training_step(batch)
                
         
            ############## Tensorboard ###############
            # Evolution thoughout a mcmc simulation
            self.tb_mcmc_simulation()

            # Log a generaed sample of images
            self.tb_mcmc_images(batch_size=25, evaluation=True)

            # Force tensorboard to write to disk (to be sure)
            self.tb_writer.flush()

            ############# Other logs ################
            # Print logged measures
            #self.flush_logs()
                        
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
        
        is_training = self.cnn.training
        self.cnn.eval()
        
        # Initial batch of noise / images: starting point of mcmc chain
        def sample_s_t_0():
            if self.mcmc_init_type == 'persistent' and not evaluation:
                rand_inds = torch.randperm(self.mcmc_persistent_data.shape[0])[0:batch_size]
                return self.mcmc_persistent_data[rand_inds], rand_inds
            elif self.mcmc_init_type == 'data' and not evaluation:
                raise RuntimeError("EBM train: Not implmented error")
                #return torch.Tensor(self.train_set.sample_toy_data(batch_size)), None
            elif self.mcmc_init_type == 'uniform' or evaluation:
                return  (2 * torch.rand((batch_size,) + self.img_shape, device=self.device) - 1) , None
            elif self.mcmc_init_type == 'gaussian' and not evaluation:
                return torch.randn((batch_size,) + self.img_shape, device=self.device) , None
            else:
                raise RuntimeError('Invalid method for "init_type" (use "persistent", "data", "uniform", or "gaussian")')
        
        x, rand_inds = sample_s_t_0()
        x = torch.autograd.Variable(x.clone(), requires_grad=True).to(self.device)
        original_x = x.clone().detach()
        
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

            # Re-init noise tensor
            noise.normal_(mean=0.0, std=noise_scale)
            out = self.cnn(x)
            grad = autograd.grad(out.sum(), x, only_inputs=True)[0]
            # grad is in "device" by default
            
            # Avoid NaN gradients
#             if torch.any(torch.isnan(grad)):
#                 self.tb_writer.flush()
#                 raise RuntimeError("Langevin grad has some NaN values!")

            x = x - self.mcmc_step_size * grad + noise

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

            
            
            
            
class EBMLangVanilla(DeepEnergyModel):
    """"Vanilla Langevin Dynamics"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class EBMLang2Ord(DeepEnergyModel):
    """SGHMC: Second order Langevin Dynamics, with leapfrog"""
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
            
            x = x + self.mcmc_step_size * self.mass * momentum
            momentum = momentum - self.mass * momentum * self.mcmc_step_size * self.C - self.mcmc_step_size * grad + noise
           

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