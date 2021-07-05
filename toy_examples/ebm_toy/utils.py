import os

import numpy as np
from numpy import random
from scipy.stats import kstwobign, pearsonr

import torch as t
import torchvision as tv
from torch.utils.data import Dataset
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter



##################
# ## PLOTTING ## #
##################

# visualize negative samples synthesized from energy
def plot_ims(p, x): tv.utils.save_image(t.clamp(x, -1., 1.), p, normalize=True, nrow=int(x.shape[0] ** 0.5))

# plot diagnostics for learning
def plot_diagnostics(batch, en_diffs, grad_mags, exp_dir, fontsize=10):
    # axis tick size
    matplotlib.rc('xtick', labelsize=6)
    matplotlib.rc('ytick', labelsize=6)
    fig = plt.figure()

    def plot_en_diff_and_grad_mag():
        # energy difference
        ax = fig.add_subplot(221)
        ax.plot(en_diffs[0:(batch+1)].data.cpu().numpy())
        ax.axhline(y=0, ls='--', c='k')
        ax.set_title('Energy Difference', fontsize=fontsize)
        ax.set_xlabel('batch', fontsize=fontsize)
        ax.set_ylabel('$d_{s_t}$', fontsize=fontsize)
        # mean langevin gradient
        ax = fig.add_subplot(222)
        ax.plot(grad_mags[0:(batch+1)].data.cpu().numpy())
        ax.set_title('Average Langevin Gradient Magnitude', fontsize=fontsize)
        ax.set_xlabel('batch', fontsize=fontsize)
        ax.set_ylabel('$r_{s_t}$', fontsize=fontsize)

    def plot_crosscorr_and_autocorr(t_gap_max=2000, max_lag=15, b_w=0.35):
        t_init = max(0, batch + 1 - t_gap_max)
        t_end = batch + 1
        t_gap = t_end - t_init
        max_lag = min(max_lag, t_gap - 1)
        # rescale energy diffs to unit mean square but leave uncentered
        en_rescale = en_diffs[t_init:t_end] / t.sqrt(t.sum(en_diffs[t_init:t_end] * en_diffs[t_init:t_end])/(t_gap-1))
        # normalize gradient magnitudes
        grad_rescale = (grad_mags[t_init:t_end]-t.mean(grad_mags[t_init:t_end]))/t.std(grad_mags[t_init:t_end])
        # cross-correlation and auto-correlations
        cross_corr = np.correlate(en_rescale.cpu().numpy(), grad_rescale.cpu().numpy(), 'full') / (t_gap - 1)
        en_acorr = np.correlate(en_rescale.cpu().numpy(), en_rescale.cpu().numpy(), 'full') / (t_gap - 1)
        grad_acorr = np.correlate(grad_rescale.cpu().numpy(), grad_rescale.cpu().numpy(), 'full') / (t_gap - 1)
        # x values and indices for plotting
        x_corr = np.linspace(-max_lag, max_lag, 2 * max_lag + 1)
        x_acorr = np.linspace(0, max_lag, max_lag + 1)
        t_0_corr = int((len(cross_corr) - 1) / 2 - max_lag)
        t_0_acorr = int((len(cross_corr) - 1) / 2)

        # plot cross-correlation
        ax = fig.add_subplot(223)
        ax.bar(x_corr, cross_corr[t_0_corr:(t_0_corr + 2 * max_lag + 1)])
        ax.axhline(y=0, ls='--', c='k')
        ax.set_title('Cross Correlation of Energy Difference\nand Gradient Magnitude', fontsize=fontsize)
        ax.set_xlabel('lag', fontsize=fontsize)
        ax.set_ylabel('correlation', fontsize=fontsize)
        # plot auto-correlation
        ax = fig.add_subplot(224)
        ax.bar(x_acorr-b_w/2, en_acorr[t_0_acorr:(t_0_acorr + max_lag + 1)], b_w, label='en. diff. $d_{s_t}$')
        ax.bar(x_acorr+b_w/2, grad_acorr[t_0_acorr:(t_0_acorr + max_lag + 1)], b_w, label='grad. mag. $r_{s_t}}$')
        ax.axhline(y=0, ls='--', c='k')
        ax.set_title('Auto-Correlation of Energy Difference\nand Gradient Magnitude', fontsize=fontsize)
        ax.set_xlabel('lag', fontsize=fontsize)
        ax.set_ylabel('correlation', fontsize=fontsize)
        ax.legend(loc='upper right', fontsize=fontsize-4)

    # make diagnostic plots
    plot_en_diff_and_grad_mag()
    plot_crosscorr_and_autocorr()
    # save figure
    plt.subplots_adjust(hspace=0.6, wspace=0.6)
    plt.savefig(os.path.join(exp_dir, 'diagnosis_plot.pdf'), format='pdf')
    plt.close()


#####################
# ## TOY DATASET ## #
#####################

class ToyDataset(Dataset):
    def __init__(self, dataset_len=60000, toy_type='gmm', toy_groups=8, toy_sd=0.15, toy_radius=1, viz_res=500, kde_bw=0.05):
        # import helper functions
        from scipy.stats import gaussian_kde
        from scipy.stats import multivariate_normal
        self.gaussian_kde = gaussian_kde
        self.mvn = multivariate_normal
        
        # dataset class property
        self.dataset_len = dataset_len

        # toy dataset parameters
        self.toy_type = toy_type
        self.toy_groups = toy_groups
        self.toy_sd = toy_sd
        self.toy_radius = toy_radius
        self.weights = np.ones(toy_groups) / toy_groups
        if toy_type == 'gmm':
            means_x = np.cos(2*np.pi*np.linspace(0, (toy_groups-1)/toy_groups, toy_groups)).reshape(toy_groups, 1, 1, 1)
            means_y = np.sin(2*np.pi*np.linspace(0, (toy_groups-1)/toy_groups, toy_groups)).reshape(toy_groups, 1, 1, 1)
            self.means = toy_radius * np.concatenate((means_x, means_y), axis=1)
        else:
            self.means = None

        # ground truth density
        if self.toy_type == 'gmm':
            def true_density(x):
                density = 0
                for k in range(toy_groups):
                    density += self.weights[k]*self.mvn.pdf(np.array([x[1], x[0]]), mean=self.means[k].squeeze(),
                                                            cov=(self.toy_sd**2)*np.eye(2))
                return density
        elif self.toy_type == 'rings':
            def true_density(x):
                radius = np.sqrt((x[1] ** 2) + (x[0] ** 2))
                density = 0
                for k in range(toy_groups):
                    density += self.weights[k] * self.mvn.pdf(radius, mean=self.toy_radius * (k + 1),
                                                              cov=(self.toy_sd**2))/(2*np.pi*self.toy_radius*(k+1))
                return density
        else:
            raise RuntimeError('Invalid option for toy_type (use "gmm" or "rings")')
        self.true_density = true_density

        # viz parameters
        self.viz_res = viz_res
        self.kde_bw = kde_bw
        if toy_type == 'rings':
            self.plot_val_max = toy_groups * toy_radius + 4 * toy_sd
        else:
            self.plot_val_max = toy_radius + 4 * toy_sd

        # save values for plotting groundtruth landscape
        self.xy_plot = np.linspace(-self.plot_val_max, self.plot_val_max, self.viz_res)
        self.z_true_density = np.zeros(self.viz_res**2).reshape(self.viz_res, self.viz_res)
        for x_ind in range(len(self.xy_plot)):
            for y_ind in range(len(self.xy_plot)):
                self.z_true_density[x_ind, y_ind] = self.true_density([self.xy_plot[x_ind], self.xy_plot[y_ind]])
    
    @property
    def tile_side(self):
        return self.xy_plot[1] - self.xy_plot[0]
    @property
    def plot_side(self):
        return np.abs(2*self.plot_val_max)
    
    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, idx):
        return (self.sample_toy_data(1).squeeze(axis=0), 0) # (example, label)
    
    def sample_toy_data(self, num_samples):
        toy_sample = np.zeros(0).reshape(0, 2, 1, 1)
        sample_group_sz = np.random.multinomial(num_samples, self.weights)
        if self.toy_type == 'gmm':
            for i in range(self.toy_groups):
                sample_group = self.means[i] + self.toy_sd * np.random.randn(2*sample_group_sz[i]).reshape(-1, 2, 1, 1)
                toy_sample = np.concatenate((toy_sample, sample_group), axis=0)
        elif self.toy_type == 'rings':
            for i in range(self.toy_groups):
                sample_radii = self.toy_radius*(i+1) + self.toy_sd * np.random.randn(sample_group_sz[i])
                sample_thetas = 2 * np.pi * np.random.random(sample_group_sz[i])
                sample_x = sample_radii.reshape(-1, 1) * np.cos(sample_thetas).reshape(-1, 1)
                sample_y = sample_radii.reshape(-1, 1) * np.sin(sample_thetas).reshape(-1, 1)
                sample_group = np.concatenate((sample_x, sample_y), axis=1)
                toy_sample = np.concatenate((toy_sample, sample_group.reshape(-1, 2, 1, 1)), axis=0)
        else:
            raise RuntimeError('Invalid option for toy_type ("gmm" or "rings")')

        return toy_sample
    
    def ebm_learned_energy(self, f):
        xy_plot_torch = t.Tensor(self.xy_plot).view(-1, 1, 1, 1).to(next(f.parameters()).device)
        # y values for learned energy landscape of descriptor network
        z_learned_energy = np.zeros([self.viz_res, self.viz_res])
        for i in range(len(self.xy_plot)):
            y_vals = float(self.xy_plot[i]) * t.ones_like(xy_plot_torch)
            vals = t.cat((xy_plot_torch, y_vals), 1)
            z_learned_energy[i] = f(vals).data.cpu().numpy()
        
        return z_learned_energy
    
    def plot_learned_energy_surf(self, f, mcmc_lr):
        # Learned energy
        z_learned_energy = self.ebm_learned_energy(f)
        
        fig = plt.figure(figsize=(10,10))
        ax = fig.gca(projection='3d')

        # Make data.
        X = self.xy_plot
        Y = self.xy_plot
        X, Y = np.meshgrid(X, Y)
        Z = z_learned_energy

        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False, alpha=0.7)

        # Customize the z axis.
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Rotate plot
        ax.view_init(30, 30) # Rotation of the 3d plot
        ax.set_title(f"Energy landscape. $\eta={mcmc_lr:.0e}$")


        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()
        
    
    def ebm_learned_density(self, f, epsilon=0.0):
        z_learned_energy = self.ebm_learned_energy(f)
        
        # transform learned energy into learned density
        z_learned_density_unnormalized = np.exp(- (z_learned_energy - np.min(z_learned_energy)))
        bin_area = (self.xy_plot[1] - self.xy_plot[0]) ** 2
        z_learned_density = z_learned_density_unnormalized / (bin_area * np.sum(z_learned_density_unnormalized))
        
        return z_learned_density
    
    def ebm_kl_divergence(self, f):
        """Compute KL[p || q]"""
        p = self.z_true_density
        q = self.ebm_learned_density(f)
        bin_area = (self.xy_plot[1] - self.xy_plot[0]) ** 2
        return bin_area * np.sum(np.where(p != 0, p * np.log(p / q), 0))
        

    def plot_toy_density(self, plot_truth=False, f=None, epsilon=0.0, x_s_t=None, save_path='toy.pdf'):
        num_plots = 0
        if plot_truth:
            num_plots += 1

        # density of learned EBM
        if f is not None:
            num_plots += 1
            xy_plot_torch = t.Tensor(self.xy_plot).view(-1, 1, 1, 1).to(next(f.parameters()).device)
            # y values for learned energy landscape of descriptor network
            z_learned_energy = np.zeros([self.viz_res, self.viz_res])
            for i in range(len(self.xy_plot)):
                y_vals = float(self.xy_plot[i]) * t.ones_like(xy_plot_torch)
                vals = t.cat((xy_plot_torch, y_vals), 1)
                z_learned_energy[i] = f(vals).data.cpu().numpy()

            # transform learned energy into learned density
            z_learned_density_unnormalized = np.exp(- (z_learned_energy - np.min(z_learned_energy)))
            bin_area = (self.xy_plot[1] - self.xy_plot[0]) ** 2
            z_learned_density = z_learned_density_unnormalized / (bin_area * np.sum(z_learned_density_unnormalized))
            
        # kernel density estimate of shortrun samples
        if x_s_t is not None:
            num_plots += 1
            density_estimate = self.gaussian_kde(x_s_t.squeeze().cpu().numpy().transpose(), bw_method=self.kde_bw)
            z_kde_density = np.zeros([self.viz_res, self.viz_res])
            for i in range(len(self.xy_plot)):
                for j in range(len(self.xy_plot)):
                    z_kde_density[i, j] = density_estimate((self.xy_plot[j], self.xy_plot[i]))

        # plot results
        plot_ind = 0
        fig = plt.figure()

        # true density
        if plot_truth:
            plot_ind += 1
            ax = fig.add_subplot(2, num_plots, plot_ind)
            ax.set_title('True density')
            plt.imshow(self.z_true_density, cmap='viridis')
            plt.axis('off')
            ax = fig.add_subplot(2, num_plots, plot_ind + num_plots)
            ax.set_title('True log-density')
            plt.imshow(np.log(self.z_true_density + 1e-10), cmap='viridis')
            plt.axis('off')
        # learned ebm
        if f is not None:
            plot_ind += 1
            ax = fig.add_subplot(2, num_plots, plot_ind)
            ax.set_title('EBM density')
            plt.imshow(z_learned_density, cmap='viridis')
            plt.axis('off')
            ax = fig.add_subplot(2, num_plots, plot_ind + num_plots)
            ax.set_title('EBM log-density')
            plt.imshow(np.log(z_learned_density + 1e-10), cmap='viridis')
            plt.axis('off')
        # shortrun kde
        if x_s_t is not None:
            plot_ind += 1
            ax = fig.add_subplot(2, num_plots, plot_ind)
            ax.set_title('Short-run KDE')
            plt.imshow(z_kde_density, cmap='viridis')
            plt.axis('off')
            ax = fig.add_subplot(2, num_plots, plot_ind + num_plots)
            ax.set_title('Short-run log-KDE')
            plt.imshow(np.log(z_kde_density + 1e-10), cmap='viridis')
            plt.axis('off')

        plt.tight_layout()
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path, bbox_inches='tight', format='pdf')
        
        plt.close()
        

        
        

        
        
###################################
### Kolmogorov Smirnov distance ###
###################################

def ksDist(ebmModel, trainSet, n_samples=1000, benchmark=False):
    """Computes Kolmogorov-Smirnov distance. Wrapper of ks2d2s
    If benchmark = True, computes p-value and D from two samples of
    the true dist.
    Returns: 
        - Two-tailed (approximated) p-value.
        - KS statistic (dist).
        - pval, KS dist of two samples of the true dist: optional.
    
    Small p-values means that the two samples are significantly different. 
    Note that the p-value is only an approximation as the analytic distribution is unkonwn. 
    The approximation is accurate enough when N > ~20 and p-value < ~0.20 or so. 
    When p-value > 0.20, the value may not be accurate, but it certainly implies that the two
    samples are not significantly different. (cf. Press 2007)
    """
    # Sample from fitted density
    negative_samples = ebmModel.generate_samples(evaluation=True, batch_size=n_samples)
    neg_samples = negative_samples.cpu().numpy().squeeze(-1).squeeze(-1)
    
    # Sample from ground truth density
    positive_samples = trainSet.sample_toy_data(n_samples).squeeze(-1).squeeze(-1)
    
    
    pval, d = ks2d2s(positive_samples[:, 0], 
                    positive_samples[:, 1], 
                    neg_samples[:, 0],
                    neg_samples[:, 1],
                    extra=True)

    if benchmark:
        # Recycle the name "neg_samples"
        neg_samples = trainSet.sample_toy_data(n_samples).squeeze(-1).squeeze(-1)
        pval_b, d_b = ks2d2s(positive_samples[:, 0], 
                                positive_samples[:, 1], 
                                neg_samples[:, 0],
                                neg_samples[:, 1],
                                extra=True)
        return pval, d, pval_b, d_b
    # if not benchmark
    return pval, d


def ks2d2s(x1, y1, x2, y2, extra=False):
    '''Two-dimensional Kolmogorov-Smirnov test on two samples. 
    Parameters
    ----------
    x1, y1 : ndarray, shape (n1, )
        Data of sample 1.
    x2, y2 : ndarray, shape (n2, )
        Data of sample 2. Size of two samples can be different.
    extra: bool, optional
        If True, KS statistic is also returned. Default is False.
    Returns
    -------
    p : float
        Two-tailed p-value.
    D : float, optional
        KS statistic. Returned if keyword `extra` is True.
    Notes
    -----
    This is the two-sided K-S test. Small p-values means that the two samples are significantly different. 
    Note that the p-value is only an approximation as the analytic distribution is unkonwn. 
    The approximation is accurate enough when N > ~20 and p-value < ~0.20 or so. 
    When p-value > 0.20, the value may not be accurate, but it certainly implies that the two
    samples are not significantly different. (cf. Press 2007)
    References
    ----------
    https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwj8j8nm7NfwAhWJ2hQKHcdSAkoQFjAAegQIAxAD&url=https%3A%2F%2Faip.scitation.org%2Fdoi%2Fpdf%2F10.1063%2F1.4822753&usg=AOvVaw0MJ3m8vCKG1h3RzVmqOuKT
    Peacock, J.A. 1983, Two-Dimensional Goodness-of-Fit Testing in Astronomy, 
    Monthly Notices of the Royal Astronomical Society, vol. 202, pp. 615-627
    Fasano, G. and Franceschini, A. 1987, A Multidimensional Version of the Kolmogorov-Smirnov Test, 
    Monthly Notices of the Royal Astronomical Society, vol. 225, pp. 155-170
    Press, W.H. et al. 2007, Numerical Recipes, section 14.8
    '''
    assert (len(x1) == len(y1)) and (len(x2) == len(y2))
    n1, n2 = len(x1), len(x2)
    D = avgmaxdist(x1, y1, x2, y2)

    sqen = np.sqrt(n1 * n2 / (n1 + n2))
    r1 = pearsonr(x1, y1)[0]
    r2 = pearsonr(x2, y2)[0]
    r = np.sqrt(1. - 0.5 * (r1**2 + r2**2))
    d = D * sqen / (1. + r * (0.25 - 0.75 / sqen))
    p = kstwobign.sf(d)
   
    if extra:
        return p, D
    else:
        return p


def avgmaxdist(x1, y1, x2, y2):
    D1 = maxdist(x1, y1, x2, y2)
    D2 = maxdist(x2, y2, x1, y1)
    return (D1 + D2) / 2


def maxdist(x1, y1, x2, y2):
    n1 = len(x1)
    D1 = 0.0 
    for i in range(n1):
        a1, b1, c1, d1 = quadct(x1[i], y1[i], x1, y1)
        a2, b2, c2, d2 = quadct(x1[i], y1[i], x2, y2)
        D1 = np.max([D1, np.abs(a1-a2), np.abs(b1-b2), np.abs(c1-c2), np.abs(d1-d2)]) 
    return D1

def quadct(x, y, xx, yy):
    n = len(xx)
    ix1, ix2 = yy >= y, xx >= x
    a = np.sum(ix1 & ix2) / n
    b = np.sum(ix1 & ~ix2) / n
    c = np.sum(~ix1 & ix2) / n
    d = 1 - a - b - c
    return a, b, c, d



        
#######################################################
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
