import math

import torch
import matplotlib.pyplot as plt
import numpy as np

class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError

    def plot_distribution(self, samples):
        plt.hist(samples[0, :, 0], bins=50, alpha=0.5)
        plt.title('Distribution of Samples')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.show()


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
        "uniform": UniformSampler,
        "bigaussian": BiGaussianSampler,
        "biuniform": BiUniformSampler
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


def sample_transformation(eigenvalues, normalize=False):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t


class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b

class UniformSampler(DataSampler):
    
    def __init__(self, n_dims, lower=None, upper=None):
       super().__init__(n_dims)
       self.lower = lower
       self.upper = upper

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        # Generating random points in the unit cube
        if seeds is None:
            xs_b = torch.rand(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.rand(n_points, self.n_dims, generator=generator)
        # Scaling to the desired range
        if self.lower is not None:
            xs_b = xs_b * (self.upper - self.lower) + self.lower
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b

class BiGaussianSampler(DataSampler):
    def __init__(self, n_dims, bias1=None, bias2=None, scale1=None, scale2=None):
        super().__init__(n_dims)
        self.bias1 = bias1 if bias1 is not None else torch.zeros(self.n_dims)
        self.bias2 = bias2 if bias2 is not None else torch.zeros(self.n_dims)
        self.scale1 = scale1 if scale1 is not None else torch.ones(self.n_dims)
        self.scale2 = scale2 if scale2 is not None else torch.ones(self.n_dims)
        #support for float inputs and broadcasting
        self.bias1 = torch.as_tensor(self.bias1).float().expand(self.n_dims)
        self.bias2 = torch.as_tensor(self.bias2).float().expand(self.n_dims)
        self.scale1 = torch.as_tensor(self.scale1).float().expand(self.n_dims)
        self.scale2 = torch.as_tensor(self.scale2).float().expand(self.n_dims)
    def sample_xs(self, n_points, b_size, mode=0, n_dims_truncated=None, seeds=None):
        '''
        mode 0: both Gaussians
        mode 1: only first Gaussian
        mode 2: only second Gaussian
        '''
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        
        if mode == 0:
            # Sample from both Gaussians
            mask = torch.randint(0, 2, (b_size, n_points, 1)).float()
            #test...
            #prob = torch.full((b_size, n_points, 1), 0.125)  # Probability of 1 is 0.3 (30%)
            #mask = torch.bernoulli(prob).float()  # Generates 1 with probability 0.3 and 0 with probability 0.7
            #end test..
            # mask = 0.5*torch.ones(b_size, n_points, 1).float()
            samples1 = xs_b * self.scale1 + self.bias1
            samples2 = xs_b * self.scale2 + self.bias2
            xs_b = mask * samples1 + (1 - mask) * samples2
        elif mode == 1:
            # Sample only from the first Gaussian
            xs_b = xs_b * self.scale1 + self.bias1
        elif mode == 2:
            # Sample only from the second Gaussian
            xs_b = xs_b * self.scale2 + self.bias2
        else:
            raise ValueError("Invalid mode, choose between 0, 1, and 2.")
        
        if n_dims_truncated is not None:
            xs_b = xs_b[..., :n_dims_truncated]
        
        return xs_b

class BiUniformSampler(DataSampler):
    def __init__(self, n_dims, lower1=None, upper1=None, lower2=None, upper2=None):
        super().__init__(n_dims)
        self.lower1 = lower1 if lower1 is not None else torch.zeros(n_dims)
        self.upper1 = upper1 if upper1 is not None else torch.ones(n_dims)
        self.lower2 = lower2 if lower2 is not None else torch.zeros(n_dims)
        self.upper2 = upper2 if upper2 is not None else torch.ones(n_dims)

        # support for input in float (broadcasted) and tensor inputs
        self.lower1 = torch.as_tensor(self.lower1).float().expand(n_dims)
        self.upper1 = torch.as_tensor(self.upper1).float().expand(n_dims)
        self.lower2 = torch.as_tensor(self.lower2).float().expand(n_dims)
        self.upper2 = torch.as_tensor(self.upper2).float().expand(n_dims)
      
    def sample_xs(self, n_points, b_size, mode=0, n_dims_truncated=None, seeds=None):
        """
        mode 0: both Uniform distributions
        mode 1: only first Uniform distribution
        mode 2: only second Uniform distribution
        """
        if seeds is None:
            xs_b = torch.rand(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.rand(n_points, self.n_dims, generator=generator)

        if mode == 0:
            choices = np.random.choice([0, 1], size=(b_size, n_points))
            
            xs_b_1 = xs_b * (self.upper1 - self.lower1) + self.lower1
            xs_b_2 = xs_b * (self.upper2 - self.lower2) + self.lower2
            
            # Use choices to select between xs_b_1 and xs_b_2
            choices = torch.from_numpy(choices).unsqueeze(-1).expand(-1, -1, self.n_dims)
            xs_b = torch.where(choices == 0, xs_b_1, xs_b_2)
        elif mode == 1:
            xs_b = xs_b * (self.upper1 - self.lower1) + self.lower1
        elif mode == 2:
            xs_b = xs_b * (self.upper2 - self.lower2) + self.lower2
        else:
            raise ValueError("Invalid mode. mode 0: both Uniform distributions, mode 1: only first Uniform, mode 2: only second Uniform")

        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0

        return xs_b

def test_bi_uniform_sampler():
    n_dims = 3
    b_size = 2
    n_points = 10000
    
    # Test initialization
    sampler1 = BiUniformSampler(n_dims, 
                               lower1=torch.tensor([-3, -3, -3]), 
                               upper1=torch.tensor([0, 0, 0]),
                               lower2=torch.tensor([2, 2, 2]),
                               upper2=torch.tensor([3, 3, 3]))
    #with broadcasting
    sampler2 = BiUniformSampler(n_dims, 
                               lower1=-3, 
                               upper1=0,
                               lower2=2,
                               upper2=3)
    
    # Test sampling
    for mode in [0, 1, 2]:
        samples1 = sampler1.sample_xs(n_points, b_size, mode=mode)
        samples2 = sampler2.sample_xs(n_points, b_size, mode=mode)
        
        assert samples1.shape == (b_size, n_points, n_dims), f"Shape mismatch for mode {mode}"
        assert samples2.shape == (b_size, n_points, n_dims), f"Shape mismatch for mode {mode}"
        #plot the distribution of the samples
        plt.figure()
        sampler1.plot_distribution(samples1)
        plt.close()
        plt.figure()
        sampler2.plot_distribution(samples2)
        plt.close()

def test_bi_gaussian_sampler():
    n_dims = 3
    b_size = 2
    n_points = 10000
    
    # Test initialization
    sampler1 = BiGaussianSampler(n_dims, 
                               bias1=torch.tensor([-2, -3, -3]), 
                               bias2=torch.tensor([2, 2, 2]),
                               scale1=torch.tensor([1., 1., 1.]),
                               scale2=torch.tensor([0.5, 0.5, 0.5]))
    #with broadcasting
    sampler2 = BiGaussianSampler(n_dims, 
                               bias1=-2, 
                               bias2=2,
                               scale1=1.,
                               scale2=1)
    #with broadcasting
    # Test sampling
    for mode in [0, 1, 2]:
        samples1 = sampler1.sample_xs(n_points, b_size, mode=mode)
        samples2 = sampler2.sample_xs(n_points, b_size, mode=mode)
        
        assert samples1.shape == (b_size, n_points, n_dims), f"Shape mismatch for mode {mode}"
        assert samples2.shape == (b_size, n_points, n_dims), f"Shape mismatch for mode {mode}"
        
        # Plot the distribution of the samples
        plt.figure()
        sampler1.plot_distribution(samples1)
        plt.close()
        plt.figure()
        sampler2.plot_distribution(samples2)
        plt.close()
# test_bi_uniform_sampler()
#test_bi_gaussian_sampler()