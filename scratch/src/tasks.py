import math

import torch
import torch.linalg as LA

def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()


def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()


def cross_entropy(ys_pred, ys):
    output = sigmoid(ys_pred)
    target = (ys + 1) / 2
    return bce_loss(output, target)


class Task:
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(
    task_name, n_dims, batch_size, pool_dict=None, num_tasks=None, **kwargs
):
    task_names_to_classes = {
        "linear_regression": LinearRegression,
        "sparse_linear_regression": SparseLinearRegression,
        "linear_classification": LinearClassification,
        "noisy_linear_regression": NoisyLinearRegression,
        "quadratic_regression": QuadraticRegression,
        "relu_2nn_regression": Relu2nnRegression,
        "decision_tree": DecisionTree,
        "toy_linear_regression": ToyLinearRegression,
        "toy_quadratic_regression": ToyQuadraticRegression,
        "affine_regression": AffineRegression,
        "toy_affine_regression": ToyAffineRegression,
        "polynomial_regression": PolynomialRegression,
        "toy_polynomial_regression": ToyPolynomialRegression,
        "chebyshev_polynomial_regression": ChebyshevPolynomialRegression,
    }
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        if num_tasks is not None:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")
            pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
        return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError


class LinearRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.w_b = torch.randn(self.b_size, self.n_dims, 1)
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class SparseLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        sparsity=3,
        valid_coords=None,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(SparseLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.sparsity = sparsity
        if valid_coords is None:
            valid_coords = n_dims
        assert valid_coords <= n_dims

        for i, w in enumerate(self.w_b):
            mask = torch.ones(n_dims).bool()
            if seeds is None:
                perm = torch.randperm(valid_coords)
            else:
                generator = torch.Generator()
                generator.manual_seed(seeds[i])
                perm = torch.randperm(valid_coords, generator=generator)
            mask[perm[:sparsity]] = False
            w[mask] = 0

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
    
class NoisyLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        noise_std=0,
        renormalize_ys=False,
    ):
        """noise_std: standard deviation of noise added to the prediction."""
        super(NoisyLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.noise_std = noise_std
        self.renormalize_ys = renormalize_ys

    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        ys_b_noisy = ys_b + torch.randn_like(ys_b) * self.noise_std
        if self.renormalize_ys:
            ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()

        return ys_b_noisy


class QuadraticRegression(LinearRegression):
    def evaluate(self, xs_b):
        self.w_b = torch.ones(self.b_size, self.n_dims, 1)
        w_b = self.w_b.to(xs_b.device)
        norm_xs = LA.vector_norm(xs_b, ord=2, dim=2)        
        # TODO  : ADD WEIGHTS
        ys_b_quad = norm_xs
        print(ys_b_quad.shape)

        x_1d = (xs_b[:, :, 0] +2)*(xs_b[:, :, 0] - 3)*xs_b[:, :, 0]**2*(xs_b[:, :, 0]-2) - 1 #  4*xs_b[:, :, 0]**2 + xs_b[:, :, 0]
        #         ys_b_quad = ys_b_quad * math.sqrt(self.n_dims) / ys_b_quad.std()
        # Renormalize to Linear Regression Scale
        # ys_b_quad = ys_b_quad / math.sqrt(3)
        ys_b_quad = self.scale * ys_b_quad
        # print("shapes : ")
        # print(xs_b.shape)
        # print(ys_b_quad.shape)
        # return ys_b_quad
        return x_1d


class Relu2nnRegression(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        hidden_layer_size=100,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(Relu2nnRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.hidden_layer_size = hidden_layer_size

        if pool_dict is None and seeds is None:
            self.W1 = torch.randn(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.randn(self.b_size, hidden_layer_size, 1)
        elif seeds is not None:
            self.W1 = torch.zeros(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.zeros(self.b_size, hidden_layer_size, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.W1[i] = torch.randn(
                    self.n_dims, hidden_layer_size, generator=generator
                )
                self.W2[i] = torch.randn(hidden_layer_size, 1, generator=generator)
        else:
            assert "W1" in pool_dict and "W2" in pool_dict
            assert len(pool_dict["W1"]) == len(pool_dict["W2"])
            indices = torch.randperm(len(pool_dict["W1"]))[:batch_size]
            self.W1 = pool_dict["W1"][indices]
            self.W2 = pool_dict["W2"][indices]

    def evaluate(self, xs_b):
        W1 = self.W1.to(xs_b.device)
        W2 = self.W2.to(xs_b.device)
        # Renormalize to Linear Regression Scale
        ys_b_nn = (torch.nn.functional.relu(xs_b @ W1) @ W2)[:, :, 0]
        ys_b_nn = ys_b_nn * math.sqrt(2 / self.hidden_layer_size)
        ys_b_nn = self.scale * ys_b_nn
        #         ys_b_nn = ys_b_nn * math.sqrt(self.n_dims) / ys_b_nn.std()
        return ys_b_nn

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        return {
            "W1": torch.randn(num_tasks, n_dims, hidden_layer_size),
            "W2": torch.randn(num_tasks, hidden_layer_size, 1),
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class DecisionTree(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, depth=4):

        super(DecisionTree, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.depth = depth

        if pool_dict is None:

            # We represent the tree using an array (tensor). Root node is at index 0, its 2 children at index 1 and 2...
            # dt_tensor stores the coordinate used at each node of the decision tree.
            # Only indices corresponding to non-leaf nodes are relevant
            self.dt_tensor = torch.randint(
                low=0, high=n_dims, size=(batch_size, 2 ** (depth + 1) - 1)
            )

            # Target value at the leaf nodes.
            # Only indices corresponding to leaf nodes are relevant.
            self.target_tensor = torch.randn(self.dt_tensor.shape)
        elif seeds is not None:
            self.dt_tensor = torch.zeros(batch_size, 2 ** (depth + 1) - 1)
            self.target_tensor = torch.zeros_like(dt_tensor)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.dt_tensor[i] = torch.randint(
                    low=0,
                    high=n_dims - 1,
                    size=2 ** (depth + 1) - 1,
                    generator=generator,
                )
                self.target_tensor[i] = torch.randn(
                    self.dt_tensor[i].shape, generator=generator
                )
        else:
            raise NotImplementedError

    def evaluate(self, xs_b):
        dt_tensor = self.dt_tensor.to(xs_b.device)
        target_tensor = self.target_tensor.to(xs_b.device)
        ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for i in range(xs_b.shape[0]):
            xs_bool = xs_b[i] > 0
            # If a single decision tree present, use it for all the xs in the batch.
            if self.b_size == 1:
                dt = dt_tensor[0]
                target = target_tensor[0]
            else:
                dt = dt_tensor[i]
                target = target_tensor[i]

            cur_nodes = torch.zeros(xs_b.shape[1], device=xs_b.device).long()
            for j in range(self.depth):
                cur_coords = dt[cur_nodes]
                cur_decisions = xs_bool[torch.arange(xs_bool.shape[0]), cur_coords]
                cur_nodes = 2 * cur_nodes + 1 + cur_decisions

            ys_b[i] = target[cur_nodes]

        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class ToyLinearRegression(LinearRegression):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        super(ToyLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )

    def evaluate(self, xs_b):
        # w_b = 2.4*torch.ones(self.b_size, self.n_dims, 1).to(xs_b.device)
        # ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        # return ys_b
        return 2.4*xs_b[:, :, 0]
    
class ToyQuadraticRegression(LinearRegression):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        super(ToyQuadraticRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )

    def evaluate(self, xs_b):
        # w_b = 2.4*torch.ones(self.b_size, self.n_dims, 1).to(xs_b.device)
        # ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        # return ys_b
        x = xs_b[:, :, 0]
        return (x-1)*(x+1)
    
class AffineRegression():
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        super(AffineRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.w_b = torch.randn(self.b_size, self.n_dims, 1)
        self.w_c = torch.randn(self.b_size, 1)

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        w_c = self.wc.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0] + w_c
        return ys_b
    
class ToyAffineRegression(AffineRegression):
    def evaluate(self, xs_b):
        w_b = 2.4*torch.ones(self.b_size, self.n_dims, 1).to(xs_b.device)
        w_c = -2*torch.ones(self.b_size, 1).to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0] + w_c
        return ys_b
    
# class PolynomialRegression(Task):
#     def __init__(self,  n_dims, batch_size, pool_dict=None, seeds=None,scale=1, max_dim=2):
#         super(PolynomialRegression, self).__init__(
#             n_dims, batch_size, pool_dict=None, seeds=None
#         )
#         self.max_dim = max_dim
#         self.scale = scale

#         if pool_dict is None and seeds is None:
#             self.coefficients = torch.randn(batch_size, n_dims, max_dim + 1)
#         elif seeds is not None:
#             self.coefficients = torch.zeros(batch_size,n_dims, max_dim + 1)
#             generator = torch.Generator()
#             assert len(seeds) == batch_size
#             for i, seed in enumerate(seeds):
#                 generator.manual_seed(seed)
#                 self.coefficients[i] = torch.randn(n_dims, max_dim+1, generator=generator)
#         else:
#             assert "w" in pool_dict
#             indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
#             self.coefficients = pool_dict["w"][indices]


#     def evaluate(self, xs_b):
        
#         # Compute the polynomial function
#         ys_b = torch.zeros(self.b_size, xs_b.shape[1]).to(xs_b.device)
#         for i in range(self.max_dim + 1):
#             x_powered = xs_b ** (self.max_dim - i)
#             x_powered = x_powered.to(xs_b.device)
#             coef = self.coefficients[:, :, i].to(xs_b.device)
#             mul = torch.bmm(x_powered,coef.unsqueeze(2)).squeeze(2)
#             ys_b += mul
        
#         # Scale the output
#         ys_b = self.scale * ys_b
        
#         return ys_b
    
#     @staticmethod
#     def get_metric():
#         return squared_error

#     @staticmethod
#     def get_training_metric():
#         return mean_squared_error
class PolynomialRegression(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        max_dim=2,
        distribution="bigaussian",
        lower=0.0,
        upper=1.0,
        bias1=1.0,
        bias2=-1.0,
        scale1=1.0,
        scale2=1.0,
    ):
        super(PolynomialRegression, self).__init__(
            n_dims, batch_size, pool_dict=None, seeds=None
        )
        self.max_dim = max_dim
        self.scale = scale
        self.distribution = distribution
        self.lower = lower
        self.upper = upper
        self.bias1= bias1
        self.bias2 = bias2
        self.scale1 = scale1
        self.scale2 = scale2

        if pool_dict is None and seeds is None:
            if self.distribution == "bigaussian":
                mask = torch.randint(0, 2, (batch_size, n_dims, max_dim + 1)).float()
                t1 = self.scale1 * torch.randn(batch_size, n_dims, max_dim + 1) + self.bias1
                t2 = self.scale2 * torch.randn(batch_size, n_dims, max_dim + 1) + self.bias2
                self.coefficients = mask * t1  + ( 1-mask) * t2
            elif self.distribution == "normal":
                self.coefficients = torch.randn(batch_size, n_dims, max_dim + 1)
            elif self.distribution == "uniform":
                self.coefficients = torch.rand(batch_size, n_dims, max_dim + 1) * (self.upper - self.lower) + self.lower
            else:
                raise ValueError(
                    f"Unsupported distribution '{self.distribution}'. Choose 'normal' or 'uniform'."
                )
        elif seeds is not None:
            self.coefficients = torch.zeros(batch_size, n_dims, max_dim + 1)
            generator = torch.Generator()
            assert len(seeds) == batch_size, "Length of seeds must match batch_size."
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                if self.distribution == "normal":
                    self.coefficients[i] = torch.randn(
                        n_dims, max_dim + 1, generator=generator
                    )
                elif self.distribution == "uniform":
                    self.coefficients[i] = torch.rand(
                        n_dims, max_dim + 1, generator=generator
                    ) * (self.upper - self.lower) + self.lower
                else:
                    raise ValueError(
                        f"Unsupported distribution '{self.distribution}'. Choose 'normal' or 'uniform'."
                    )
        else:
            assert "w" in pool_dict, "pool_dict must contain key 'w'."
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.coefficients = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        # Compute the polynomial function
        ys_b = torch.zeros(self.b_size, xs_b.shape[1]).to(xs_b.device)
        for i in range(self.max_dim + 1):
            x_powered = xs_b ** (self.max_dim - i)
            x_powered = x_powered.to(xs_b.device)
            coef = self.coefficients[:, :, i].to(xs_b.device)
            mul = torch.bmm(x_powered, coef.unsqueeze(2)).squeeze(2)
            ys_b += mul

        # Scale the output
        ys_b = self.scale * ys_b

        return ys_b

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error    
    
class ToyPolynomialRegression(PolynomialRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        max_dim=2,
        distribution="normal",
        lower=0.0,
        upper=1.0,
        bias1=1.0,
        bias2=-1.0,
        scale1=1.0,
        scale2=1.0,
    ):
        super().__init__(
            n_dims=n_dims,
            batch_size=batch_size,
            pool_dict=pool_dict,
            seeds=seeds,
            scale=scale,
            max_dim=max_dim,
            distribution=distribution,
            lower=lower,
            upper=upper,
            bias1=bias1,
            bias2=bias2,
            scale1=scale1,
            scale2=scale2,
        )
        self.coefficients = torch.ones(batch_size, n_dims, max_dim + 1)
        
        # -1/48 + x/24 + (5 x^2)/16 - (5 x^3)/12 - (11 x^4)/12 + x^5
        if max_dim == 5:
            self.coefficients[:, :, -1] = -1/48
            self.coefficients[:, :, -2] = 1/24
            self.coefficients[:, :, -3] = 5/16
            self.coefficients[:, :, -4] = -5/12
            self.coefficients[:, :, -5] = -11/12
            self.coefficients[:, :, 0] = 1
        elif max_dim == 4:
            self.coefficients[:, :, -1] = 4
            self.coefficients[:, :, -2] = 0
            self.coefficients[:, :, -3] = -5
            self.coefficients[:, :, -4] = 2
            self.coefficients[:, :, -5] = 1
        elif max_dim == 3:
            self.coefficients[:, :, -1] = 4
            self.coefficients[:, :, -2] = -4
            self.coefficients[:, :, -3] = -1
            self.coefficients[:, :, -4] = 1
        elif max_dim == 2:
            self.coefficients[:, :, -1] = 20
            self.coefficients[:, :, -2] = 0
            self.coefficients[:, :, -3] = -1
        elif max_dim == 1:
            self.coefficients[:, :, -1] = 0
            self.coefficients[:, :, -2] = 1
        else:
            raise NotImplementedError
        
        # print(self.coefficients)
        
import random         
class LinearClassification(PolynomialRegression):
    def evaluate(self, xs_b):
        #ys_b = super().evaluate(xs_b)
        #return ys_b.sign()
        ys_b = xs_b[:,:,0]
        positive_mask = ys_b > 0

        #AND
        result = positive_mask.cumprod(dim=1)
        return result.float() * 2 - 1
        
        #OR
        # positive_mask = ys_b > 0
        # # Use cumulative sum to check if any previous element was positive
        # has_positive = torch.cumsum(positive_mask, dim=1) > 0
        # # Convert to 1 or -1
        # result = torch.where(has_positive, torch.tensor(1, dtype=torch.int8), torch.tensor(-1, dtype=torch.int8))
        # return result

        #50% AND 50% OR
        # if random.choice([True, False]):  # 50% de chance pour chaque code
        #     result = positive_mask.cumprod(dim=1)
        #     return result.float() * 2 - 1
        # else:
        #     has_positive = torch.cumsum(positive_mask, dim=1) > 0
        #     result = torch.where(has_positive, torch.tensor(1, dtype=torch.int8), torch.tensor(-1, dtype=torch.int8))
        #     return result
        
        
        # # Check if all values in the entire batch are True
        # result = torch.all(positive_mask, dim=1)  # Result will be [batch] (True/False)
        # result_expanded = result.unsqueeze(1).expand(-1, ys_b.size(1))  # Expand to shape [batch, points]
        # return result_expanded
    
    
        #result = torch.where(torch.all(positive_mask, dim=1, keepdim=True), torch.tensor(1.0), torch.tensor(-1.0))
        
        # Now, expand the result to [batch, points] with either all True or all False
        

        #print(result_expanded.shape)  # Should output: torch.Size([64, 40])
        #print(result_expanded)  # Should be a tensor of shape [batch, points] with True or False
        #print(f"result.shape: {result.shape}")
        


    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy
        
# class SignClassification(PolynomialRegression):
#     def evaluate(self, xs_b):
#         ys_b = super().evaluate(xs_b)
#         #return ys_b.sign()
#         return torch.all(ys_b > 0, dim=1)

#     @staticmethod
#     def get_metric():
#         return accuracy

#     @staticmethod
#     def get_training_metric():
#         return cross_entropy
    
    

class ChebyshevPolynomialRegression(Task):
    def __init__(self,  n_dims, batch_size, pool_dict=None, seeds=None,scale=1, max_dim=2):
        super(ChebyshevPolynomialRegression, self).__init__(
            n_dims, batch_size, pool_dict=None, seeds=None
        )
        self.max_dim = max_dim
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.linear_coefficients = torch.randn(batch_size, n_dims, max_dim + 1)
        elif seeds is not None:
            self.linear_coefficients = torch.zeros(batch_size,n_dims, max_dim + 1)
            generator = torch.Generator()
            assert len(seeds) == batch_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.linear_coefficients[i] = torch.randn(n_dims, max_dim+1, generator=generator)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.linear_coefficients = pool_dict["w"][indices]
       
        assert max_dim >= 2, "Chebyshev Polynomial requires max_dim >= 2"

        self.Chebyshev_func = lambda x,n: torch.cos(n * torch.acos(x))
        
        self.polynomial_func = lambda x: torch.sum(torch.stack([self.linear_coefficients[:,:,i] * self.Chebyshev_func(x, i) for i in range(max_dim + 1)]), dim=0)

        assert self.polynomial_func(torch.tensor([0.5])).shape == torch.Size([batch_size, n_dims])
        
    def evaluate(self, xs_b):
        ys_b = self.polynomial_func(xs_b)
        return ys_b
    
    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error