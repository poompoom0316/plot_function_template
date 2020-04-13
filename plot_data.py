import pandas as pd
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics.regression import r2_score

# setting up model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(ard_num_dims=train_x.shape[1]) +
                                                         gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPMixturModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(GPMixturModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=train_x.shape[1])+
                gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1])
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train(train_x_pre, train_y_pre, training_iter=100):
    train_x = train_x_pre.cuda()
    train_y = train_y_pre.cuda()

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    likelihood = likelihood.cuda()
    model = model.cuda()

    # training the model
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # adamで最適化
    optimizer = torch.optim.Adam([
        {"params": model.parameters()},
    ], lr=0.05)

    # defining loss og GPs (marginal likelihood)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        print(f"Iter {i + 1}/{training_iter} - Loss: {loss.item():.3f}  noise:{model.likelihood.noise.item():.3f}")
        optimizer.step()

    return model, likelihood


def train2(train_x_pre, train_y_pre, training_iter=100):
    train_x = train_x_pre.cuda()
    train_y = train_y_pre.cuda()

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPMixturModel(train_x, train_y, likelihood)

    likelihood = likelihood.cuda()
    model = model.cuda()

    # training the model
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # adamで最適化
    optimizer = torch.optim.Adam([
        {"params": model.parameters()},
    ], lr=0.05)

    # defining loss og GPs (marginal likelihood)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        print(f"Iter {i + 1}/{training_iter} - Loss: {loss.item():.3f}  noise:{model.likelihood.noise.item():.3f}")
        optimizer.step()

    return model, likelihood


def test(test_x, model, likelihood):
    model.eval()
    likelihood.eval()
    test_x = test_x.cuda()

    # Te
    # st points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))
        mean = observed_pred.mean
        lower, upper = observed_pred.confidence_region()

    return observed_pred, mean, lower, upper


def main():
    path1 = "analysis/wine/log_data.csv"
    path2 = "analysis/wine/pc_data.csv"

    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    np.random.seed(777)
    randint = np.random.choice(df1.index, size=250)
    test_x = torch.from_numpy(df1.loc[randint].drop('quality', axis=1).values).float()
    test_y = torch.from_numpy(df1.loc[randint, 'quality'].values).float()
    train_x_pre = torch.from_numpy(df1.drop(randint).drop('quality', axis=1).values).float()
    train_y_pre = torch.from_numpy(df1.drop(randint).loc[:,'quality'].values).float()

    model, likelihood = train(train_x_pre, train_y_pre, training_iter=1000)

    observed_pred, mean, lower, upper = test(test_x, model, likelihood)

    sns.scatterplot(x=test_y.numpy(), y=mean.cpu().numpy())
    r2_score(test_y.numpy(), mean.cpu().numpy())

    # test_x2 = torch.from_numpy(df2.loc[randint].drop('quality', axis=1).values).float()
    # test_y2 = torch.from_numpy(df2.loc[randint, 'quality'].values).float()
    # train_x_pre2 = torch.from_numpy(df2.drop(randint).drop('quality', axis=1).values).float()
    # train_y_pre2 = torch.from_numpy(df2.drop(randint).loc[:,'quality'].values).float()
    model2, likelihood2 = train2(train_x_pre, train_y_pre, training_iter=1000)

    observed_pred2, mean2, lower2, upper2 = test(test_x, model2, likelihood2)
    sns.scatterplot(x=test_y.numpy(), y=mean2.cpu().numpy())
    r2_score(test_y.numpy(), mean2.cpu().numpy())


