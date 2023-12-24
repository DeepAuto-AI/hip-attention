from torch import nn, optim
import torch

class VAEEncoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        self.training = True
        
    def forward(self, x):
        h_       = self.LeakyReLU(self.FC_input(x))
        h_       = self.LeakyReLU(self.FC_input2(h_))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)
        
        return mean, log_var

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h     = self.LeakyReLU(self.FC_hidden(x))
        h     = self.LeakyReLU(self.FC_hidden2(h))
        
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat

class VAEModel(nn.Module):
    def __init__(self, indim, outdim, hiddim, latentdim):
        super().__init__()
        self.Encoder = VAEEncoder(indim, hiddim, latentdim)
        self.Decoder = VAEDecoder(latentdim, hiddim, outdim)
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(var.device)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
          
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)
        
        return x_hat, mean, log_var

    def kl_loss(self, mean, log_var):
        klloss = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
        return klloss