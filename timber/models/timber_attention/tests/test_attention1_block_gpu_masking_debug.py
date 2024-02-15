from timber.models.timber_attention.attention1_block_gpu import masking_iteration
import torch

x = torch.load('/home/ainl/masking_iteration_input_args.pt')
t = masking_iteration(**x)
torch.cuda.synchronize()
print(t)