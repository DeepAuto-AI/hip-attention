# x: different k values
# y: accuracy    
from matplotlib import pyplot as plt
import torch
import os
import matplotlib

plt.style.use('seaborn-bright')
matplotlib.rcParams['font.family'] = 'Noto Sans, DejaVu Sans'

model = 'bert'
if model=='bert':
    trained_k = [7, 13, 25]
elif 'opt' in model:
    trained_k = [32, 64, 128]
    
def main():
    data = {}
    for k in trained_k:
        os.makedirs('./plots/main/figure_dynamic_k', exist_ok=True)
        
        path = f'./plots/main/figure_dynamic_k/k_{k}.pth'
        state = torch.load(path, map_location='cpu')
        k_acc_dict = state['k_acc_bucket']
        
        data[k] = k_acc_dict
    
    COLORS = {7:'magenta', 13:'purple', 25:'blue'}
    BASELINE = {7:0.81844, 13:0.834946, 25:0.842893}
    for k in trained_k:
        data[k][k] = BASELINE[k]*100
    
    plt.figure(figsize=(3.5,3.0))
    for k in trained_k:
        k_acc_dict = data[k]
        
        ts = sorted(k_acc_dict.items(), key=lambda x: x[0])
        xs = [t[0] for t in ts]
        ys = [t[1] for t in ts]
        plt.plot(
            xs, ys,
            label=f'Adjusted From $k$={k}', 
            linestyle='--', 
            linewidth=0.75,
            marker='*',
            color=COLORS[k]
        )
    
    for k in trained_k:
        k_acc_dict = data[k]
        xs = [min(k_acc_dict.keys()), max(k_acc_dict.keys())]
        ys = [k_acc_dict[k], ] * 2
        
        plt.plot(
            xs, ys,
            linestyle=':',
            linewidth=1.5,
            color=COLORS[k]
        )
        
        if k < 25:
            if k < 10:
                plt.annotate(f"k={k}", (52, ys[0] + 0.1), fontweight=500, color='#444')
            else:
                plt.annotate(f"k={k}", (52, ys[0] - 0.3), fontweight=500, color='#444')
        else:
            plt.annotate(f"k={k}", (4, ys[0] + 0.1), fontweight=500, color='#444')
        
    plt.title(
        f'Accuracy After Adjusting $k$', 
        fontsize=12,
        fontweight=500,
        pad=8,
    )
    # plt.subplots_adjust(hspace=0.2)
    plt.xlabel(f'$k$', fontsize=10,  fontweight=500)
    plt.ylabel(f'Acc. ↑', fontsize=10, fontweight=500)
    plt.ylim(80, 85.2)
    plt.xlim(3, 60)
    plt.grid()
    plt.legend(fontsize=9)
    
    root = f'./plots/main/figure_dynamic_k/'
    os.makedirs(root, exist_ok=True)
    
    path = os.path.join(root, f'plot_dynamic_k.png')
    plt.savefig(path, bbox_inches='tight')
    print('saved', path)
    path = os.path.join(root, f'plot_dynamic_k.pdf')
    plt.savefig(path, bbox_inches='tight')
    print('saved', path)

if __name__ == '__main__':
    main()