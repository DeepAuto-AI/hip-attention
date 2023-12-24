import matplotlib.pyplot as plt
plt.style.use('seaborn-bright')
import argparse, math, os

# count flops to calculate attention approximation
def calc_perlin(H, T, D, M, K):
    flops = {}
    # performer (H, T, D)
    # - K(H, D, T) x V(H, T, D)
    # - Q x $
    flops['performer'] = \
        H*D*T*D +\
        H*T*D*D
    # mlp x D->M (H,T,D) x (D, M)
    flops['mlp'] = \
        H*T*D*M
    # conv pixels * 2*(KIN*KOUT*KH*KW), head is KIN dim
    # two resnet
    # pixel shuffle, then conv
    flops['cnn'] = \
        T*M*(H*4*H*9)/4 +\
        T*M*(4*H*4*H*9)/4 +\
        T*M*(4*H*4*H*9)/4 +\
        H*T*M +\
        T*M*(4*H*4*H*9)/4 +\
        T*M*(4*H*4*H*9)/4 +\
        H*T*M +\
        T*M*(H*H*9)
    # topk
    flops['topk'] = H*T*M*K*math.log(K)
    # interp calc addr
    flops['interp'] = H*T*T
    # dot product
    flops['dot'] = H*T*K*D
    # softmax
    flops['softmax'] = H*T*K*2
    # scaler (H, T, D) x (D, 1)
    flops['scaler'] = H*T*D*1
    # average pool
    # - avg((H, T, D), dim=1)
    # - weight (H, T, D) x (D, 1)
    # - weighted sum (H, T, D) + (H, T, D)
    flops['pool'] = \
        H*T*D +\
        H*T*D*1 +\
        4*H*T*D
    # masked A(H, T, K) x V(H, K, D)
    flops['masked_value'] = \
        H*T*K*D
    flops['all'] = sum(flops.values())
    return flops

def calc_dense(H, T, D, **kwargs):
    flops = {}
    # Q(H, T, D) x K(H, D, T)
    flops['attention'] = H*T*D*T
    # softmax
    flops['softmax'] = H*T*T
    # A(H, T, T) x V(H, T, D)
    flops['value'] = H*T*T*D
    flops['all'] = sum(flops.values())
    return flops

def main_calc():
    parser = argparse.ArgumentParser()
    parser.add_argument('--t', type=int, default=2048)
    parser.add_argument('--m', type=int, default=128)
    parser.add_argument('--h', type=int, default=12)
    parser.add_argument('--d', type=int, default=64)
    parser.add_argument('--k', type=int, default=16)
    args = parser.parse_args()
    
    print('for', args)
    
    kwargs = {
        'H': args.h,
        'T': args.t,
        'D': args.d,
        'M': args.m,
        'K': args.k,
    }

    print('flops perlin', {k: v / (1024**3) for k, v in calc_perlin(**kwargs).items()})
    print('flops dense', {k: v / (1024**3) for k, v in calc_dense(**kwargs).items()})

def main_plot():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, default=64)
    parser.add_argument('--h', type=int, default=12)
    parser.add_argument('--d', type=int, default=64)
    args = parser.parse_args()
    
    ts = [2**(x/10) for x in range(50, 140)]
    ks = [2**x for x in range(5, 8)]
    
    perlins = [
        [calc_perlin(H=args.h, T=t, D=args.d, M=args.m, K=k)['all'] / (1024**3) for t in ts]
        for k in ks
    ]
    denses = [calc_dense(H=args.h, T=t, D=args.d)['all'] / (1024**3) for t in ts]
    
    plt.plot(ts, denses, label='Quadratic')
    for ik, k in enumerate(ks):
        plt.plot(ts, perlins[ik], label=f'Ours(k={k}, K=64)')
    root = './saves/poc/neko/calc_flops_comp/'
    os.makedirs(root, exist_ok=True)
    plt.grid()
    plt.legend()
    plt.xlabel('Sequence Length')
    plt.ylabel('GFLOPs')
    # plt.yscale('log')
    plt.title('FLOPs Comparision Between SEA (Ours) And Quadratic Attention')
    path = os.path.join(root, 'plot')
    plt.savefig(path+'.png', dpi=300, bbox_inches='tight')
    plt.savefig(path+'.pdf', dpi=300, bbox_inches='tight')
    print('saved', path+'.png')

if __name__ == '__main__':
    main_calc()
    main_plot()