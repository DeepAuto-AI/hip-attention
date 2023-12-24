H = 12
T = 8192
Tm = 256
d = 64
TOPK = 64
Q = K = V = (T, d)

def t(a):
    return (a[1], a[0])

def mm(a, b):
    assert a[1] == b[0]
    return a[0] * a[1] * b[1]

def layernorm(x):
    return x[0] * x[1] * 2

def conv(x, outch, k=3):
    C, H, W = x
    return k*k*C*outch*H*W

def calc_performer():
    # V_concat = [V_pos, V_atten]
    # Q x (K^t x V_concat)
    V_concat = (T, 2*d)
    flops = 0
    
    flops += mm(Q, (d, d)) #phi, nbf8
    flops += mm(K, (d, d)) #phi, nbf8
    flops += mm(t(K), V_concat)
    flops += mm(Q, (K[1], V_concat[-1]))
    
    return flops * H

def calc_predictor():
    print(T, d, H, Tm)
    
    flops = 0
    #attention_predictor_enc
    Vinp = (T, 3*d)
    flops += mm(Vinp, (3*d, 2*d)) * H
    X = (T, 2*d)
    flops += layernorm(X) * H
    flops += mm(X, (2*d, Tm*2)) * H
    X = (H*4, T, Tm/2)
    conv_flops = 0
    conv_flops += conv(X, 4*H) / 2 #divide by stride
    X = (H*4, T/2, Tm/2)
    conv_flops += conv(X, 4*H)
    X = (H*4, T, Tm)
    conv_flops += conv(X, H)
    
    print(flops/1000000, conv_flops/1000000)
    flops += conv_flops
    
    # attention_predictor_dec_scaler
    X = (T, 2*d)
    flops += mm(X, (2*d, 2)) * H
    
    return flops

def topk(x, k):
    # T * (Tm + k)
    return x[0] * (x[1] + k)

def calc_mask():
    flops = 0
    
    # topk (t+k), which k is negligible, per-query
    X = (T, Tm)
    flops += topk(X, round(TOPK*(Tm/T)))
    
    return flops * H

def calc_interp():
    flops = 0
    
    # point location calculation
    X = (T, Tm)
    flops += X[0] * X[1] * 16
    
    # allocation
    X = (T, TOPK)
    flops += X[0] * X[1] * 2
    
    return flops * H

def calc_attention():
    flops = 0
    
    print('atten', T, TOPK, d)
    
    #mbmm
    X = (T, TOPK)
    flops += X[0] * X[1] * d
    #softmax
    X = (T, TOPK)
    flops += X[0] * X[1] * 4
    #elmul
    X = (T, TOPK)
    flops += X[0] * X[1]
    #sdbmm
    flops += X[0] * X[1] * d
    X = (T, d)
    
    # weighted pool
    flops += X[0] * X[1] * 4
    
    return flops * H

data = {
    'performer': calc_performer(),
    'perdictor': calc_predictor(),
    'mask': calc_mask(),
    'interp': calc_interp(),
    'attention': calc_attention(),
}

GFLOPS = 1000000000
for k, v in data.items():
    print(k, f"{int(v)/GFLOPS:,} gmac")
    
dense = data['performer'] + data['perdictor'] + data['mask']
sparse = data['interp'] + data['attention']

print('dense', f"{int(dense)/GFLOPS:,} gmac", dense / (dense+sparse))
print('sparse', f"{int(sparse)/GFLOPS:,} gmac", sparse / (dense+sparse))