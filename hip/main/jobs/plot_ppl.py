import matplotlib.pyplot as plt

def proc_copy_paste(t: str):
    return list(map(float, t.split()))

FLASH_ATTN = 'flash_attn'
HIP_ATTN = 'hip_attn'
BIGBIRD = 'bigbird'
SLLM = 'streaming_llm'
H2O = 'h2o'
HYPER_ATTN = 'hyper_attention'

prefill_data = {
    FLASH_ATTN: proc_copy_paste('3.82	13.49	53.47	213.46	861.40'),
    HIP_ATTN: proc_copy_paste('3.22	7.67	18.14	42.20	95.69'),
    BIGBIRD: proc_copy_paste('1.41	2.87	6.30	14.47	30.97'),
    SLLM: proc_copy_paste('178.8	359.5	721.2	NaN	NaN'),
    H2O: proc_copy_paste('36.6	142.6	569.5	2563	12576'),
    HYPER_ATTN: proc_copy_paste('15.65	36.40	80.52	NaN	NaN'),
}

decode_data = {
    FLASH_ATTN: proc_copy_paste('0.0360	0.0713	0.1418	0.2822	0.5639'),
    HIP_ATTN: proc_copy_paste('0.0149	0.0159	0.0168	0.0180	0.0188'),
    BIGBIRD: proc_copy_paste('0.0131	0.0135	0.0137	0.0138	0.0138'),
    SLLM: proc_copy_paste('0.0134	0.0134	0.0134	NaN	NaN'),
    H2O: proc_copy_paste('0.1941	0.1946	0.1996	0.1972	0.1949'),
    HYPER_ATTN: proc_copy_paste('NaN	NaN	NaN	NaN	NaN'),
}

pg19_data = {
    FLASH_ATTN: proc_copy_paste('8.7684	8.5071	8.3100	8.1655	8.1151'),
    HIP_ATTN: proc_copy_paste('8.7994	8.6028	8.5057	8.4810	8.6499'),
    
}