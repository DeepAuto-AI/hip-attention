from neuronxcc import nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

@nki.jit
def scan_stage_neuron(
    Q,
    K,
    INDICES_LEFT,
    INDICES_RIGHT,
    CHUNK_SCORES,
    POS_TDST,
    
    BLOCK_SIZE_Q,
    BLOCK_CHUNK,
):
    BSZ, TDST, HEAD, HID = Q.shape
    _, TSRC, HEAD_KV, _ = K.shape
    _, BDST, _, N_CHUNK = INDICES_LEFT.shape
    assert INDICES_LEFT.shape == INDICES_RIGHT.shape
    assert INDICES_LEFT.shape == CHUNK_SCORES.shape
    HEAD_GROUP = HEAD // HEAD_KV
    
    BN_CHUNK = N_CHUNK // BLOCK_CHUNK
    
    out_indices_left = nl.ndarray(
        INDICES_LEFT.shape, 
        dtype=INDICES_LEFT.dtype,
        buffer=nl.shared_hbm
    )
    out_indices_right = nl.ndarray(
        INDICES_RIGHT.shape, 
        dtype=INDICES_RIGHT.dtype,
        buffer=nl.shared_hbm
    )
    out_chunk_scores = nl.ndarray(
        CHUNK_SCORES.shape, 
        dtype=CHUNK_SCORES.dtype,
        buffer=nl.shared_hbm
    )
    
    idx_bsz = nl.program_id(0)
    idx_bdst = nl.program_id(1)
    idx_head = nl.program_id(2)
    idx_bchunk = nl.program_id(3)
    
    # for idx_bsz in nl.sequential_range(BSZ):
    #     for idx_bdst in nl.sequential_range(BDST):
    #         for idx_head in nl.sequential_range(HEAD):
    #             for idx_bchunk in nl.sequential_range(BN_CHUNK):
    
    idx_tdst = nl.arange(BLOCK_SIZE_Q)[:, None] + idx_bdst * BLOCK_SIZE_Q
    idx_hid = nl.arange(0, HID)
    queries = nl.load(
        Q[
            idx_bsz,
            idx_tdst,
            idx_head,
            idx_hid[None, :]
        ]
    )
    pos_tdst = nl.load(
        POS_TDST[
            idx_bsz,
            idx_tdst,
        ]
    )
    
    idx_chunk = nl.arange(0, BLOCK_CHUNK)[None, :] + idx_bchunk * BLOCK_CHUNK
    indices_left = nl.load(
        INDICES_LEFT[
            idx_bsz,
            idx_bdst,
            idx_head,
            idx_chunk,
        ]
    )
    indices_right = nl.load(
        INDICES_RIGHT[
            idx_bsz,
            idx_bdst,
            idx_head,
            idx_chunk,
        ]
    )
    
    out_scores = nl.load(
        CHUNK_SCORES[
            idx_bsz,
            idx_bdst,
            idx_head,
            idx_chunk,
        ],
    )
    
    # max_chunk_size = 256
    # while max_chunk_size >= 1:
    #     max_chunk_size /= 2
    
    indices_center = (indices_left + indices_right) // 2
    
    k_left = nl.load(
        K[
            idx_bsz,
            ((indices_left + indices_center) // 2),
            idx_head // HEAD_GROUP,
            idx_hid[:, None],
        ]
    )
    scores_left_psum = nl.matmul(
        queries, nl.transpose(k_left),
    )
    scores_left = nl.copy(
        scores_left_psum,
        dtype=CHUNK_SCORES.dtype
    )
    mask_left_lhs = nl.zeros((BLOCK_CHUNK, BLOCK_SIZE_Q), dtype=indices_left.dtype) + ((indices_left + indices_center) // 2)
    mask_left_rhs = nl.zeros((BLOCK_SIZE_Q, BLOCK_CHUNK), dtype=indices_left.dtype) + pos_tdst
    mask_left = mask_left_lhs <= nl.transpose(mask_left_rhs)
    scores_left = nl.where(mask_left, nl.transpose(scores_left), -32000.0) 
    scores_left = nl.max(scores_left, axis=-1)
    
    k_right = nl.load(
        K[
            idx_bsz,
            ((indices_center + indices_right) // 2),
            idx_head // HEAD_GROUP,
            idx_hid[:, None],
        ]
    )
    scores_right_psum = nl.matmul(
        queries, nl.transpose(k_right),
    )
    scores_right = nl.copy(
        scores_right_psum,
        dtype=CHUNK_SCORES.dtype
    )
    mask_right_lhs = nl.zeros((BLOCK_CHUNK, BLOCK_SIZE_Q), dtype=indices_left.dtype) + ((indices_center + indices_right) // 2)
    mask_right_rhs = nl.zeros((BLOCK_SIZE_Q, BLOCK_CHUNK), dtype=indices_left.dtype) + pos_tdst
    mask_right = mask_right_lhs <= nl.transpose(mask_right_rhs)
    scores_right = nl.where(mask_right, nl.transpose(scores_right), -32000.0)
    scores_right = nl.max(scores_right, axis=-1)
    
    indices_left = nl.where(
        nl.greater(scores_left, scores_right),
        indices_left,
        indices_center,
    )
    indices_right = nl.where(
        nl.greater(scores_left, scores_right),
        indices_center,
        indices_right,
    )
    out_scores[...] = nl.maximum(
        nl.maximum(scores_left, scores_right), 
        out_scores
    )
    
    nl.store(
        out_indices_left[
            idx_bsz,
            idx_bdst,
            idx_head,
            idx_chunk,
        ],
        value=indices_left,
    )
    nl.store(
        out_indices_right[
            idx_bsz,
            idx_bdst,
            idx_head,
            idx_chunk,
        ],
        value=indices_right,
    )
    nl.store(
        out_chunk_scores[
            idx_bsz,
            idx_bdst,
            idx_head,
            idx_chunk,
        ],
        value=out_scores,
    )
    
    return out_indices_left, out_indices_right, out_chunk_scores