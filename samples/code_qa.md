You are a professional low-level code developer, that very skilled with GPU kernel programming. Recently your project is using OpenAI Triton, which is pythonic GPU kernel writing languages.

I will give code snippets about HiP Attention. HiP Attention is a sparse attention mechanism to reduce complexity of attention mechanism. I will provide the paper related to HiP Attention (HiP Attention, InfiniteHiP).

### Paper: `A Training-free Sub-quadratic Cost Transformer Model Serving Framework With Hierarchically Pruned Attention`

```latex
\begin{abstract}
% Transformer-based generative models have revolutionized machine learning, empowering various applications such as advanced chatbots.

% 1. Needs for long context length in recent LLMs & heavy-complexity problem
% TODO: HJ
% 2. Prior work(sub-quadratic methods)'s limitations:  needs for model-specific further training
% (1) Linear attention methods
% (2) Sparse attention methods
% 3. Our method

In modern large language models (LLMs), increasing the context length is crucial for improving comprehension and coherence in long-context, multi-modal, and retrieval-augmented language generation.
While many recent transformer models attempt to extend their context length over a million tokens, they remain impractical due to the quadratic time and space complexities.
Although recent works on linear and sparse attention mechanisms can achieve this goal, their real-world applicability is often limited by the need to re-train from scratch and significantly worse performance. In response, we propose a novel approach, Hierarchically Pruned Attention (HiP), which reduces the time complexity of the attention mechanism to $O(T \log T)$ and the space complexity to $O(T)$, where $T$ is the sequence length.
We notice a pattern in the attention scores of pretrained LLMs where tokens close together tend to have similar scores, which we call ``attention locality''. Based on this observation, we utilize a novel tree-search-like algorithm that estimates the top-$k$ key tokens for a given query on the fly, which is mathematically guaranteed to have better performance than random attention pruning. In addition to improving the time complexity of the attention mechanism, we further optimize GPU memory usage by implementing KV cache offloading, which stores only $O(\log T)$ tokens on the GPU while maintaining similar decoding throughput. Experiments on benchmarks show that HiP, with its training-free nature, significantly reduces both prefill and decoding latencies, as well as memory usage, while maintaining high-quality generation with minimal degradation.
HiP enables pretrained LLMs to scale up to millions of tokens on commodity GPUs, potentially unlocking long-context LLM applications previously deemed infeasible.
\end{abstract}

\vspace{-0.24in}
\section{Introduction}
\vspace{-0.5em}
\label{sec:introduction}
%  suggested overall structure:
%       (1) Background (motivation)
%       (2) Prior works's limitations
%       (3) Our HiP: {method, effect}
%       (4) Experimental results
%       (5) Summary of main contributions

%% Background: needs for long-sequence and limitations of the existing self-attention mechanism from the perspective of complexity
Large Transformer-based generative language models (LLM) trained on huge datasets have recently demonstrated remarkable abilities in various problem domains, such as natural language understanding~\citep{touvron_llama_2023}, code generation~\citep{roziere_code_2024}, and multi-modal question answering~\citep{liu_improved_2023}.
This is made possible by the effectiveness of the attention mechanism, which learns $T^2$ pairwise relationships between all tokens in a sequence of $T$ tokens.
Despite their success, the quadratic complexity of the attention mechanism makes it increasingly challenging to meet growing resource demands when processing longer sequences.

%% prior works's limitations
Various approaches have been suggested to handle longer sequences efficiently to overcome this limitation.
FlashAttention~\citep{dao_flashattention_2022, dao_flashattention-2_2023} has reduced the space complexity to $O(T)$ by fusing the component computations to avoid storing $T^2$ attention scores at one time.
However, its time complexity remains $O(T^2)$, making it less applicable to inference tasks with long contexts.
Many other methods~\citep{lee_sea_2023, beltagy_longformer_2020, zaheer_big_2020, tay_sparse_2020, kitaev_reformer_2019, tay_synthesizer_2021, liu_transformer_2021} tackle the issue by sparsifying the attention matrix or approximate the attention mechanism using kernel methods to reduce its quadratic complexity.
However, these works are not widely employed in real-world LLM serving frameworks because they often lead to performance degradation due to drastic changes in the computation flow and are too complex to implement efficiently for actual speedups.
Moreover, they often require extensive fine-tuning or even pre-training from scratch, which can be prohibitively expensive and prevent the timely deployment of production-ready pre-trained models.

\begin{figure}[t]
\centering
\vspace{-0.5in}
\includegraphics[width=1.0\linewidth]{figures/hip_intro_concept.pdf}
\vspace{-1.8em}
\caption{\small \textbf{HiP Attention.} HiP dynamically prunes block sparse attention depending on a given query token in sub-quadratic cost by utilizing the hierarchy and locality of natural language.}
\label{fig:intro_concept}
\vspace{-0.1in}
\end{figure}

%% Ours effectiveness (1) ??
In this paper, we define and achieve three fundamental objectives for frameworks tailored to long-context transformer serving frameworks: (1) minimizing the algorithmic complexity of attention mechanisms, (2) enhancing GPU compute efficiency, particularly through TensorCore utilization, and (3) maximizing the effective use of limited GPU memory capacity.

First, to serve long sequence in a timely manner, we propose \textbf{Hi}erarchically \textbf{P}runed Attention (HiP), an efficient training-free attention mechanism reducing the quadratic time complexity to $O(T\log T)$ by approximating the top-$k$ key tokens in a sequence.
HiP exploits ``attention locality'', where neighboring tokens often have similar attention scores, as shown in~\cref{fig:intro_concept} (Left).
Therefore, as shown in~\cref{fig:intro_concept} (Center), HiP divides the input sequence into $2k$ chunks, and the center token in each chunk is chosen to represent its neighbors, driven by the attention locality within the chunk.
HiP computes the attention scores of these representative tokens to approximate the importance of each chunk for a given query.
% Only the top-$k$ most important chunks are selected, and this process is applied iteratively, refining the selection until each chunk contains a single token.
HiP iteratively refines its selection by starting with the top-$k$ most important chunks and progressively narrowing them down until each chunk contains a single token.
This hierarchical top-$k$ key estimation takes $O(T \log T)$ time, which is used for sparse attention computation that costs $O(T)$, making the overall complexity of our attention mechanism log-linear.
We provide mathematical proof demonstrating that our HiP outperforms random selection, supported by empirical evidence from attention score statistics in~\cref{sec:theory}.

Second, we introduce hardware-aware optimizations to enhance GPU compute efficiency for our HiP through block-wise key sparsity, as illustrated in~\cref{fig:intro_concept} (Right).
Specifically, our top-k approximation is implemented in a tiled manner~\citep{tillet_triton_2019} so that it can fully utilize matrix multiplier units (MMUs; e.g., TensorCores~\citep{nvidia2024tensorcore}) and achieve the highest possible token-processing throughput.
Additionally, we integrate our attention mechanism into throughput-optimized LLM serving frameworks, such as vLLM~\citep{kwon_vllm_2023} and SGlang~\citep{zheng2024sglang}, further enhancing deployment efficiency.

Lastly, to serve extremely long sequences within the limited GPU memory, we propose a KV cache management strategy that stores only $O(\log T)$ tokens in GPU memory (HBM) and offloads the remaining tokens to host memory (DRAM).
The $O(\log T)$ tokens stored in GPU memory are the ones accessed most frequently and are meant to provide quick access for the GPU's MMUs.
In contrast, other less frequently accessed tokens reside in main memory and are transferred to GPU memory only upon token access misses.
With a high access hit ratio in HiP, our memory management scheme effectively meets the demand for limited HBM capacity while leveraging the larger DRAM capacity, preventing token access from becoming a bottleneck.

We validate HiP on various benchmarks by applying it to Llama3.1-8B~\citep{dubey2024llama3}.
In LongBench~\citep{bai_longbench_2023}, HiP maintains 96\% of its relative performance while achieving almost \textbf{$2.7\times$} speedup in the prefill stage and \textbf{$16.5\times$} speedup attention computation in the decode stage with 32k context length compared to Flash Attention.
Additionally, in passkey retrieval tasks such as RULER~\citep{hsieh2024ruler}, HiP preserves its original effective context length, while all baselines fail to do so.
We also evaluate the effectiveness of the proposed KV cache offloading framework.
On a machine capable of serving up to a 16k context length with Flash Attention, our method extends the context length up to 64k by offloading the KV cache without significant throughput degradation.
% Despite some CPU memory being accessed from the GPU during decoding, we are able to maintain competitive latency compared to setups without KV cache offloading.

% In conclusion, by fusing three proposed solutions to a long-context serving framework, we can provide the usability and transparency of a long-context framework that internally manages compute and memory resources wisely.
% We believe that this increased context length within the same space and compute budget greatly benefits long-context applications such as question answering with long textbooks~\citep{kryscinski_booksum_2022}, multi-agent~\citep{hu2024ADAS} chatbots, enhanced retrieval-augmented reasoning, long video data summarization, and more.
% Furthermore, due to its training-free nature, our method can be applied directly to pretrained LLMs without further training. Therefore, we expect our work to be highly practical for long-context LLM applications.
In conclusion, by integrating the three proposed solutions, we present a single long-context serving framework that efficiently manages compute and memory resources while being transparent and easily usable.
This extension of serving context length, achieved within the constraints of limited space and compute budgets, delivers substantial benefits for long-context applications, such as question answering with long texts~\citep{kryscinski_booksum_2022}, multi-agent chatbots~\citep{hu2024ADAS}, enhanced retrieval-augmented reasoning, and long video data summarization.
Furthermore, since our approach is training-free, HiP can be seamlessly applied to pretrained LLMs without requiring additional training.
As a result, we expect our method to be highly practical for a wide range of long-context LLM applications.


% \newpage
Our contributions within the proposed framework can be summarized as follows:
\vspace{-0.25em}
\begin{itemize}[itemsep=0.5mm, parsep=2pt, leftmargin=12pt]
\item We propose a novel, training-free hierarchically pruned attention mechanism that uses hierarchical score-locality-aware top-$k$ approximation to accelerate LLM serving, reducing the quadratic cost of the attention mechanism to $O(T\log T)$ time and $O(T)$ space complexity (\cref{subsec:mask_estimation}).
\item We further optimize our HiP mechanism with a hardware-aware block-wise tiled optimization using OpenAI Triton, achieving up to speed up to $6.83\times$ speedup in end-to-end decoding for 128k context. (\cref{subsec:block_approx}, \cref{fig:decode_speedup_longbench})
\item We implement KV cache offloading to reduce GPU memory efficiency further, increasing serving context from 16k up to 64k tokens in an RTX 4090 with 8B model
(\cref{sec:method_kv_cache_offloading}).
\end{itemize}

\vspace{-0.8em}
\section{Related Works}
\vspace{-0.3em}
\label{sec:related_works}

% \input{figure_srcs/main_concept}

% In prior works, several attention approximation methods with linear complexity were proposed using kernel methods or sparse attention mechanisms.
% By low-rank approximation of softmax attention using kernel method~\citep{choromanski_rethinking_2022, qin_cosformer_2022} could achieve extremely fast inference speed with linear complexity.
% However, since the low-rank approximation changes the inference data flow graph by a large amount, the performance degradation of the kernel-based approaches is not negligible and hard to recover from.
% In contrast to low-rank approximation, sparse attention methods use attention pruning. The sparse attention methods can maintain trained attention scores; they recover well after a simple replacement (plug-and-play) of the pre-trained attention mechanisms.
% Still, sparse attention requires further fine-tuning in order to adapt to the new static attention patterns~\citep{beltagy_longformer_2020, bigbird_2020, xiao_streamingllm_2023}, or train the attention estimator~\citep{lee_sea_2023, liu_transformer_2021}.
% Furthermore, most implementations of them are not as efficient as fused attention~\citep{dao_flashattention_2022, dao_flashattention-2_2023}, because they cannot utilize tensor processing unit due to their fine-grained sparsities.
% A tensor processing unit (block matrix multiplication unit) is a critical feature of modern accelerators that computes a part of matrix multiplication in one or a few cycles instead of computing every fused-multiply-add one by one.
% We are especially inspired by \cite{lee_sea_2023}, the sparse attention framework using an attention mask estimator. Please take a look \cref{sec:related_works_appendix} for further discussion.

Previous studies proposed several attention approximations with linear complexity using either kernel methods or sparse attention.
Low-rank approximations of softmax attention via kernel methods \citep{choromanski_rethinking_2022, qin_cosformer_2022} achieve faster inference speeds but significantly alter the data flow, leading to performance degradation that is hard to mitigate.
In contrast, sparse attention methods, which use attention pruning to preserve trained attention scores, allow for simple replacement of pre-trained mechanisms.
However, they often require additional fine-tuning to adapt to static attention patterns \citep{beltagy_longformer_2020, bigbird_2020, xiao_streamingllm_2023} or the training of an attention estimator \citep{lee_sea_2023, liu_transformer_2021}.
These methods are generally less efficient than fused attention techniques \citep{dao_flashattention_2022, dao_flashattention-2_2023} due to their fine-grained sparsity, which prevents optimal MMU utilization.
% Modern MMUs can compute blocks of matrix multiplication in a few clock cycles, greatly surpassing the speed of scalar or even vector operations.
% We are particularly inspired by \cite{lee_sea_2023}, which proposes a sparse attention framework using an attention mask estimator.
For more details, see \cref{sec:related_works_appendix}.

\vspace{-0.3em}
\section{Methodology}
\vspace{-0.3em}
\label{sec:method}

\input{figure_srcs/main_concept}

Given query, key, and value sequences $\bm{Q}, \bm{K}, \bm{V} \in \mathbb{R}^{T\times d}$, the conventional single-head attention output $\bm{O}$ is computed as
$\bm{S} = \bm{Q}\bm{K}^\top \in \mathbb{R}^{T\times T}$,
$\bm{P} = \mathrm{softmax}(\bm{S}) \in \mathbb{R}^{T\times T}$,
$\bm{O} = \bm{P}\bm{V} \in \mathbb{R}^{T\times d}$,
where $d$ denotes embedding dimension, and softmax is applied row-wise. The causal masking and constant scaling are omitted for brevity.
The $\bm{S}$ and $\bm{P}$ matrices are respectively called the \textit{attention scores} and \textit{probabilities}.
We focus on the fact that, due to the nature of the softmax function, only the highest attention scores significantly impact the output.
Therefore, a promising approach to approximating $\bm{S}$ in a sparse format and reducing the complexity from $O(T^2)$ is to retain only its top-$k$ elements, as detailed in the following equations:
\begin{gather}
    \bm{M} = \mathrm{top\_}k\mathrm{\_mask} \left( \bm{Q}\bm{K}^\top \right) \in \{0, 1\}^{T\times T}, \label{eq:def_mask}\\
    \widehat{\bm{S}} = \mathrm{mask}_{\bm{M}}(\bm{Q}\bm{K}^\top) \in \mathbb{R}^{T\times T},\quad
    \widehat{\bm{P}} = \mathrm{softmax}(\widehat{\bm{S}}) \in \mathbb{R}^{T\times T},\quad
    \widehat{\bm{O}} = \widehat{\bm{P}}\bm{V} \in \mathbb{R}^{T\times d},\label{eq:phat_ohat}\\
    \text{where }[\mathrm{mask}_{\bm{M}}(\bm{S})]_{i,j} := \begin{cases} \emS_{i,j} & \text{if } \emM_{i,j} = 1 \\ -\infty & \text{if } \emM_{i,j} = 0 \end{cases},\label{eq:mask_m}
\end{gather}
where $\mathrm{top\_}k\mathrm{\_mask} (\cdot)$ denotes a binary mask which selects the top-$k$ largest elements for each row of the given matrix.
Since $\widehat{\bm{S}}$ is a sparse matrix with only $kT$ valid elements, $\widehat{\bm{S}}$ and $\widehat{\bm{O}}$ in \Cref{eq:phat_ohat} can be computed in $O(T)$ time using sparse matrix operations.

However, obtaining the binary mask $\bm{M}$ in sub-quadratic time is no easy task.
To address this challenging problem, we exploit what we call ``attention locality''. Observation of attention scores reveal that the scores tend to exhibit local similarity, a phenomenon we refer to as attention locality.
We exploit this observation by performing a tree-based search for the top-$k$ tokens.
We divide the sequence into $2k$ chunks, and then select a representative token from each chunk.
Due to attention locality, a representative token have similar scores to other tokens in its chunk - thereby ``representing'' that chunk.
We select the top-$k$ most important chunks based on the attention scores of the representative tokens.
By repeating this process, we refine the tokens until we can no longer divide chunks.
Exact details of our method are shown in \Cref{subsec:mask_estimation}.
We only cover the single-head non-causal case here, but note that our method can easily be extended to causal multi-head attention.

% \subsection{Hierarchical Approximate Top-$k$ Key Selection}
\subsection{Hierarchical Score-Locality-Aware Top-$k$ Estimation}
\label{subsec:mask_estimation}

% \input{figure_srcs/masking_iteration_concept}

As shown in \Cref{eq:def_mask}, our goal is to select the top-$k$ largest elements of each row of pre-trained attention score $S$ without computing the entire matrix.
To this end, we use a greedy binary tree search algorithm, as illustrated in the left side of \Cref{fig:concept}.
The complete algorithm for mask estimation is presented in \Cref{alg:mask_estimation}.

For a given query $\bm{q} \in \mathbb{R}^d$, at the first iteration, we divide the key sequence $\bm{K}\in\mathbb{R}^{T\times d}$ along the time dimension into $k$ equal-sized chunks $(f^{(1)}_1:l^{(1)}_1), (f^{(1)}_2:l^{(1)}_2), \dots, (f^{(1)}_k:l^{(1)}_k)$,
where $f^{(1)}_j = \round{\frac {(j-1)\cdot T}{k}}+1$ and $l^{(1)}_j = \round{\frac{j \cdot T}{k}}$ are the first and last indices of the $j$th chunk, each.\footnote{\round{\cdot} denotes rounding to the nearest integer.} The superscripts denote the iteration number.
At each iteration $i$, we further divide each of the $k$ chunks into two equal-sized \textit{branches}:
$$\mathcal{B}^{(i)}_{2j - 1} = (f^{(i)}_j , m^{(i)}_j - 1),
~~\mathcal{B}^{(i)}_{2j} = (m^{(i)}_j , l^{(i)}_j),
\text{ where }m^{(i)}_j = \round{(f^{(i)}_j + l^{(i)}_j) / 2},
\text{ for } j = 1~..~k.$$
A representative key index $r^{(i)}_j$ is the center key token index for each branch $\mathcal{B}^{(i)}_{j}$. We assume that this representative key represents the entire branch. Thus, among the $2k$ branches, the top $k$ branches whose representative key's scores are the highest are chosen for the next iteration:
\begin{align}
(f^{(i+1)}_j, l^{(i+1)}_j) := \mathcal{B}^{(i)}_{t_j} \text{ for } j = 1~..~k,\text{ where } \{t_1, \dots, t_k\} := \underset{j\in[1~..~2k]}{\mathrm{argtop}_k} \left[ \bm{q}^\top \bm{K}_{r^{(i)}_j,:} \right].
\label{eq:branch_selection}
\end{align}
We repeat the above iteration $n_{it} := \left\lceil\log_2 T\right\rceil$ times, i.e., until the length of each branch all becomes 1. In the end, we obtain a set of indices $\mathcal{I} = \{ f_1^{(n_{it})}, \dots, f_k^{(n_{it})} \}$, which is our estimation of the top-$k$ indices of $\bm{K}$ which have the largest attention scores with the query $\bm{q}$. Thus, we obtain $\widehat{\bm{m}}$, an estimation of a row of the attention mask $\bm{M}$\footnote{$\mathds{1}_{\mathcal{A}}(x)$, where $\mathcal{A}$ is a set, denotes the indicator function: $\mathds{1}_{\mathcal{A}}(x) = 1$ if $x \in \mathcal{A}$, and otherwise $\mathds{1}_{\mathcal{A}}(x) = 0$.}:
\begin{align}
\widehat{\bm{m}} = \mathrm{estimate\_attn\_mask}_k(\bm{q}, \bm{K}) := \left[\mathds{1}_{\mathcal{I}}(1), \mathds{1}_{\mathcal{I}}(2), \dots, \mathds{1}_{\mathcal{I}}(d)\right].
\end{align}
In conclusion, this algorithm takes $O(T\log T)$ time in total because the total number of iterations is $\log_2 T$ where each iteration takes constant time $O(k)$, and we do this for each of the $T$ queries.

\subsection{Block Approximation of Top-$k$ Estimation}
\label{subsec:block_approx}

%\input{figure_srcs/block_approx_concept}

Despite the log-linear complexity, obtaining competitive latency to the state-of-the-art implementations of dense attention on an accelerator (e.g., GPU) is difficult.
This is because the matrix multiplier unit (MMU) inside accelerators is optimized for dense attention, where they compute fixed-size blocks of matrix multiplication in a few clock cycles.
In contrast, the attention score computation in the top-$k$ estimation of HiP cannot be performed with traditional matrix multiplication because a different key matrix is used to compute the dot product for each query vector.
To utilize MMU, we use a technique called \textit{block approximation} during top-$k$ estimation, illustrated in \Cref{fig:concept} (Right).

In top-$k$ estimation, we replace $\bm{K} \in \mathbb{R}^{T\times d}$ with its tiled version $\bm{\mathsf{K}} \in \mathbb{R}^{T/b_k\times b_k\times d}$, and $\bm{Q}$ with its tiled version $\bm{\mathsf{Q}} \in \mathbb{R}^{T/b_q\times b_q\times d}$, where $b_k$ and $b_q$ are the size of a key block and a query block.
The top-$k$ estimation iterations are done similarly to before, except that the division and branching of the key sequence are done block-wise (using the first dimension of $\bm{\mathsf{K}}$). Importantly, instead of $k$, $k / b_k$ chunks are maintained at each iteration in order to select $k$ tokens, and the score calculation in \Cref{eq:branch_selection} is replaced with $\max_{\scriptstyle m\in [1:b_q], \scriptstyle n\in [1:b_k]}\left(\bm{q}_{m, :}^\top \bm{\mathsf{K}}_{l_j^{(i)}, n, :}\right),$
where $\bm{q} \in \mathbb{R}^{b_q \times d}$ is the given query block.
While this modification enables HiP to reduce the cost further, we internally sample the blocks with stride $b_{sq}$ in the query dimension and $b_{sk}$ in the key dimension instead of using the full $b_q \times b_k$ block.
%The sampled values are packed in local memory so we can still fully utilize the MMU.
%To summarize, we compare the maximum score values in the representative $(b_q/b_{sq}) \times (b_k/b_{sk})$-sized block of each branch.

As a result of this optimization, the estimated mask $\widehat{\bm{M}}$ becomes block-sparse.
Therefore, each $(b_q / b_{sq}) \times d$-block of the query can be matrix-multiplied with the same $(k / b_{sk}) \times d$ key matrix to obtain $(b_q / b_{sq}) \times (k / b_{sk})$ elements of $\widehat{\bm{S}}$.
Thus, $b_q$ and $b_{sq}$ are critical for the most efficient utilization of the MMU:
we can achieve a considerable latency reduction if we set $b_q / b_{sq}$ to a multiple of 16 or 32, as shown in \cref{sec:ablation_bq_bk}.
While the choice of $b_k$ and $b_{sk}$ is irrelevant to the MMU utilization, it helps reduce the number of top-$k$ estimation iterations.
% In \Cref{sec:analysis_blocksize}, we analyze the effect of block size on the latency and accuracy of the model.

% \subsubsection{HiP on Large Multimodal Models}
% % Since our HiP can seamlessly substitute the self-attention operation in any large language models, we can easily apply HiP to the language model part~(\eg, LLaMA) within large multimodal models, \eg, LLaVA-1.6~\citep{liu2024llavanext}.
% % In large multimodal models (LMMs) that process inputs beyond language tokens, such as image~\citep{liu_llava_2023,liu2024llavanext}, video~\citep{damonlpsg2023videollama}, and audio~\citep{lyu2023macaw} modalities, the computational demand increases significantly. As these models integrate large language models (LLMs) to process diverse modalities' tokens, the burden on the self-attention mechanisms grows. Thus, our HiP can seamlessly substitute the self-attention in these LMMs, ensuring that the efficiency gains observed in LLMs can be extended to LMMs as well. This capability allows HiP to enhance the overall efficiency of LMMs by reducing computational overhead and facilitating faster processing speeds.
% The large multimodal models (LMMs) process inputs beyond language tokens from domains such as  image~\citep{liu_llava_2023,liu2024llavanext}, video~\citep{damonlpsg2023videollama}, and audio~\citep{lyu2023macaw} modalities, and therefore the burden on the self-attention mechanism grows and need to handle longer context in general. Our HiP can seamlessly substitute the self-attention in these LMMs as well, ensuring that the efficiency gains observed in LLMs can be also extended to LMMs.

\begingroup
\setlength{\columnsep}{6pt}%

\subsection{KV Cache Offloading}
\label{sec:method_kv_cache_offloading}

\begin{wrapfigure}[12]{r}{0.52\textwidth}
\setlength{\columnsep}{0pt}%
\vspace{-2.82em}
\centering
\resizebox{1.0\linewidth}{!}{
\includegraphics[trim=0mm 0mm 39mm 0mm,clip]{figures/kv_cache_offload.pdf}
}
\vspace{-1.8em}
\caption{\small \textbf{Flow of KV Cache Offloading with HiP.}}
\label{fig:offloading_diagram}
\vspace{-0.0in}
\end{wrapfigure}

Thanks to our top-$k$ estimation algorithm, HiP only accesses $(k / b_{sk}) \log{T}$ key states per attention head.
Moreover, the algorithm's memory access pattern exhibits strong temporal locality.
Using this fact, we can further enhance efficiency by exploiting the memory hierarchy: we offload less frequently accessed key-value (KV) states from the GPU to the main memory.
This involves caching frequently accessed KV states (hot tokens) by tracking state access patterns of top-$k$ estimation and sparse attention using the estimated HiP mask.

\begin{wrapfigure}[14]{r}{0.21\textwidth}
\setlength{\columnsep}{0pt}%
\vspace{-0.5em}
\centering
\resizebox{1.0\linewidth}{!}{
\includegraphics[trim=90mm 0mm 0mm 0mm,clip]{figures/kv_cache_offload.pdf}
}
\vspace{-1.8em}
\caption{\small \textbf{KV Token Index Translation}}
\label{fig:offloading_index_transition_diagram}
\vspace{-0.0in}
\end{wrapfigure}

Our GPU cache that holds the hot tokens consists of two components: a \textit{token bank} containing the actual KV states and a \textit{page table} with the token-bank index mapping, as shown in~\cref{fig:offloading_index_transition_diagram}.
One straightforward implementation for the page table would be a vector map: a simple length-$T$ array of pointers.
While this approach is practical for typical sequence lengths (e.g., 128k - 1M), its space complexity is $O(T)$.
We employ a linear probing hash table to reduce the space complexity, achieving $O(\log T)$ space complexity.
However, empirical results show that GPU hash map lookups introduce additional latency compared to using a simpler vector-based page table.

Given the distinct memory access patterns in top-$k$ estimation and in sparse attention, we maintain two separate offloading contexts, each containing a page table and a set of GPU-resident hot tokens, as illustrated as two separate GPU loaded KV caches in~\cref{fig:offloading_diagram}.
For the top-$k$ estimation stage, $k_{\text{cache}} := c \cdot (k / b_{sk}) \log T$ key states are held in VRAM, where $c$ is a hyperparameter determining the cache size. For sparse attention, $k$ key and value states are held.
In summary, we need to hold $(k_{\text{cache}} / 2 + k)$ tokens' equivalent of KV states in the GPU.
The kernel first queries the GPU cache when accessing key or value tokens.
Upon a cache miss (which is unavoidable due to the dynamic nature of the attention access pattern), the system attempts to retrieve tokens from the main memory. % using CUDA unified virtual memory (UVM).
By using our cache, we can significantly speed up memory access compared to directly accessing CPU memory from the GPU.
% This is because we can minimize UVM’s overhead during page faults, which involves (1) CPU-GPU interrupts for page table synchronization and (2) Updating non-GPU-optimized page tables.

In conclusion, we reduce the GPU memory footprint for KV tokens from $O(T)$ to $O(\log T)$, but this comes with page table overhead that can range between $O(T)$ and $O(\log T)$ depending on the data structure used.
The overall space complexity is thus determined by the type of page table, allowing for a configurable trade-off between GPU memory efficiency and latency.
However, we suggest that users use vector maps in many practical long-context ranges (32-512k) to achieve competitive latency compared to Flash attention.
% With our proposed KV cache offloading method, we could extend serving context from 16k up to 64k with a single card of RTX 4090 while maintaining 93\% of decoding throughput.
Please refer to \cref{sec:experiments_offload} for detailed benchmarks.
\endgroup

\section{Theoretical Analysis}
\label{sec:theory}

In this section, we justify the design choices of our HiP's approximate top-$k$ key selection algorithm by answering the following questions: (1) Is HiP's key selection algorithm better than the random selection baseline at finding keys with the biggest scores? (2) How should the representative token in each branch be chosen?
%How should the representative token be selected? Also, is the tree-based hierarchical structure necessary? %Can't we randomly sample $k$ tokens and hence avoid all the overhead resulting from the hierarchical structure and top-$k$ selection?
We answer these questions by providing a probabilistic analysis of HiP's key selection algorithm in a simplified setting ($k$ = 1), based on the assumption of attention locality.

\paragraph{Observation: keys closer together exhibit higher similarity in attention scores.}
In each attention head of a layer in an LLM, a key sequence $\bm K \in \mathbb{R}^{T \times d}$ is used for computing the attention mechanism. Given a query vector $\bm q \in \mathbb{R}^d$, the scores for each key $\bm s = \bm K \bm q \in \mathbb{R}^T$ can be computed. We investigate how much locality these scores exhibit by studying the correlation between their distance $\Delta := |i - j|$ and the score difference $\delta_\Delta := s_i - s_j$ for every $i, j \in [1..T]$, with a sample natural language data.
As shown in \Cref{fig:attn_distribution}, our empirical observation shows that $\delta_\Delta$ generally follows a normal distribution, whose mean is almost zero and the standard deviation is an increasing function of distance $\Delta$. More details regarding this observation are provided in \Cref{sec:theoretical_assumptions}.

\input{figure_srcs/hip_theory_attn_distribution}

\paragraph{Analysis.} Based on this observation, we assume that we can approximate the difference in attention scores between two keys separated by $\Delta$ tokens as a scalar random variable $\delta_\Delta \sim \mathcal{N} \left( 0 , \sigma(\Delta)^2 \right)$, where $\sigma(\Delta)$ is an increasing function of $\Delta$.
This can be interpreted as keys that are closer together are more likely to have a similar attention score, which fits well with our observation and attention locality assumption. With this assumption, the following \Cref{thrm:hip_iteration} can be shown.

\vspace{.3em}
\begin{theorem}[Informal]
\label{thrm:hip_iteration}
    Consider the case of finding the location of the top-$1$ key token with the maximum attention score in a context of $T$ tokens. Suppose that our locality assumption holds true.
    We divide the context into two branches with $T/2$ keys each. Then, the branch whose center token has the bigger attention score is more likely to contain the top-$1$ key token.
\end{theorem}
\vspace{-.5em}

The above shows the effectiveness of one iteration of HiP's key selection algorithm.
By recursive application of HiP's key selection iterations, we can intuitively see that the probability of HiP's key selection algorithm finding the location of the top-$1$ key would be higher than that of uniform random selection as well.
Therefore, under the attention locality assumption, on average, HiP's key selection algorithm on average finds the best key tokens more often than random selection. This is also the basis for choosing the center key token as the representative in our algorithm. See \Cref{sec:theoretical_sketch} for the proof sketch and \Cref{sec:theoretical_proofs} for the formal statement and proof of the theorem.

\section{Experiments}
\label{sec:experiments}
\subsection{Experiment Settings}
Large Language Models (LLMs) are one of the most prominent models that utilize the attention mechanism.
Thus, we first apply our proposed HiP to Llama3.1-8B~\citep{touvron_llama_2023}, a pretrained LLM that is reported to perform well on various long-context natural language understanding tasks up to 128k context tokens, to evaluate the effectiveness of our HiP mechanism.
% In this section, we outline our experimental setup.
We replace all, but the initial $l_d$ attention layers with HiP in the pretrained LLM, where $L$ is the total number of layers, and $l_d$ denotes the remaining dense attention layers.
We choose $l_d$ through an ablation study (\cref{sec:dense_layer}).
During LLM decoding, we cache the sparse attention mask from the previous step and refresh it every $r_m$ step to reduce the decoding latency.
The latency-performance tradeoff of $r_m$ is discussed in~\cref{subsec:latency_breakdown}.
For a detailed description of HiP’s decoding process, see \cref{alg:decoding} in the appendix.
Further details on the hyperparameters are in \cref{sec:hyperparam}.

\textbf{Baselines.} We use several sparse attention baselines: \baselineA, StreamingLLM (SLLM)~\citep{xiao_streamingllm_2023}, \baselineAVD~\citep{jiang2024minference, li2024snapkv}, BigBird~\citep{bigbird_2020}, HyperAttention~\citep{han2024hyperattention}, and \baselineHeavyHeater~\citep{zhang_h2o_2023}, chosen for their training-free and sub-quadratic properties.
% These properties are inspired by prior works \citep{xiao_streamingllm_2023, bigbird_2020, beltagy_longformer_2020, zhang_h2o_2023, han2024hyperattention, jiang2024minference, li2024snapkv}.
Both StreamingLLM and \baselineA~use a combination of global sink tokens and sliding window~\citep{beltagy_longformer_2020}, with StreamingLLM additionally using rolling RoPE indexing \citep{xiao_streamingllm_2023}.
\baselineAVD~retains key vertical and diagonal lines in the prefill attention mask based on snapshot scores on top of \baselineA. As it is a prefill-oriented method, \baselineA~is used for decoding.
BigBird uses random masking along with the \baselineA~pattern.
HyperAttention is a token-clustering-style~\citep{kitaev_reformer_2019} attention mechanism.
Finally, \baselineHeavyHeater~retains the top-$k$ high-scoring KV tokens for the next step's KV cache.

% \subsection{Performance Evaluation}

\input{figure_srcs/latency_ppl}

\subsection{Language Modeling Performance Evaluation}
% We use the commonly used PG19 and WikiText2~\citep{merity_pointer_2016} dataset to evaluate the performance of HiP.
% We also fine-tune the pretrained models using LoRA~\citep{hu_lora_2021} on the Arxiv corpus~\citep{together2023redpajama} and perform the same evaluation.
% Additionally, we measure the latency in the two stages of text generation: (1) The initial pass (prompt, a.k.a. prefill), where the forward pass is computed on the entire prompt, and (2) the subsequent passes (decode), which are performed with cached key-value pairs and the query is only one token long each time.
% In \cref{fig:latency_ppl}, our proposed HiP attention is $9.00\times$ faster in prompt latency and $29.99\times$ faster in decoding latency in Llama3.1-8B while only suffering +0.5348 {\scriptsize (8.5057)} increase in PG19~\citep{raecompressive2019pg19} perplexity.
% Since our method fully utilizes the tensor processing unit by block approximation, our method is significantly faster than quadratic baselines and achieves near-linear decoding latency like BigBird.
% We describe further detail about the experiment setting in \cref{sec:hyperparam}.

We evaluate HiP on the PG19~\citep{raecompressive2019pg19} datasets.
We measure latency in two stages: (1) the initial pass (prefill), where the forward pass covers the entire prompt, and (2) subsequent passes (decode), which process one token at a time with a KV cache.
In \cref{fig:latency_ppl}, HiP attention is $9.00\times$ faster in prompt latency and $29.99\times$ faster in decoding latency on Llama3.1-8B, with only a +0.5348 increase in perplexity on PG19 {\scriptsize (8.1151 $\rightarrow$ 8.6499)}.
Our method leverages block approximation to maximize MMU efficiency, outperforming quadratic baselines and achieving near-linear decoding latency.
Further details on experimental settings are in \cref{sec:hyperparam}.

% We also test our proposed HiP and baselines on MMLU~\citep{mmlu} in \cref{sec:mmlu_result}. We also perform a comparison against linear attention methods which require training: Reformer~\citep{kitaev_reformer_2019} and SEA~\citep{lee_sea_2023} in \Cref{subsec:reformer_sea}, showing that the methods requiring training are far behind ours and other baselines within the same training budget.

% \input{figure_srcs/performance_radar}

\input{figure_srcs/passkey_table}

\subsection{Long Context Performance}

In this section, we investigate the performance of our HiP, comparing its latency and accuracy against baselines on various benchmarks.
Mainly, we build two kinds of benchmark sets: (1) long-context utilization to verify our method can retrieve the information in a given context using a needle in a haystack (NIAH) and (2) long-context natural language understanding to show that our method can preserve reasoning and text generation performance of original long-context LLM.
We apply the efficient attention method to mimic various deployment settings by replacing prefill, decode, or prefill-decode flash attention.
We can find our HiP performs robustly in every scenario compared to baselines, by applying efficient attention methods in different phases separately.

\input{figure_srcs/ruler_table}
\textbf{Passkey and RULER.} First, we analyze the result of long-context utilization performance using passkey retrieval in \cref{fig:passkey_result,,fig:ruler_result}.
Our passkey retrieval test is a simple test to find a five-digit passkey in a repeated haystack sentence.
RULER~\citep{hsieh2024ruler} is a more complex benchmark containing NIAH tests, such as finding multiple passkeys and tracking variable changes inside complicated essay-style haystack  sentences.
In \cref{fig:passkey_result}, our method is the strongest in every deployment setting.
Dense prefill in general scores high in this benchmark because the model has no chance of overlooking the passkey tokens.
However, interestingly, \baselineAVD~shows an almost perfect score with sparse prefill + dense decode. We think this is because the snapshot heuristic that captures important tokens during prefill is a perfect fit for this benchmark.
However, because of this aspect, it performs poorly on more complex tasks such as RULER and LongBench.
The combination of HiP and \baselineAVD~slightly increases the performance from regular HiP, achieving 100\% accuracy in passkey up to 64k context length.

\input{figure_srcs/longbench_table}

\textbf{LongBench.} We then use the LongBench benchmark~\citep{bai_longbench_2023} to evaluate the long context prompt and decoding performance of HiP in \Cref{fig:longbench_result}.
We believe that this benchmark is the most important because it shows both long context generation performance and knowledge retrieval performance, which are critical in many LLM applications, such as multi-turn assistants and in-context learning.
Compared to passkey, the dense decode setting scores higher because this benchmark is much more decoding-heavy.
This means that real-world natural language question answering and long context text generation rely more on decoding accuracy rather than prefill.
Therefore, we can see non-decode-friendly baselines such as StreamingLLM, \baselineAVD~and \baselineA~failing to recover long-generation performance in GovReport and MultiNews subtasks, which decode 512 tokens.
Interestingly, \baselineAVD~completely fails on those two subsets while it works moderately well on some QA tasks.
We think this is because \baselineAVD~fails to capture complex reasoning and long-term context due to its restrictive attention mask patterns.
In \Cref{sec:analysis_summary_sllm_hip}, we illustrate this long context knowledge retrieval ability by using an example from LongBench. HiP outperforms every baseline, and with a small amount of fine-tuning with an unrelated dataset, it even recovers the original model's performance (`HiP \textsc{heal}'). See \cref{sec:hyperparam} for more details and discussion about healing.

\input{tables/table_booksum}

\textbf{BookSum.}
We use the BookSum benchmark \citep{kryscinski_booksum_2022} to assess the long-context and long-response generation capabilities of HiP.
We report the average ROUGE F1-scores \citep{lin-2004-rouge} for the generated summaries in \Cref{tab:booksum}.
To simulate a realistic long-context decoding scenario and demonstrate the effectiveness of KV cache offloading, we put a limit on the GPU KV memory size to 8K tokens. This represents a practical context length on a 24GB GPU with an 8B model without KV offloading.
Specifically, for FlashAttention and BigBird, we truncate the context to 8K tokens, and \baselineAVD~uses an 8K token length sliding window.
With our method, with KV cache offloading, we can expand the effective context length only limited by the main memory's capacity, which is much cheaper.
HiP outperforms all other baselines in this VRAM-limited setting while maintaining high decoding speed: over $7\times$ faster than regular FlashAttention.
Although FlashAttention with a truncated context is faster, it suffers from significant performance degradation and, most importantly, breaks the user's expectation that the model can access the entire context.
We observe that HiP with a context window of only 512 still outperforms \baselineAVD~with an 8k window.

\input{figure_srcs/latency_breakdown_decoding_speedup}

\vspace{-0.7em}
\subsection{Latency Breakdown and End-to-end Decoding Speedup} \label{subsec:latency_breakdown}
We evaluate the trade-off between attention latency and the model performance with HiP in \Cref{fig:latency_ppl}.
We observe that our HiP's latency-optimized setting shows about 9.00$\times$ speedup of attention decoding latency but only increases the perplexity by 0.5348 in PG19~\citep{raecompressive2019pg19}, compared to FlashAttention2.
In \Cref{fig:latency_breakdown}, we show the latency breakdown of the HiP-applied transformer model.
Our proposed method contains two major components that contribute to the overall latency: (1) top-$k$ estimation iterations and (2) fused sparse attention.
We observe that the HiP top-$k$ estimation kernel is the only scaling part as the sequence grows; the sparse attention and linear layer shows constant time for each decoding step.
Since the top-$k$ estimation iteration results can be cached and reused $r_m$ times, the latency of the HiP method is dominated by fused sparse attention in most practical scenarios, as shown in~\cref{fig:latency_breakdown}.
On the other hand, the $r_m$ hyperparameter trades off the generation quality for latency, especially for long decoding, as shown in~\Cref{fig:decode_speedup_longbench}.
HiP achieves 6.83 times end-to-end decoding speedup with 128k context while maintaining 96.0\% relative performance in LongBench.
We can speed up further to 14.30$\times$ when we allow a moderate amount of performance degradation (-3.6\%p).
% Please refer to~\cref{fig:longbench_result} for a more detailed performance-latency trade-off analysis.
% Therefore, for users, $k$ is the most important efficiency factor of HiP.

\vspace{-0.7em}
\subsection{KV Cache Offloading Benchmark}
\label{sec:experiments_offload}

In \cref{tab:offload_cache}, we evaluate the latency and memory usage of our KV offloading framework.
The \textsc{UVM} variants use the CUDA unified virtual memory API to offload the whole KV cache to the main memory.
Our HiP has two variants that depend on the type of cache implementation.
We use Llama3.1-8B with 16-bit weights, and the KV states are stored in 8-bit floats.
We use a single RTX 4090 24GB for the graph on the left, and to additionally test our method up to 512k tokens, we also test on a single A100 80GB GPU.
We set $l_d=0$, and choose the last token for the representative key to reduce the memory access in this test.
See \cref{sec:hyperparam} for details.

\input{tables/table_offload_cache}

As shown in~\cref{tab:offload_cache}, with UVM, both ours and Flash Attention slow down decoding about 5 to 7 times compared to full GPU runtime.
However, we could serve until 64k context, while the same machine can serve only 16k at maximum.
Since memory access is significantly more costly with UVM, the trend of logarithmic scaling of decode throughput is clearer than when working with pure GPU memory.
So, at 64k context length, ours is more than 50 times faster than Flash Attention with UVM.
However, UVM slows down both methods too much compared to full GPU runtime.

We test two types of cache implementation: vector map and hash map.
A vector map uses a $T$-sized vector of pointers pointing to the allocated bank to store the mapping between a token index and a bank index.
Our GPU-loaded KV offloading cache (Vector Map) shines by achieving 93\% decoding throughput compared to no KV offloading at all.
Without a significant slowdown, we could extend the serving context from 16k to 64k on an RTX 4090, which is 4.17$\times$ higher decoding throughput compared to HiP$_{\text{UVM}}$ and 49.97$\times$ higher decoding throughput compared to Flash Attention$_{\text{UVM}}$, as shown in~\cref{tab:additional_offloading}.
However, with the vector map, the space complexity is $O(T)$.
To reduce the space complexity to $O(\log T)$, we use a linear probing hash map to store the index mapping.
This way, we can reduce the GPU memory consumption by 40.8\% on 512k context length.
However, since the hash map lookup is not friendly to the GPU, it slows down token accesses more than naive UVM.

We present our KV offloading framework on a standard gaming PC equipped with a single RTX 4090.
Our experiments confirm that the PCIe 4.0x8 bandwidth is sufficient to manage offloading traffic through KV accesses using UVM.
Furthermore, when scaled up to a single A100 80GB, our framework demonstrates its ability to extend serving context length, even on server-grade hardware.
We anticipate that our HiP's KV offloading framework will effectively increase serviceable context length across a wide range of deployments, from on-device setups to cloud-based environments.

\vspace{-0.7em}
\section{Conclusion}
\label{sec:conclusion}
In this study, we present HiP Attention, a novel framework for accelerating pretrained Transformer-based models without any training, with a focus on the acceleration of LLMs for long-context tasks.
Our proposed HiP rapidly estimates the top-$k$ context keys for computing sparse attention, drastically reducing the computation required for long context inference and fine-tuning from $O(T^2)$ to $O(T \log T)$.
Our HiP attention is a drop-in replacement for the core of any Transformer-based model, such as language and multimodal models, and does not require modifying the existing weights.
This is a practical and meaningful improvement as it allows pre-trained models to be fine-tuned and executed much more efficiently in long sequences without sacrificing quality.
We are looking forward to contributing to open-source LLM serving frameworks by combining various efficient decoding strategies with HiP attention.
% We expect a synergy effect with speculative decoding, KV cache eviction, and compression strategies since they are orthogonal to our method.
% In \cref{sec:hip_possible_improvements}, we outline possible future research directions, aiming to improve the accuracy of top-$k$ estimation by leveraging its tree structure.

%\newpage
\section*{Reproducibility Statement}

We provide every experiment code and kernel code in the attached supplementary file.
We also provide detailed instructions on how to run experiments in readme markdown files, so please read those files.
And we put detailed experiment settings in \cref{sec:hyperparam}.
We will try our best to resolve further reproducibility problems.
Inside the HiP library, we have multiple versions of HiP kernels, all written with OpenAI Triton.
The upstream kernel path is \texttt{\small hip / models / hip\_attention / attention2\_draft\_prefetch.py}.
Additionally, you can see the evolution of our HiP from the very first HiP implementation \texttt{\small hip / models / hip\_attention / attention1.py}; please feel free to enjoy our codebases.
We left them all for research purposes when someone needs various settings, such as dynamic retention ratios, that are only supported by old versions.
Our main experiment entry file is \texttt{\small hip / main / model\_eval.py}.
Please execute \texttt{\small --help} option to gather further information.
Our offloading experiment entry file is \texttt{\small hip / models / hip\_attention / offload\_runner / offload\_runner.py}.
For Longbench and RULER, we modified the official code to run our method with vLLM.
Please refer to \texttt{\small HiPAttentionArgs} class to investigate full settings, including every subtle configuration.
\baselineA, \baselineAVD~and BigBird are using the same HiP kernel since they are the same block sparse attention.
We just modify the block masks that passed to block sparse attention.
StreamingLLM is implemented in \texttt{\small hip/models/sink\_attention/sink\_attention.py}.
About HiP-related environment variables of vLLM and SGlang, please refer to \texttt{\small HiPAttentionEnvs} in vLLM and SGlang attention backend implementations.


\section{Theoretical Analysis}\label{sec:theory_appendix}
\input{sections/appendix_theory}

\section{Detailed Methodology Descriptions}

\subsection{Hierarchical Sparse Attention Mask Estimation Algorithm}
\input{algorithms/mask_estimation}
In \cref{alg:mask_estimation}, we describe the complete algorithm used for mask estimation. The peak amount of memory used during the mask estimation process is in $O(T)$, since at each iteration, only the immediately previous iteration's node indices are needed, and the rest can be discarded.

\subsection{HiP Decoding Algorithm}
\input{algorithms/full_alg}
In \cref{alg:decoding}, we show a rough sketch of the decoding process with HiP. In particular, note the function of the mask estimation period $r_m$ and the number of initial dense layers $l_d$, as well as the time and space complexities of each component.

\begin{figure}[t]
\centering
% \vspace{1.0em}
\includegraphics[width=1.0\linewidth]{figures/full-page-diagram.pdf}
\caption{\textbf{Detailed Flowchart of Hierarchically Pruned Attention.} We illustrate the internal steps of each program of hierarchical attention pruning. We instantiate a single program for each attention row due to row-level synchronization on the top-k operation.}
\label{fig:appendix_flow}
\end{figure}

\subsection{Detailed Flow-diagram of HiP}

{In \cref{fig:appendix_flow}, we illustrate hierarchical attention pruning step-by-step.}

% \subsection{Sparsity Inducing Attention Regularization}
% Since our approximation assumes that the attention probabilities are sparse, we employ an extra regularization term that encourages the attention probabilities to be sparse during the healing process. To this end, we aim to compute the $L_p$ norm with $p = 0.5$ on the attention probability distributions $\bm{P}$ for each layer. However, since this computation is quadratic and expensive, we instead sample $n_{reg}=1024$ query indices for each layer at each step and compute the $L_p$ norm for those indices only. We add the following regularization term to the loss only during the healing process:
% \begin{align}
%     R_{sparsity} = \lambda_{sparsity} \cdot \sum_{l=1}^{L} \sum_{h=1}^{H_l} \sum_{i=1}^{n_{reg}} \frac{1}{L \cdot H_l \cdot n_{reg}} \cdot L_p\left( \bm{P}^{(l,h)}_{\mathrm{randperm}_i,:} \right),
% \end{align}
% where $L$ is the number of layers in the model, $H_l$ is the number of attention heads in the $l$th layer, $\bm{P}^{(l,h)}$ is the attention probability map at layer $l$, head $h$, and $\lambda_{sparsity}$ is a hyperparameter. $\mathrm{randperm}_i$ indicates the $i$th value of a random shuffling of the integers in the range $1~..~T$. We set $\lambda_{sparsity} = 0.001$ throughout the experiments.

\subsection{Additional Optimization Techniques}
\subsubsection{Top-r Approximation}
\label{subsec:sparq}
In order to reduce the cost of the mask estimator even further, we take inspiration from SparQ Attention~\citep{ribar_sparq_2023}, where global memory (HBM or GDDR) transfer is reduced by selectively fetching only the most relevant components of the key vectors.
Specifically, when computing the inequality condition in \Cref{eq:branch_selection}, instead of fetching all $d$ components of the key vectors, we only fetch $r \ll d$ most prominent components estimated by the query vector $\bm{q}$.
Thus, we compute the following as an approximation:
\begin{align}
    \bm{q}^\top \bm{k}_{t} &\approx \sum_{l=1}^r \bm{q}_{p_l} \cdot \bm{k}_{p_l}
\end{align}
where $\{p_1, p_2, \dots, p_r\} = \mathrm{argtop\_}r(|\bm{q}|)$.
By using the top-$r$ approximation, the total number of global memory accesses in the mask estimation stage can be drastically reduced.
However, we disable this approximation by default.

\subsubsection{Block Sparse Flash Attention}

We utilize the Flash Attention \citep{dao_flashattention_2022} mechanism to reduce the latency of sparse attention and use a small size sliding window to reduce performance degradation on the side of the sparse attention kernel.
Following the receipt of StreamingLLM \citep{xiao_streamingllm_2023}, local sliding window and global sink attention are also added during block sparse flash attention operation.
% The sliding window and sink attention are fixed sizes (256, 16) for every experiment in this paper.

% \newpage

\subsection{Training Downstream Tasks with HiP}
\label{sec:training}

In this section, we describe the HiP attention training strategy for downstream tasks.
We discovered that direct fine-tuning after applying HiP could not achieve the performance of the fine-tuned vanilla attention.
Empirically, HiP's highly sparse attention matrices show excellent performance approximation during test time but not in train gradients.
Since our method heavily prunes the attention matrix, the gradient cannot flow through dense attention probabilities.
This incomplete and unstable gradient of the attention matrix leads to significant training performance degradation because HiP forces the model to have attention patterns similar to those of the pretrained model rather than adopting them for the downstream task.

However, we could achieve the same performance as vanilla by using the following two-step training strategy: (1) fine-tuning with vanilla first and (2) further fintuning after applying HiP (healing).
First, we train the pretrained model with vanilla attention to the downstream task as usual.
Then, we load the finetuned weight, apply HiP, and further finetuning with the downstream task dataset with just a few optimizer steps.
We call this a healing process because we only finetune a few steps from the finetuned weight.
For training details about each experiment, we describe hyperparameter and optimization setups in \cref{sec:hyperparam}.


\section{Additional Experimental Results}

\subsection{Large Multimodal Model with HiP}
\input{figure_srcs/lmms_table}
Since large multi-modal models (LMM)~\citep{liu_llava_2023, liu2024llavanext} leverage pre-trained LLMs as a backbone for their NLU ability, without changes, except for the tokenizer, we also evaluated our HiP on top of LLaVA-1.6-13B \citep{liu2024llavanext}, a large multi-modal model, using LMMs-eval~\citep{lmms_eval2024}, which provides extensive benchmarks for large multimodal model suites.
As we show in \cref{tab:lmms_eval}, our method scores 95.9\% relative scores, which is similar to the performance recovery ratio of HiP on NLU tasks.
% Surprisingly, StreamingLLM fails on this task, while it achieves competitive performance recovery in NLU tasks.
% Our method outperforms StreamingLLM in every multimodal dataset, and ours sometimes even scores 3.75$\times$ higher.

\subsection{Massive Multitask Language Understanding (MMLU)}
\label{sec:mmlu_result}
\input{figure_srcs/mmlu_table}
Next, we evaluate HiP on the MMLU benchmark~\citep{hendrycks_measuring_2021} to show that our method does not negatively affect the NLU ability of the pretrained model.
The results show that our HiP preserves the NLU performance of the original model.
All tested methods are able to recover original MMLU scores without significant loss here.
This is probably due to the nature of the MMLU task: the answer is only dependent on the most recent span of tokens rather than the entire prompt (which contains few-shot examples in the beginning).

% \subsection{Detailed Results on LongBench with HiP and StreamingLLM}

% In \cref{tab:longbench}, we evaluate our method and baselines on LongBench~\citep{bai_longbench_2023}.

% \input{tables/table_longbench}

\subsection{Comparison with Reformer and SEA}
\label{subsec:reformer_sea}
\input{tables/table_sea_reformer}
We compare the performance of HiP against Reformer~\citep{kitaev_reformer_2019} and SEA~\citep{lee_sea_2023} using the Llama2-7b model and show the results in \Cref{tab:ppl_additional}. Even though HiP requires no fine-tuning, unlike Reformer and SEA, which need fine-tuning, our HiP attention is far superior in language modeling performance compared to these two baselines.

For a fair comparison, we fine-tune Reformer and SEA using LoRA (Low-rank adapters), which have the same rank as the healed version of HiP. Due to this, the Reformer and SEA's performance converges to a much-degraded value compared to the values reported in their respective original papers. We conclude that both methods need much modification to the original weights in order to perform well, whereas since HiP performs well even without any modification to the weights whatsoever, a small amount of LoRA training can even close this small gap.

% \subsection{Detailed LMMs-Eval Results with LLaVA1.6}

% \input{tables/table_lmmseval}

% In \Cref{tab:lmms_eval}, we evaluate our method and baselines on various multimodal benchmarks on LMMs-Eval~\citep{lmms_eval2024}.

% \newpage
\subsection{Context Extention with Self-Extend}

Since our method has a considerable advantage in long-context decoding, we need the pre-trained long-context model.
Unfortunately, however, not all pretrained transformer models support long contexts.
Therefore, many previous studies~\citep{peng2023yarnefficientcontextwindow, jin2024selfextend} try to extend maximum position embeddings of the trained model to extend the context size of LLM.
We adopt the SelfExtend~\citep{jin2024selfextend} method into our method because it is also a training-free context extension method.
We picked Gemma2~\citep{gemmateam2024gemma2improvingopen} to target LLM and extend the context model because the model is a hybrid of sliding windows and dense attention.
Therefore, it will have the advantage of a long context model with HiP by saving the KV cache of sliding window layers.
The Gemma2 repeats the attention layer by repeating the stack of different attention blocks: sliding window and dense attention.
To evaluate the effectiveness of the combination of HiP and SelfExtend, we apply them to attention layers.

\begin{table}[t]\centering
% \vspace{1em}
\caption{\textbf{Self-Extend Result with HiP.} We apply Self-Extend with HiP $k$=256 on Gemma2-2B-it. We extend the maximum context length of Gemma2 from 8k up to 128k tokens. We measure the perplexity of Wiktiext2.}
\label{tab:appendix_self_extend}
\resizebox{\linewidth}{!}{
\begin{tabular}{lllrrrrrrrrr}\toprule
Method &
    \makecell{Self\\Extend} &
    \makecell{Sliding\\Window} &
    $T$=1k &
    $T$=2k &
    $T$=4k &
    $T$=8k &
    $T$=16k &
    $T$=32k &
    $T$=64k &
    $T$=128k \\
\midrule
FA2 &
    \xmark &
    \cmark &
    18.90 &
    14.48 &
    12.35 &
    \textbf{\underline{11.40}} &
    \textit{46.66} &
    \textit{167.40} &
    \textit{363.02} &
    - \\
FA2 &
    \xmark &
    \xmark &
    18.90 &
    14.48 &
    \textbf{\underline{12.35}} &
    \textit{30.17} &
    \textit{198.84} &
    \textit{638.65} &
    \textit{1310.26} &
    - \\
HiP &
    \cmark~(x4) &
    \xmark &
    18.98 &
    14.77 &
    13.00 &
    12.14 &
    \textbf{\underline{12.00}} &
    \textit{46.51} &
    \textit{277.54} &
    - \\
HiP &
    \cmark~(x8) &
    \xmark &
    18.98 &
    14.90 &
    13.30 &
    12.57 &
    \textbf{12.22} &
    \underline{12.63} &
    \textit{49.26} &
    - \\
HiP &
    \cmark~(x4-8) &
    \xmark &
    18.98 &
    14.81 &
    13.15 &
    12.36 &
    \textbf{12.06} &
    \underline{13.62} &
    \textit{116.18} &
    - \\
HiP &
    \cmark~(x8-16) &
    \xmark &
    18.98 &
    15.00 &
    13.51 &
    12.83 &
    \textbf{12.53} &
    12.55 &
    \underline{14.32} &
    - \\
HiP &
    \cmark~(x8) &
    \cmark &
    19.28 &
    14.86 &
    12.87 &
    11.95 &
    11.53 &
    \textbf{11.36} &
    \underline{11.69} &
    - \\
HiP &
    \cmark~(x16) &
    \cmark &
    19.28 &
    14.98 &
    13.11 &
    12.25 &
    11.86 &
    11.69 &
    \textbf{11.54} &
    \underline{11.68} \\
\bottomrule
\end{tabular}
}
\vspace{1em}
\end{table}

We can observe Gemma2 explode after its pretrained context length, which is 8192 (First row of the~\cref{tab:appendix_self_extend}).
We can see the model fails after the sliding window context length, which is 4096 for the sliding window layer (Second row of the~\cref{tab:appendix_self_extend}).
Therefore, we know that treating sliding windows especially is quite essential for performance.
In the third and fourth rows of the~\cref{tab:appendix_self_extend}, we apply the same Self-Extend group size for every attention layer, including the layer that was originally a sliding window, before replacing it with HiP.
We could observe the settings are struggling to recover performance right after $\text{Self-Extend Group Size} \times \text{Sliding Window Size}$, so we apply twice the larger Self Extend group size for the HiP layers originally sliding window.
The modified group size application is in the fifth and sixth rows of the~\cref{tab:appendix_self_extend}.
We could observe that we can extend the context window as expected, not bounded by sliding window size.
However, the above replacements are quite restarted setting because they are not even allowed to use sliding window attention, which is usually very efficient in practical LLM frameworks.
Therefore, we replace only dense attention to HiP, and we could observe a significant performance boost from all layer replacements while extending context length up to 128k, as shown in the last two rows of the~\cref{tab:appendix_self_extend}.

\subsection{Ensemble Hierarchical Top-$k$ Approximation}
\label{sec:ensemble_result}
% Here we provide an in-depth explanation of how our ensemble method can improve inference performance, along with empirical results that show the potential to replace the dense layers $l_d$ in HiP with comparable performance.
For the first attempt to address the challenges in HiP described in \cref{sec:discussion_possible_improvements}, we perform an ensemble of HiP masks by introducing slight randomness to the branching decisions within a given iteration.
Our empirical results on Wikitext2 indicate that the ensemble of HiP with no dense layers ($l_d=0$) achieves comparable perplexity to HiP with dense layers ($l_d=3$) across context lengths from 4k to 16k. This highlights how considering different branching strategies within a given iteration can improve inference performance and suggests the potential for replacing the dense layers in HiP with comparable performance.

\textbf{Methodology.}
First, we generate $n_e$ HiP attention mask samples by adding randomness $r_e$ to the branch direction when the middle iteration branches out to the next iteration.
As $r_e$ slightly adjusts the selected node in the iteration, each sample can take a slightly different path during tree branching, leading to a diverse construction of attention masks.

Second, we count the number of agreements of the indices per query and retain the indices that exceed or equal to the agreement threshold $\theta_{\text{vote}}$.
Therefore, $\theta_{\text{vote}}=1$ functions as a union operator and $\theta_{\text{vote}} = n_e$ as an intersection operator. By adjusting $\theta_{\text{vote}}$ from 1 to $n_e$, we can perform an operation that lies between union and intersection, prioritizing indices with a higher number of votes.
To prevent the union-like $\theta_{\text{vote}}$ increasing the number of active attention elements too much, we truncate the number of the final selected indices by original $k$ when $\tau \in \{0, 1\}$ is $1$,  prioritizing the indices having more votes.

Lastly, we perform the introduced ensemble method for the first $l_e$ layers, just like we do with $l_d$.

% ensemble: all set with l_d=0
\textbf{Experiments.}
We first provide experimental evidence on how ensemble enables end-to-end sub-quadratic complexity and then give details on our hyperparameter tuning process. The experiments are conducted with the LLaMA2-7B model.

\begin{figure}[h]
\vspace{1em}
\centering
\begin{subfigure}[b]{0.32\textwidth}
\includegraphics[width=\linewidth]{figures/plot_ppl_long_context.pdf}
\end{subfigure}%
\hspace{0.02\textwidth}
\begin{subfigure}[b]{0.32\textwidth}
\includegraphics[width=\linewidth]{figures/plot_sparsity_ppl_long_context.pdf}
\end{subfigure}
\caption{
\textbf{Perplexity Evaluation on Long Context.} \textbf{(Left)} Perplexity result in Wikitext2 with different ensemble settings ($\theta_\text{vote},~\tau$) where $r_e = 5.0,~l_e = all$. \textbf{(Right)} Perplexity comparison between full HiP ($l_d = 0$) and ensemble ($\theta_{\text{vote}} = 1,~\tau = 0,~r_e=5.0,~l_e=all$) using same relative retention ratio in each sequence length.
}
\label{fig:ensemble_performance_long_context}
\end{figure}

\begin{figure}[h]
\centering
\vspace{1em}
\begin{subfigure}[b]{0.32\linewidth}
\includegraphics[width=\linewidth]{figures/plot_ppl_thresh_randn.pdf}
\end{subfigure}
\hspace{0.02\textwidth}
\begin{subfigure}[b]{0.32\linewidth}
\includegraphics[width=\linewidth]{figures/plot_ppl_till_what_layer.pdf}
\end{subfigure}%
\caption{
\textbf{Effect of Randomness in HiP sampling and Ensemble-Enabled Layers $l_e$.} Perplexity in Wikitext2 with $T=4k$. \textbf{(Left)} We adjust randomness $r_e$ with fixed $l_e=all$. \textbf{(Right)} We adjust number of $l_e$ with fixed $\theta_\text{vote}=1,~r_e=5.0$. The dashed horizontal line shows the performance of HiP ($l_d = 0,~l_d = 3$).
}
\label{fig:ensemble_performance_thresh_randn_layer_till}
\end{figure}

\textbf{Performance Comparison with Original HiP.}
To show that ensemble enables end-to-end sub-quadratic complexity with comparable performance to our default HiP ($l_d=3$), we compare the full HiP ($l_d=0$), the default HiP ($l_d=3$), and the ensemble with $l_d=0$.
We fix $r_e=5.0$, $l_e=\text{all}$ that gave the best performance in $T=4096$, as shown in \cref{fig:ensemble_performance_thresh_randn_layer_till}.
The result indicates that ensemble with $\theta_{\text{vote}}=1$, $\tau=0$ outperforms both full HiP and default HiP, as shown in \cref{fig:ensemble_performance_long_context} (Left), and therefore this suggests that ensemble could not only improve the performance but also replace the dense layers with comparable performance.

Moreover, we provide a comparison with full HiP ($l_d=0$) at the same level of sparsity as the ensemble to demonstrate that the improvement is not solely due to the increased number of selected indices resulting from our ensemble method.
As shown in \cref{fig:ensemble_performance_long_context} (Right), our ensemble method is Pareto frontier compared to HiP, with performance measured against the retention ratio.

\textbf{Latency of Ensemble.}
Since we sample multiple HiP masks by $n_e$ and perform voting operations across $n_e \times k$ number of indices, the ensemble costs a few times more than the original HiP.
However, since the cost of dense attention grows quadratically, the ensemble will become more efficient compared to the dense attention as the context length increases. Therefore, we think that the use of the ensemble method could be particularly advantageous in extremely long contexts.

\begin{figure}[h]
\vspace{1em}
\centering
\begin{subfigure}[b]{0.32\linewidth}
\includegraphics[width=\linewidth]{figures/plot_sparsity_long_context.pdf}
\end{subfigure}
\begin{subfigure}[b]{0.32\linewidth}
\includegraphics[width=\linewidth]{figures/plot_sparsity_thresh_randn.pdf}
\end{subfigure}
\begin{subfigure}[b]{0.32\linewidth}
\includegraphics[width=\linewidth]{figures/plot_sparsity_till_what_layer.pdf}
\end{subfigure}
\vspace{-.5em}
\caption{
\textbf{Relative Retention Ratio and Ensemble Factors.} Relative retention ratio after mask ensemble. The ratio is computed by dividing the number of active indices after the ensemble by before. \textbf{(Left)} We adjust the sequence length $T$ with fixed $r_e=5.0,~l_e=all$. \textbf{(Center)} We adjust the randomness $r_e$ in $T=4k$ with fixed $l_e=all$. \textbf{(Right)} We adjust the number of ensemble enabled layer $l_e$ in $T=4k$ with fixed $\theta_\text{vote}=1,~r_e=5.0$. Ensemble disabled layers are computed as 1.0.
}
\label{fig:relative_retention_ratio}
\vspace{.5em}
\end{figure}

\begin{figure}[t]
\centering
\vspace{-0.25in}
\includegraphics[width=0.75\linewidth]{figures/figure_ensemble_mask_comparison.pdf}
\caption{
\textbf{Attention Mask Ensemble Visualization.} Visualization of attention mask in $T=16k$ for (\textbf{Left}) HiP ($l_d=0$).
\textbf{(Center)} ensemble ($\theta_{\text{vote}}=1$, $\tau=0$).
\textbf{(Right)} ensemble ($\theta_{\text{vote}}=1$, $\tau=1$).
Red indicates indices added by our ensemble method, yellow indicates indices from HiP, and green indicates where attention will not be computed.
% Full attention visualization is max pooled by 32x32 kernel size, and the zoom-in visualization is without max pooling.
}
\label{fig:ensemble_mask_comparison}
\vspace{.5em}
\end{figure}

\textbf{Ablation Study of Ensemble Hyperparameter: $\theta_{\text{vote}}$, $\tau$, $r_e$, $l_e$.}
As shown in \cref{fig:ensemble_performance_thresh_randn_layer_till} (Left), $\theta_{\text{vote}}=1$ with $\tau=0$, and $r_e >= 5.0$ gives the highest score.
This indicates that performing the union operation with large randomness in sampling gives the highest performance.
Also, in \cref{fig:ensemble_performance_thresh_randn_layer_till} (Right), with $\theta_\text{vote}=1$, we show that $l_e=\text{all}$ works the best, therefore if we want to replace the few layers of vanilla attention, then we would have to apply ensemble in every layer.

\textbf{Correlation Between the Relative Retention Ratio and Ensemble Factors.} In \cref{fig:relative_retention_ratio}, we show that the relative retention ratio shows no correlation with sequence length, while randomness shows a positive correlation.
Moreover, when measuring the relative retention ratio changes over $l_e$ with $\theta_\text{vote}=1$, since we treat the ensemble disabled layer as having a 1.0 relative retention ratio, more ensemble-enabled layers lead to a higher relative retention ratio.

\textbf{Analysis with Visualization.}
In \cref{fig:ensemble_mask_comparison}, we provide a visual analysis to show how the ensemble selects indices that HiP missed to fill up a complete attention pattern and how it enables dynamic sparsity.
We can see how the ensemble catches missed important indices such as diagonal, vertical, and stride attention patterns in (a), (b), and (c) of \cref{fig:ensemble_mask_comparison}.
Moreover, compared to HiP (left), the union operation (center) enables dynamic sparsity per head.
Especially in (c), we can see that the ensemble is effective for filling missed indices in a long sequence while providing dynamic sparsity for each row (red pixels are gradually frequent in the bottom). Lastly, in \cref{fig:ensemble_mask_comparison} (center, right), we show how $\tau=1$ selects indices that receive more votes compared to those selected by $\tau=0$.

\section{Detailed Experimental Settings}
\label{sec:hyperparam}

% 4090 Machine
% 2x RTX 4090 24GB, 128GB DDR5-3600, Ryzen 7950x

% A100 Machine
% 8x A100 80GB SMX4, ???GB DDR?-????, ???? ????

\textbf{Computation Resources.} We use two machines to run experiments and training. (1) 4090 Machine. We use this local development machine. Most of the micro-benchmark and kernel optimization is done with this machine. (2x RTX 4090 24GB, 128GB DDR5-3600, Ryzen 7950x), (2) A100 Machine. We use this AWS cloud node as the main computation horse. All training and most long context benchmarks are measured with this machine. We use different GPU architectures for kernel development because we could not get an H100 AWS node due to a lack of quota. Therefore, our kernel's CUDA hyper-parameters, such as the number of warps per program and block sizes, are not optimal in the A100 machine. To overcome these mismatches between the development machine and the computation horse, we used \texttt{triton.autotune} as much as possible. However, for the above reasons, the optimization of the GPU kernel may not be optimal for every GPU architecture.

\textbf{Experiment Settings.}
By default, for the HiP experiment, we use $b_q=32, b_k=2, k=512, l_d=3~\texttt{if}~\text{7B}~\texttt{else}~4, r_m=8$. For StreamingLLM, we use $\text{num\_sink}=4$.

We show overall experiment settings, such as the number of GPUs and model IDs to which the experiment is introduced (e.g., the caption of the figure and table). To reference, we leave the huggingface model path in \cref{tab:hf_model_path}.
% We used 4-bit quantization from huggingface transformers during generation.
We used an instruction model from the same provider for instruction following ability-required tasks such as passkey, LongBench, and RULER.

\begin{table}[h]
\centering
\vspace{.5em}
\caption{\textbf{Model IDs on Huggingface.} We use large language models trained on long context inputs in order to demonstrate our method's effectiveness at long context lengths.}
\label{tab:hf_model_path}
\vspace{-.5em}
\resizebox{0.9\textwidth}{!}{%
\begin{tabular}{l l c}
\toprule
Model & Huggingface ID & \makecell{Maximum\\context length} \\
\midrule
LLaMA2-7B & \texttt{togethercomputer/LLaMA-2-7B-32K} & 32K \\
LLaMA2-13B & \texttt{Yukang/Llama-2-13b-chat-longlora-32k-sft} & 32K \\
Qwen1.5-7B & \texttt{Qwen/Qwen1.5-7B-Chat} & 32K \\
Qwen1.5-14B & \texttt{Qwen/Qwen1.5-14B-Chat} & 32K \\
Luxia-21.4B & \texttt{saltlux/luxia-21.4b-alignment-v1.1} & 32K \\
Llama3.1-8B & \texttt{meta-llama/Meta-Llama-3.1-8B} & 128K \\
Llama3.1-8B-Instruct & \texttt{meta-llama/Meta-Llama-3.1-8B-Instruct} & 128K \\
Gemma2-2B-it & \texttt{google/gemma-2-2b-it} & 8K \\
Gemma2-9B-it & \texttt{google/gemma-2-9b-it} & 8K \\
Exaone3-7.8B & \texttt{LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct} & 4K \\
\bottomrule
\end{tabular}
}
\vspace{.5em}
\end{table}

\textbf{Experiment Details.}
% We provide the following experiment details for \Cref{tab:wikitext2}.
We measure every latency measures with a single NVIDIA RTX 4090, PCIe 4.0 x8, 128GB DDR5-5600, and Ryzen 7950x.
The batch size is 32 for decoding and 1 for prefilling.
The official implementation of HyperAttention is not available for decoding; therefore, we did not measure the decoding latency and decoding benchmark.

For FlashAttention, we use \texttt{\small flash\_attn==2.6.3} for every baseline that requires a FlashAttention backend, such as vLLM, SGlang, and HyperAttention.

We do not use HyperAttention in another experiment because it fails to recover the perplexity of PG19, the most basic metric of language modeling.
For HyperAttention~\citep{han2024hyperattention} we used {\small \texttt{lsh\_num\_projs}=7, \texttt{block\_size}=64, \texttt{sample\_size}=1024, \texttt{min\_seq\_len}=32}.
We select \texttt{min\_seq\_len} to match the size of the MMU's block size ($32$) rather than $4096$, which is suggested by the authors' code repository.
Since in sortLSH~\citep{han2024hyperattention} it processes shorter block size of \texttt{min\_seq\_len} with vanilla attention.
Therefore, we have to reduce its size to a smaller size than $4096$. % which is context length in perplexity measurement of \cref{tab:wikitext2}.

Since StreamingLLM does not use block-wise computation just like HiP's block approximation, it cannot utilize TensorCore in GPU.
This downside degrades the throughput significantly compared to HW-aware algorithms such as ours and Flash Attention~\citep{dao_flashattention-2_2023}.
However, since the method requires a different RoPE index for every query-key dot product, we cannot easily adopt block sparsity on their method.
This will slow down attention computation more than twice because the RoPE re-computation costs twice as much as attention score computation.

In the \cref{fig:latency_ppl}, the latency is measured with our latency measure machine (RTX 4090).
StreamingLLM shows OOM over 32k context due to the overhead of the COO sparse matrix.
HyperAttention shows invalid kernel parameters over 32k context due to heavily nested tensors in a long context.
It uses high-dimensional striding to perform reformer-style token clustering, but the backbone flash attention kernel does not support that high-dimensional striding with a larger tensor.

In the \cref{tab:offload_cache}, the latency is measured with our latency measure machine (RTX 4090).
The machine has about 4GB of VRAM available for the KV cache, excluding model weight and temporary buffers.
We limit the size of the CPU offloaded KV cache to 32GB.
The tested model is Llama3.1-8B.

\textbf{Training Details (Healing HiP in Arxiv).}
The HiP healing in \cref{fig:longbench_result} is done as follows.
For the Llama3.1-8B model, after applying HiP, we fine-tune the pretrained model on the Arxiv dataset for 500 steps with AdamW optimizer with learning rate 1e-5, and with batch size 32, LoRA rank 256, and HiP's hyperparameters set to $b_k=2, b_q=64, b_sq=2, b_sk=1, k=512, l_d=3$. We use the Arxiv dataset in Redpajama~\citep{together2023redpajama}. The inputs are truncated to a maximum of 8192 tokens to speed up the process.

The purpose of this fine-tuning, which we call healing, is to make the model adapt to the slight differences in the activations that appear when the original dense attention layers are replaced with sparse HiP attention. As shown in \cref{fig:longbench_result}, the healed model performs slightly better than the plug-and-play (unhealed) model on LongBench. However, we emphasize that HiP is meant to be training-free, and healing is just an additional option for extra performance, as our method already works near-perfectly without training.

\section{Additional Analysis}

\subsection{More Discussion on Related Works}
\input{tables/table_comparison}
\input{figure_srcs/baseline_vis}

In \cref{tab:comp}, we compare efficient attention methods that we do not include in our main benchmarks. For each method, we show the time and space complexities, whether it is training-free, and whether it uses dynamic attention.
Dynamic attention methods can attend to each other content dynamically rather than using static attention patterns such as a sliding window.
Besides HiP, HyperAttention is the only method that satisfies all four criteria, but HyperAttention suffers from substantial performance degradation (see \cref{fig:latency_ppl}).
In \cref{fig:baselines_visualization}, we conceptually visualize the various sparse attention baselines' attention patterns for extra clarity.

\label{sec:related_works_appendix}

\paragraph{StreamingLLM~\citep{xiao_streamingllm_2023}.} StreamingLLM uses a sliding window attention with an attention sink, which processes the input sequence in linear complexity without resetting the KV cache; they call this process `streaming.'
StreamingLLM introduces the attention sink, which is similar to the global attention token in Longformer~\citep{beltagy_longformer_2020}, and streams the KV cache using RoPE indexing.
However, due to the sliding window, the method cannot perform long-context knowledge retrieval.
Therefore, this method cannot utilize the full context, and they do not extend the context window of the model by any amount.
Since the method loses the key-value memory as time passes, it cannot take advantage of the Transformer's strength: its powerful past knowledge retrieval ability.
Furthermore, since they use a different RoPE indexing for every query-key dot-product, they cannot utilize a MMU, which is a critical speedup factor in modern accelerators.

\paragraph{HyperAttention~\citep{han2024hyperattention}.} HyperAttention introduces \textit{sortLSH}, improved version of LSH~\citep{kitaev_reformer_2019}, to work as plug-and-play.
The method uses block sparsity to utilize MMU.
It is training-free, has sub-quadratic time complexity (near-linear), and has the ability to potentially access to every past key token, much like our method.
However, HyperAttention struggles to recover vanilla performance when replacing most of the layers in the trained model in a training-free manner (see \cref{fig:latency_ppl}).

\paragraph{Sparse Linear Attention with Estimated Attention Mask (SEA)~\citep{lee_sea_2023}.} Inspired by SEA's framework, which introduces linear complexity attention estimation and sparse matrix interpolation, we aimed to improve its efficiency.
SEA estimates each query's attention probabilities over the keys with a fixed-size vector, turns it into a sparse mask by selecting the top-k elements, and resizes it; this process is done with linear complexity.
However, the method is difficult to implement efficiently due to its extra modules, mainly the estimator and sparse matrix interpolation.
Furthermore, the method does not support block sparsity; thus, it cannot utilize the MMU.
We were motivated to improve this work drastically by introducing a fused and train-free attention mask estimator, HiP.

\subsection{Analysis of Summarizing Result between StreamingLLM and HiP}
\label{sec:analysis_summary_sllm_hip}

\begin{figure}[h]
\centering
\includegraphics[width=\textwidth]{figures/rouge_compare.pdf}
\caption{\textbf{Summarizing Example of GovReport Dataset from LongBench.} We sample random examples from GovReport summarization results with Qwen1.5-14B.
% The hyperparameter is the same as \Cref{tab:longbench}. We use $k=1024$ for both methods.
}
\label{fig:summary_analysis}
\end{figure}

In \Cref{fig:summary_analysis}, we analyze one example generation result from GovReport in LongBench.
We pick four important keywords from the human ground truth summary: \textit{MYP}, \textit{BBC}, \textit{DOD}, and \textit{10 U.S.C 2306b}.
We pick two keywords (\textit{MYP}, \textit{DOD}) that appear in every summary and two keywords (\textit{BBC}, \textit{10 U.S.C 2306b}) that appear only in the ground truth and HiP result.
The result clearly shows that StreamingLLM is struggling to gather information beyond its single-layer window size, $k=1024$.
StreamingLLM should be accessible to a much longer distance than $k$ because information keeps exchanging across time dimensions in each layer, like Mistral.
In contrast to StreamingLLM, our proposed HiP attention shows successful knowledge retrieval in summary from long-range in-context documents, with the same \textit{plug-and-play} manner.
Also, quantitatively, ROUGE-* scores show that the summary generated by HiP is much better in coherence with ground truth than StreamingLLM.

\subsection{Hierarchical Attention Mask Pruning Visualization}

\begin{figure}[h]
    \centering
    \includegraphics[width=1.0\textwidth]{figures/masking_visualization.pdf}
    \caption{\textbf{Visualization of Hierarchical Attention Mask Pruning.} Yellow indicates a non-zero entry of the attention matrix, and green indicates an empty entry of the attention matrix. We use $k=512, b_q=32, b_k=2, T=4k$.}
    \label{fig:masking_visualization}
\end{figure}

In \Cref{fig:masking_visualization}, we demonstrate real-world examples of hierarchical attention mask pruning.
We sample the Q, K, and V tensors from the first layer of LLaMA2-7B with a random text sample from Wikitext-2.
Note that each attention mask in the masking iteration is not the final attention mask.
The final attention mask generated by this process is from iteration 3.
In an earlier iteration, the sparsity of the mask is low because the group size of blocks is very large (8), so the $8*2$ key values are treated as single groups.
The attention score of that group is represented by the attention score between the query and the group's first block ($b_k$).

\subsection{Ablation Study on Block Size}
\label{sec:ablation_bq_bk}

\input{tables/table_ablation_blocksize_ppl}
\input{tables/table_ablation_blocksize_decoding}
\input{tables/table_ablation_blocksize_specdec}

We perform an ablation study on block sizes ($b_q, b_k$) using our method.
Block size $b_q$ determines how many queries are grouped into the block during the masking iteration and sparse attention.
And block size $b_k$ determines how many keys are grouped.
Block size is a really important factor in utilizing MMU (e.g., NVIDIA TensorCore) in modern accelerators.
MMU enables matrix multiplication and tensor operations to be performed in single or fewer cycles rather than processing one by one using floating point multiplication and addition.
This kind of accelerator trend leads to mismatching of wall-clock latency and FLOPs in modern hardware.
Therefore, we check the performance and latency trade-off among grouping queries and keys by block size $b_q, b_k$.

In \cref{tab:ablation_blocksize_ppl}, we show that perplexity gets better as $b_q$ increases while it gets worse as $b_k$ increases.
It is not intuitive that increasing $b_q$ shows better perplexity than before because they lose the resolution across the query dimension in attention mask estimation.
However, the result shows that more block size (more averaging) across the query (time) dimension shows better performance.
In contrast to this observation, $b_k$ works as expected, like that less resolution in key (past knowledge or memory) dimension leads to worse performance.

This phenomenon makes our method speed up without any performance loss, even achieving better performance.
In \cref{tab:ablation_blocksize_decoding}, we measure the micro latency benchmark of our attention operation during the decoding stage, which feeds a single query into the attention operator.
With a single query, we cannot utilize the MMU fully because, during sparse attention and attention score estimation in masking iteration, we cannot matrix multiply between the $Q$ group and $K$ group.
We have a single query vector; therefore, we need a vector-matrix multiplier instead of matrix-matrix multiplication, which is the main key feature of MMU.
However, in \cref{tab:ablation_blocksize_specdec}, we measure the micro latency benchmark of our attention operation during the decoding stage with a speculative decoding strategy, which feeds multiple queries into the attention operator.
We feed 32 query vectors within a query dimension in the input tensor; therefore, now we can utilize a matrix-matrix multiplier in an MMU.
With these multiple queries and MMU utilization, our method could achieve a 10.23 times speedup on 12k sequence length compared to PyTorch naive implementation (using bmm).

We use $b_q=32, b_k=2$ by default, according to the ablation study across the block sizes.
We choose $b_q=32$ because increasing $b_q$ leads to better latency and perplexity.
However, we stopped increasing $b_q$ by 32 because the current modern GPU, especially the NVIDIA Ampere series, usually does not support matrix-matrix multiplication larger than 32.
And maybe in the future, some variants will support larger matrix multiplication, just like Google TPU.
However, larger blocks need more register allocation for block masking and address calculation.
Therefore, considering implementation limitations, we think there is no benefit to increasing $b_q$ infinitely.
Also, from a performance perspective, we do not think this trend will keep over $b_q > 32$.
We choose $b_k=2$ because latency speedup from $b_k=1$ to $b_k=2$ is huge respect to perplexity loss.

Additionally, we measure the latency with $r_m=1$, which means without mask caching.
Therefore, this speedup will be amplified with $r_m$ in a practical setting.
% As we show in \cref{tab:wikitext2}, an attention speedup due to sub-quadratic complexity is more than 36 times.

\subsection{Ablation Study on Dense Layer Choice}
\label{sec:dense_layer}

\begin{figure}[h]
\centering
\includegraphics[width=0.5\textwidth]{figures/plot_ablation_ld.pdf}
\vspace{-1em}
\caption{\textbf{How Many Layers Should be Remained as Dense Layer $l_d$?} We use $l_d = 3$ as the default value. The Y-axis means how many first layers of the Transformer model are kept as dense attention layers rather than replaced with HiP. We use Llama2 7B 32k for PPL evaluation on Wikitext2}
\label{fig:ablation_ld}
\vspace{.5em}
\end{figure}

We do not replace the first few layers ($l_d$) of the Transformer because the first few layers tend to have dense attention probabilities rather than sparse attention probabilities.
This phenomenon is well described in previous results~\citep {ribar_sparq_2023}.
The first few layers exchange information globally and uniformly across the tokens.

Therefore, in \cref{fig:ablation_ld}, we perform an ablation study on how many first layers should remain as dense attention.
We observe that the first layers are kept as dense attention and then more perplexity.
In other words, if we replace the original transformer block with HiP attention, we could minimize the performance degradation.
However, for maximum practical speedup, we want to minimize the number of dense layers for the experiment.
Therefore, we run the ablations study on different numbers of dense layers, and we choose 3.
For 2 to 3, the performance (perplexity) improvement is maximized.
In conclusion, the practical effect of dense layers on latency is minimal because the number of dense layers (e.g., 3, 4) is small compared to the number of HiP layers (e.g., 29, 36).
We show that we could achieve practical end-to-end speedup compared to baselines in \cref{fig:latency_breakdown}.

% \subsection{Ablation Study of Mask Refreshing Interval in Decoding}
% \label{sec:ablation_refresh_interval}

% % \input{tables/table_vllm_quality}

% Also, we perform an ablation study on the mask refresh interval $r_m$ in \Cref{tab:refresh_mask}.
% By caching the mask and reusing it for a few decoding steps, we can avoid re-computing the attention mask frequently while losing a bit of accuracy.
% However, the accuracy degradation is not significant compared to the large speedup, as shown in \cref{tab:refresh_mask}.
% With our default setting $r_m = 8$, we could speed up 1.7$\times$ and achieve only a 0.52\%p degradation in the Booksum ROUGE-1 score compared to without mask caching.

\subsection{Ablation Study on Representative Token Location}
\label{sec:repr_token}

In the theoretical analysis section, we have mathematically proved that selecting the middle token as the representative token of a section guarantees better performance than random selection. In this subsection, we perform a brief ablation study on the location of the representative token to see if this claim is true. In the following experiment, we measured the perplexity values of the Llama3.1 8B model for various representative token locations. We used the PG19 dataset as the input for the model.

\input{tables/table_ablation_representative_token}

\cref{tab:ablation_reprtoken} shows the experimental results. As can be seen in the table, the perplexity value is minimized when the middle token is selected as the representative token. This closely matches our expectations, and it asserts that our theoretical analysis of the representative token location is valid. Therefore, this ablation study justifies the selection of the middle token as the representative token.

\subsection{Discussion about KV Cache Eviction and Compression Strategy}
\label{sec:discussion_kv_cache_eviction}

We think the KV eviction and compression strategy is an orthogonal method to our proposed HiP method, and we can cooperate with KV cache strategies.
Users can use sparse linear attention methods like ours with a KV eviction strategy.
If the KV eviction strategy is careful enough, our method should retain the same performance as vanilla attention.

Also, the typical retention ratio ($512/32000=1.6\%$) of our method is much more extreme than state-of-art eviction strategies ($10$ to $20\%$~\citep{zhang_h2o_2023}).
Moreover, the KV eviction strategy loses information permanently, which should be a problem.
We think we can solve the memory pressure from the KV cache should be solved with the memory hierarchy of the computer system. NVMe storage should be large enough to store everything.
We think KV eviction has limitations because we cannot estimate which information will be important in the future.
Therefore, we should store every important piece of knowledge somewhere in our memory.
During the storage of the KV cache, we can utilize a partial KV cache eviction strategy.

We believe KV cache offloading is the future direction to tackle the memory limitation of the attention mechanism, as we proposed in the main section.
% In the CUDA UVM experiment \cref{sec:uvm}, we show that the KV cache offloading strategy is way much more feasible with our method, even if the offloading method is on-demand (UVM).
% In future work, we will tackle this issue more precisely.

\subsection{Discussion about Speculative Decoding}
\label{sec:discussion_speculative}

We think that HiP can cooperate with many other speculative decoding strategies orthogonal~\citep{leviathan_specdec_2023, miao_specinfer_tree_2024, fu_lookahead_2024, cai_medusa_2024} because they are working with the output of LLM, which is logits.
Also, the speculative decoding method tries to decode multiple queries simultaneously to verify the speculative generation candidates.
This characteristic of speculative decoding will take advantage of additional speedup with the large batches in HiP.

% \subsection{Discussion on Possible Improvements in HiP}
% \label{sec:hip_possible_improvements}
% \input{sections/appendix_ensemble}

\subsection{Remaining Challenges in HiP and Potential Solutions}
\label{sec:discussion_possible_improvements}

Although HiP successfully replaces the existing vanilla attention, there is room for improvement as follows:
\begin{itemize}[leftmargin=15pt, itemsep=2pt]
    \item While HiP outperforms the baselines and performs similarly with Flash Attention in long context evaluation, HiP still underperforms Flash Attention with smaller $k$ (lower compute budget).
    \item As shown in \cref{sec:dense_layer}, HiP uses a few layers ($l_d$) of quadratic dense attention, which could lead to a higher cost as the context length increases.
    \item Since HiP enforces every row in the attention matrix to have the same sparsity, this is not optimal to handle dynamic sparsity of attention~\citep{ribar_sparq_2023, lee_sea_2023}.
\end{itemize}

As shown in \cref{fig:concept}, because HiP discards the bottom chunks for unselected branches, it is impossible to select tokens from the actual top-$k$ set if they happen to be within those discarded chunks. We refer to this as 'branch early termination' in HiP, and addressing this issue could help resolve above improvement points.
Therefore, we propose two possible research directions that tackle the branch early termination problem while also enabling dynamic sparsity: (1) an ensemble hierarchical top-$k$ approximation and (2) an improved tree traverse strategy.
% Both approaches focus on diversifying branch locations within HiP to ensure it considers indices that may have been overlooked due to its binary search strategy, which could be more helpful when handling extremely large context lengths.

First, for the ensemble hierarchical top-$k$ approximation, we generate multiple HiP masks by injecting randomness into branching decisions during a specific iteration and create a final ensemble mask by aggregating indices from these masks.
The ensemble method demonstrates that applying different branching in a given iteration can enhance inference accuracy and indicates the potential to replace the dense layers $l_d$ in HiP with comparable performance, as shown in \cref{sec:ensemble_result}. Moreover, \cref{fig:ensemble_mask_comparison} illustrates how the ensemble method enables dynamic sparsity across layers and heads, addressing the limitation of uniform sparsity in HiP.

Second, we could explore more diverse traversal methods, rather than strictly relying on binary branching and selecting the top-$k$ tokens at each iteration. In~\cref{sec:ensemble_result}, we examine the effectiveness of applying ensemble techniques to HiP masks with slight randomness. However, this approach incurs additional computational costs due to the oversampling of the mask, which can be quite expensive.
Therefore, to achieve a similar effect, we could diversify the tree branching beyond the binary structure, similar to an n-beam search. Another potential solution specifically tailored to HiP is to apply multi-branching in a certain iteration and oversample the chunks in subsequent iterations, maintaining multiple paths until the end. By doing so, the final iteration would include more than $k$ candidates, resolving the branch early termination issue and allowing us to decide how many tokens to select for dynamic sparsity across the layers.
% creating more paths to the final top-$k$ approximated candidates
% By exploring more diverse algorithms for tree construction and traversal, we aim to achieve a more precise approximation of the top-$k$ selection.
% Second, we may try to investigate more diverse ways to traverse rather than strictly utilizing binary branching and selecting top-$k$ tokens in each iteration.
% In~\cref{sec:ensemble_result}, we investigate the effectiveness of applying ensemble to HiP masks with slight randomness.
% However, this approach requires additional computational cost $n_e \times k$ due to the oversampling of the mask, which is expensive.
% We can achieve a similar effect by diversifying the tree branching beyond that of a binary tree, akin to an n-beam search.
% Another potential solution uniquely designed for HiP is a multi-branching and then oversampling block in HiP masking iterations.
% After a certain iteration of HiP, we may multi-branch rather than binary branching to make more paths to the final top-$k$ approximated candidates to keep multiple paths to the end.
% Therefore, by investigating more diverse algorithms for constructing and traversing trees, we hope to approximate the top-$k$ selection more precisely.

\subsection{Unique GPU Resource Demand Pattern of HiP Compared to Flash Attention}

Our novel HiP attention is quite different from any other attention implementations.
We do not heavily rely on ALUs (floating operations) like Flash Attention.
Also, we do not heavily rely on memory bandwidth like previous sparse attention methods like \textsf{H$_2$O} (it has to store attention scores in global memory).
The HiP relies on thread resources because of the top-$k$ operator in between HiP iterations.
Moreover, we have highly complex algorithms compared to traditional GPU applications like graphics shaders.
Due to this highly complex algorithm, we heavily rely on thread resources, even if we are heavily optimized with MMU for attention score sampling.

Thanks to the reduced complexity of HiP, $O(T \log T)$, we are winning every configuration with long context prefill and decoding.
Since the decoding phase is highly memory-dependent for Flash Attention, we also always win in most practical context lengths.
However, we sometimes lose if Flash Attention is way too much faster because of the trillions of floating operation specifications of high-end server-grade GPUs.
Moreover, Flash Attention 2 and 3 utilize special floating point valuation resources in GPU, especially on H100; FA3 is way too much faster in another setting.
Therefore, we are starting to lose in short context ($T$=32k) to FA2 and FA3 because of the speedup of Flash Attention on H100.

This phenomenon is disappointing to us. Therefore, we try to investigate why HiP in H100 is so slower than others, even compared to consumer-grade GPUs such as RTX 4090.
We think the high demand for CUDA thread resources is due to our internal sorting algorithm.
Since we must remain top-$k$ blocks in every iteration, we must perform $O(k \log k)$ cost sorting operation $O(\log T)$ times.
Therefore, as $k$ grows, we are staving to allocate worker thread for score comparison.

\begin{table}[h]\centering
\vspace{1.0em}
\caption{\textbf{Prefill Latency Speedup of HiP by Removing Sorting on RTX 4090 and H100 with different $k$ and $T$.} The time unit is milliseconds.}\label{tab:appendix_4090_h100_latency}
\resizebox{\linewidth}{!}{
\setlength{\columnsep}{2pt}
\begin{tabular}{r|ccc|ccc|ccc|ccc}\toprule
\makecell[r]{Device} &\multicolumn{6}{c|}{4090} &\multicolumn{6}{c}{H100} \\
% \cmidrule{2-13}
\makecell[r]{Context Length} &\multicolumn{3}{c|}{32k} &\multicolumn{3}{c|}{128k} &\multicolumn{3}{c|}{32k} &\multicolumn{3}{c}{128k} \\
% \cmidrule{2-13}
\makecell[r]{Prefill Latency} &w/o Sort &w/ Sort &Speedup &w/o Sort &w/ Sort &Speedup &w/o Sort &w/ Sort &Speedup &w/o Sort &w/ Sort &Speedup \\\midrule
Flash Attention &\multicolumn{3}{c|}{56.0} &\multicolumn{3}{c|}{855.2} &\multicolumn{3}{c|}{24.05} &\multicolumn{3}{c}{430.79} \\\midrule
HiP k=512 &16.48 &19.33 &\textbf{1.173} &85.62 &103.13 &\textbf{1.204} &16.55 &19.80 &\textbf{1.196} &86.65 &108.38 &\textbf{1.251} \\
HiP k=1024 &26.96 &33.70 &\textbf{1.250} &150.90 &190.98 &\textbf{1.266} &27.56 &36.30 &\textbf{1.317} &153.41 &210.78 &\textbf{1.374} \\
HiP k=2048 &44.12 &62.27 &\textbf{1.411} &263.60 &386.97 &\textbf{1.468} &43.74 &69.54 &\textbf{1.590} &260.22 &434.30 &\textbf{1.669} \\
\bottomrule
\end{tabular}
}
\vspace{0.5em}
\end{table}

\begin{table}[h]\centering
\vspace{0.5em}
\caption{\textbf{Technical Specifications of 4090 and H100.} We put a comparison of prefill latency compared to RTX 4090.}\label{tab:appendix_4090_h100_spec}
\resizebox{1.0\linewidth}{!}{
\setlength{\columnsep}{2pt}
\begin{tabular}{l|rrrr|rr}
\toprule
&
    Rel. CUDA Core &
    Rel. TFLOPs &
    Rel. Mem. Bandwidth &
    Rel. Clock Speed &
    Rel. HiP Speedup &
    Rel. FA2 Speedup \\
\midrule
RTX 4090 &
    1.00 &
    1.00 &
    1.00 &
    1.00 &
    1.00 &
    1.00 \\
H100 &
    1.00 &
    3.66 &
    3.32 &
    0.71 &
    0.89 &
    1.99 \\
\bottomrule
\end{tabular}
}
\vspace{0.5em}
\end{table}

We want to show that the elimination of the sorting operation will speed up our top-$k$ estimation.
To do so, we replace sorting with an identity function.
So, in this version, we always select the first half blocks to pass the next HiP iteration.
As shown in \cref{tab:appendix_4090_h100_latency}, eliminating sorting speed up our HiP significantly. In 4090, we could observe 46.8\% speedup, and in H100, we could observe 66.9\%.
We can see the high relation between (CUDA cores + relative clock speed) and HiP speed as shown in~\cref{tab:appendix_4090_h100_spec}.
So, we will try to investigate removing the sorting and replacing it with some approximations for more practicality of HiP.

This characteristic is quite good for cost-effectiveness.
Nvidia does not usually reduce CUDA cores on consumer-grade GPUs; therefore, we could achieve the same speed as H100 while reducing GPU costs more than ten times.
Even in server-grade GPUs, there are some great cost-effective alternatives.
For example, L40s has more ALU than 4090 and the same amount of CUDA core.
Therefore, L40s will offer A100-level linear layer computation while offering H100-level attention, which is way more after than flash attention on L40s.
In the L40s, flash attention will be extremely slow, like in A100 and 4090, because they have similar FLOPs due to the price tag.
We wanted to test L40s during submission, but unfortunately, we could not find any possible option to get L40s.

The lower-grade GPUs often struggle with the small size of VRAM.
However, the tiny memory of lower-grade GPU is not a problem with our method due to the powerful KV cache offloading feature without decoding throughput degradation.
We have already shown that we can serve 64K context length with a single RTX 4090 card, and if you put 8 of them together, then we can serve around 512K context length with high decoding throughput.
For example, the tinygrad company offers 8x 4090 workstations with only 40,000\$~\citep{tinygrad2024tinybox} \textit{(we are not them, just for clarification)}.
The price is almost similar to a single H100 card, but you can serve 512K context length with more than twice TFLOPs!
This means that if you have two nodes of that machine, you can actually run Google Gemini class~\citep{googlegemini2024million} long context LLM in the home.
And if the tensor parallelism is linearly scaled with two nodes, you can decode 1,527 tokens with 64k context length.
Since our method is almost a logarithm scale with context length during decoding, we can expect to decode around 1K tokens per second with a one million context length.
So, we are really excited to introduce our KV cache offloading feature with HiP in many practical aspects.

\section{Potential Negative Social Impact}
In this paper, we do not perform a careful investigation on LLM alignment performance with HiP.
There is the potential that HiP might break the LLM safety guard.
However, as far as we observed during the experiment, the HiP could preserve most of the behavior of trained LLM.
Furthermore, we could always adopt the third-party LLM safety guard model such as LLaMA Guard~\citep{inan_llama_guard_2023}.
% Maybe our model can degrade the performance of the guard of LLM., so please be careful.
```

### Paper: `InfiniteHiP: Extending LLM Context Beyond Millions`

```latex
\begin{abstract}
In modern large language models (LLMs), handling very long context lengths presents significant challenges as it causes slower inference speeds and increased memory costs. Additionally, most existing pre-trained LLMs fail to generalize beyond their original training sequence lengths. To enable efficient and practical long-context utilization, we introduce \textit{InfiniteHiP}, a novel and practical LLM inference framework that accelerates processing by dynamically eliminating irrelevant context tokens through a modular hierarchical token pruning algorithm. Our method also allows generalization to longer sequences by selectively applying various RoPE adjustment methods according to the internal attention patterns within LLMs. Furthermore, we offload the key-value cache to host memory during inference, significantly reducing GPU memory pressure. As a result, InfiniteHiP enables the processing of up to 3 million tokens on a single L40s 48GB GPU -- 3x larger -- without any permanent loss of context information. Our framework achieves an 18.95x speedup in attention decoding for a 1 million token context without requiring additional training. We implement our method in the SGLang framework and demonstrate its effectiveness and practicality through extensive evaluations.
\end{abstract}

\section{Introduction}
\label{introduction}

In modern Transformer-based generative large language models (LLMs), extending the context length is essential for improving comprehension and coherence in long-context, multi-modal, and retrieval-augmented language generation. However, achieving this poses significant challenges, primarily due to the attention mechanism~\citep{vaswani_attention_2023}, a fundamental component of these models. The attention mechanism computes relationships between each input token and all preceding tokens, causing computational and memory costs to scale quadratically as the input sequence length increases. Another problem arising from the attention mechanism is the key-value (KV) cache. During generation, previously computed attention keys and values are cached on GPU memory for reuse. However, the KV cache size scales linearly with context length, creating a challenge for long context inference.

Various methods have been proposed to reduce the high costs of the attention mechanism.
FlashAttention (FA2)~\citep{dao_flashattention_2022} significantly reduces memory consumption and bandwidth utilization by avoiding writing the entire attention score matrix to global GPU memory. However, it does not reduce the arithmetic computation cost. Other approaches~\citep{xiao_efficient_2024, lee_training-free_2024} selectively attend to a fixed number of key tokens, either statically or dynamically, during attention inference.

\input{figures/fig_concept}

Many efforts have also been made to mitigate the memory burden of the KV cache. KV cache eviction methods selectively `forget' past contexts to conserve GPU memory~\citep{zhang_h_2o_2023, oren_transformers_2024}. However, these methods permanently erase past contexts, which may be needed again later. HiP attention~\citep{lee_training-free_2024} offloads infrequently accessed `cold' tokens to larger and cheaper host memory, dynamically fetching them back to GPU during generation only when needed while keeping only frequently accessed `hot' tokens on the GPU.

Despite these optimizations, another problem with context extension still remains: pre-trained LLMs cannot handle inputs longer than their trained context length. Since the attention mechanism is permutation invariant, they utilize positional embedding methods such as Rotary Positional Embeddings (RoPE)~\cite{su_roformer_2023} to model the temporal order of tokens. However, as LLMs are typically pre-trained on sequences truncated to a fixed length, they fail to adapt to unseen positions when prompted with longer contexts.

One option for overcoming this problem is long context fine-tuning~\citep{roziere_code_2024}, i.e., fine-tuning the model on a set of longer inputs. However, fine-tuning, especially on long sequences, requires exorbitant training costs and high-quality training data. Thus, \textit{out-of-length} (OOL) generalization, i.e., the capability for pre-trained models to perform well beyond their pre-trained limits without training, becomes increasingly important. Self-Extend~\citep{jin_llm_2024} proposes a training-free way of scaling the RoPE embeddings beyond the pre-trained limit.

In this paper, we propose \textit{\ours}, a long-context LLM framework that combines the strengths of all the above methods. To alleviate the computational burden of attention, \ours proposes a novel modular sparse attention scheme that minimizes computation for less important contexts. For optimizing KV cache offloading, \ours enhances HiP attention~\citep{lee_training-free_2024}'s offloading strategy with a sophisticated LRU-based cache policy. Finally, \ours achieves OOL generalization by carefully applying various RoPE adjustment strategies within different components of LLMs according to their internal attention patterns. By providing a unified solution to all the aforementioned problems as a whole, \ours demonstrates strong practicality and is well suited for real-world deployment.

What sets \ours apart is its novel use of pruning modules, as illustrated in \Cref{fig:concept}. These modules employ a novel modular hierarchical pruning algorithm to selectively discard less important input tokens. The algorithm leverages common patterns observed in attention matrices of popular LLMs -- namely, their sparsity and the spatial locality of nonzero entries within a sequence -- to prune irrelevant tokens effectively. Each pruning module partitions the input sequence into chunks of fixed length $b_k$, and efficiently identifies the approximate top-1 token with the highest attention score within each chunk in parallel. Only the top-$K$ most significant chunks (where $K$ is constant) are passed to the next module, while the rest are discarded. By stacking multiple pruning modules, \ours iteratively refines a block sparse attention mask.

While our work is based upon HiP~\citep{lee_training-free_2024}, we overhaul several key mechanisms. First, our novel hierarchical pruning modules achieve higher accuracy compared to HiP's heuristic-based hierarchical pruning. Second, the pruning algorithm within each module is significantly faster due to its enhanced parallelizability. Lastly, its modular design enables fine-grained control over pruning-stage caches, leading to much faster decoding than HiP.

\ours enables extremely long-context inference with pre-trained LLMs, surpassing their original context length limits without quality degradation while overcoming GPU memory limitations with efficient KV cache offloading. As a training-free solution, \ours can be used as a drop-in replacement for any pretrained Transformer-based LLM, providing faster inference and extending usable context length at both the model and hardware levels.

Our contributions can be summarized as follows:
\begin{itemize}[itemsep=0.5mm, parsep=2pt, leftmargin=12pt, topsep=0pt]
\item We propose a modular, highly parallelizable training-free hierarchically pruned attention mechanism that enables out-of-length generalization while significantly speeding up LLM inference on long contexts.
\item We demonstrate that our method does not degrade the LLM's long-context language understanding, reasoning, and text generation capabilities compared to other SoTA efficient long-context inference methods.
\item We efficiently implement \ours on the SGLang LLM serving framework, achieving a 7.24$\times$ speedup in end-to-end decoding on a 3M token context while using only 3.34\% of the VRAM required by FA2, and design an efficient KV cache offloading algorithm that utilizes modular pruning algorithm, making it practical for real-world scenarios.
\end{itemize}

\section{Related Works}
\label{related_works}

\input{figures/fig_pruning}

Previous studies have proposed dynamic token selection for efficient LLM inference for long contexts. MInference~\citep{jiang_minference_2024} classifies attention heads into two types to estimate the sparse attention pattern, which is used to drop less important tokens before the dot product.
%: vertical-slash heads and block-sparse heads. Vertical-slash heads use the last few queries to estimate the attention pattern for the rest of the queries, whereas block-sparse heads use mean-pooled queries and keys to estimate a block-sparse attention pattern.
While this method considerably speeds up the prefill stage, it cannot be applied in the decoding stage, which takes up most of the inference time.
%
HiP Attention~\citep{lee_training-free_2024} estimates the top-k context blocks with the highest attention scores in a hierarchical and iterative manner, significantly speeding up both prefill and decoding in long contexts. However, the iterative algorithm involves many global thread synchronizations, which hinders parallelism.
Quest~\citep{tang_quest_2024} divides the context into fixed-size pages and estimates the maximum attention score by using cached element-wise min and max vectors.
%TokenSelect~\citep{xiao_infllm_2024} selects tokens by performing a global top-$k$ operation, and updates the selected token mask whenever the cosine distance of the query exceeds a threshold during decoding. For computational efficiency, it ignores the difference between attention heads.
InfLLM~\citep{xiao_infllm_2024} divides the context sequence into blocks and selects representative tokens in each block. For each new query, the top-k blocks whose representative tokens give the highest attention scores are selected.
In contrast to our \ours, the representative tokens of each block are prechosen and do not change with the current query.
Both HiP Attention and InfLLM enable KV cache offloading, which makes long context inference possible within a single GPU.

\section{From HiP to \ours}
\label{methodology}


In this section, we describe three major problems identified in HiP~\citep{lee_training-free_2024} and our proposed changes to address those problems in \ours.

\begin{tcolorbox}[
    enhanced,                 % Enable advanced features
    colframe=black,           % Black frame
    colback=white,            % White background
    boxrule=1.5pt,            % Thickness of the border
    width=\textwidth,         % Full width
    before skip=-0.0em plus 3.0pt minus 0.0pt,          % Space before box
    after skip=1.0em,           % Space after box
    top=1em,
    parbox=false,
    breakable,
    break at=0pt/10cm,
    vfill before first,
    title after break=,
    pad at break=1mm,
    overlay unbroken and first={%
        % Header box overlay
        \node[anchor=north west, fill=black, text=white, inner sep=1.5mm,
        xshift=4mm, yshift=3.5mm, rounded corners=1mm]
        at (frame.north west) {Problem 1: Low Parallelizability of Hierarchical Top-$k$ Estimation};
    }
]
\textbf{Problem\quad} In HiP~\citep{lee_training-free_2024}'s Hierarchical Top-$k$ Estimation, each iteration contains a global top-$k$ operation which selects $k$ chunks out of the $2k$ candidates, which limits the degree of parallelism per attention head to $k$ threads (typically set to 1024), regardless of the context length. Furthermore, each iteration requires a global synchronization, so $O(\log_2 T)$ global synchronizations are needed (where $T$ is the number of context tokens).

% \textbf{Chunk sparsity of the attention mechanism.}
% To devise an algorithm that estimates the locations of the top-$k$ key tokens for block sparse attention, we first analyze the characteristics of the attention score distribution.
% We observe distinct patterns in the distribution of top-$k$ tokens within a typical LLM attention context.

% % TODO:: Improve clarity

% \Cref{fig:motivation} suggests that the top-$k$ tokens are concentrated in a small number of context chunks. As shown in the left chart, fewer than 2\% of the chunks contain more than 12.5\% of the top-2K tokens in a 128K-token context. Furthermore, the right chart tells us that around 75\% of the 64-token context chunks do not contain any top-2K tokens at all.
% %These observations suggest that we can effectively utilize the top-$k$ tokens by using the few context chunks containing them.
% These observations suggest that selecting the few context chunks containing top-$k$ tokens can act as a good approximation for selecting the individual top-$k$ tokens.
% To this end, we devise an efficient algorithm that divides the context into fixed-size chunks and filters out irrelevant chunks based on their estimated maximum attention scores.

\textbf{Solution\quad} \ours overhauls the token pruning algorithm to allow a higher degree of parallelism and require fewer global thread synchronizations.
This is done by splitting the context sequence into $O(T)$ chunks of fixed size, instead of $O(1)$ chunks of variable size as in HiP.
This is motivated by the chunk sparsity of attention scores, which suggests that the top-$k$ tokens are concentrated in few contiguous context chunks, shown in \Cref{fig:motivation}.

Also, just few (3 in our default setting) global thread synchronizations are required at each of our novel pruning stage.
While this change increases the time complexity of the pruning algorithm from HiP's $O(\log T)$ to $O(T)$, the increased parallelizability means that \ours's pruning algorithm runs faster on modern GPUs in practice.
See \Cref{subsec:pruning} for an in-depth description of our token pruning algorithm.
\end{tcolorbox}
\vspace{-0.7em}
\begin{tcolorbox}[
    enhanced,                 % Enable advanced features
    colframe=black,           % Black frame
    colback=white,            % White background
    boxrule=1.5pt,            % Thickness of the border
    width=\textwidth,         % Full width
    before skip=1.2em plus 3.0pt minus 0.0pt,          % Space before box
    after skip=1.0em,           % Space after box
    top=1em,
    parbox=false,
    overlay={%
        % Header box overlay
        \node[anchor=north west, fill=black, text=white, inner sep=1.5mm,
        xshift=4mm, yshift=3.5mm, rounded corners=1mm]
        at (frame.north west) {Problem 2: No Out-of-length Generalization Capability};
    }
]
\textbf{Problem\quad} Speeding up long-context inference with 3 million tokens is not useful if existing pre-trained models' generation quality drops significantly after a 32K token threshold. HiP is capable of doing the former, but its usefulness is severely limited by pre-trained models that do not support out-of-length generalization.

\textbf{Solution\quad} \ours employs a dynamic RoPE adjustment trick to allow OOL generalization of any pre-trained short-context model. See \Cref{subsec:rope} for more details.
\vspace{-0.7em}
\end{tcolorbox}
\begin{tcolorbox}[
    enhanced,                 % Enable advanced features
    colframe=black,           % Black frame
    colback=white,            % White background
    boxrule=1.5pt,            % Thickness of the border
    width=\textwidth,         % Full width
    before skip=1.2em plus 3.0pt minus 0.0pt,          % Space before box
    after skip=0.0em,           % Space after box
    top=1em,
    parbox=false,
    overlay={%
        % Header box overlay
        \node[anchor=north west, fill=black, text=white, inner sep=1.5mm,
        xshift=4mm, yshift=3.5mm, rounded corners=1mm]
        at (frame.north west) {Problem 3: Inefficient KV Cache Offloading};
    }
]
\textbf{Problem\quad} While HiP proposes a preliminary method to offload the KV cache to the host memory to reduce pressure on the GPU VRAM. However, it incurs a large overhead during top-$k$ estimation process, because which elements will be accessed from the host memory is inherently unpredictable.

\textbf{Solution\quad}
\ours addresses this problem by caching each pruning stage's output candidates, and refreshing each of them at different intervals.
The first pruning stage, which is the most costly, is refreshed least frequently, and each subsequent stages are refreshed more often.
This strikes a balance between performance and accuracy. Furthermore, we use the Least Recently Used (LRU) policy for efficient GPU cache management. More details are described in \Cref{subsec:offload}.
\end{tcolorbox}

\subsection{Efficient Multi-Stage Context Pruning}
\label{subsec:pruning}

In this section, we introduce a novel and efficient design for context pruning.
The complete description of our algorithm is detailed in \Cref{sec:algorithm}.
Here, we describe the overview of our design.

\textbf{Background.}
Given query, key, and value sequences $\bm{Q}, \bm{K}, \bm{V} \in \mathbb{R}^{H\times T\times d}$, the conventional multi-head attention output $\bm{O}$ is computed as
$\bm{O} = \text{Concat}[\bm{O}_1, \dots, \bm{O}_H]$, where
$\bm{S}_h = \bm{Q}_h\bm{K}_h^\top \in \mathbb{R}^{T\times T}$,
$\bm{P}_h = \mathrm{softmax}(\bm{S}_h) \in \mathbb{R}^{T\times T}$,
$\bm{O}_h = \bm{P}_h\bm{V}_h \in \mathbb{R}^{T\times d}$ for all $h = 1..H$,
where $H$ denotes the number of attention heads, $T$ denotes the sequence length, $d$ denotes the embedding dimension, and softmax is applied row-wise~\cite{vaswani_attention_2023}. The causal masking and constant scaling are omitted for brevity.
The $\bm{S}$ and $\bm{P}$ matrices are each called the \textit{attention scores} and \textit{probabilities}.

Note that the initial $n_\text{sink}$ tokens (\textit{sink} tokens) and $n_\text{stream}$ most recent tokens (\textit{streaming} tokens) are always included. We sparsely select the tokens in between the sink and streaming tokens.
We aim to find a block sparse attention mask that approximately selects the top-$K$ key blocks with the highest attention scores for each query block. This allows us to perform efficient block sparse attention (BSA) while preserving the capabilities of the model~\citep{lee_training-free_2024}.
For conciseness, in this section, we ignore the existence of sink and streaming tokens, as well as the causal part of the self-attention mechanism.

\textbf{Efficient Modular Context Pruning.}
Unlike HiP~\citep{lee_training-free_2024}, we use multiple pruning stages to find the top-$k$ tokens, each discarding context chunks irrelevant to the current query. By applying the pruning stages, \ours generates a sparse attention mask, a good approximation for the top-$k$ tokens.
%The exact details are as follows.

%TODO:: Consider adding verbal explanation
\Cref{fig:context_pruning} illustrates how each pruning stage preserves only the most relevant contexts.
%A pruning stage narrows down the selection of the context tokens.
First, the input key tokens are partitioned into equally sized chunks of fixed size (in contrast to HiP, which divides the tokens into a fixed number of chunks).
Next, we select a representative token for each key chunk. In HiP, the middle token was always chosen for every chunk. In contrast, \ours chooses the representative token dynamically:
we use a top-1 variant of the Hierarchical Mask Selection Algorithm (HMSA)~\citep{lee_training-free_2024}. Note that our use of HMSA is to find the representative token for each chunk, not by itself for selecting the top-k, which contrasts our method to HiP's. Additionally, since this HMSA is performed locally within each chunk, there is no need for global GPU thread synchronizations.
%Leveraging the idea of attention locality introduced in \citet{lee_training-free_2024}, where nearby tokens tend to display similar attention scores, representative tokens provide an estimate for the attention scores within their chunks.

Using the attention scores of these representative tokens, max-pooled across attention heads, we select the top-$K$ key chunks and discard the rest. The surviving tokens are used as the input for the next pruning stage. By iteratively applying these pruning stages, we can effectively obtain a good estimate of the top-$k$ tokens in the form of a sparse attention mask.

In formal notation, we denote a pruning stage by $\mathcal{S}^{(i)} = (b_q^{(i)}, l_{c}^{(i)}, k^{(i)})$, where $b_q$ is the size of the query block, $l_{c}$ is the chunk size, $k$ is the number of tokens to keep, and the superscript $i = 1~..~N$ denotes the stage index.
To speed up the process by parallel processing, the queries are grouped into contiguous blocks. Specifically, in the $i$th stage, the query $\bm{Q}$ is divided into multiple $b_q^{(i)}$-sized blocks.
We denote the $m$th query block in the $h$th attention head in the $i$th pruning stage by $\bm{q}_{h,m}^{(i)} \in \mathbb{R}^{b_q\times d}$.

For the initial stage, we select all of the key indices $\mathcal{I}_{m}^{(0)} = [1, \dots, T]$ for each query block index $m$. Each pruning stage will transform this list of indices into a smaller list by discarding indices corresponding to less important contexts.

To the $i$th pruning stage, the input sequence $\mathcal{I}_{m}^{(i-1)}$ is divided into contiguous chunks of size $l_{c}^{(i)}$ where the $j$th chunk contains $\mathcal{C}^{(i)}_{m,j}
%:= \left[ \mathcal{I}^{(i-1)}_{m}[j\,l_{c}^{(i)}], \dots, \mathcal{I}^{(i-1)}_{m}[(j+1)l_{c}^{(i)}-1] \right]
$.
From each $\mathcal{C}^{(i)}_{m,j}$, we dynamically pick a representative token independently for each attention head, using a top-1 variant of the algorithm used in HiP~\citep{lee_training-free_2024}. We denote the representative token index for the $h$th attention head as $r^{(i)}_{h,m,j} = \text{SelectRep}(\bm{q}_{h,m}^{(i)}, \mathcal{C}^{(i)}_{m,j})$.

The representative tokens provide a way to estimate the maximum attention score within each chunk. We estimate each chunk's score by computing the maximum value across the attention heads and each query in the query block as
%\begin{equation}
$s^{(i)}_{m,j} := \max_{\text{{$\begin{matrix}[0.1] h=1..H, \\ t=1..b_q^{(i)}\!\!\!\!\! \end{matrix}$}}} (\bm{q}_{h,m}^{(i)})_{t}^\top \bm{k}_{h,r^{(i)}_{h,m,j}}$.
%\end{equation}
Finally, the top $K^{(i)} := k^{(i)}/l_c^{(i)}$ chunks with the highest estimated attention scores are selected for the next stage, as follows:
\begingroup%
\allowdisplaybreaks%
\begin{align}
    &\mathcal{I}^{(i)}_{m'} = \bigcup_{\hat{\jmath} \in \mathcal{T}^{(i)}_{m}} \mathcal{C}_{m, \hat{\jmath}}^{(i)},
    \text{ where } \mathcal{T}^{(i)}_{m} = \underset{j}{\text{arg\,top\,}}\!_{K^{(i)}} (s_{m,j}^{(i)}),
    \text{ and } m'=\text{\small$\begin{cases}
        \lceil m\cdot{b_q^{(i)}}/{b_q^{(i+1)}} \rceil & \text{if } i \leq N, \\
        m & \text{otherwise}.
    \end{cases}$}
\end{align}%
\endgroup

When all $N$ stages are done, we are left with sparse key indices $\mathcal{I}^{(N)}_m \in \{1, \dots, T\}^{k^{(N)}}$ for all query blocks $m = 1~..~T/b_q^{(N)}$, which can be used for efficient block sparse attention, also used in existing sparse attention methods~\citep{lee_training-free_2024,jiang_minference_2024,lai_flexprefill_2025}.

% TODO: Add Positional Embedding Scheme figure
\subsection{Dynamic RoPE for OOL Generalization}
\label{subsec:rope}

\input{tables/tab_infbench}
\input{figures/fig_infbench}

We employ a novel combination of multiple RoPE interpolation strategies for the sparse key tokens for out-of-length generalization.
During token pruning, two strategies are employed:
(1) \textbf{Chunk-indexed RoPE:}
Each key chunk is given a single position ID, where the last chunk's position ID is offset by $n_\text{stream}$ from the current query. All keys in the chunk are given the same position ID.
(2) \textbf{Relative-style RoPE:}
During the hierarchical top-1 estimation algorithm, the left branch gets a position ID offset by $n_\text{stream} + 1$ from the current query, and the right branch gets a position ID offset by $n_\text{stream}$ from the current query. For chunk score estimation, the representative key is given a position ID offset by $n_\text{stream}$ from the current query.
We apply strategy (1) for the first three layers of the LLM and strategy (2) for the rest. The reason for this choice is explained in detail in \cref{sec:visualization_streaming}.
During block sparse attention, we use the StreamingLLM-style RoPE: The selected keys, including the sink and streaming keys, are given position IDs sequentially in their original order, where the most recent token is given the same position ID as the current query~\citep{xiao_efficient_2024}.
Since this dynamic RoPE incurs some computational overhead, it can be disabled when the OOL generalization is unnecessary.

\subsection{KV Cache Offloading}
\label{subsec:offload}

We improve the KV cache offloading mechanism of HiP Attention~\citep{lee_training-free_2024} by enhancing its cache management policy.
Similarly to HiP Attention, we manage the KV cache on the unified memory space while keeping a smaller key bank on the GPU memory, which acts as a cache.
Note that we maintain two different key banks on the GPU for the mask-selection and block sparse-attention processes.
We also keep a page table, which maps the global key index to an index within the GPU key bank, in the GPU memory as well.
Upon a cache miss, the missing keys are fetched from the unified memory space and placed on the GPU bank.
Unlike HiP Attention~\citep{lee_training-free_2024}, we use the Least Recently Used (LRU) policy as the eviction mechanism.

\textbf{Sparse Attention Mask Caching.}
To further reduce latency during decoding, we cache the sparse attention mask for each pruning stage. We observe that the sparse attention mask exhibits temporal locality. Therefore, instead of recomputing it every decoding step, we update the output attention mask for the $i$th pruning stage periodically every $n_\text{refresh}^{(i)}$ steps using the latest query block. Additional details are provided in \Cref{sec:algorithm}.

%The KV cache is dynamically offloaded from the GPU to the host memory by the Least Recently Used (LRU) policy.
%We maintain a key bank with size $n_\text{GPU}$ on the GPU, and a key-value bank with size $n_\text{host}$, where $n_\text{GPU} \ll n_\text{host}$.
%We also maintain a page table that maps the global page index to the GPU bank or contains \texttt{null} in case the token is offloaded to host memory.
%During mask selection, the GPU kernel first tries to load the required keys from the GPU bank by checking the page table. In case of a cache miss, missing keys are loaded from the host memory, and caches the loaded keys replacing other keys in the GPU bank.
%For the block sparse attention step, a separate key-value bank is maintained in a similar manner.
%
%Our KV cache offloading framework consists of a key bank on the GPU, key-value banks on the host memory, and a page table that connects these banks.
%We denote the sizes of the key bank and the key-value banks as  $n_\text{GPU}$ and $n_\text{host}$ respectively, where $n_\text{GPU} \ll n_\text{host}$.
%Separate key-value banks are maintained for the sparse mask selection step and the sparse attention step.
%The page table maps the global page index to either the GPU bank or \texttt{null} when the token is offloaded to host memory.
%We employ a Least Recently Used (LRU) policy as the eviction mechanism for our KV cache offloading framework.
%During mask selection, the GPU kernel first checks the page table to load the required keys.
%If a cache miss occurs, the missing keys are fetched from the host memory and placed in the GPU bank according to the LRU policy.
%Similarly, a separate key-value bank is maintained for the block sparse attention step, following the same offloading and caching procedure.


\textbf{Implementation.}
We implement the GPU kernels for our method using the Triton language~\citep{tillet_triton_2019}. We implement a single GPU kernel for the pruning stage, which can be reused for all stages just with different parameters. For block sparse attention, we implement a method similar to FlashAttention~\citep{dao_flashattention_2022} for prefill and Flash Decoding~\citep{dao_flash_decoding} for decoding. We also combine PagedAttention~\citep{kwon_efficient_2023} to alleviate the overhead from KV cache memory management. To implement dynamic loading and offloading with host memory, we use Nvidia UVM (Unified Virtual Memory).

\section{Experiments}
\label{sec:experiments}

\input{tables/tab_latency_offload_compact}
\input{tables/tab_latency_compact}

\subsection{Experiment Setting}
\label{sec:exp_setting}

\textbf{Hyperparameters and Baselines.}
We describe hyperparameter details in~\cref{sec:appendix_hyperparameter}.
We compare the performance of \ours against the following baselines, mostly chosen for their long-context capabilities.
(1) \textbf{Truncated FA2}: The input context is truncated in the middle to fit in each model's pre-trained limit, and we perform dense attention with FlashAttention2 (FA2)~\citep{dao_flashattention_2022}.
(2) \textbf{DynamicNTK}~\citep{bloc97_ntk-aware_2023} and (3) \textbf{Self-Extend}~\citep{jin_llm_2024} adjust the RoPE for OOL generalization. We perform dense attention with FA2 without truncating the input context for these baselines.
Both (4) \textbf{LM-Infinite}~\citep{han_lm-infinite_2024} and (5) \textbf{StreamingLLM}~\citep{xiao_efficient_2024} use a combination of sink and streaming tokens while also adjusting the RoPE for OOL generalization.
(6) \textbf{H2O}~\citep{zhang_h_2o_2023} is a KV cache eviction strategy which retains the top-$k$ KV tokens at each decoding step.
(7) \textbf{InfLLM}~\citep{xiao_infllm_2024} selects a set of representative tokens for each chunk of the context, and uses them for top-$k$ context selection.
%(8) \textbf{Double Sparse Attention}~\citep{yang_post-training_2024} estimates the top-$k$ tokens by sampling few channels of the key vectors.
(8) \textbf{HiP Attention}~\citep{lee_training-free_2024} uses a hierarchical top-$k$ token selection algorithm based on attention locality.

\input{tables/tab_longbench}

\textbf{Benchmarks.}
We evaluate the performance of \ours on mainstream long-context benchmarks.
(1) LongBench~\citep{bai_longbench_2023}, whose sequence length averages at around 32K tokens,
and (2) $\infty$Bench~\citep{zhang_inftybench_2024} with a sequence length of over 100K tokens.
Both benchmarks feature a diverse range of tasks, such as long document QA, summarization, multi-shot learning, and information retrieval.
We apply our method to the instruction-tuned Llama 3 8B~\citep{grattafiori_llama_2024} and Mistral 0.2 7B models~\citep{jiang_mistral_2023}. As our framework is training-free, applying our method to these models has zero extra cost.


\subsection{Results}
\textbf{LongBench.}
In \Cref{tab:longbench}, our method achieves about 7.17\%p better relative score using Llama 3 and 3.19\%p better using Mistral 0.2 compared to the best-performing baseline, InfLLM.
This is significant because our method processes 4$\times$ fewer key tokens through sparse attention compared to InfLLM, leading to better decoding latency as shown in \cref{tab:latency}.

\textbf{$\infty$Bench.}
We show our results on $\infty$Bench in \Cref{tab:infbench}. Our \textit{3K-fast} and \textit{3K-flash} options use the same setting as \textit{3K} except they use longer mask refreshing intervals as detailed in \Cref{sec:exp_setting}.
Our method achieves 9.99\%p better relative score using Llama 3 and 4.32\%p better using Mistral 0.2 compared to InfLLM. The performance gain is larger than in LongBench, which has a fourfold shorter context. This suggests that our method is able to better utilize longer contexts than the baselines.

To further demonstrate our method's superior OOL generalization ability, we compare $\infty$Bench's En.MC score in various context lengths with Llama 3.1 8B in \cref{fig:infbench_llama3.1}.
While \ours keeps gaining performance as the context length gets longer, baselines with no OOL generalization capability degrade significantly beyond the pretrained context length (128K).
In \cref{fig:infbench_gemma_exaone}, we experiment with other short-context LLMs: Exaone 3 (4K)~\citep{research_exaone3_2024}, Exaone 3.5 (32K)~\citep{research_exaone_2024} and Gemma2 (8K)~\citep{team_gemma_2024}.
We observe the most performance gain in an extended context with these short-context models. For instance, with Gemma2, we gain an impressive +24.45\%p in En.MC and +22.03\%p in En.QA compared to FA2.

\begin{wrapfigure}[5]{r}{0.4\linewidth}
\vspace{-1.75em}
\captionof{table}{\textbf{LongVideoBench Result.}}
\vspace{-0.25em}
\label{tab:longvideobench}
\resizebox{\linewidth}{!}{
\begin{tabular}{lrrrr}
\toprule
&T (k) &FA &Ours \\
\midrule
Llama4 Scout 109B & 256 &52.27 &51.07 \\
Qwen2.5 VL 72B & 128 & 56.15 &54.28 \\
\bottomrule
\end{tabular}
}
\end{wrapfigure}

\textbf{LongVideoBench.} We show our performance with multi-modality on Llama 4 Scout~\citep{meta_llama4} and Qwen2.5 VL 32B~\citep{qwen25_report} on LongVideoBench~\citep{wu2024longvideobench}. We could recover most of the full dense attention performance with only 1.54\% degradation in average.

\subsection{Analysis}
\input{figures/fig_sglang}
\input{figures/fig_topk_recall}
In this section, we analyze the latency and the effect of each of the components of our method.

\textbf{Latency.}
We analyze the latency of our method on a 1-million-token context and compare it against baselines with settings that yield similar benchmark scores. In \cref{tab:latency}, we measure the latencies of attention methods.
%InfLLM uses a 12K context window, HiP uses a 1K window, and ours uses the 3K window setting.
During a 1M token prefill, our method is 20.29$\times$ faster than FlashAttention2 (FA2), 6\% faster than InfLLM, and achieves similar latency with the baseline HiP.
During decoding with a 1M token context, our method significantly outperforms FA2 by 19.85$\times$, InfLLM by 4.98$\times$, and HiP by 92\%.
With context extension (dynamic RoPE) enabled, our method slows down about 1.6$\times$ in prefill and 5\% in decoding due to overheads incurred by additional memory reads of precomputed cos and sin vectors.
Therefore, our method is 50\% slower than InfLLM in context extension-enabled prefill, but it is significantly faster in decoding because decoding is memory-bound:
Our method with a 3K token context window reads fewer context tokens than InfLLM with a 12K token context window.

\textbf{Latency with KV Offloading.} In \cref{tab:latency_offload}, we measure the decoding latency with KV cache offloading enabled on a Passkey retrieval task sample.
We keep FA2 in the table for reference, even though FA2 with UVM offloading is 472$\times$ slower than the baseline HiP.
Among the baseline methods, only InfLLM achieves KV cache offloading in a practical way.
In 256K context decoding, we outperform InfLLM by 3.64$\times$.
With KV cache offloading, the attention mechanism is extremely memory-bound, because accessing the CPU memory over PCIe is 31.5$\times$ more expensive in terms of latency than accessing VRAM.
InfLLM chooses not to access the CPU memory while executing its attention kernel, so it has to sacrifice the precision of its top-k estimation algorithm. This makes larger block and context window sizes necessary to maintain the model's performance on downstream tasks.
In contrast, we choose to access the CPU memory during attention kernel execution like baseline HiP.
This allows more flexibility for the algorithm design, performing better in downstream NLU tasks.
Moreover, our UVM implementation makes the KV cache offloaded attention mechanism a graph-capturable operation, which allows us to avoid CPU overheads, unlike InfLLM.
In contrast to the offloading framework proposed by \citet{lee_training-free_2024}, we cache the sparse attention mask separately for each pruning stage.
This enables us to reduce the frequency of calling the costly initial pruning stage, which scales linearly.

\textbf{Throughput.} In \cref{fig:sglang_decoding}, we present the decoding throughput of our method using RTX 4090 (24GB) and L40S (48GB) GPUs. On the 4090, our method achieves a throughput of 3.20$\times$ higher at a 1M context length compared to the estimated decoding throughput of SRT (SGlang Runtime with FlashInfer). Similarly, on the L40S, our method surpasses SRT by 7.25$\times$ at a 3M context length.
Due to hardware limitations, we estimated the decoding performance since a 1M and 3M context requires approximately 64GB and 192GB of KV cache, respectively, which exceeds the memory capacities of 24GB and 48GB GPUs.
We further demonstrate that adjusting the mask refreshing interval significantly enhances decoding throughput without substantially affecting performance. The \textit{Flash} configuration improves decoding throughput by approximately 3.14$\times$ in a 3M context compared to the \textit{Fast} configuration.

\textbf{Accuracy of top-$k$ estimation.}
In \cref{fig:topk_recall}, we demonstrate our method has better coverage of important tokens, which means higher recall of attention probabilities of selected key tokens.
Our method performs 1.57\%p better than InfLLM and 4.72\%p better than baseline HiP.
The better recall indicates our method follows pretrained attention patterns more closely than the baselines.

%\input{tables/tab_stage_ablation}
\textbf{Ablation on Depth of Stage Modules.}
In \cref{tab:stage_ablation}, we perform an ablation study on a number of stages ($N$) that are used in ours. The latency-performance optimal pruning module combination for each setting is found empirically.

\textbf{Ablation on RoPE interpolation strategies.}
In Table \labelcref{tab:rope_ablation}, we perform an ablation study on the dynamic RoPE extrapolation strategy in masking and sparse attention.
We choose the best-performing RT/ST combination for our method.

\section{Conclusion, Limitations, Future Work, and Broader Impact}
\label{conclusion}
In this paper, we introduced \textit{\ours}, a training-free LLM inference framework for efficient long context inference that supports out-of-length generalization and dynamic KV cache offloading.
\ours effectively addresses the three major challenges that arise in long context LLM inference:
(1) Efficient inference with long contexts,
(2) Out-of-length generalization,
(3) GPU memory conservation through KV cache offloading without `forgetting'.
The experiments on LongBench and $\infty$Bench, and the latency benchmarks demonstrate our method's superior performance and practicality over previous state-of-the-art methods.

\textbf{Broader Impact} We believe our easy-to-apply method can significantly enhance energy efficiency, reduce inference latency and hardware cost of production LLM serving machines, which may contribute to the reduction of environmental cost of running LLMs. However, the reduction in monetary cost may accelerate the societal risks accompanied by LLMs as well.

\textbf{Limitations and Future Work.} Please see \cref{sec:appendix_limitations}.

%We believe our method can significantly enhance energy efficiency and reduce inference latency. Since our approach focuses solely on accelerating the existing Transformer model without altering its trained behavior, we do not expect any notable social impact concerns. Additionally, our method demonstrates strong results in performance recovery, indicating that it can maintain performance levels comparable to the original Transformer while achieving faster processing. We anticipate that this method will offer substantial benefits for production use in the future.
```

### `hip_attn/v1_2/attention_extend.py`

```py
import math
import os
import warnings
from typing import Optional

import cv2
import numba
import numba.cuda
import numpy as np
import torch
import triton
import triton.language as tl
from matplotlib import pyplot as plt
from torch import Tensor

from hip_attn.utils.rope import adjust_rope
from hip_attn.v1_2.attention_decode_bsa import decode_block_sparse_attention
from hip_attn.v1_2.attention_extend_bsa import block_sparse_attention
from hip_attn.v1_2.attention_extend_bsa_tilelang import block_sparse_attention_tilelang
from hip_attn.v1_2.attention_metadata import (
    EnsembleScoreStage,
    EvalScoreStage,
    HiPAttentionArgs,
    HiPAttentionCacheAccessStatistics,
    HiPAttentionOutputMetadata,
    HiPAttentionStageInputCache,
    HiPAttentionState,
    NopStage,
    ScanStage,
    safe_stride,
)
from hip_attn.v1_2.compute_scores_landmark import compute_scores_landmark
from hip_attn.v1_2.compute_v_cos import compute_v_cos
from hip_attn.v1_2.eval_stage import calculate_chunk_score
from hip_attn.v1_2.landmark_sample import landmark_sample
from hip_attn.v1_2.scan_stage import chunk_controllable_sampling_mask
from hip_attn.v1_2.stage_prologue import stage_prologue

try:
    import torch.distributed as dist
    from sglang.srt.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        model_parallel_is_initialized,
        split_tensor_along_last_dim,
        tensor_model_parallel_all_gather,
        tensor_model_parallel_all_reduce,
    )

    SGLANG_DIST_ACTIVATED = True
except ImportError as ex:
    SGLANG_DIST_ACTIVATED = False


def get_local_rank() -> int:
    if SGLANG_DIST_ACTIVATED:
        if not model_parallel_is_initialized():
            return 0
        return get_tensor_model_parallel_rank()
    else:
        return 0


def get_world_size() -> int:
    if SGLANG_DIST_ACTIVATED:
        if not model_parallel_is_initialized():
            return 1
        return get_tensor_model_parallel_world_size()
    else:
        return 1


_NUM_STREAMING_MULTIPROCESSOR = None

DEFAULT_VALUE_HIP_HEAD_REDUCE = "1"


def num_streaming_multiprocessor():
    global _NUM_STREAMING_MULTIPROCESSOR
    if _NUM_STREAMING_MULTIPROCESSOR is None:
        _NUM_STREAMING_MULTIPROCESSOR = (
            numba.cuda.get_current_device().MULTIPROCESSOR_COUNT
        )
    return _NUM_STREAMING_MULTIPROCESSOR


def get_block_sparse_backend(
    args: HiPAttentionArgs, q: torch.Tensor
) -> type(block_sparse_attention):
    # return block_sparse_attention_tilelang

    block_sparse_attention_backend = block_sparse_attention

    # Use flashdecode
    # print(q.shape, int(os.getenv("HIP_FLASHDECODE_THRESH", "32")), (not os.environ.get("HIP_DISABLE_FLASHDECODE", "0") == "1"), (not args.disable_flashdecode))
    if (
        (q.shape[1] < int(os.getenv("HIP_FLASHDECODE_THRESH", "32")))
        and (not os.environ.get("HIP_DISABLE_FLASHDECODE", "0") == "1")
        and (not args.disable_flashdecode)
    ):
        block_sparse_attention_backend = decode_block_sparse_attention

    return block_sparse_attention_backend


@numba.njit(parallel=True)
def render_plot(out_indices_cpu, debug, DEBUG_HEAD, BLOCK_SIZE_Q):
    for i in numba.prange(out_indices_cpu.shape[1]):
        for j in range(out_indices_cpu.shape[-1]):
            # if j >= out_indices_cpu.shape[-1]: continue
            t = out_indices_cpu[0, i, DEBUG_HEAD, j] // BLOCK_SIZE_Q
            debug[i, t : t + 1] = 1


@numba.njit(parallel=True)
def render_plot_dynamic(
    out_indices_cpu,
    debug,
    DEBUG_HEAD,
    BLOCK_SIZE_Q,
    stage_k,
    chunk_size,
    causal_mask=False,
    sliding_window_size=0,
):
    for i in numba.prange(out_indices_cpu.shape[1]):
        for j in range(math.ceil(stage_k / chunk_size)):
            if j >= out_indices_cpu.shape[-1]:
                continue
            t = out_indices_cpu[0, i, DEBUG_HEAD, j] // BLOCK_SIZE_Q
            if causal_mask and ((t + sliding_window_size // BLOCK_SIZE_Q) >= i):
                continue
            tt = t + math.ceil(chunk_size / BLOCK_SIZE_Q)
            if causal_mask:
                tt = min(tt, i + 1)
            debug[i, t:tt] = 1


@numba.njit(parallel=True)
def render_plot_sampled(
    out_indices_cpu,
    debug,
    DEBUG_HEAD,
    BLOCK_CHUNK,
    chunk_count,
    TDST,
    sink_token_size,
):
    for i in numba.prange(out_indices_cpu.shape[1]):
        t_chunk_size = math.ceil(TDST / chunk_count * BLOCK_CHUNK)
        # print(i, t_chunk_size)
        for j in range(max(0, out_indices_cpu.shape[-1])):
            if j >= out_indices_cpu.shape[-1]:
                continue
            t = (
                out_indices_cpu[0, i, DEBUG_HEAD, j] - sink_token_size
            ) // BLOCK_CHUNK + sink_token_size // BLOCK_CHUNK
            t = t // t_chunk_size * t_chunk_size
            debug[i, t : t + t_chunk_size] = 1


@numba.njit(parallel=True)
def render_plot_ks(indices, ks, debug, DEBUG_HEAD, BLOCK_SIZE_Q):
    for i in numba.prange(indices.shape[1]):
        k = ks[DEBUG_HEAD, i]
        for j in range(indices.shape[-1]):
            if j >= k:
                continue
            t = indices[DEBUG_HEAD, i, j] // BLOCK_SIZE_Q
            debug[i, t : t + 1] = 1


DEBUG = os.getenv("HIP_DEBUG", "0") == "1"
DEBUG_LOGALL = os.getenv("HIP_DEBUG_LOGALL", "0") == "1"
__logall_index = 0
DEBUG_RENDER = os.getenv("HIP_DEBUG_RENDER", "1") == "1"


from .utils import capture


@capture
def dual_stage_quadratic_hip_attention(
    q: Tensor,
    k: Optional[Tensor],
    v: Optional[Tensor],
    args: HiPAttentionArgs,
    cached_metadata: Optional[HiPAttentionOutputMetadata] = None,
):
    global __logall_index
    global DEBUG
    DEBUG_HEAD = -1

    # HIP_LANDMARK_BASED_SCAN_STAGE = (
    #     os.getenv("HIP_LANDMARK_BASED_SCAN_STAGE", "1") == "1"
    # )

    require_state = args.using_landmark or any(
        [s.using_landmark if isinstance(s, ScanStage) else False for s in args.stages]
    )

    if require_state and (not args.is_decode):
        # if q.shape[1] > 1: print('using cached state')
        if (cached_metadata is not None) and (cached_metadata.state is not None):
            state = cached_metadata.state
        else:
            state = HiPAttentionState.from_args(q, args, k)
    else:
        state = None

    flatten_paged_cache = False
    if q.shape[1] == 1:
        pass
    # elif HIP_DEBUG_LANDMARK_BASED_SCAN_STAGE:
    #     # FIXME: just for dev
    #     if k is None:
    #         flatten_paged_cache = True
    #         seq_len = args.position_ids.amax().item() + 1
    #         k = args.gather_k_from_paged_cache(
    #             chunk_size=args.stages[0].stage_chunk_size,
    #             disable_gqa=True,
    #             gqa_q=q,
    #         )
    #         k = k[:, :seq_len]
    #         # v = args.gather_v_from_paged_cache(
    #         #     chunk_size=args.stages[0].stage_chunk_size,
    #         #     disable_gqa=True,
    #         #     gqa_q=q,
    #         # )
    #         # v = v[:, :seq_len]

    if args.q_mask is None:
        q_bsa = q
    else:
        q_bsa = q
        q = args.q_mask
    if args.k_mask is None:
        k_mask = k
    else:
        k_mask = args.k_mask

    k_mask_original = k_mask

    BSZ, TDST, HEAD, HID = q.shape
    if k is not None:
        BSZ, TSRC, HEAD_KV, HID = k.shape
        if v is not None:
            assert v.shape[0] == k.shape[0]
            assert v.shape[1] == k.shape[1]
            assert v.shape[2] == k.shape[2]
        MAX_TSRC = TSRC
    else:
        # MAX_TSRC = args.k_cache.shape[0] * args.k_cache.shape[1]
        # MAX_TSRC = int(os.getenv('EXTEND_LEN', '128')) * 1024
        MAX_TSRC = args.extend_context_length
        if args.k_cache is not None:
            HEAD_KV = args.k_cache.shape[-2]
        else:
            HEAD_KV = args.offload_cache.k_uvm.bank_cpu.shape[-2]
        TSRC = MAX_TSRC

    assert len(args.stages) > 0
    STAGE_STRIDE = args.stages[0].stage_stride
    BLOCK_SIZE_Q = args.stages[0].stage_block_size_q
    BDST = triton.cdiv(TDST, BLOCK_SIZE_Q)
    BDST_SCAN = triton.cdiv(BDST, STAGE_STRIDE)
    BLOCK_CHUNK = int(os.getenv("SCAN_BLOCK_CHUNK", "64"))
    chunk_size = args.stages[0].stage_chunk_size
    chunk_count = triton.cdiv(
        max(0, MAX_TSRC - args.sink_token_size - args.sliding_window_size), chunk_size
    )

    args = args.clone()
    args.mask_k = args.stages[0].stage_chunk_size
    original_sliding_window_size = args.sliding_window_size
    # args.sliding_window_size = max(0, args.sliding_window_size - args.mask_k)

    if args.rope_range is None:
        args.rope_range = (0, HID)

    if args.rope_is_neox_style is None:
        # warnings.warn(
        #     "Deprecated: Please specify `rope_is_neox_style`. Defaulting to True."
        # )
        args.rope_is_neox_style = True

    if args.rope_range[0] == 0 and args.rope_range[1] == HID:
        HID_BLOCK = triton.next_power_of_2(HID)
    else:
        assert triton.next_power_of_2(args.rope_range[0]) == args.rope_range[0]
        assert args.rope_range[1] == HID
        HID_BLOCK = args.rope_range[0]

    if torch.cuda.is_current_stream_capturing() or args.position_ids is not None:
        assert args.position_ids is not None
        position_ids = args.position_ids
    else:
        position_ids = (torch.arange(0, TDST, device=q.device) + (TSRC - TDST))[
            None, :
        ].expand(BSZ, TDST)
        args = args.clone()
        args.position_ids = position_ids
    assert position_ids.shape == (BSZ, TDST), position_ids.shape

    if args.using_paged_cache:
        MAX_PAGE = args.paged_cache_page_count
    else:
        MAX_PAGE = MAX_TSRC

    if args.require_cache_statistics:
        mask_access_counter = torch.zeros(
            (BSZ, HEAD_KV, MAX_PAGE), dtype=torch.int32, device=q.device
        )
        mask_cache_miss_counter = torch.zeros(
            (BSZ, HEAD_KV, MAX_PAGE), dtype=torch.int32, device=q.device
        )
        sa_access_counter = torch.zeros(
            (BSZ, HEAD_KV, MAX_PAGE), dtype=torch.int32, device=q.device
        )
        sa_cache_miss_counter = torch.zeros(
            (BSZ, HEAD_KV, MAX_PAGE), dtype=torch.int32, device=q.device
        )
    else:
        sa_cache_miss_counter = sa_access_counter = mask_cache_miss_counter = (
            mask_access_counter
        ) = None

    stage_caches = (
        []
        if (cached_metadata is None) or (cached_metadata.stage_caches is None)
        else cached_metadata.stage_caches
    )
    if not args.require_stage_caches:
        stage_caches = None

    if (cached_metadata is None) or (cached_metadata.indices is None):
        # loop carrying variables: indices_left, indices_right, out_scores
        if (
            (cached_metadata is None)
            or (cached_metadata.stage_caches is None)
            or (stage_caches is None)
        ):
            indices_left = torch.zeros(
                (BSZ, BDST_SCAN, HEAD, chunk_count), device=q.device, dtype=torch.int64
            )

            indices_left[:, :, :, :] = (
                torch.floor(
                    torch.arange(0, chunk_count, device=q.device, dtype=torch.float64)
                    * chunk_size
                    + args.sink_token_size
                ).to(indices_left.dtype)
            )[None, None, None, :]
            indices_right = indices_left + chunk_size
            indices_right.clamp_max_(MAX_TSRC - args.sliding_window_size)

            out_scores = torch.full(
                (BSZ, BDST_SCAN, HEAD, triton.next_power_of_2(chunk_count)),
                device=q.device,
                dtype=torch.float32,
                fill_value=-32000.0,
            )
        else:
            assert cached_metadata is not None
            assert cached_metadata.stage_caches is not None
            assert len(stage_caches) <= len(args.stages)

            last_stage_cache = stage_caches[-1]

            indices_left = last_stage_cache.indices_left.clone()
            indices_right = last_stage_cache.indices_right.clone()
            out_scores = last_stage_cache.out_scores.clone()

        landmark_scores = None

        for i_stage, stage_info in enumerate(args.stages):
            # if stage_chunk_size > chunk_size: continue
            # if stage_k > TSRC: continue

            stage_block_stride_q = stage_info.stage_block_stride_q
            stage_chunk_size = stage_info.stage_chunk_size
            stage_k = stage_info.stage_k

            if i_stage < (len(stage_caches if stage_caches is not None else []) - 1):
                # print('stage cached pass', i_stage)
                continue
            elif i_stage == (len(stage_caches if stage_caches is not None else []) - 1):
                # print('last cached stage', i_stage)
                pass
            elif i_stage > 0:
                (
                    indices_left,
                    indices_right,
                    out_scores,
                    BLOCK_SIZE_Q,
                    BDST,
                    STAGE_STRIDE,
                ) = stage_prologue(
                    q,
                    indices_left,
                    indices_right,
                    out_scores,
                    stage_k,
                    stage_chunk_size,
                    chunk_size,
                    stage_info,
                    args,
                    TDST,
                    BDST,
                    STAGE_STRIDE,
                    BLOCK_SIZE_Q,
                )
            else:
                assert stage_info.stage_k is None, "first stage always quadratic"
                assert isinstance(
                    stage_info, ScanStage
                ), f"frist stage always scan {stage_info}"
                STAGE_STRIDE = stage_info.stage_stride

            if (stage_caches is not None) and (i_stage >= len(stage_caches)):
                if i_stage == 0:
                    # NOTE: do not cache first stage input, because it is meaning less.
                    stage_caches.append(
                        HiPAttentionStageInputCache(
                            indices_left=None,
                            indices_right=None,
                            out_scores=None,
                        )
                    )
                else:
                    stage_caches.append(
                        HiPAttentionStageInputCache(
                            indices_left=indices_left.clone(),
                            indices_right=indices_right.clone(),
                            out_scores=out_scores.clone(),
                        )
                    )

            chunk_size = stage_chunk_size
            chunk_count = indices_left.shape[-1]
            BLOCK_CHUNK = max(16, triton.next_power_of_2(min(chunk_count, BLOCK_CHUNK)))

            pre_device = torch.cuda.current_device()
            torch.cuda.set_device(q.device)

            if isinstance(stage_info, ScanStage):
                extend_backend = (
                    args.scan_extend_backend
                    if stage_info.stage_extend_backend is None
                    else stage_info.stage_extend_backend
                )

                # if args.offload_cache is not None:
                #     print('before masking')
                #     args.offload_cache.mask_k_cache._verify_cache()

                # B T H D
                # if k_mask_original is not None:
                #     B, T, H, D = k.shape
                #     wind_size = args.stages[i_stage + 1].stage_chunk_size // 2 - 1 if (i_stage + 1) < len(args.stages) else 0
                #     if wind_size > 0:
                #         k_max = torch.nn.functional.max_pool1d(k_mask_original.permute(0, 2, 3, 1).reshape(-1, 1, T), kernel_size=wind_size*2 + 1, padding=wind_size, stride=1)
                #         k_min = -torch.nn.functional.max_pool1d((-k_mask_original).permute(0, 2, 3, 1).reshape(-1, 1, T), kernel_size=wind_size*2 + 1, padding=wind_size, stride=1)
                #         k_mask = ((k_min + k_max) / 2).view(B, H, D, T).permute(0, 3, 1, 2).contiguous()
                #         del k_max, k_min
                #     else:
                #         k_mask = k_mask_original

                debug_exclude_landmark = []
                if "HIP_DEBUG_EXCLUDE_LANDMARK" in os.environ:
                    debug_exclude_landmark = list(
                        map(
                            lambda x: int(x),
                            os.environ["HIP_DEBUG_EXCLUDE_LANDMARK"].split(","),
                        )
                    )

                assert q.shape[1] <= BDST * BLOCK_SIZE_Q
                if (
                    (args.using_landmark or stage_info.using_landmark)
                    and (not args.is_decode)
                    and (BDST > 1)
                    and (args.position_ids.shape[0] == 1)
                    and (args.layer_id not in debug_exclude_landmark)
                ):
                    assert not torch.cuda.is_current_stream_capturing()

                    if triton.next_power_of_2(q.shape[-1]) > q.shape[-1]:
                        NOPE_HID = triton.next_power_of_2(q.shape[-1]) // 2
                    else:
                        NOPE_HID = q.shape[-1]

                    # chunked sampling
                    if landmark_scores is None:
                        landmark_scores = landmark_sample(
                            q[..., :NOPE_HID],
                            k[..., :NOPE_HID] if k is not None else k,
                            state,
                            args,
                            BSZ,
                            HEAD,
                            HEAD_KV,
                            BDST,
                            DEBUG,
                            __logall_index,
                        )

                    _TSRC = TSRC
                    if k is not None:
                        _TSRC = k.shape[1]

                    landmarks = landmark_scores.view(
                        BSZ,
                        HEAD,
                        landmark_scores.shape[-1] // stage_info.stage_chunk_size,
                        stage_info.stage_chunk_size,
                    )
                    num_landmarks = args.landmark_stage_k[i_stage]
                    _, landmarks = torch.topk(landmarks, k=num_landmarks, sorted=False)
                    landmarks = landmarks.permute(0, 2, 1, 3)[
                        :, : _TSRC // stage_info.stage_chunk_size
                    ].contiguous()
                    assert landmarks.shape == (
                        BSZ,
                        _TSRC // stage_info.stage_chunk_size,
                        HEAD,
                        num_landmarks,
                    ), f"{landmarks.shape} == ({BSZ}, {_TSRC // stage_info.stage_chunk_size}, {HEAD}, {num_landmarks}), {k.shape if k is not None else None}"

                    assert indices_left.shape == (
                        BSZ,
                        BDST,
                        HEAD,
                        indices_left.shape[-1],
                    ), f"{indices_left.shape} == ({BSZ},{BDST},{HEAD},{indices_left.shape[-1]},)"

                    # k_temp = args.gather_k_from_paged_cache(
                    #     chunk_size=1,
                    #     disable_gqa=False,
                    #     gqa_q=q,
                    # )
                    scores = compute_scores_landmark(
                        q=q[..., :NOPE_HID],
                        # k=k_temp,
                        # k_cache=None,
                        k=k[..., :NOPE_HID] if k is not None else k,
                        k_cache=(
                            args.get_k_cache()[..., :NOPE_HID]
                            if args.get_k_cache() is not None
                            else None
                        ),
                        block_table=args.block_table,
                        position_ids=args.position_ids,
                        indices_left=indices_left,
                        landmarks=landmarks,
                        cos=args.rope_cos,
                        sin=args.rope_sin,
                        BLOCK_SIZE_Q=stage_info.stage_block_size_q,
                        BLOCK_STRIDE_Q=stage_info.stage_block_stride_q,
                        CHUNK_SIZE=stage_info.stage_chunk_size,
                        SLIDING_WINDOW_SIZE=args.sliding_window_size,
                    )
                    assert (
                        args.sink_token_size % stage_info.stage_chunk_size
                    ) == 0, f"{args.sink_token_size} % {stage_info.stage_chunk_size}"
                    # scores = scores[:, :, :, args.sink_token_size // stage_info.stage_chunk_size:]

                    out_scores[:, :, :, : scores.shape[-1]] = scores
                    out_scores[:, :, :, scores.shape[-1] :].fill_(float("-inf"))
                    # indices_left = (indices_left + indices_right) // 2
                    # indices_right = indices_left.clone()

                    # print('landmark based sampling', args.layer_id)
                elif (
                    os.getenv("HIP_DEBUG_TOPKMEAN", "0") == "1"
                    and (i_stage == 0)
                    and (BDST > 1)
                    and ((q.shape[1] % BLOCK_SIZE_Q) == 0)
                    and (args.position_ids.shape[0] == 1)
                ):
                    debug_topk_window = int(os.getenv("HIP_DEBUG_TOPK_WINDOW", "8"))
                    k_dense = args.gather_k_from_paged_cache(
                        chunk_size=chunk_size, disable_gqa=True, gqa_q=q
                    )
                    scores = torch.matmul(
                        q.permute(0, 2, 1, 3), k_dense.permute(0, 2, 3, 1)
                    )[:, args.sink_token_size : -args.sliding_window_size, :, :]
                    mask = (
                        args.position_ids[0][:, None]
                        >= (
                            args.sink_token_size
                            + torch.arange(
                                0, k_dense.shape[1], dtype=q.dtype, device=q.device
                            )
                        )[None, :]
                    )[None, None, :, :]
                    scores = torch.where(mask, scores, -32000.0)
                    scores = scores.view(
                        scores.shape[0],
                        scores.shape[1],
                        scores.shape[2] // BLOCK_SIZE_Q,
                        BLOCK_SIZE_Q,
                        scores.shape[3] // chunk_size,
                        chunk_size,
                    )
                    scores = torch.amax(scores, dim=3)
                    topk_scores, _ = torch.topk(scores, dim=-1, k=debug_topk_window)
                    scores = topk_scores.mean(dim=-1)
                    scores = scores.permute(0, 2, 1, 3)
                    out_scores[:, :, :, : scores.shape[-1]] = scores
                elif (
                    os.getenv("HIP_DEBUG_SOFTMAXMEAN", "0") == "1"
                    and (i_stage == 0)
                    and (BDST > 1)
                    and ((q.shape[1] % BLOCK_SIZE_Q) == 0)
                    and (args.position_ids.shape[0] == 1)
                ):

                    def rotate_half(vec):
                        # assert len(vec.shape) == 1
                        out = torch.zeros_like(vec)
                        x1 = vec[..., : vec.shape[-1] // 2]
                        x2 = vec[..., vec.shape[-1] // 2 :]
                        out[..., : vec.shape[-1] // 2] = -x2
                        out[..., vec.shape[-1] // 2 :] = x1
                        return out

                    def apply_rope(vec, cos, sin):
                        vec_rope = (vec * cos) + (rotate_half(vec) * sin)
                        return vec_rope

                    k_dense = args.gather_k_from_paged_cache(
                        chunk_size=chunk_size, disable_gqa=True, gqa_q=q
                    )[:, args.sink_token_size : -args.sliding_window_size, :, :]
                    k_dense = apply_rope(
                        k_dense,
                        args.rope_cos[None, 0 : 0 + 1, None, :],
                        args.rope_sin[None, 0 : 0 + 1, None, :],
                    )
                    q_dense = apply_rope(
                        q,
                        args.rope_cos[None, 1024 : 1024 + 1, None, :],
                        args.rope_sin[None, 1024 : 1024 + 1, None, :],
                    )
                    scores = torch.matmul(
                        q_dense.permute(0, 2, 1, 3), k_dense.permute(0, 2, 3, 1)
                    )
                    mask = (
                        args.position_ids[0][:, None]
                        >= (
                            args.sink_token_size
                            + torch.arange(
                                0,
                                k_dense.shape[1],
                                dtype=q_dense.dtype,
                                device=q_dense.device,
                            )
                        )[None, :]
                    )[None, None, :, :]
                    scores = torch.where(mask, scores, -32000.0).float()
                    scores = scores.softmax(dim=-1)
                    scores = torch.where(mask, scores, -32000.0)
                    scores = scores.view(
                        scores.shape[0],
                        scores.shape[1],
                        scores.shape[2] // BLOCK_SIZE_Q,
                        BLOCK_SIZE_Q,
                        scores.shape[3] // chunk_size,
                        chunk_size,
                    )
                    scores = (
                        scores.permute(0, 1, 2, 4, 3, 5)
                        .contiguous()
                        .view(
                            scores.shape[0],
                            scores.shape[1],
                            scores.shape[2],
                            scores.shape[4],
                            -1,
                        )
                    )
                    mask = scores > -30000.0
                    scores = (scores * mask).sum(dim=-1) / (
                        mask.float().sum(dim=-1) + 1e-12
                    )
                    scores.masked_fill_(mask.float().sum(dim=-1) == 0, -32000.0)
                    scores = scores.permute(0, 2, 1, 3)
                    # print(scores[0,:,0,:])
                    out_scores[:, :, :, : scores.shape[-1]] = scores
                elif (
                    os.getenv("HIP_DEBUG_FLATTENMEAN", "0") == "1"
                    and (i_stage == 0)
                    and (BDST > 1)
                    and ((q.shape[1] % BLOCK_SIZE_Q) == 0)
                    and (args.position_ids.shape[0] == 1)
                ):

                    def rotate_half(vec):
                        # assert len(vec.shape) == 1
                        out = torch.zeros_like(vec)
                        x1 = vec[..., : vec.shape[-1] // 2]
                        x2 = vec[..., vec.shape[-1] // 2 :]
                        out[..., : vec.shape[-1] // 2] = -x2
                        out[..., vec.shape[-1] // 2 :] = x1
                        return out

                    def apply_rope(vec, cos, sin):
                        vec_rope = (vec * cos) + (rotate_half(vec) * sin)
                        return vec_rope

                    k_dense = args.gather_k_from_paged_cache(
                        chunk_size=chunk_size, disable_gqa=True, gqa_q=q
                    )[:, args.sink_token_size : -args.sliding_window_size, :, :]
                    k_dense = apply_rope(
                        k_dense,
                        args.rope_cos[None, 0 : 0 + 1, None, :],
                        args.rope_sin[None, 0 : 0 + 1, None, :],
                    )
                    q_dense = apply_rope(
                        q,
                        args.rope_cos[None, 1024 : 1024 + 1, None, :],
                        args.rope_sin[None, 1024 : 1024 + 1, None, :],
                    )
                    scores = torch.matmul(
                        q_dense.permute(0, 2, 1, 3), k_dense.permute(0, 2, 3, 1)
                    )
                    mask = (
                        args.position_ids[0][:, None]
                        >= (
                            args.sink_token_size
                            + torch.arange(
                                0,
                                k_dense.shape[1],
                                dtype=q_dense.dtype,
                                device=q_dense.device,
                            )
                        )[None, :]
                    )[None, None, :, :]
                    scores = torch.where(mask, scores, -32000.0)
                    scores = scores.view(
                        scores.shape[0],
                        scores.shape[1],
                        scores.shape[2] // BLOCK_SIZE_Q,
                        BLOCK_SIZE_Q,
                        scores.shape[3] // chunk_size,
                        chunk_size,
                    )
                    scores = (
                        scores.permute(0, 1, 2, 4, 3, 5)
                        .contiguous()
                        .view(
                            scores.shape[0],
                            scores.shape[1],
                            scores.shape[2],
                            scores.shape[4],
                            -1,
                        )
                    )
                    mask = scores > -30000.0
                    scores = (scores * mask).sum(dim=-1) / (
                        mask.float().sum(dim=-1) + 1e-12
                    )
                    scores.masked_fill_(mask.float().sum(dim=-1) == 0, -32000.0)
                    scores = scores.permute(0, 2, 1, 3)
                    # print(scores[0,:,0,:])
                    out_scores[:, :, :, : scores.shape[-1]] = scores
                elif (
                    os.getenv("HIP_DEBUG_FLATTENTOPKMEAN", "0") == "1"
                    and (i_stage == 0)
                    and (BDST > 1)
                    and ((q.shape[1] % BLOCK_SIZE_Q) == 0)
                    and (args.position_ids.shape[0] == 1)
                ):

                    def rotate_half(vec):
                        # assert len(vec.shape) == 1
                        out = torch.zeros_like(vec)
                        x1 = vec[..., : vec.shape[-1] // 2]
                        x2 = vec[..., vec.shape[-1] // 2 :]
                        out[..., : vec.shape[-1] // 2] = -x2
                        out[..., vec.shape[-1] // 2 :] = x1
                        return out

                    def apply_rope(vec, cos, sin):
                        vec_rope = (vec * cos) + (rotate_half(vec) * sin)
                        return vec_rope

                    debug_topk_window = int(os.getenv("HIP_DEBUG_TOPK_WINDOW", "8"))
                    k_dense = args.gather_k_from_paged_cache(
                        chunk_size=chunk_size, disable_gqa=True, gqa_q=q
                    )[:, args.sink_token_size : -args.sliding_window_size, :, :]
                    k_dense = apply_rope(
                        k_dense,
                        args.rope_cos[None, 0 : 0 + 1, None, :],
                        args.rope_sin[None, 0 : 0 + 1, None, :],
                    )
                    q_dense = apply_rope(
                        q,
                        args.rope_cos[None, 1024 : 1024 + 1, None, :],
                        args.rope_sin[None, 1024 : 1024 + 1, None, :],
                    )
                    scores = torch.matmul(
                        q_dense.permute(0, 2, 1, 3), k_dense.permute(0, 2, 3, 1)
                    )
                    mask = (
                        args.position_ids[0][:, None]
                        >= (
                            args.sink_token_size
                            + torch.arange(
                                0,
                                k_dense.shape[1],
                                dtype=q_dense.dtype,
                                device=q_dense.device,
                            )
                        )[None, :]
                    )[None, None, :, :]
                    scores = torch.where(mask, scores, -32000.0)
                    scores = scores.view(
                        scores.shape[0],
                        scores.shape[1],
                        scores.shape[2] // BLOCK_SIZE_Q,
                        BLOCK_SIZE_Q,
                        scores.shape[3] // chunk_size,
                        chunk_size,
                    )
                    scores = (
                        scores.permute(0, 1, 2, 4, 3, 5)
                        .contiguous()
                        .view(
                            scores.shape[0],
                            scores.shape[1],
                            scores.shape[2],
                            scores.shape[4],
                            -1,
                        )
                    )
                    topk_scores, _ = torch.topk(scores, dim=-1, k=debug_topk_window)
                    topk_scores_mask = topk_scores > -30000.0
                    scores = (topk_scores * topk_scores_mask).sum(dim=-1) / (
                        topk_scores_mask.float().sum(dim=-1) + 1e-12
                    )
                    scores = torch.where(
                        topk_scores_mask.int().sum(dim=-1) != 0, scores, -32000.0
                    )
                    scores = scores.permute(0, 2, 1, 3)
                    out_scores[:, :, :, : scores.shape[-1]] = scores
                else:
                    chunk_controllable_sampling_mask(
                        args,
                        chunk_count,
                        BLOCK_CHUNK,
                        TDST,
                        BLOCK_SIZE_Q,
                        STAGE_STRIDE,
                        HEAD,
                        BSZ,
                        q,
                        k_mask,
                        position_ids,
                        indices_left,
                        indices_right,
                        out_scores,
                        mask_access_counter,
                        mask_cache_miss_counter,
                        MAX_TSRC,
                        HID,
                        HID_BLOCK,
                        stage_block_stride_q,
                        HEAD_KV,
                        extend_backend,
                    )

                # TODO: OPTIMIZE THIS. Add head unified version of HiP.
                HEAD_REDUCE_MODE = os.getenv(
                    "HIP_HEAD_REDUCE", DEFAULT_VALUE_HIP_HEAD_REDUCE
                )
                if (
                    # always reduce the head.
                    (HEAD_REDUCE_MODE == "1")
                    or
                    # reduce only when decode. this is for handling flash-decode kernel.
                    (HEAD_REDUCE_MODE == "2" and BDST == 1)
                    or
                    # reduce only within tp. this will be incorrect in tp size
                    (HEAD_REDUCE_MODE == "3")
                ):
                    ori_shape = out_scores.shape
                    # out_scores = out_scores.softmax(dim=2) # NOTE: not good idea
                    # out_scores, _ = torch.max(out_scores, keepdim=True, dim=2)

                    if (
                        SGLANG_DIST_ACTIVATED
                        and get_world_size() > 1
                        and HEAD_REDUCE_MODE in ["1", "2"]
                    ):
                        warnings.warn("TP all gather is used for head reduce, this may degrade throughput.")

                        out_scores_tp = out_scores
                        out_scores = (
                            tensor_model_parallel_all_gather(
                                out_scores_tp.permute(0, 1, 3, 2).contiguous()
                            )
                            .permute(0, 1, 3, 2)
                            .contiguous()
                        )

                    out_scores = torch.amax(out_scores, keepdim=True, dim=2)
                    out_scores = torch.broadcast_to(out_scores, ori_shape).contiguous()
                else:
                    args.disable_flashdecode = True

                if args.offload_cache is not None:
                    # print('after masking')
                    args.offload_cache.mask_k_cache.verify_cache()
            elif isinstance(stage_info, EvalScoreStage):
                raise Exception()  # TODO: handle new args
                extend_backend = (
                    args.scan_extend_backend
                    if stage_info.stage_extend_backend is None
                    else stage_info.stage_extend_backend
                )

                grid = (
                    BSZ
                    * triton.cdiv(BDST, stage_info.stage_stride)
                    * HEAD,  # SCAN_STRIDE = 1
                )
                calculate_chunk_score[grid](
                    q_mask,
                    *q_mask.stride(),
                    k_mask,
                    *safe_stride(k_mask, 4),
                    position_ids,
                    *position_ids.stride(),
                    args.rope_cos,
                    *safe_stride(args.rope_cos, 2),
                    args.rope_sin,
                    *safe_stride(args.rope_sin, 2),
                    *args.args_paged_kv_cache(),
                    *args.args_offload_cache(True),
                    indices_left,
                    *indices_left.stride(),
                    indices_right,
                    *indices_right.stride(),
                    out_scores,
                    *out_scores.stride(),
                    # model_context_length if (not scan_extend_backend == 'streaming') else 0,
                    args.model_context_length,
                    args.sliding_window_size,
                    args.sink_token_size,
                    chunk_size,
                    TDST,
                    BDST,
                    triton.cdiv(BDST, stage_info.stage_stride),  # SCAN STRIDE == 1
                    HEAD,
                    chunk_count,
                    HEAD // HEAD_KV,
                    USING_EXTEND=args.using_extend,
                    NEED_APPLY_ROPE=args.need_apply_rope,
                    EXTEND_BACKEND=extend_backend,
                    BLOCK_HID=BLOCK_HID,
                    BLOCK_SIZE_Q=BLOCK_SIZE_Q,
                    BLOCK_STRIDE_Q=stage_block_stride_q,
                    BLOCK_SIZE_K=args.stage_early_terminate,
                    BLOCK_STRIDE_K=args.block_stride_k,
                    SCAN_STRIDE=stage_info.stage_stride,
                    BLOCK_CHUNK=stage_info.block_chunk,
                )
            elif isinstance(stage_info, EnsembleScoreStage):
                raise Exception()
            elif isinstance(stage_info, NopStage):
                pass
            else:
                raise Exception()

            torch.cuda.set_device(pre_device)

            if stage_info.require_post_sort:
                apply_v_dot = os.getenv("APPLY_V_DOT", "0") == "1"
                # apply_v_dot = apply_v_dot and (i_stage == (len(stages) - 1))
                apply_v_dot = apply_v_dot and (i_stage != 0)
                if apply_v_dot:
                    v_scores = torch.zeros_like(out_scores, dtype=torch.float32)
                    V_BLOCK_SIZE_K = 8
                    V_BLOCK_STRIDE_Q = 1
                    V_BLOCK_STRIDE_K = 1
                    V_GROUP_K = 64 // V_BLOCK_SIZE_K
                    # V_GROUP_K = indices_left.shape[3]
                    grid = (
                        v_scores.shape[0]
                        * v_scores.shape[1]
                        * v_scores.shape[2]
                        * triton.cdiv(indices_left.shape[3], V_GROUP_K),
                    )
                    compute_v_cos[grid](
                        v,
                        *safe_stride(v, 4),
                        indices_left,
                        *indices_left.stride(),
                        position_ids,
                        *position_ids.stride(),
                        v_scores,
                        *v_scores.stride(),
                        *args.args_paged_kv_cache(),
                        *args.args_offload_cache(is_masking=True),
                        sa_access_counter,
                        *safe_stride(sa_access_counter, 3),
                        sa_cache_miss_counter,
                        *safe_stride(sa_cache_miss_counter, 3),
                        TDST,
                        MAX_TSRC,
                        HEAD,
                        indices_left.shape[3],
                        HEAD_GROUP=HEAD // HEAD_KV,
                        GROUP_K=V_GROUP_K,
                        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
                        BLOCK_SIZE_K=V_BLOCK_SIZE_K,
                        BLOCK_STRIDE_Q=V_BLOCK_STRIDE_Q,
                        BLOCK_STRIDE_K=V_BLOCK_STRIDE_K,
                        BLOCK_HID=q.shape[-1],
                    )

                    if out_scores.dtype != torch.float32:
                        out_scores = out_scores.to(torch.float32)
                    out_scores = (
                        out_scores - out_scores.min(dim=-1, keepdim=True).values
                    )

                    # print(indices_left[0, -1, DEBUG_HEAD, :])
                    # print(out_scores[0, -1, DEBUG_HEAD, :])
                    # print(v_scores[0, -1, DEBUG_HEAD, :])

                    if DEBUG and DEBUG_RENDER:
                        img = v_scores[0, :, DEBUG_HEAD, :].cpu().float().numpy()
                        plt.clf()
                        plt.imshow(img)
                        plt.colorbar()
                        plt.savefig("dummy_v_scores.png")

                    # out_scores = torch.where(
                    #     torch.isnan(v_scores),
                    #     out_scores,
                    #     out_scores * v_scores
                    # )

                    # out_scores = out_scores * v_scores

                    out_scores = out_scores + v_scores

                if i_stage < (len(args.stages) - 1):
                    # print(indices_left.shape, (stages[i_stage + 1].stage_k // stages[i_stage + 1].stage_chunk_size))
                    next_stage_k = (
                        args.stages[i_stage + 1].stage_k
                        // args.stages[i_stage].stage_chunk_size
                    )
                else:
                    next_stage_k = (
                        args.second_stage_k
                        // args.stages[i_stage].stage_chunk_size
                    )
                next_stage_k = min(next_stage_k, indices_left.shape[-1])
                _, t_indices = out_scores[..., : indices_left.shape[-1]].topk(
                    k=next_stage_k,
                    dim=-1,
                    sorted=False,
                    largest=True,
                )
                # else:
                #     _, t_indices = out_scores[..., : indices_left.shape[-1]].sort(
                #         dim=-1, descending=True, stable=False
                #     )
                indices_left = indices_left.gather(dim=-1, index=t_indices)
                indices_right = indices_right.gather(dim=-1, index=t_indices)

            if (
                DEBUG
                and DEBUG_RENDER
                and not torch.cuda.is_current_stream_capturing()
                and get_local_rank() == 0
            ):
                if (i_stage + 1) < len(args.stages):
                    next_stage_k = args.stages[i_stage + 1].stage_k
                else:
                    next_stage_k = args.second_stage_k
                out_indices_cpu = (
                    indices_left.repeat_interleave(STAGE_STRIDE, 1)[:, -BDST:]
                    .contiguous()
                    .cpu()
                    .numpy()
                )
                debug = np.zeros(
                    (triton.cdiv(TDST, BLOCK_SIZE_Q), triton.cdiv(TSRC, BLOCK_SIZE_Q))
                )
                render_plot_dynamic(
                    out_indices_cpu,
                    debug,
                    DEBUG_HEAD,
                    BLOCK_SIZE_Q,
                    next_stage_k,
                    chunk_size,
                    causal_mask=True,
                    sliding_window_size=args.sliding_window_size,
                )
                if DEBUG_LOGALL and (BDST > 1):
                    __logall_index += 1
                    os.makedirs("./cache/mask_log", exist_ok=True)
                    # cv2.imwrite(
                    #     f"./cache/mask_log/{__logall_index:04d}_dummy_sampled_stage_{i_stage}.png",
                    #     debug * 255,
                    # )
                else:
                    cv2.imwrite(f"dummy_sampled_stage_{i_stage}.png", debug * 255)
                # print(f'saved dummy_sampled_stage_{i_stage}.png')

        if STAGE_STRIDE > 1:
            indices_left = indices_left.repeat_interleave(STAGE_STRIDE, 1)[
                :, -BDST:
            ].contiguous()
            indices_right = indices_right.repeat_interleave(STAGE_STRIDE, 1)[
                :, -BDST:
            ].contiguous()
            out_scores = out_scores.repeat_interleave(STAGE_STRIDE, 1)[
                :, -BDST:
            ].contiguous()

        assert (args.second_stage_k % chunk_size) == 0
        # if DEBUG:
        #     print('indices_left', indices_left[0, -1])
        #     print('out_scores', out_scores[0, -1], args.second_stage_k, indices_left.shape, chunk_size)
        indices = (
            indices_left[..., : args.second_stage_k // chunk_size]
            // chunk_size
            * chunk_size
        )

        # NOTE: union head masks
        if os.getenv("HIP_DEBUG_UNION_HEAD", "0") == "1":
            assert os.getenv("HIP_HEAD_REDUCE", DEFAULT_VALUE_HIP_HEAD_REDUCE) == "0"
            # args.disable_flashdecode = True
            # B BDST H CHUNK
            indices = indices.flatten(-2, -1).unsqueeze(-2).repeat(1, 1, HEAD, 1)

        # NOTE: sampled indices might be delayed
        if os.getenv("HIP_DEBUG_ADD_DELAY_WINDOW", "0") == "1":
            delayed_indices = [
                indices,
            ]
            delay_window = 64
            for i_delay in range(0, delay_window, chunk_size):
                delayed_indices.append(indices - i_delay - chunk_size)
            # print(indices.shape)
            indices = torch.cat(delayed_indices, dim=-1)
            # print(indices.shape)

        # NOTE: performing SnapKV
        if (os.getenv("HIP_DEBUG_SNAP_KV", "0") == "1") and (BDST > 1):
            is_paged = False
            if k_mask_original is None:
                is_paged = True
                k_mask = args.gather_k_from_paged_cache(chunk_size=chunk_size)
            else:
                k_mask = k_mask_original
            scores = torch.matmul(
                q.permute(0, 2, 1, 3)[:, :, -128:, :],
                k_mask.permute(0, 2, 3, 1).repeat(1, HEAD // HEAD_KV, 1, 1),
            )
            # if is_paged:
            tsrcs = torch.arange(0, scores.shape[-1], device=q.device)
            tsrc_mask = tsrcs[None, :] > args.position_ids[:, -1, None]
            scores = scores.masked_fill_(tsrc_mask[:, None, None, :], float("-inf"))
            scores = scores.amax(dim=-2)  # B H TSRC
            snap_window = 127
            scores = torch.nn.functional.max_pool1d(
                scores, kernel_size=snap_window * 2 + 1, stride=1, padding=snap_window
            )
            scores = scores.view(scores.shape[0], scores.shape[1], -1, chunk_size)
            scores = scores.amax(dim=-1)
            # print(scores.shape)
            _, snap_indices = scores.topk(
                k=min(scores.shape[-1], 131072 // chunk_size), dim=-1
            )
            snap_indices = snap_indices * chunk_size
            snap_indices = snap_indices.unsqueeze(1).expand(
                snap_indices.shape[0],
                indices.shape[1],
                snap_indices.shape[1],
                snap_indices.shape[2],
            )
            indices = torch.concat([indices, snap_indices], dim=-1)
            if is_paged:
                k_mask = None

        # NOTE: add sliding window indices
        if args.sliding_window_indices is not None:
            sw_indices = (args.sliding_window_indices // chunk_size) * chunk_size
            assert position_ids.shape == (BSZ, TDST), position_ids.shape
            assert sw_indices.shape[0] == HEAD, sw_indices.shape
            args.disable_flashdecode = True
            warnings.warn("Flash Decode is disabled due to experimental feature")
            sw_indices = (
                position_ids[:, ::BLOCK_SIZE_Q, None, None]
                + sw_indices[None, None, :, :]
            )
            sw_indices = (sw_indices // chunk_size) * chunk_size
            sw_indices.clamp_min_(0)
            indices = torch.concat([indices, sw_indices], dim=-1)

        # NOTE: adding important Ks
        if (os.getenv("HIP_DEBUG_IMPORTANT_K", "0") == "1") and (BDST > 1):
            k_seq = args.gather_k_from_paged_cache(chunk_size=chunk_size)
            k_bos = k_seq[:, :1, :, :].contiguous().permute(0, 2, 1, 3)
            k_seq = k_seq.permute(0, 2, 1, 3)
            k_seq = k_seq / k_seq.square().sum(dim=-1, keepdim=True).sqrt()
            k_bos = k_bos / k_bos.square().sum(dim=-1, keepdim=True).sqrt()
            scores = torch.matmul(k_bos, k_seq.permute(0, 1, 3, 2)).squeeze(2)  # B H T
            tsrcs = torch.arange(0, scores.shape[-1], device=q.device)
            tsrc_mask = (
                tsrcs[None, :] + original_sliding_window_size
            ) > args.position_ids[:, -1, None]
            scores.masked_fill_(tsrc_mask[:, None, :], float("-inf"))
            scores[:, :, : args.sink_token_size].fill_(float("-inf"))
            scores = scores.view(scores.shape[0], scores.shape[1], -1, chunk_size)
            # scores = scores.amax(dim=1, keepdim=True)
            scores = scores.amax(dim=-1)
            _, important_indices = torch.topk(scores, k=8192 // chunk_size, dim=-1)
            important_indices = (
                important_indices.repeat_interleave(
                    HEAD // important_indices.shape[1], 1
                )
                * chunk_size
            )
            important_indices = important_indices.unsqueeze(1).expand(
                important_indices.shape[0],
                indices.shape[1],
                important_indices.shape[1],
                important_indices.shape[2],
            )
            indices = torch.concat([indices, important_indices], dim=-1)

        if (
            DEBUG
            and DEBUG_RENDER
            and not torch.cuda.is_current_stream_capturing()
            and (BDST > 10)
            and get_local_rank() == 0
        ):
            out_indices_cpu = indices.cpu().numpy()
            debug = np.zeros(
                (triton.cdiv(TDST, BLOCK_SIZE_Q), triton.cdiv(TSRC, BLOCK_SIZE_Q))
            )
            render_plot(out_indices_cpu, debug, DEBUG_HEAD, BLOCK_SIZE_Q)
            debug = debug * 255
            debug = debug.astype(np.uint8)
            debug = np.repeat(debug[:, :, None], 3, axis=2)
            cv2.putText(
                debug,
                f"Layer: {args.layer_id}",
                (320, 256),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 255, 0),
                2,
            )
            tdst_start = position_ids[0, 0].item() // BLOCK_SIZE_Q
            debug = cv2.line(
                debug,
                (
                    tdst_start,
                    0,
                ),
                (tdst_start + debug.shape[0], debug.shape[0]),
                thickness=1,
                color=(0, 255, 0),
            )

            if DEBUG_LOGALL and (BDST > 1):
                os.makedirs("./cache/mask_log", exist_ok=True)
                __logall_index += 1
                cv2.imwrite(
                    f"./cache/mask_log/{__logall_index:04d}_dummy_sampled_final.png",
                    debug,
                )
            else:
                cv2.imwrite("dummy_sampled_final.png", debug)
            # print('saved dummy_sampled_final.png')

        args = args.clone()
        args.block_size_q = args.stages[-1].stage_block_size_q
        block_sparse_block_size_q = min(
            args.block_sparse_block_size_q, args.block_size_q
        )
        args.sliding_window_size += args.mask_k
        args.block_size_k = chunk_size
        args.mask_k = args.second_stage_k
        args.using_extend = args.using_extend and True

        # NOTE: convert format and taking unique in indices
        indices = indices.permute(0, 2, 1, 3).flatten(0, 1)

        require_post_unique = BDST > 1
        if require_post_unique:
            indices, t_sort_1 = indices.sort(dim=-1)
            indices = indices // args.block_size_k * args.block_size_k

            unique_mask = torch.roll(indices, shifts=1, dims=-1) != indices
            indices = torch.where(unique_mask, indices, torch.iinfo(indices.dtype).max)
            indices, t_sort_2 = indices.sort(dim=-1)

        active_mask = indices < (
            position_ids[:, :: args.block_size_q, None].repeat_interleave(HEAD, 0)
            + args.block_size_q
        )
        ks = active_mask.int().sum(-1)
        ks_count = ks.unsqueeze(-1)
        ks_start_end = torch.zeros(
            (ks.shape[0], ks.shape[1], 2), dtype=torch.int32, device=q.device
        )
        ks_start_end[:, :, -1] = ks

        # print(args.layer_id, round(ks.float().mean().item() * args.block_size_k))

        if (args.low_percent > 0) and (args.low_k_ratio < 1):
            scores = (
                out_scores[..., : args.second_stage_k // chunk_size]
                .permute(0, 2, 1, 3)
                .flatten(0, 1)
            )
            scores = scores.gather(dim=-1, index=t_sort_1)
            scores = scores.gather(dim=-1, index=t_sort_2)
            scores = torch.where(active_mask, scores, -32000.0)

            masked_scores = torch.where(scores > -16000.0, scores, 0)
            # masked_scores = torch.softmax(scores, dim=-1)
            scores_std, scores_mean = torch.std_mean(masked_scores, dim=-1)

            # TODO: TEST SENSITIVITY

            if dim_to_lower == "head":
                dim_to_lower = 0
                values_to_sort = (scores_std).mean(dim=1)
            elif dim_to_lower == "seq":
                dim_to_lower = 1
                values_to_sort = scores_std
            else:
                raise Exception()

            _, lowk = values_to_sort.topk(
                k=int(scores_mean.shape[dim_to_lower] * args.low_percent),
                dim=dim_to_lower,
                largest=False,
                sorted=False,
            )
            # print(lowk[:, -1])
            if lowk.ndim == 2:
                lowk = lowk[:, :, None].expand(-1, -1, scores.shape[-1])
            if lowk.ndim == 1:
                lowk = lowk[:, None, None].expand(
                    -1, scores.shape[-2], scores.shape[-1]
                )
            _, t_sort_score = torch.topk(
                scores.gather(dim=dim_to_lower, index=lowk),
                dim=-1,
                k=int(scores.shape[-1] * (1 - args.low_k_ratio)),
                largest=False,
            )
            # print(t_sort_score.shape)
            N, BDST = scores_mean.shape
            indices.scatter_(
                dim=dim_to_lower,
                index=lowk,
                src=indices.gather(dim=dim_to_lower, index=lowk).scatter(
                    dim=-1, index=t_sort_score, value=987654321
                ),
            )
            indices, t_sort_2 = indices.sort(dim=-1)
            active_mask = indices < (
                position_ids[:, :: args.block_size_q, None].repeat_interleave(HEAD, 0)
                + args.block_size_q
            )
            # print(indices[1, -1, :])
            # print(active_mask[1, -1, :])
            ks = active_mask.int().sum(-1)
            ks_count = ks.unsqueeze(-1)
            ks_start_end = torch.zeros(
                (ks.shape[0], ks.shape[1], 2), dtype=torch.int32, device=q.device
            )
            ks_start_end[:, :, -1] = ks

            if (
                DEBUG
                and DEBUG_RENDER
                and not torch.cuda.is_current_stream_capturing()
                and (BDST > 10)
            ):
                indices_cpu = indices.cpu().numpy()
                ks_cpu = ks.cpu().numpy()
                debug = np.zeros(
                    (triton.cdiv(TDST, BLOCK_SIZE_Q), triton.cdiv(TSRC, BLOCK_SIZE_Q))
                )
                render_plot_ks(indices_cpu, ks_cpu, debug, DEBUG_HEAD, BLOCK_SIZE_Q)
                cv2.imwrite("dummy_sampled_final_lowk.png", debug * 255)
                print("saved dummy_sampled_final_lowk.png", DEBUG_HEAD)

                # print(ks[:, -1])

                plt.clf()
                plt.plot(scores_std[:3, :].float().cpu().numpy().T)
                # plt.ylim(0, 0.01)
                plt.savefig("dummy_stat_std.png")
                plt.clf()
                plt.plot(scores_mean[:3, :].float().cpu().numpy().T)
                plt.savefig("dummy_stat_mean.png")
                plt.clf()
                plt.plot(ks[DEBUG_HEAD, :].float().cpu().numpy())
                plt.savefig("dummy_stat_ks.png")

        if (
            DEBUG
            and DEBUG_RENDER
            and not torch.cuda.is_current_stream_capturing()
            and (BDST > 10)
            and get_local_rank() == 0
        ):
            try:
                input(f"[{args.layer_id}] >")
            except EOFError:
                print()

        # NOTE: break-down to fit BSA block size
        if (block_sparse_block_size_q is not None) and (
            triton.cdiv(TDST, block_sparse_block_size_q)
            != triton.cdiv(TDST, args.block_size_q)
        ):
            assert (BLOCK_SIZE_Q % block_sparse_block_size_q) == 0
            indices = indices.repeat_interleave(
                BLOCK_SIZE_Q // block_sparse_block_size_q, 1
            )
            ks = ks.repeat_interleave(BLOCK_SIZE_Q // block_sparse_block_size_q, 1)
            ks_count = ks_count.repeat_interleave(
                BLOCK_SIZE_Q // block_sparse_block_size_q, 1
            )
            ks_start_end = ks_start_end.repeat_interleave(
                BLOCK_SIZE_Q // block_sparse_block_size_q, 1
            )
            args.block_size_q = block_sparse_block_size_q

        if args.mask_only:
            return None, None
    else:
        args = args.clone()
        args.sliding_window_size += args.mask_k
        args.block_size_k = args.stages[-1].stage_chunk_size
        args.mask_k = args.second_stage_k
        args.using_extend = args.using_extend and True

        assert cached_metadata is not None
        require_cache_clone = False
        if require_cache_clone:
            indices = cached_metadata.indices.clone()
            ks = cached_metadata.ks.clone()
            ks_count = cached_metadata.ks_count.clone()
            ks_start_end = cached_metadata.ks_start_end.clone()
        else:
            indices = cached_metadata.indices
            ks = cached_metadata.ks
            ks_count = cached_metadata.ks_count
            ks_start_end = cached_metadata.ks_start_end

    args.block_size_q = min(args.block_size_q, triton.next_power_of_2(TDST))

    if args.sliding_window_size == 777:
        args.sliding_window_size = (
            args.model_context_length
            - args.sink_token_size
            - args.second_stage_k
            - args.block_size_q
        )
    elif args.sliding_window_size > 0:
        args.sliding_window_size += args.block_size_q

    if flatten_paged_cache:
        k = None
        v = None

    block_sparse_attention_backend = get_block_sparse_backend(args, q_bsa)
    # from hip_attn.v1_2.attention_extend_bsa_tilelang import block_sparse_attention as tilelang_bsa
    # block_sparse_attention_backend = tilelang_bsa

    if args.bsa_sliding_window_size > 0:
        args = args.clone()
        args.sliding_window_size = args.bsa_sliding_window_size

    context = block_sparse_attention_backend(
        q=q_bsa,
        k=k,
        v=v,
        seq_lens=position_ids[:, -q_bsa.shape[1] :] + 1,
        indices=indices,
        ks=ks,
        ks_count=ks_count,
        ks_start_end=ks_start_end,
        args=args,
        access_counter=sa_access_counter,
        cache_miss_counter=sa_cache_miss_counter,
        EXTEND_BACKEND=args.sa_extend_backend,  # streaming works way much better in Gemma2, than dynamic_extend
        model_context_length=args.model_context_length,
        extend_context_length=args.extend_context_length,
        offload_update_cache=(cached_metadata is None) and args.online_update_cache,
        return_running_statistics=args.bsa_return_running_statistics,
        k_descale=args.k_descale,
        v_descale=args.v_descale,
        # offload_update_cache=args.online_update_cache,
        # offload_update_cache=False,
    )
    if args.offload_cache is not None:
        args.offload_cache.sa_kv_cache.verify_cache()

    # if DEBUG:
    #     print('context', context[0, :, DEBUG_HEAD, :], context.shape)
    #     print('indices', indices[0 + DEBUG_HEAD, -1], indices.shape)
    #     print('ks', ks[0 + DEBUG_HEAD, -1], ks.shape)

    metadata = HiPAttentionOutputMetadata(
        indices=indices,
        ks=ks,
        ks_count=ks_count,
        ks_start_end=ks_start_end,
        mask_cache_statistics=(
            HiPAttentionCacheAccessStatistics(
                access_counter=mask_access_counter,
                cache_miss_counter=mask_cache_miss_counter,
            )
            if (cached_metadata is None) or (cached_metadata.indices is None)
            else None
        ),
        sa_cache_statistics=HiPAttentionCacheAccessStatistics(
            access_counter=sa_access_counter,
            cache_miss_counter=sa_cache_miss_counter,
        ),
        stage_caches=stage_caches,
        state=state,
    )

    # if BDST > 1:
    #     print(id(metadata), type(state))

    return context, metadata
```

This files references to following modules. I will also provide those files too.

```
hip_attn.utils.rope
hip_attn.v1_2.attention_decode_bsa
hip_attn.v1_2.attention_extend_bsa
hip_attn.v1_2.attention_extend_bsa_tilelang
hip_attn.v1_2.attention_metadata
hip_attn.v1_2.compute_scores_landmark
hip_attn.v1_2.compute_v_cos
hip_attn.v1_2.eval_stage
hip_attn.v1_2.landmark_sample
hip_attn.v1_2.scan_stage
hip_attn.v1_2.stage_prologue
```

### `hip_attn/v1_2/attention_decode_bsa.py`

```py
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import torch
import triton
import triton.language as tl
from torch import Tensor

from hip_attn.v1_2.attention_extend_bsa import block_sparse_attention_cuda_step
from hip_attn.v1_2.attention_metadata import safe_stride
from hip_attn.v1_2.utils import capture
from hip_attn.v1_2.uvm_gpu_cache import load_tokens

if TYPE_CHECKING:
    from hip_attn.v1_2.attention_metadata import HiPAttentionArgs

DEFAULT_EXTEND_BACKEND: tl.constexpr = "streaming"
MAX_INT: tl.constexpr = 2_147_483_647


@triton.jit
def load_queries(
    cur_batch,
    cur_head,
    idx_tdst,
    offs_d,
    mask_h,
    mask_d,
    cur_batch_seq_len,
    Lk: tl.constexpr,
    Q,
    stride_q_bsz,
    stride_q_tdst,
    stride_q_head,
    stride_q_hid,
    COS,
    stride_cos_t,
    stride_cos_hid,
    SIN,
    stride_sin_t,
    stride_sin_hid,
    sink_token_size,
    sliding_window_size,
    sparse_token_size,
    model_context_length,
    rope_range_begin: tl.constexpr,
    rope_range_end: tl.constexpr,
    rope_is_neox_style: tl.constexpr,
    USING_EXTEND: tl.constexpr,
    NEED_APPLY_ROPE: tl.constexpr,
    EXTEND_BACKEND: tl.constexpr,
):
    offs_q = (
        cur_batch.to(tl.int64) * stride_q_bsz
        + idx_tdst * stride_q_tdst
        + cur_head[:, None].to(tl.int64) * stride_q_head
        + offs_d[None, :].to(tl.int64) * stride_q_hid
    )
    q = tl.load(
        Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0
    )  # [BLOCK_H, BLOCK_DMODEL]
    if q.dtype == tl.float8e5:
        q = q.to(tl.float16)

    if USING_EXTEND and NEED_APPLY_ROPE:
        ROPE_DIM = rope_range_end - rope_range_begin

        idx_rope_range = offs_d - rope_range_begin
        rope_mask = (rope_range_begin <= offs_d) & (offs_d < rope_range_end)
        if rope_is_neox_style:
            rope_rot_idx = tl.where(
                rope_mask,
                (offs_d - rope_range_begin + ROPE_DIM // 2) % ROPE_DIM
                + rope_range_begin,
                offs_d,
            )
            cos_sin_idx = idx_rope_range % (ROPE_DIM // 2)
            rope_mult = ((idx_rope_range + ROPE_DIM // 2 < ROPE_DIM) * (-2) + 1).to(
                q.dtype
            )
        else:
            flip = tl.where(idx_rope_range & 1 == 0, 1, -1)
            rope_rot_idx = tl.where(
                rope_mask,
                idx_rope_range + flip + rope_range_begin,
                offs_d,
            )
            cos_sin_idx = idx_rope_range // 2
            rope_mult = ((idx_rope_range % 2 == 0) * (-2) + 1).to(q.dtype)

        # rope_tdst = cur_batch_seq_len - 1
        if EXTEND_BACKEND == "streaming":
            rope_tdst = cur_batch_seq_len - 1
            activate_len = sink_token_size + sliding_window_size + sparse_token_size
            rope_tdst = rope_tdst - cur_batch_seq_len + activate_len
            rope_tdst = tl.minimum(tl.maximum(0, rope_tdst), model_context_length)
        else:
            rope_tdst = cur_batch_seq_len - 1

        queries_rot = tl.load(
            Q
            + cur_batch.to(tl.int64) * stride_q_bsz
            + idx_tdst * stride_q_tdst
            + cur_head[:, None].to(tl.int64) * stride_q_head
            + rope_rot_idx[None, :].to(tl.int64) * stride_q_hid,
            mask=(mask_h[:, None]) & (mask_d[None, :] & rope_mask[None, :]),
            other=0.0,
        )  # [BLOCK_H, BLOCK_DMODEL]
        if queries_rot.dtype == tl.float8e5:
            queries_rot = queries_rot.to(tl.float16)

        cos_new = tl.load(
            COS
            + rope_tdst.to(tl.int64) * stride_cos_t
            + cos_sin_idx[None, :].to(tl.int64) * stride_cos_hid,
            mask=mask_d[None, :] & rope_mask[None, :],
            other=0.0,
        ).to(
            q.dtype
        )  # [1, BLOCK_DMODEL]
        sin_new = tl.load(
            SIN
            + rope_tdst.to(tl.int64) * stride_sin_t
            + cos_sin_idx[None, :].to(tl.int64) * stride_sin_hid,
            mask=mask_d[None, :] & rope_mask[None, :],
            other=0.0,
        ).to(
            q.dtype
        )  # [1, BLOCK_DMODEL]

        queries_rot *= rope_mult[None, :]

        q = tl.where(
            rope_mask[None, :],
            (q * cos_new + queries_rot * sin_new).to(q.dtype),
            q,
        )

    return q


@triton.jit
def _fwd_kernel_stage1(
    Q,
    stride_q_bsz,
    stride_q_tdst,
    stride_q_head,
    stride_q_hid,
    K,
    stride_k_bsz,
    stride_k_tsrc,
    stride_k_head,
    stride_k_hid,
    V,
    stride_v_bsz,
    stride_v_tsrc,
    stride_v_head,
    stride_v_hid,
    K_DESCALE,
    V_DESCALE,
    B_Seqlen,
    stride_pos_bsz,
    stride_pos_tdst,
    INDICES,  # Warning: first dim is a flattened axis of (batch, q_head)
    stride_indices_b,
    stride_indices_bdst,
    stride_indices_bk,
    KS_START_END,  # Warning: first dim is a flattened axis of (batch, q_head)
    stride_ks_start_end_b,
    stride_ks_start_end_bdst,
    stride_ks_start_end_g,
    ATTN_LOGITS,
    stride_attn_logits_bsz,
    stride_attn_logits_tdst,
    stride_attn_logits_head,
    stride_attn_logits_kv_split,
    stride_attn_logits_hid,
    q_head_num: tl.constexpr,
    BK: tl.constexpr,
    num_query,
    MAX_TDST,
    MAX_TSRC,
    kv_group_num: tl.constexpr,
    sliding_window_size: tl.constexpr,
    sink_token_size: tl.constexpr,
    LOGIT_SOFTCAP: tl.constexpr,
    USING_EXTEND: tl.constexpr,
    NEED_APPLY_ROPE: tl.constexpr,
    COS,
    stride_cos_t,
    stride_cos_hid,
    SIN,
    stride_sin_t,
    stride_sin_hid,
    rope_range_begin: tl.constexpr,
    rope_range_end: tl.constexpr,
    rope_is_neox_style: tl.constexpr,
    model_context_length,
    # paged attention args template
    USING_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    K_CACHE,
    stride_k_cache_page,
    stride_k_cache_offset,
    stride_k_cache_kv_head,
    stride_k_cache_hid,
    V_CACHE,
    stride_v_cache_page,
    stride_v_cache_offset,
    stride_v_cache_kv_head,
    stride_v_cache_hid,
    BLOCK_TABLE,
    stride_block_table_bsz,
    stride_block_table_page,
    CACHE_SEQ_LENS,
    stride_cache_seq_lens_b,
    USING_OFFLOAD_CACHE: tl.constexpr,
    OFFLOAD_CACHE_KV_PACKED: tl.constexpr,
    GPU_BANK_COUNT,
    OFFLOAD_CACHE_UVM_METADATA,
    stride_offload_cache_uvm_metadata_token,
    stride_offload_cache_uvm_metadata_k,
    OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
    stride_offload_cache_gpu_global_metadata_k,
    stride_offload_cache_gpu_global_metadata_pad,
    OFFLOAD_CACHE_GPU_BANK,
    stride_offload_cache_gpu_bank_token,
    stride_offload_cache_gpu_bank_hid,
    OFFLOAD_CACHE_GPU_METADATA,
    stride_offload_cache_gpu_metadata_token,
    stride_offload_cache_gpu_metadata_k,
    OFFLOAD_CACHE_GPU_TABLE,
    stride_offload_cache_gpu_table_head_kv,
    stride_offload_cache_gpu_table_token,
    strdie_offload_cache_gpu_table_k,
    ACCESS_COUNTER,
    stride_access_counter_bsz,
    stride_access_counter_head_kv,
    stride_access_counter_tsrc,
    CACHE_MISS_COUNTER,
    stride_cache_miss_counter_bsz,
    stride_cache_miss_counter_head_kv,
    stride_cache_miss_counter_tsrc,
    TDST_NEXT_POWER_OF_2,
    IS_CAUSAL: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    Lk: tl.constexpr,  # hidden dim of key
    Lv: tl.constexpr,  # hidden dim of value
    # autotuning parameters
    BLOCK_BK: tl.constexpr,  # = BLOCK_N / BLOCK_SIZE_K
    NUM_SPARSE_KV_SPLITS: tl.constexpr,
    NUM_SINK_KV_SPLITS: tl.constexpr,
    NUM_SLIDING_KV_SPLITS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_DMODEL_0: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    EXTEND_BACKEND: tl.constexpr,
    UPDATE_CACHE: tl.constexpr,
    CHUNKED_SW: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    TOTAL_HEAD_BLOCKS = tl.cdiv(q_head_num, tl.minimum(BLOCK_H, kv_group_num))
    idx_head_block = pid % TOTAL_HEAD_BLOCKS
    pid = pid // TOTAL_HEAD_BLOCKS

    TOTAL_SPLITS = NUM_SPARSE_KV_SPLITS + NUM_SINK_KV_SPLITS + NUM_SLIDING_KV_SPLITS
    idx_split = pid % TOTAL_SPLITS
    pid = pid // TOTAL_SPLITS

    idx_tdst = pid % num_query
    idx_batch = pid // num_query

    # cur_batch = tl.program_id(0).to(tl.int64)
    # cur_head_id = tl.program_id(1).to(tl.int64)
    # cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    # split_kv_id = tl.program_id(2).to(tl.int64)

    cur_batch = idx_batch
    cur_head_id = idx_head_block
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = idx_split

    sink_split_kv_id = split_kv_id - NUM_SPARSE_KV_SPLITS
    sliding_split_kv_id = split_kv_id - NUM_SPARSE_KV_SPLITS - NUM_SINK_KV_SPLITS
    sparse_token_size = BK * BLOCK_SIZE_K

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head_begin = cur_head_id * VALID_BLOCK_H
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    # FIXME: current implementation is incorrect across heads
    cur_flattened_batch = cur_batch * q_head_num + cur_head_begin  # [BLOCK_H]

    ROPE_DIM = rope_range_end - rope_range_begin

    BLOCK_DMODEL_1: tl.constexpr = Lk - BLOCK_DMODEL_0

    offs_d_0 = tl.arange(0, BLOCK_DMODEL_0)
    mask_d_0 = offs_d_0 < Lk
    rope_mask_0 = (rope_range_begin <= offs_d_0) & (offs_d_0 < rope_range_end)
    idx_rope_range_q0 = offs_d_0 - rope_range_begin
    if rope_is_neox_style:
        rope_rot_idx_0 = tl.where(
            rope_mask_0,
            (idx_rope_range_q0 + ROPE_DIM // 2) % ROPE_DIM + rope_range_begin,
            offs_d_0,
        )
    else:
        flip = tl.where(idx_rope_range_q0 % 2 == 0, 1, -1)
        rope_rot_idx_0 = tl.where(
            rope_mask_0,
            idx_rope_range_q0 + flip + rope_range_begin,
            offs_d_0,
        )

    if BLOCK_DMODEL_1 > 0:
        offs_d_1 = BLOCK_DMODEL_0 + tl.arange(0, BLOCK_DMODEL_1)
        mask_d_1 = offs_d_1 < Lk
        rope_mask_1 = (rope_range_begin <= offs_d_1) & (offs_d_1 < rope_range_end)
        idx_rope_range_q1 = offs_d_1 - rope_range_begin
        if rope_is_neox_style:
            rope_rot_idx_1 = tl.where(
                rope_mask_1,
                (idx_rope_range_q1 + ROPE_DIM // 2) % ROPE_DIM + rope_range_begin,
                offs_d_1,
            )
        else:
            flip = tl.where(idx_rope_range_q1 % 2 == 0, 1, -1)
            rope_rot_idx_1 = tl.where(
                rope_mask_1,
                idx_rope_range_q1 + flip + rope_range_begin,
                offs_d_1,
            )
    else:
        offs_d_1 = None
        mask_d_1 = None

    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dv = offs_dv < Lv

    cur_batch_seq_len = tl.load(
        B_Seqlen + cur_batch.to(tl.int64) * stride_pos_bsz + idx_tdst * stride_pos_tdst
    )
    # cur_batch_req_idx = tl.load(B_req_idx + cur_batch)

    if K_DESCALE is not None:
        k_descale = tl.load(
            K_DESCALE +
            cur_batch.to(tl.int64) * (q_head_num // kv_group_num) +
            (cur_head // kv_group_num).to(tl.int64),
        )
        v_descale = tl.load(
            V_DESCALE +
            cur_batch.to(tl.int64) * (q_head_num // kv_group_num) +
            (cur_head // kv_group_num).to(tl.int64),
        )
    else:
        k_descale = None
        v_descale = None

    q_0 = load_queries(
        cur_batch,
        cur_head,
        idx_tdst,
        offs_d_0,
        mask_h,
        mask_d_0,
        cur_batch_seq_len,
        Lk,
        Q,
        stride_q_bsz,
        stride_q_tdst,
        stride_q_head,
        stride_q_hid,
        COS,
        stride_cos_t,
        stride_cos_hid,
        SIN,
        stride_sin_t,
        stride_sin_hid,
        sink_token_size,
        sliding_window_size,
        sparse_token_size,
        model_context_length,
        rope_range_begin,
        rope_range_end,
        rope_is_neox_style,
        USING_EXTEND and (rope_range_begin < BLOCK_DMODEL_0),
        NEED_APPLY_ROPE,
        EXTEND_BACKEND,
    )

    if BLOCK_DMODEL_1 > 0:
        q_1 = load_queries(
            cur_batch,
            cur_head,
            idx_tdst,
            offs_d_1,
            mask_h,
            mask_d_1,
            cur_batch_seq_len,
            Lk,
            Q,
            stride_q_bsz,
            stride_q_tdst,
            stride_q_head,
            stride_q_hid,
            COS,
            stride_cos_t,
            stride_cos_hid,
            SIN,
            stride_sin_t,
            stride_sin_hid,
            sink_token_size,
            sliding_window_size,
            sparse_token_size,
            model_context_length,
            rope_range_begin,
            rope_range_end,
            rope_is_neox_style,
            USING_EXTEND,
            NEED_APPLY_ROPE,
            EXTEND_BACKEND,
        )
    else:
        q_1 = None

    if q_0.dtype == tl.float8e5:
        q_0 = q_0.to(tl.float16)
        if q_1 is not None:
            q_1 = q_1.to(tl.float16)

    # Start and end indices to the `indices` tensor
    range_start = tl.load(
        KS_START_END
        + cur_flattened_batch.to(tl.int64) * stride_ks_start_end_b
        + 0 * stride_ks_start_end_bdst
        + 0 * stride_ks_start_end_g,
        mask=cur_head_begin < q_head_num,
        other=0,
    )
    range_end = tl.load(
        KS_START_END
        + (
            (
                cur_flattened_batch.to(tl.int64) * stride_ks_start_end_b
                + 0 * stride_ks_start_end_bdst
            )
            + 1 * stride_ks_start_end_g
        ),
        mask=cur_head_begin < q_head_num,
        other=0,
    )
    if BK <= 0:
        range_start = 0
        range_end = 0

    if BK > 0:
        kv_blocks_per_split = tl.cdiv(BK, NUM_SPARSE_KV_SPLITS)
        split_kv_block_start = kv_blocks_per_split * split_kv_id
        split_kv_block_end = tl.minimum(split_kv_block_start + kv_blocks_per_split, BK)
    else:
        kv_blocks_per_split = 0
        split_kv_block_start = 0
        split_kv_block_end = 0

    e_max = tl.full([BLOCK_H, 1], float("-inf"), dtype=tl.float32)  # m_i
    e_sum = tl.full([BLOCK_H, 1], 1.0, dtype=tl.float32)  # l_i
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if ((BK > 0) & (split_kv_block_end > split_kv_block_start)) and True:
        for i_bk in range(split_kv_block_start, split_kv_block_end, BLOCK_BK):
            idx_bk = i_bk + tl.arange(0, BLOCK_BK)  # [BLOCK_BK]
            mask_bk = (range_start <= idx_bk) & (
                idx_bk < tl.minimum(range_start + BK, range_end)
            )  # [BLOCK_BK]

            if (range_start <= i_bk + BLOCK_BK) & (i_bk < range_end):
                idx_tsrc_start = tl.load(
                    INDICES
                    + cur_flattened_batch.to(tl.int64) * stride_indices_b
                    + 0 * stride_indices_bdst
                    + idx_bk.to(tl.int64) * stride_indices_bk,
                    mask=mask_bk & (cur_head_begin < q_head_num),
                    other=0,
                )  # [BLOCK_BK]
                idx_tsrc_start = tl.where(mask_bk, idx_tsrc_start, MAX_TSRC + 1)
                idx_tsrc = idx_tsrc_start[:, None] + tl.arange(0, BLOCK_SIZE_K)[None, :]
                idx_tsrc = tl.reshape(idx_tsrc, (BLOCK_BK * BLOCK_SIZE_K))
                mask_tsrc_from_bk = mask_bk[:, None] & tl.full(
                    (1, BLOCK_SIZE_K), 1, dtype=tl.int1
                )
                mask_tsrc_from_bk = tl.reshape(
                    mask_tsrc_from_bk, (BLOCK_BK * BLOCK_SIZE_K)
                )
                mask_tsrc = (
                    ((MAX_TSRC * 0) <= idx_tsrc)
                    & (idx_tsrc < (MAX_TSRC * 1))
                    & mask_tsrc_from_bk
                )
                idx_tsrc = idx_tsrc % MAX_TSRC  # [BLOCK_BK * BLOCK_SIZE_K]
                mask_tsrc = (
                    (sink_token_size <= idx_tsrc)
                    & (idx_tsrc < cur_batch_seq_len)
                    & mask_tsrc
                )

                keys_0 = load_tokens(
                    K,
                    stride_k_bsz,
                    stride_k_tsrc,
                    stride_k_head,
                    stride_k_hid,
                    USING_PAGES,
                    PAGE_SIZE,
                    K_CACHE,
                    stride_k_cache_page,
                    stride_k_cache_offset,
                    stride_k_cache_kv_head,
                    stride_k_cache_hid,
                    BLOCK_TABLE,
                    stride_block_table_bsz,
                    stride_block_table_page,
                    CACHE_SEQ_LENS,
                    stride_cache_seq_lens_b,
                    USING_OFFLOAD_CACHE,
                    OFFLOAD_CACHE_KV_PACKED,
                    GPU_BANK_COUNT,
                    False,
                    OFFLOAD_CACHE_UVM_METADATA,
                    stride_offload_cache_uvm_metadata_token,
                    stride_offload_cache_uvm_metadata_k,
                    OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                    stride_offload_cache_gpu_global_metadata_k,
                    stride_offload_cache_gpu_global_metadata_pad,
                    OFFLOAD_CACHE_GPU_BANK,
                    stride_offload_cache_gpu_bank_token,
                    stride_offload_cache_gpu_bank_hid,
                    OFFLOAD_CACHE_GPU_METADATA,
                    stride_offload_cache_gpu_metadata_token,
                    stride_offload_cache_gpu_metadata_k,
                    OFFLOAD_CACHE_GPU_TABLE,
                    stride_offload_cache_gpu_table_head_kv,
                    stride_offload_cache_gpu_table_token,
                    strdie_offload_cache_gpu_table_k,
                    ACCESS_COUNTER,
                    stride_access_counter_bsz,
                    stride_access_counter_head_kv,
                    stride_access_counter_tsrc,
                    CACHE_MISS_COUNTER,
                    stride_cache_miss_counter_bsz,
                    stride_cache_miss_counter_head_kv,
                    stride_cache_miss_counter_tsrc,
                    cur_batch,
                    idx_tsrc[None, :],
                    cur_kv_head,
                    offs_d_0[:, None],
                    mask_tsrc[None, :],
                    q_head_num // kv_group_num,
                    BLOCK_SIZE_K,
                    BLOCK_DMODEL_0,
                    Lk,
                    IS_BSA=True,
                    UPDATE_CACHE=UPDATE_CACHE,
                    V_CACHE=V_CACHE,
                    stride_v_cache_page=stride_v_cache_page,
                    stride_v_cache_offset=stride_v_cache_offset,
                    stride_v_cache_kv_head=stride_v_cache_kv_head,
                    stride_v_cache_hid=stride_v_cache_hid,
                )

                if BLOCK_DMODEL_1 > 0:
                    keys_1 = load_tokens(
                        K,
                        stride_k_bsz,
                        stride_k_tsrc,
                        stride_k_head,
                        stride_k_hid,
                        USING_PAGES,
                        PAGE_SIZE,
                        K_CACHE,
                        stride_k_cache_page,
                        stride_k_cache_offset,
                        stride_k_cache_kv_head,
                        stride_k_cache_hid,
                        BLOCK_TABLE,
                        stride_block_table_bsz,
                        stride_block_table_page,
                        CACHE_SEQ_LENS,
                        stride_cache_seq_lens_b,
                        USING_OFFLOAD_CACHE,
                        OFFLOAD_CACHE_KV_PACKED,
                        GPU_BANK_COUNT,
                        False,
                        OFFLOAD_CACHE_UVM_METADATA,
                        stride_offload_cache_uvm_metadata_token,
                        stride_offload_cache_uvm_metadata_k,
                        OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                        stride_offload_cache_gpu_global_metadata_k,
                        stride_offload_cache_gpu_global_metadata_pad,
                        OFFLOAD_CACHE_GPU_BANK,
                        stride_offload_cache_gpu_bank_token,
                        stride_offload_cache_gpu_bank_hid,
                        OFFLOAD_CACHE_GPU_METADATA,
                        stride_offload_cache_gpu_metadata_token,
                        stride_offload_cache_gpu_metadata_k,
                        OFFLOAD_CACHE_GPU_TABLE,
                        stride_offload_cache_gpu_table_head_kv,
                        stride_offload_cache_gpu_table_token,
                        strdie_offload_cache_gpu_table_k,
                        ACCESS_COUNTER,
                        stride_access_counter_bsz,
                        stride_access_counter_head_kv,
                        stride_access_counter_tsrc,
                        CACHE_MISS_COUNTER,
                        stride_cache_miss_counter_bsz,
                        stride_cache_miss_counter_head_kv,
                        stride_cache_miss_counter_tsrc,
                        cur_batch,
                        idx_tsrc[None, :],
                        cur_kv_head,
                        offs_d_1[:, None],
                        mask_tsrc[None, :],
                        q_head_num // kv_group_num,
                        BLOCK_SIZE_K,
                        BLOCK_DMODEL_1,
                        Lk,
                        IS_BSA=True,
                        UPDATE_CACHE=UPDATE_CACHE,
                        V_CACHE=V_CACHE,
                        stride_v_cache_page=stride_v_cache_page,
                        stride_v_cache_offset=stride_v_cache_offset,
                        stride_v_cache_kv_head=stride_v_cache_kv_head,
                        stride_v_cache_hid=stride_v_cache_hid,
                    )
                else:
                    keys_1 = None

                if USING_EXTEND and NEED_APPLY_ROPE:
                    if rope_range_begin < BLOCK_DMODEL_0:
                        keys_rot_0 = load_tokens(
                            K,
                            stride_k_bsz,
                            stride_k_tsrc,
                            stride_k_head,
                            stride_k_hid,
                            USING_PAGES,
                            PAGE_SIZE,
                            K_CACHE,
                            stride_k_cache_page,
                            stride_k_cache_offset,
                            stride_k_cache_kv_head,
                            stride_k_cache_hid,
                            BLOCK_TABLE,
                            stride_block_table_bsz,
                            stride_block_table_page,
                            CACHE_SEQ_LENS,
                            stride_cache_seq_lens_b,
                            USING_OFFLOAD_CACHE,
                            OFFLOAD_CACHE_KV_PACKED,
                            GPU_BANK_COUNT,
                            False,
                            OFFLOAD_CACHE_UVM_METADATA,
                            stride_offload_cache_uvm_metadata_token,
                            stride_offload_cache_uvm_metadata_k,
                            OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                            stride_offload_cache_gpu_global_metadata_k,
                            stride_offload_cache_gpu_global_metadata_pad,
                            OFFLOAD_CACHE_GPU_BANK,
                            stride_offload_cache_gpu_bank_token,
                            stride_offload_cache_gpu_bank_hid,
                            OFFLOAD_CACHE_GPU_METADATA,
                            stride_offload_cache_gpu_metadata_token,
                            stride_offload_cache_gpu_metadata_k,
                            OFFLOAD_CACHE_GPU_TABLE,
                            stride_offload_cache_gpu_table_head_kv,
                            stride_offload_cache_gpu_table_token,
                            strdie_offload_cache_gpu_table_k,
                            ACCESS_COUNTER,
                            stride_access_counter_bsz,
                            stride_access_counter_head_kv,
                            stride_access_counter_tsrc,
                            CACHE_MISS_COUNTER,
                            stride_cache_miss_counter_bsz,
                            stride_cache_miss_counter_head_kv,
                            stride_cache_miss_counter_tsrc,
                            cur_batch,
                            idx_tsrc[None, :],
                            cur_kv_head,
                            rope_rot_idx_0[:, None],
                            mask_tsrc[None, :],
                            q_head_num // kv_group_num,
                            BLOCK_SIZE_K,
                            BLOCK_DMODEL_0,
                            Lk,
                            IS_BSA=True,
                            UPDATE_CACHE=UPDATE_CACHE,
                            V_CACHE=V_CACHE,
                            stride_v_cache_page=stride_v_cache_page,
                            stride_v_cache_offset=stride_v_cache_offset,
                            stride_v_cache_kv_head=stride_v_cache_kv_head,
                            stride_v_cache_hid=stride_v_cache_hid,
                        )
                    else:
                        keys_rot_0 = None

                    if BLOCK_DMODEL_1 > 0:
                        keys_rot_1 = load_tokens(
                            K,
                            stride_k_bsz,
                            stride_k_tsrc,
                            stride_k_head,
                            stride_k_hid,
                            USING_PAGES,
                            PAGE_SIZE,
                            K_CACHE,
                            stride_k_cache_page,
                            stride_k_cache_offset,
                            stride_k_cache_kv_head,
                            stride_k_cache_hid,
                            BLOCK_TABLE,
                            stride_block_table_bsz,
                            stride_block_table_page,
                            CACHE_SEQ_LENS,
                            stride_cache_seq_lens_b,
                            USING_OFFLOAD_CACHE,
                            OFFLOAD_CACHE_KV_PACKED,
                            GPU_BANK_COUNT,
                            False,
                            OFFLOAD_CACHE_UVM_METADATA,
                            stride_offload_cache_uvm_metadata_token,
                            stride_offload_cache_uvm_metadata_k,
                            OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                            stride_offload_cache_gpu_global_metadata_k,
                            stride_offload_cache_gpu_global_metadata_pad,
                            OFFLOAD_CACHE_GPU_BANK,
                            stride_offload_cache_gpu_bank_token,
                            stride_offload_cache_gpu_bank_hid,
                            OFFLOAD_CACHE_GPU_METADATA,
                            stride_offload_cache_gpu_metadata_token,
                            stride_offload_cache_gpu_metadata_k,
                            OFFLOAD_CACHE_GPU_TABLE,
                            stride_offload_cache_gpu_table_head_kv,
                            stride_offload_cache_gpu_table_token,
                            strdie_offload_cache_gpu_table_k,
                            ACCESS_COUNTER,
                            stride_access_counter_bsz,
                            stride_access_counter_head_kv,
                            stride_access_counter_tsrc,
                            CACHE_MISS_COUNTER,
                            stride_cache_miss_counter_bsz,
                            stride_cache_miss_counter_head_kv,
                            stride_cache_miss_counter_tsrc,
                            cur_batch,
                            idx_tsrc[None, :],
                            cur_kv_head,
                            rope_rot_idx_1[:, None],
                            mask_tsrc[None, :],
                            q_head_num // kv_group_num,
                            BLOCK_SIZE_K,
                            BLOCK_DMODEL_1,
                            Lk,
                            IS_BSA=True,
                            UPDATE_CACHE=UPDATE_CACHE,
                            V_CACHE=V_CACHE,
                            stride_v_cache_page=stride_v_cache_page,
                            stride_v_cache_offset=stride_v_cache_offset,
                            stride_v_cache_kv_head=stride_v_cache_kv_head,
                            stride_v_cache_hid=stride_v_cache_hid,
                        )
                    else:
                        keys_rot_1 = None

                else:
                    keys_rot_0 = None
                    keys_rot_1 = None

                if k_descale is not None:
                    keys_0 *= k_descale
                    keys_rot_0 *= k_descale
                    if keys_1 is not None:
                        keys_1 *= k_descale
                        keys_rot_1 *= k_descale

                values = load_tokens(
                    V,
                    stride_v_bsz,
                    stride_v_tsrc,
                    stride_v_head,
                    stride_v_hid,
                    USING_PAGES,
                    PAGE_SIZE,
                    V_CACHE,
                    stride_v_cache_page,
                    stride_v_cache_offset,
                    stride_v_cache_kv_head,
                    stride_v_cache_hid,
                    BLOCK_TABLE,
                    stride_block_table_bsz,
                    stride_block_table_page,
                    CACHE_SEQ_LENS,
                    stride_cache_seq_lens_b,
                    USING_OFFLOAD_CACHE,
                    OFFLOAD_CACHE_KV_PACKED,
                    GPU_BANK_COUNT,
                    True,
                    OFFLOAD_CACHE_UVM_METADATA,
                    stride_offload_cache_uvm_metadata_token,
                    stride_offload_cache_uvm_metadata_k,
                    OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                    stride_offload_cache_gpu_global_metadata_k,
                    stride_offload_cache_gpu_global_metadata_pad,
                    OFFLOAD_CACHE_GPU_BANK,
                    stride_offload_cache_gpu_bank_token,
                    stride_offload_cache_gpu_bank_hid,
                    OFFLOAD_CACHE_GPU_METADATA,
                    stride_offload_cache_gpu_metadata_token,
                    stride_offload_cache_gpu_metadata_k,
                    OFFLOAD_CACHE_GPU_TABLE,
                    stride_offload_cache_gpu_table_head_kv,
                    stride_offload_cache_gpu_table_token,
                    strdie_offload_cache_gpu_table_k,
                    ACCESS_COUNTER,
                    stride_access_counter_bsz,
                    stride_access_counter_head_kv,
                    stride_access_counter_tsrc,
                    CACHE_MISS_COUNTER,
                    stride_cache_miss_counter_bsz,
                    stride_cache_miss_counter_head_kv,
                    stride_cache_miss_counter_tsrc,
                    cur_batch,
                    idx_tsrc[:, None],
                    cur_kv_head,
                    offs_dv[None, :],
                    mask_tsrc[:, None],
                    q_head_num // kv_group_num,
                    BLOCK_SIZE_K,
                    BLOCK_DV,
                    Lv,
                    IS_BSA=True,
                    UPDATE_CACHE=UPDATE_CACHE,
                    V_CACHE=K_CACHE,
                    stride_v_cache_page=stride_k_cache_page,
                    stride_v_cache_offset=stride_k_cache_offset,
                    stride_v_cache_kv_head=stride_k_cache_kv_head,
                    stride_v_cache_hid=stride_k_cache_hid,
                )

                if v_descale is not None:
                    values *= v_descale

                acc, e_sum, e_max = block_sparse_attention_cuda_step(
                    q_0,  # FIXME: q is [BLOCK_H, BLOCK_DMODEL]: the first axis is head, not time
                    q_1,
                    keys_0,
                    keys_1,
                    keys_rot_0,
                    keys_rot_1,
                    values,
                    idx_tsrc,
                    mask_tsrc,
                    tl.zeros([1], dtype=tl.int32) + idx_tdst,
                    tl.full((1,), 1, dtype=tl.int1),
                    acc,
                    e_sum,
                    e_max,
                    sliding_window_size,
                    sink_token_size,
                    sparse_token_size,
                    (range_end - range_start) * BLOCK_SIZE_K,  # mask_k
                    True,
                    False,
                    LOGIT_SOFTCAP,
                    USING_EXTEND,
                    NEED_APPLY_ROPE,
                    COS,
                    stride_cos_t,
                    stride_cos_hid,
                    SIN,
                    stride_sin_t,
                    stride_sin_hid,
                    rope_range_begin,
                    rope_range_end,
                    rope_is_neox_style,
                    model_context_length,
                    tl.reshape(
                        idx_bk[:, None] * BLOCK_SIZE_K
                        + tl.arange(0, BLOCK_SIZE_K)[None, :],
                        BLOCK_SIZE_K * BLOCK_BK,
                    )
                    + sink_token_size,
                    cur_batch_seq_len,
                    offs_d_0,
                    offs_d_1,
                    IS_CAUSAL,
                    Lk,
                    BLOCK_SIZE_Q,
                    BLOCK_BK * BLOCK_SIZE_K,
                    BLOCK_SIZE_K,
                    EXTEND_BACKEND=EXTEND_BACKEND,
                )
            else:
                pass

    # process sink tokens
    if sink_token_size > 0:
        sink_tokens_per_split = tl.cdiv(sink_token_size, NUM_SINK_KV_SPLITS)
        split_sink_start = sink_tokens_per_split * sink_split_kv_id
        split_sink_end = tl.minimum(
            split_sink_start + sink_tokens_per_split, sink_token_size
        )
    else:
        sink_tokens_per_split = 0
        split_sink_start = 0
        split_sink_end = 0
    if (
        (sink_token_size > 0)
        & (0 <= sink_split_kv_id)
        & (sink_split_kv_id < NUM_SINK_KV_SPLITS)
        & (split_sink_end > split_sink_start)
    ) and True:
        for i_tsrc in range(split_sink_start, split_sink_end, BLOCK_BK * BLOCK_SIZE_K):
            idx_tsrc = i_tsrc + tl.arange(0, BLOCK_BK * BLOCK_SIZE_K)
            mask_tsrc = idx_tsrc < tl.minimum(cur_batch_seq_len, split_sink_end)

            keys_0 = load_tokens(
                K,
                stride_k_bsz,
                stride_k_tsrc,
                stride_k_head,
                stride_k_hid,
                USING_PAGES,
                PAGE_SIZE,
                K_CACHE,
                stride_k_cache_page,
                stride_k_cache_offset,
                stride_k_cache_kv_head,
                stride_k_cache_hid,
                BLOCK_TABLE,
                stride_block_table_bsz,
                stride_block_table_page,
                CACHE_SEQ_LENS,
                stride_cache_seq_lens_b,
                USING_OFFLOAD_CACHE,
                OFFLOAD_CACHE_KV_PACKED,
                GPU_BANK_COUNT,
                False,
                OFFLOAD_CACHE_UVM_METADATA,
                stride_offload_cache_uvm_metadata_token,
                stride_offload_cache_uvm_metadata_k,
                OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                stride_offload_cache_gpu_global_metadata_k,
                stride_offload_cache_gpu_global_metadata_pad,
                OFFLOAD_CACHE_GPU_BANK,
                stride_offload_cache_gpu_bank_token,
                stride_offload_cache_gpu_bank_hid,
                OFFLOAD_CACHE_GPU_METADATA,
                stride_offload_cache_gpu_metadata_token,
                stride_offload_cache_gpu_metadata_k,
                OFFLOAD_CACHE_GPU_TABLE,
                stride_offload_cache_gpu_table_head_kv,
                stride_offload_cache_gpu_table_token,
                strdie_offload_cache_gpu_table_k,
                ACCESS_COUNTER,
                stride_access_counter_bsz,
                stride_access_counter_head_kv,
                stride_access_counter_tsrc,
                CACHE_MISS_COUNTER,
                stride_cache_miss_counter_bsz,
                stride_cache_miss_counter_head_kv,
                stride_cache_miss_counter_tsrc,
                cur_batch,
                idx_tsrc[None, :],
                cur_kv_head,
                offs_d_0[:, None],
                mask_tsrc[None, :],
                q_head_num // kv_group_num,
                BLOCK_SIZE_K,
                BLOCK_DMODEL_0,
                Lk,
                IS_BSA=True,
                UPDATE_CACHE=UPDATE_CACHE,
                V_CACHE=V_CACHE,
                stride_v_cache_page=stride_v_cache_page,
                stride_v_cache_offset=stride_v_cache_offset,
                stride_v_cache_kv_head=stride_v_cache_kv_head,
                stride_v_cache_hid=stride_v_cache_hid,
            )

            if BLOCK_DMODEL_1 > 0:
                keys_1 = load_tokens(
                    K,
                    stride_k_bsz,
                    stride_k_tsrc,
                    stride_k_head,
                    stride_k_hid,
                    USING_PAGES,
                    PAGE_SIZE,
                    K_CACHE,
                    stride_k_cache_page,
                    stride_k_cache_offset,
                    stride_k_cache_kv_head,
                    stride_k_cache_hid,
                    BLOCK_TABLE,
                    stride_block_table_bsz,
                    stride_block_table_page,
                    CACHE_SEQ_LENS,
                    stride_cache_seq_lens_b,
                    USING_OFFLOAD_CACHE,
                    OFFLOAD_CACHE_KV_PACKED,
                    GPU_BANK_COUNT,
                    False,
                    OFFLOAD_CACHE_UVM_METADATA,
                    stride_offload_cache_uvm_metadata_token,
                    stride_offload_cache_uvm_metadata_k,
                    OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                    stride_offload_cache_gpu_global_metadata_k,
                    stride_offload_cache_gpu_global_metadata_pad,
                    OFFLOAD_CACHE_GPU_BANK,
                    stride_offload_cache_gpu_bank_token,
                    stride_offload_cache_gpu_bank_hid,
                    OFFLOAD_CACHE_GPU_METADATA,
                    stride_offload_cache_gpu_metadata_token,
                    stride_offload_cache_gpu_metadata_k,
                    OFFLOAD_CACHE_GPU_TABLE,
                    stride_offload_cache_gpu_table_head_kv,
                    stride_offload_cache_gpu_table_token,
                    strdie_offload_cache_gpu_table_k,
                    ACCESS_COUNTER,
                    stride_access_counter_bsz,
                    stride_access_counter_head_kv,
                    stride_access_counter_tsrc,
                    CACHE_MISS_COUNTER,
                    stride_cache_miss_counter_bsz,
                    stride_cache_miss_counter_head_kv,
                    stride_cache_miss_counter_tsrc,
                    cur_batch,
                    idx_tsrc[None, :],
                    cur_kv_head,
                    offs_d_1[:, None],
                    mask_tsrc[None, :],
                    q_head_num // kv_group_num,
                    BLOCK_SIZE_K,
                    BLOCK_DMODEL_1,
                    Lk,
                    IS_BSA=True,
                    UPDATE_CACHE=UPDATE_CACHE,
                    V_CACHE=V_CACHE,
                    stride_v_cache_page=stride_v_cache_page,
                    stride_v_cache_offset=stride_v_cache_offset,
                    stride_v_cache_kv_head=stride_v_cache_kv_head,
                    stride_v_cache_hid=stride_v_cache_hid,
                )
            else:
                keys_1 = None

            if USING_EXTEND and NEED_APPLY_ROPE:
                if rope_range_begin < BLOCK_DMODEL_0:
                    keys_rot_0 = load_tokens(
                        K,
                        stride_k_bsz,
                        stride_k_tsrc,
                        stride_k_head,
                        stride_k_hid,
                        USING_PAGES,
                        PAGE_SIZE,
                        K_CACHE,
                        stride_k_cache_page,
                        stride_k_cache_offset,
                        stride_k_cache_kv_head,
                        stride_k_cache_hid,
                        BLOCK_TABLE,
                        stride_block_table_bsz,
                        stride_block_table_page,
                        CACHE_SEQ_LENS,
                        stride_cache_seq_lens_b,
                        USING_OFFLOAD_CACHE,
                        OFFLOAD_CACHE_KV_PACKED,
                        GPU_BANK_COUNT,
                        False,
                        OFFLOAD_CACHE_UVM_METADATA,
                        stride_offload_cache_uvm_metadata_token,
                        stride_offload_cache_uvm_metadata_k,
                        OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                        stride_offload_cache_gpu_global_metadata_k,
                        stride_offload_cache_gpu_global_metadata_pad,
                        OFFLOAD_CACHE_GPU_BANK,
                        stride_offload_cache_gpu_bank_token,
                        stride_offload_cache_gpu_bank_hid,
                        OFFLOAD_CACHE_GPU_METADATA,
                        stride_offload_cache_gpu_metadata_token,
                        stride_offload_cache_gpu_metadata_k,
                        OFFLOAD_CACHE_GPU_TABLE,
                        stride_offload_cache_gpu_table_head_kv,
                        stride_offload_cache_gpu_table_token,
                        strdie_offload_cache_gpu_table_k,
                        ACCESS_COUNTER,
                        stride_access_counter_bsz,
                        stride_access_counter_head_kv,
                        stride_access_counter_tsrc,
                        CACHE_MISS_COUNTER,
                        stride_cache_miss_counter_bsz,
                        stride_cache_miss_counter_head_kv,
                        stride_cache_miss_counter_tsrc,
                        cur_batch,
                        idx_tsrc[None, :],
                        cur_kv_head,
                        rope_rot_idx_0[:, None],
                        mask_tsrc[None, :],
                        q_head_num // kv_group_num,
                        BLOCK_SIZE_K,
                        BLOCK_DMODEL_0,
                        Lk,
                        IS_BSA=True,
                        UPDATE_CACHE=UPDATE_CACHE,
                        V_CACHE=V_CACHE,
                        stride_v_cache_page=stride_v_cache_page,
                        stride_v_cache_offset=stride_v_cache_offset,
                        stride_v_cache_kv_head=stride_v_cache_kv_head,
                        stride_v_cache_hid=stride_v_cache_hid,
                    )
                else:
                    keys_rot_0 = None

                if BLOCK_DMODEL_1 > 0:
                    keys_rot_1 = load_tokens(
                        K,
                        stride_k_bsz,
                        stride_k_tsrc,
                        stride_k_head,
                        stride_k_hid,
                        USING_PAGES,
                        PAGE_SIZE,
                        K_CACHE,
                        stride_k_cache_page,
                        stride_k_cache_offset,
                        stride_k_cache_kv_head,
                        stride_k_cache_hid,
                        BLOCK_TABLE,
                        stride_block_table_bsz,
                        stride_block_table_page,
                        CACHE_SEQ_LENS,
                        stride_cache_seq_lens_b,
                        USING_OFFLOAD_CACHE,
                        OFFLOAD_CACHE_KV_PACKED,
                        GPU_BANK_COUNT,
                        False,
                        OFFLOAD_CACHE_UVM_METADATA,
                        stride_offload_cache_uvm_metadata_token,
                        stride_offload_cache_uvm_metadata_k,
                        OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                        stride_offload_cache_gpu_global_metadata_k,
                        stride_offload_cache_gpu_global_metadata_pad,
                        OFFLOAD_CACHE_GPU_BANK,
                        stride_offload_cache_gpu_bank_token,
                        stride_offload_cache_gpu_bank_hid,
                        OFFLOAD_CACHE_GPU_METADATA,
                        stride_offload_cache_gpu_metadata_token,
                        stride_offload_cache_gpu_metadata_k,
                        OFFLOAD_CACHE_GPU_TABLE,
                        stride_offload_cache_gpu_table_head_kv,
                        stride_offload_cache_gpu_table_token,
                        strdie_offload_cache_gpu_table_k,
                        ACCESS_COUNTER,
                        stride_access_counter_bsz,
                        stride_access_counter_head_kv,
                        stride_access_counter_tsrc,
                        CACHE_MISS_COUNTER,
                        stride_cache_miss_counter_bsz,
                        stride_cache_miss_counter_head_kv,
                        stride_cache_miss_counter_tsrc,
                        cur_batch,
                        idx_tsrc[None, :],
                        cur_kv_head,
                        rope_rot_idx_1[:, None],
                        mask_tsrc[None, :],
                        q_head_num // kv_group_num,
                        BLOCK_SIZE_K,
                        BLOCK_DMODEL_1,
                        Lk,
                        IS_BSA=True,
                        UPDATE_CACHE=UPDATE_CACHE,
                        V_CACHE=V_CACHE,
                        stride_v_cache_page=stride_v_cache_page,
                        stride_v_cache_offset=stride_v_cache_offset,
                        stride_v_cache_kv_head=stride_v_cache_kv_head,
                        stride_v_cache_hid=stride_v_cache_hid,
                    )
                else:
                    keys_rot_1 = None
            else:
                keys_rot_0 = None
                keys_rot_1 = None

            if k_descale is not None:
                keys_0 *= k_descale
                keys_rot_0 *= k_descale
                if keys_1 is not None:
                    keys_1 *= k_descale
                    keys_rot_1 *= k_descale

            values = load_tokens(
                V,
                stride_v_bsz,
                stride_v_tsrc,
                stride_v_head,
                stride_v_hid,
                USING_PAGES,
                PAGE_SIZE,
                V_CACHE,
                stride_v_cache_page,
                stride_v_cache_offset,
                stride_v_cache_kv_head,
                stride_v_cache_hid,
                BLOCK_TABLE,
                stride_block_table_bsz,
                stride_block_table_page,
                CACHE_SEQ_LENS,
                stride_cache_seq_lens_b,
                USING_OFFLOAD_CACHE,
                OFFLOAD_CACHE_KV_PACKED,
                GPU_BANK_COUNT,
                True,
                OFFLOAD_CACHE_UVM_METADATA,
                stride_offload_cache_uvm_metadata_token,
                stride_offload_cache_uvm_metadata_k,
                OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                stride_offload_cache_gpu_global_metadata_k,
                stride_offload_cache_gpu_global_metadata_pad,
                OFFLOAD_CACHE_GPU_BANK,
                stride_offload_cache_gpu_bank_token,
                stride_offload_cache_gpu_bank_hid,
                OFFLOAD_CACHE_GPU_METADATA,
                stride_offload_cache_gpu_metadata_token,
                stride_offload_cache_gpu_metadata_k,
                OFFLOAD_CACHE_GPU_TABLE,
                stride_offload_cache_gpu_table_head_kv,
                stride_offload_cache_gpu_table_token,
                strdie_offload_cache_gpu_table_k,
                ACCESS_COUNTER,
                stride_access_counter_bsz,
                stride_access_counter_head_kv,
                stride_access_counter_tsrc,
                CACHE_MISS_COUNTER,
                stride_cache_miss_counter_bsz,
                stride_cache_miss_counter_head_kv,
                stride_cache_miss_counter_tsrc,
                cur_batch,
                idx_tsrc[:, None],
                cur_kv_head,
                offs_dv[None, :],
                mask_tsrc[:, None],
                q_head_num // kv_group_num,
                BLOCK_SIZE_K,
                BLOCK_DV,
                Lv,
                IS_BSA=True,
                UPDATE_CACHE=UPDATE_CACHE,
                V_CACHE=K_CACHE,
                stride_v_cache_page=stride_k_cache_page,
                stride_v_cache_offset=stride_k_cache_offset,
                stride_v_cache_kv_head=stride_k_cache_kv_head,
                stride_v_cache_hid=stride_k_cache_hid,
            )

            if v_descale is not None:
                values *= v_descale

            acc, e_sum, e_max = block_sparse_attention_cuda_step(
                q_0,
                q_1,
                keys_0,
                keys_1,
                keys_rot_0,
                keys_rot_1,
                values,
                idx_tsrc,
                mask_tsrc,
                tl.zeros([1], dtype=tl.int32) + idx_tdst,
                tl.full((1,), 1, dtype=tl.int1),
                acc,
                e_sum,
                e_max,
                sliding_window_size,
                sink_token_size,
                sparse_token_size,
                (range_end - range_start) * BLOCK_SIZE_K,
                True,
                True,
                LOGIT_SOFTCAP,
                USING_EXTEND,
                NEED_APPLY_ROPE,
                COS,
                stride_cos_t,
                stride_cos_hid,
                SIN,
                stride_sin_t,
                stride_sin_hid,
                rope_range_begin,
                rope_range_end,
                rope_is_neox_style,
                model_context_length,
                idx_tsrc,
                cur_batch_seq_len,
                offs_d_0,
                offs_d_1,
                IS_CAUSAL,
                Lk,
                BLOCK_SIZE_Q,
                BLOCK_BK * BLOCK_SIZE_K,
                BLOCK_SIZE_K,
                EXTEND_BACKEND=EXTEND_BACKEND,
            )

    # process sliding window
    i_tsrc_range_start = tl.maximum(
        0, cur_batch_seq_len - sliding_window_size - BLOCK_SIZE_Q
    )
    sliding_tokens_per_split = tl.cdiv(
        cur_batch_seq_len - i_tsrc_range_start, NUM_SLIDING_KV_SPLITS
    )
    split_sliding_start = (
        i_tsrc_range_start + sliding_tokens_per_split * sliding_split_kv_id
    )
    split_sliding_end = tl.minimum(
        split_sliding_start + sliding_tokens_per_split, cur_batch_seq_len
    )
    if (
        (sliding_window_size > 0)
        & (0 <= sliding_split_kv_id)
        & (sliding_split_kv_id < NUM_SLIDING_KV_SPLITS)
        & (split_sliding_end > split_sliding_start)
    ) and True:
        for i_tsrc in range(
            split_sliding_start, split_sliding_end, BLOCK_BK * BLOCK_SIZE_K
        ):
            idx_tsrc = i_tsrc + tl.arange(0, BLOCK_BK * BLOCK_SIZE_K)
            mask_tsrc = (0 <= idx_tsrc) & (idx_tsrc < split_sliding_end)

            # idx_n = idx_b * G + idx_group
            keys_0 = load_tokens(
                K,
                stride_k_bsz,
                stride_k_tsrc,
                stride_k_head,
                stride_k_hid,
                USING_PAGES,
                PAGE_SIZE,
                K_CACHE,
                stride_k_cache_page,
                stride_k_cache_offset,
                stride_k_cache_kv_head,
                stride_k_cache_hid,
                BLOCK_TABLE,
                stride_block_table_bsz,
                stride_block_table_page,
                CACHE_SEQ_LENS,
                stride_cache_seq_lens_b,
                USING_OFFLOAD_CACHE,
                OFFLOAD_CACHE_KV_PACKED,
                GPU_BANK_COUNT,
                False,
                OFFLOAD_CACHE_UVM_METADATA,
                stride_offload_cache_uvm_metadata_token,
                stride_offload_cache_uvm_metadata_k,
                OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                stride_offload_cache_gpu_global_metadata_k,
                stride_offload_cache_gpu_global_metadata_pad,
                OFFLOAD_CACHE_GPU_BANK,
                stride_offload_cache_gpu_bank_token,
                stride_offload_cache_gpu_bank_hid,
                OFFLOAD_CACHE_GPU_METADATA,
                stride_offload_cache_gpu_metadata_token,
                stride_offload_cache_gpu_metadata_k,
                OFFLOAD_CACHE_GPU_TABLE,
                stride_offload_cache_gpu_table_head_kv,
                stride_offload_cache_gpu_table_token,
                strdie_offload_cache_gpu_table_k,
                ACCESS_COUNTER,
                stride_access_counter_bsz,
                stride_access_counter_head_kv,
                stride_access_counter_tsrc,
                CACHE_MISS_COUNTER,
                stride_cache_miss_counter_bsz,
                stride_cache_miss_counter_head_kv,
                stride_cache_miss_counter_tsrc,
                cur_batch,
                idx_tsrc[None, :],
                cur_kv_head,
                offs_d_0[:, None],
                mask_tsrc[None, :],
                q_head_num // kv_group_num,
                BLOCK_SIZE_K,
                BLOCK_DMODEL_0,
                Lk,
                IS_BSA=True,
                UPDATE_CACHE=UPDATE_CACHE,
                V_CACHE=V_CACHE,
                stride_v_cache_page=stride_v_cache_page,
                stride_v_cache_offset=stride_v_cache_offset,
                stride_v_cache_kv_head=stride_v_cache_kv_head,
                stride_v_cache_hid=stride_v_cache_hid,
            )

            if BLOCK_DMODEL_1 > 0:
                keys_1 = load_tokens(
                    K,
                    stride_k_bsz,
                    stride_k_tsrc,
                    stride_k_head,
                    stride_k_hid,
                    USING_PAGES,
                    PAGE_SIZE,
                    K_CACHE,
                    stride_k_cache_page,
                    stride_k_cache_offset,
                    stride_k_cache_kv_head,
                    stride_k_cache_hid,
                    BLOCK_TABLE,
                    stride_block_table_bsz,
                    stride_block_table_page,
                    CACHE_SEQ_LENS,
                    stride_cache_seq_lens_b,
                    USING_OFFLOAD_CACHE,
                    OFFLOAD_CACHE_KV_PACKED,
                    GPU_BANK_COUNT,
                    False,
                    OFFLOAD_CACHE_UVM_METADATA,
                    stride_offload_cache_uvm_metadata_token,
                    stride_offload_cache_uvm_metadata_k,
                    OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                    stride_offload_cache_gpu_global_metadata_k,
                    stride_offload_cache_gpu_global_metadata_pad,
                    OFFLOAD_CACHE_GPU_BANK,
                    stride_offload_cache_gpu_bank_token,
                    stride_offload_cache_gpu_bank_hid,
                    OFFLOAD_CACHE_GPU_METADATA,
                    stride_offload_cache_gpu_metadata_token,
                    stride_offload_cache_gpu_metadata_k,
                    OFFLOAD_CACHE_GPU_TABLE,
                    stride_offload_cache_gpu_table_head_kv,
                    stride_offload_cache_gpu_table_token,
                    strdie_offload_cache_gpu_table_k,
                    ACCESS_COUNTER,
                    stride_access_counter_bsz,
                    stride_access_counter_head_kv,
                    stride_access_counter_tsrc,
                    CACHE_MISS_COUNTER,
                    stride_cache_miss_counter_bsz,
                    stride_cache_miss_counter_head_kv,
                    stride_cache_miss_counter_tsrc,
                    cur_batch,
                    idx_tsrc[None, :],
                    cur_kv_head,
                    offs_d_1[:, None],
                    mask_tsrc[None, :],
                    q_head_num // kv_group_num,
                    BLOCK_SIZE_K,
                    BLOCK_DMODEL_1,
                    Lk,
                    IS_BSA=True,
                    UPDATE_CACHE=UPDATE_CACHE,
                    V_CACHE=V_CACHE,
                    stride_v_cache_page=stride_v_cache_page,
                    stride_v_cache_offset=stride_v_cache_offset,
                    stride_v_cache_kv_head=stride_v_cache_kv_head,
                    stride_v_cache_hid=stride_v_cache_hid,
                )
            else:
                keys_1 = None

            if USING_EXTEND and NEED_APPLY_ROPE:
                if rope_range_begin < BLOCK_DMODEL_0:
                    keys_rot_0 = load_tokens(
                        K,
                        stride_k_bsz,
                        stride_k_tsrc,
                        stride_k_head,
                        stride_k_hid,
                        USING_PAGES,
                        PAGE_SIZE,
                        K_CACHE,
                        stride_k_cache_page,
                        stride_k_cache_offset,
                        stride_k_cache_kv_head,
                        stride_k_cache_hid,
                        BLOCK_TABLE,
                        stride_block_table_bsz,
                        stride_block_table_page,
                        CACHE_SEQ_LENS,
                        stride_cache_seq_lens_b,
                        USING_OFFLOAD_CACHE,
                        OFFLOAD_CACHE_KV_PACKED,
                        GPU_BANK_COUNT,
                        False,
                        OFFLOAD_CACHE_UVM_METADATA,
                        stride_offload_cache_uvm_metadata_token,
                        stride_offload_cache_uvm_metadata_k,
                        OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                        stride_offload_cache_gpu_global_metadata_k,
                        stride_offload_cache_gpu_global_metadata_pad,
                        OFFLOAD_CACHE_GPU_BANK,
                        stride_offload_cache_gpu_bank_token,
                        stride_offload_cache_gpu_bank_hid,
                        OFFLOAD_CACHE_GPU_METADATA,
                        stride_offload_cache_gpu_metadata_token,
                        stride_offload_cache_gpu_metadata_k,
                        OFFLOAD_CACHE_GPU_TABLE,
                        stride_offload_cache_gpu_table_head_kv,
                        stride_offload_cache_gpu_table_token,
                        strdie_offload_cache_gpu_table_k,
                        ACCESS_COUNTER,
                        stride_access_counter_bsz,
                        stride_access_counter_head_kv,
                        stride_access_counter_tsrc,
                        CACHE_MISS_COUNTER,
                        stride_cache_miss_counter_bsz,
                        stride_cache_miss_counter_head_kv,
                        stride_cache_miss_counter_tsrc,
                        cur_batch,
                        idx_tsrc[None, :],
                        cur_kv_head,
                        rope_rot_idx_0[:, None],
                        mask_tsrc[None, :],
                        q_head_num // kv_group_num,
                        BLOCK_SIZE_K,
                        BLOCK_DMODEL_0,
                        Lk,
                        IS_BSA=True,
                        UPDATE_CACHE=UPDATE_CACHE,
                        V_CACHE=V_CACHE,
                        stride_v_cache_page=stride_v_cache_page,
                        stride_v_cache_offset=stride_v_cache_offset,
                        stride_v_cache_kv_head=stride_v_cache_kv_head,
                        stride_v_cache_hid=stride_v_cache_hid,
                    )
                else:
                    keys_rot_0 = None

                if BLOCK_DMODEL_1 > 0:
                    keys_rot_1 = load_tokens(
                        K,
                        stride_k_bsz,
                        stride_k_tsrc,
                        stride_k_head,
                        stride_k_hid,
                        USING_PAGES,
                        PAGE_SIZE,
                        K_CACHE,
                        stride_k_cache_page,
                        stride_k_cache_offset,
                        stride_k_cache_kv_head,
                        stride_k_cache_hid,
                        BLOCK_TABLE,
                        stride_block_table_bsz,
                        stride_block_table_page,
                        CACHE_SEQ_LENS,
                        stride_cache_seq_lens_b,
                        USING_OFFLOAD_CACHE,
                        OFFLOAD_CACHE_KV_PACKED,
                        GPU_BANK_COUNT,
                        False,
                        OFFLOAD_CACHE_UVM_METADATA,
                        stride_offload_cache_uvm_metadata_token,
                        stride_offload_cache_uvm_metadata_k,
                        OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                        stride_offload_cache_gpu_global_metadata_k,
                        stride_offload_cache_gpu_global_metadata_pad,
                        OFFLOAD_CACHE_GPU_BANK,
                        stride_offload_cache_gpu_bank_token,
                        stride_offload_cache_gpu_bank_hid,
                        OFFLOAD_CACHE_GPU_METADATA,
                        stride_offload_cache_gpu_metadata_token,
                        stride_offload_cache_gpu_metadata_k,
                        OFFLOAD_CACHE_GPU_TABLE,
                        stride_offload_cache_gpu_table_head_kv,
                        stride_offload_cache_gpu_table_token,
                        strdie_offload_cache_gpu_table_k,
                        ACCESS_COUNTER,
                        stride_access_counter_bsz,
                        stride_access_counter_head_kv,
                        stride_access_counter_tsrc,
                        CACHE_MISS_COUNTER,
                        stride_cache_miss_counter_bsz,
                        stride_cache_miss_counter_head_kv,
                        stride_cache_miss_counter_tsrc,
                        cur_batch,
                        idx_tsrc[None, :],
                        cur_kv_head,
                        rope_rot_idx_1[:, None],
                        mask_tsrc[None, :],
                        q_head_num // kv_group_num,
                        BLOCK_SIZE_K,
                        BLOCK_DMODEL_1,
                        Lk,
                        IS_BSA=True,
                        UPDATE_CACHE=UPDATE_CACHE,
                        V_CACHE=V_CACHE,
                        stride_v_cache_page=stride_v_cache_page,
                        stride_v_cache_offset=stride_v_cache_offset,
                        stride_v_cache_kv_head=stride_v_cache_kv_head,
                        stride_v_cache_hid=stride_v_cache_hid,
                    )
                else:
                    keys_rot_1 = None

            else:
                keys_rot_0 = None
                keys_rot_1 = None

            if k_descale is not None:
                keys_0 *= k_descale
                keys_rot_0 *= k_descale
                if keys_1 is not None:
                    keys_1 *= k_descale
                    keys_rot_1 *= k_descale

            values = load_tokens(
                V,
                stride_v_bsz,
                stride_v_tsrc,
                stride_v_head,
                stride_v_hid,
                USING_PAGES,
                PAGE_SIZE,
                V_CACHE,
                stride_v_cache_page,
                stride_v_cache_offset,
                stride_v_cache_kv_head,
                stride_v_cache_hid,
                BLOCK_TABLE,
                stride_block_table_bsz,
                stride_block_table_page,
                CACHE_SEQ_LENS,
                stride_cache_seq_lens_b,
                USING_OFFLOAD_CACHE,
                OFFLOAD_CACHE_KV_PACKED,
                GPU_BANK_COUNT,
                True,
                OFFLOAD_CACHE_UVM_METADATA,
                stride_offload_cache_uvm_metadata_token,
                stride_offload_cache_uvm_metadata_k,
                OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                stride_offload_cache_gpu_global_metadata_k,
                stride_offload_cache_gpu_global_metadata_pad,
                OFFLOAD_CACHE_GPU_BANK,
                stride_offload_cache_gpu_bank_token,
                stride_offload_cache_gpu_bank_hid,
                OFFLOAD_CACHE_GPU_METADATA,
                stride_offload_cache_gpu_metadata_token,
                stride_offload_cache_gpu_metadata_k,
                OFFLOAD_CACHE_GPU_TABLE,
                stride_offload_cache_gpu_table_head_kv,
                stride_offload_cache_gpu_table_token,
                strdie_offload_cache_gpu_table_k,
                ACCESS_COUNTER,
                stride_access_counter_bsz,
                stride_access_counter_head_kv,
                stride_access_counter_tsrc,
                CACHE_MISS_COUNTER,
                stride_cache_miss_counter_bsz,
                stride_cache_miss_counter_head_kv,
                stride_cache_miss_counter_tsrc,
                cur_batch,
                idx_tsrc[:, None],
                cur_kv_head,
                offs_dv[None, :],
                mask_tsrc[:, None],
                q_head_num // kv_group_num,
                BLOCK_SIZE_K,
                BLOCK_DV,
                Lv,
                IS_BSA=True,
                UPDATE_CACHE=UPDATE_CACHE,
                V_CACHE=K_CACHE,
                stride_v_cache_page=stride_k_cache_page,
                stride_v_cache_offset=stride_k_cache_offset,
                stride_v_cache_kv_head=stride_k_cache_kv_head,
                stride_v_cache_hid=stride_k_cache_hid,
            )

            if v_descale is not None:
                values *= v_descale

            # idx_bk = (
            #     tl.arange(0, BLOCK_BK)
            #     + (i_tsrc - i_tsrc_range_start) // BLOCK_SIZE_K
            #     + (cur_batch_seq_len - 1 - sliding_window_size) // BLOCK_SIZE_K
            # )
            idx_rope = (
                idx_tsrc
                - cur_batch_seq_len
                + sliding_window_size
                + sink_token_size
                + sparse_token_size
            )
            acc, e_sum, e_max = block_sparse_attention_cuda_step(
                q_0,  # [BLOCK_H, BLOCK_DMODEL]
                q_1,  # [BLOCK_DMODEL, BLOCK_BK * BLOCK_SIZE_K]
                keys_0,
                keys_1,
                keys_rot_0,
                keys_rot_1,
                values,
                idx_tsrc,
                mask_tsrc,
                tl.zeros([1], dtype=tl.int32) + idx_tdst,
                tl.full((1,), 1, dtype=tl.int1),
                acc,
                e_sum,
                e_max,
                sliding_window_size,
                sink_token_size,
                sparse_token_size,
                (range_end - range_start) * BLOCK_SIZE_K,
                False,
                False,
                LOGIT_SOFTCAP,
                USING_EXTEND,
                NEED_APPLY_ROPE,
                COS,
                stride_cos_t,
                stride_cos_hid,
                SIN,
                stride_sin_t,
                stride_sin_hid,
                rope_range_begin,
                rope_range_end,
                rope_is_neox_style,
                model_context_length,
                idx_rope,
                cur_batch_seq_len,
                offs_d_0,
                offs_d_1,
                IS_CAUSAL,
                Lk,
                BLOCK_SIZE_Q,
                BLOCK_BK * BLOCK_SIZE_K,
                BLOCK_SIZE_K,
                EXTEND_BACKEND=EXTEND_BACKEND,
                CHUNKED_SW=CHUNKED_SW,
            )

    e_sum = tl.where(e_sum < 1e-20, 1e-20, e_sum)

    # Store results
    offs_mid_o = (
        cur_batch.to(tl.int64) * stride_attn_logits_bsz
        + idx_tdst * stride_attn_logits_tdst
        + cur_head[:, None].to(tl.int64) * stride_attn_logits_head
        + split_kv_id.to(tl.int64) * stride_attn_logits_kv_split
        + offs_dv[None, :].to(tl.int64) * stride_attn_logits_hid
    )
    tl.store(
        ATTN_LOGITS + offs_mid_o,
        value=acc / e_sum,
        mask=(mask_h[:, None]) & (mask_dv[None, :]),
    )

    offs_mid_o_1 = (
        cur_batch.to(tl.int64) * stride_attn_logits_bsz
        + idx_tdst * stride_attn_logits_tdst
        + cur_head.to(tl.int64) * stride_attn_logits_head
        + split_kv_id.to(tl.int64) * stride_attn_logits_kv_split
        + Lv * stride_attn_logits_hid
    )
    tl.store(
        ATTN_LOGITS + offs_mid_o_1[:, None],
        value=e_max + tl.math.log2(e_sum),
        mask=mask_h[:, None],
    )


def decode_block_sparse_attention_stage1(
    q: Tensor,
    k: Optional[Tensor],
    v: Optional[Tensor],
    k_descale: Optional[Tensor],
    v_descale: Optional[Tensor],
    seq_lens: Tensor,
    indices: Tensor,
    ks_start_end: Tensor,
    args: HiPAttentionArgs,
    head_num: int,
    BK: int,
    MAX_TDST: int,
    MAX_TSRC: int,
    kv_group_num: int,
    model_context_length: int,
    HID: int,
    HID_V: int,
    BLOCK_BK: int,
    extend_backend: str,
    access_counter: Tensor,
    cache_miss_counter: Tensor,
    offload_update_cache: bool,
):
    batch = q.shape[0]
    num_query = q.shape[1]
    assert q.ndim == 4
    BLOCK_H = max(16, q.shape[2])
    NUM_SM = int(os.getenv("SA_DECODE_NUM_SM", 144 + 16))  # H100 + Slack

    total_tokens = args.second_stage_k + args.sink_token_size + args.sliding_window_size
    MAX_PROGRAM = int(
        os.getenv(
            "SA_DECODE_MAX_PROGRAM", min(64, triton.cdiv(NUM_SM, batch * num_query))
        )
    )
    token_chunk = triton.cdiv(total_tokens, MAX_PROGRAM)

    BLOCK_SIZE = min(args.block_size_k * BLOCK_BK, triton.next_power_of_2(token_chunk))
    BLOCK_BK = max(
        triton.cdiv(32, args.block_size_k), triton.cdiv(BLOCK_SIZE, args.block_size_k)
    )

    NUM_SPARSE_KV_SPLITS = min(
        MAX_PROGRAM,
        max(
            1 if args.second_stage_k > 0 else 0,
            round(args.second_stage_k / token_chunk),
        ),
    )  # TODO: apply from server args
    NUM_SINK_KV_SPLITS = min(
        MAX_PROGRAM,
        max(
            1 if args.sink_token_size > 0 else 0,
            round(args.sink_token_size / token_chunk),
        ),
    )
    NUM_SLIDING_KV_SPLITS = min(
        MAX_PROGRAM,
        max(
            1 if args.sliding_window_size > 0 else 0,
            round(args.sliding_window_size / token_chunk),
        ),
    )

    NUM_TOTAL_KV_SPLITS = (
        NUM_SPARSE_KV_SPLITS + NUM_SINK_KV_SPLITS + NUM_SLIDING_KV_SPLITS
    )
    # print('asdf', batch, num_query, NUM_TOTAL_KV_SPLITS, NUM_SINK_KV_SPLITS, NUM_SPARSE_KV_SPLITS, NUM_SLIDING_KV_SPLITS)

    temp_attn_logits = torch.empty(
        (batch, num_query, head_num, NUM_TOTAL_KV_SPLITS, HID + 1),
        dtype=torch.float32,
        device=q.device,
    )

    if k_descale is not None:
        assert k_descale.is_contiguous()
        assert v_descale.is_contiguous()
        assert k_descale.shape == (batch, head_num // kv_group_num)
        assert v_descale.shape == (batch, head_num // kv_group_num)

    grid = (
        batch
        * num_query
        * NUM_TOTAL_KV_SPLITS
        * triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
    )

    if args.rope_range[0] == 0 and args.rope_range[1] == HID:
        BLOCK_DMODEL = triton.next_power_of_2(HID)
    else:
        assert triton.next_power_of_2(args.rope_range[0]) == args.rope_range[0]
        assert args.rope_range[1] == HID
        BLOCK_DMODEL = args.rope_range[0]

    BLOCK_DV = triton.next_power_of_2(HID_V)

    _fwd_kernel_stage1[grid](
        q,
        *safe_stride(q, 4),
        k,
        *safe_stride(k, 4),
        v,
        *safe_stride(v, 4),
        k_descale,
        v_descale,
        seq_lens,
        *safe_stride(seq_lens, 2),
        indices,
        *safe_stride(indices, 3),
        ks_start_end,
        *safe_stride(ks_start_end, 3),
        temp_attn_logits,
        *safe_stride(temp_attn_logits, 5),
        head_num,
        BK,
        num_query,
        MAX_TDST,
        MAX_TSRC,
        kv_group_num,
        args.sliding_window_size,
        args.sink_token_size,
        args.logit_softcap,
        *args.args_extend(),
        model_context_length,
        *args.args_paged_kv_cache(),
        *args.args_offload_cache(is_masking=False),
        access_counter,
        *safe_stride(access_counter, 3),
        cache_miss_counter,
        *safe_stride(cache_miss_counter, 3),
        triton.next_power_of_2(MAX_TDST),
        args.is_causal,
        args.block_size_q,
        args.block_size_k,
        Lk=HID,
        Lv=HID_V,
        BLOCK_BK=BLOCK_BK,
        NUM_SPARSE_KV_SPLITS=NUM_SPARSE_KV_SPLITS,
        NUM_SINK_KV_SPLITS=NUM_SINK_KV_SPLITS,
        NUM_SLIDING_KV_SPLITS=NUM_SLIDING_KV_SPLITS,
        BLOCK_H=BLOCK_H,
        BLOCK_DMODEL_0=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        EXTEND_BACKEND=extend_backend,
        UPDATE_CACHE=offload_update_cache,
        CHUNKED_SW=args.using_chunked_sliding_window,
    )

    return temp_attn_logits, NUM_TOTAL_KV_SPLITS


@triton.jit
def _fwd_kernel_stage2(
    ATTN_LOGITS,
    stride_attn_logits_bsz,
    stride_attn_logits_tdst,
    stride_attn_logits_head,
    stride_attn_logits_kv_split,
    stride_attn_logits_hid,
    O,
    stride_o_bsz,
    stride_o_tdst,
    stride_o_head,
    stride_o_hid,
    B_SEQ_LEN,
    stride_pos_bsz,
    stride_pos_tdst,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0).to(tl.int64)
    idx_tdst = tl.program_id(1).to(tl.int64)
    cur_head = tl.program_id(2).to(tl.int64)

    cur_batch_seq_len = tl.load(
        B_SEQ_LEN + cur_batch.to(tl.int64) * stride_pos_bsz + idx_tdst * stride_pos_tdst
    )

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = (
        cur_batch * stride_attn_logits_bsz
        + idx_tdst * stride_attn_logits_tdst
        + cur_head * stride_attn_logits_head
        + offs_d * stride_attn_logits_hid
    )
    offs_logic = (
        cur_batch * stride_attn_logits_bsz
        + idx_tdst * stride_attn_logits_tdst
        + cur_head * stride_attn_logits_head
        + Lv * stride_attn_logits_hid
    )

    for split_kv_id in range(0, NUM_KV_SPLITS):
        tv = tl.load(
            ATTN_LOGITS
            + offs_v.to(tl.int64)
            + split_kv_id.to(tl.int64) * stride_attn_logits_kv_split,
            mask=mask_d,
            other=0.0,
        )
        tlogic = tl.load(
            ATTN_LOGITS
            + offs_logic.to(tl.int64)
            + split_kv_id.to(tl.int64) * stride_attn_logits_kv_split
        )
        n_e_max = tl.maximum(tlogic, e_max)

        n_e_max_valid = n_e_max > -1e50
        old_scale = tl.math.exp2(e_max - n_e_max)
        exp_logic = tl.math.exp2(tlogic - n_e_max)
        acc = tl.where(n_e_max_valid, acc * old_scale + exp_logic * tv, acc)

        e_sum = tl.where(n_e_max_valid, e_sum * old_scale + exp_logic, e_sum)
        e_max = n_e_max

    e_sum = tl.where(e_sum < 1e-20, 1e-20, e_sum)

    tl.store(
        O
        + cur_batch.to(tl.int64) * stride_o_bsz
        + idx_tdst * stride_o_tdst
        + cur_head * stride_o_head
        + offs_d * stride_o_hid,
        value=acc / e_sum,
        mask=mask_d,
    )


def decode_block_sparse_attention_stage2(
    logits,
    q,
    o,
    b_seq_len,
    num_total_kv_splits,
    HID_V: int,
):
    batch, num_query, head_num = q.shape[:3]
    Lv = HID_V
    BLOCK_DV = triton.next_power_of_2(Lv)

    NUM_KV_SPLITS = num_total_kv_splits

    grid = (batch, num_query, head_num)
    _fwd_kernel_stage2[grid](
        logits,
        *safe_stride(logits, 5),
        o,
        *safe_stride(o, 4),
        b_seq_len,
        *safe_stride(b_seq_len, 2),
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        num_warps=4,
        num_stages=2,
    )


def decode_block_sparse_attention_impl(
    q: Tensor,
    k: Optional[Tensor],
    v: Optional[Tensor],
    k_descale: Optional[Tensor],
    v_descale: Optional[Tensor],
    seq_lens: Tensor,
    indices: Tensor,
    ks_start_end: Tensor,
    context: Tensor,
    args: HiPAttentionArgs,
    HEAD: int,
    BK: int,
    MAX_TDST: int,
    MAX_TSRC: int,
    KV_HEAD_REPEAT: int,
    model_context_length: int,
    HID: int,
    HID_V: int,
    BLOCK_BK: int,
    extend_backend: str,
    access_counter: Tensor,
    cache_miss_counter: Tensor,
    offload_update_cache: bool,
):
    """
    FlashDecode block sparse attention.
    :param q: (BSZ, TDST, HEAD, HID)
    :param seq_lens: (BSZ, TDST)
    :param indices: (BSZ, TDST, BK)
    :param ks_start_end: (BSZ, BSRC, 2)
    :param context: (BSZ, TDST, HEAD, HID)
    """

    attn_logits, NUM_TOTAL_KV_SPLITS = decode_block_sparse_attention_stage1(
        q,
        k,
        v,
        k_descale=k_descale,
        v_descale=v_descale,
        seq_lens=seq_lens,
        indices=indices,
        ks_start_end=ks_start_end,
        args=args,
        head_num=HEAD,
        BK=BK,
        MAX_TDST=MAX_TDST,
        MAX_TSRC=MAX_TSRC,
        kv_group_num=KV_HEAD_REPEAT,
        model_context_length=model_context_length,
        HID=HID,
        HID_V=HID_V,
        BLOCK_BK=BLOCK_BK,
        extend_backend=extend_backend,
        access_counter=access_counter,
        cache_miss_counter=cache_miss_counter,
        offload_update_cache=offload_update_cache,
    )

    decode_block_sparse_attention_stage2(
        attn_logits,
        q,
        context,
        seq_lens,
        NUM_TOTAL_KV_SPLITS,
        HID_V,
    )

    return attn_logits


@capture
def decode_block_sparse_attention(
    q: Tensor,  # [1, 1 (TDST), 32 (Q_HEAD), 128]
    k: Optional[Tensor],  # None
    v: Optional[Tensor],  # None
    seq_lens: Tensor,  # [1, 1 (TDST)], tensor([34089])
    indices: Tensor,  # [32 (BSZ*Q_HEAD), 1 (BDST), 512]
    ks: Tensor,  # [32 (BSZ*Q_HEAD), 1 (BDST)]
    ks_count: Tensor,  # [32 (BSZ*Q_HEAD), 1 (BDST), 1]
    ks_start_end: Tensor,  # [32 (BSZ*Q_HEAD), 1 (BDST), 2]
    args: HiPAttentionArgs,
    # args.block_table: [1 (BSZ), 196612]
    # args.cache_seq_lens: [1 (BSZ)], tensor([34089])
    # args.k_cache: [109527 (NUM_PAGE), 1 (PAGE_SIZE), 8 (KV_HEAD), 128 (Lk)]
    # args.v_cache: [109527 (NUM_PAGE), 1 (PAGE_SIZE), 8 (KV_HEAD), 128 (Lv)]
    # args.position_ids: [1, 1 (TDST)]
    # args.rope_cos: [196608, 128 (Lk)]
    # args.rope_sin: [196608, 128 (Lk)]
    access_counter: Tensor,  # [1, 8, 109527]
    cache_miss_counter: Tensor,  # [1, 8, 109527]
    EXTEND_BACKEND: str = DEFAULT_EXTEND_BACKEND,  # 'streaming'
    model_context_length: int = 131072,  # 131072
    extend_context_length: int = 131072,  # 196608
    offload_update_cache: bool = False,
    return_running_statistics: bool = False,
    k_descale: Tensor = None,
    v_descale: Tensor = None,
):
    assert not return_running_statistics

    BSZ, TDST, HEAD, HID = q.shape

    assert TDST < args.block_sparse_block_size_q, "TDST must be 1 for flashdecode"

    if k is not None:
        _, TSRC, KV_HEAD, _ = k.shape
        MAX_TSRC = TSRC
        HID_V = v.shape[-1]
    else:
        if args.k_cache is not None:
            NUM_PAGE, PAGE_SIZE, KV_HEAD, _ = args.k_cache.shape
            HID_V = args.v_cache.shape[-1]
        else:
            KV_HEAD = args.offload_cache.k_uvm.bank_cpu.shape[-2]
            HID_V = args.offload_cache.v_uvm.bank_cpu.shape[-1]
        MAX_TSRC = extend_context_length
    KV_HEAD_REPEAT = HEAD // KV_HEAD
    assert KV_HEAD_REPEAT * KV_HEAD == HEAD
    HID_V = args.v_hidden_dim if args.v_hidden_dim is not None else HID_V

    BK = indices.shape[-1]

    context = torch.empty((BSZ, TDST, HEAD, HID_V), dtype=q.dtype, device=q.device)

    max_block_size = int(
        os.getenv("SA_DECODE_BLOCK_SIZE", os.getenv("SA_BLOCK_SIZE", "64"))
    )
    if HID >= 512:
        # NOTE: when MLA
        max_block_size = min(
            max_block_size,
            int(os.getenv("SA_DECODE_MLA_BLOCK_SIZE", "32")),
        )

    BLOCK_BK = max_block_size // args.block_size_k
    BLOCK_BK = max(1, min(max_block_size, BLOCK_BK))
    if "SA_BLOCK_BK" in os.environ:
        BLOCK_BK = int(os.environ["SA_BLOCK_BK"])

    assert BLOCK_BK > 0, BLOCK_BK

    if args.rope_cos is not None:
        assert len(args.rope_cos.stride()) == 2
        assert len(args.rope_sin.stride()) == 2

    assert context.ndim == 4
    if ks_start_end is not None:
        assert ks_start_end.ndim == 3
    if indices is not None:
        assert indices.ndim == 3
    assert q.ndim == 4
    if k is not None:
        assert k.ndim == 4
        assert v.ndim == 4
    elif args.using_paged_cache:
        if args.k_cache is not None:
            assert args.k_cache.ndim == 4
            assert args.v_cache.ndim == 4
        else:
            assert args.offload_cache.k_uvm.bank_cpu.ndim == 3
            assert args.offload_cache.v_uvm.bank_cpu.ndim == 3
    else:
        raise Exception()
    assert seq_lens.ndim == 2

    if k_descale is not None:
        k_descale = k_descale.contiguous()
        v_descale = v_descale.contiguous()
        assert k_descale.shape == v_descale.shape
        assert k_descale.shape == (BSZ, KV_HEAD)

    pre_device = torch.get_default_device()
    torch.set_default_device(q.device)

    decode_block_sparse_attention_impl(
        q,
        k,
        v,
        k_descale=k_descale,
        v_descale=v_descale,
        seq_lens=seq_lens,
        indices=indices,
        ks_start_end=ks_start_end,
        context=context,
        args=args,
        HEAD=HEAD,
        BK=BK,
        MAX_TDST=TDST,
        MAX_TSRC=MAX_TSRC,
        KV_HEAD_REPEAT=KV_HEAD_REPEAT,
        model_context_length=model_context_length,
        HID=HID,
        HID_V=HID_V,
        BLOCK_BK=BLOCK_BK,
        extend_backend=EXTEND_BACKEND,
        access_counter=access_counter,
        cache_miss_counter=cache_miss_counter,
        offload_update_cache=offload_update_cache,
    )

    torch.set_default_device(pre_device)

    return context

```

### `hip_attn/v1_2/attention_extend_bsa.py`

```py
import os
import warnings
from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton import cdiv as cdiv_python

from hip_attn.utils.rope import adjust_rope
from hip_attn.v1_2.attention_metadata import HiPAttentionArgs, safe_stride
from hip_attn.v1_2.uvm_gpu_cache import load_tokens

DEFAULT_EXTEND_BACKEND: tl.constexpr = "streaming"


@triton.jit
def apply_rope_to_keys(
    queries,
    keys,
    keys_rot,
    # indices
    idx_tsrc,
    mask_tsrc,
    mask_tdst,
    pos_tdst,
    idx_rope,
    idx_hid,
    # configs
    COS,
    stride_cos_t,
    stride_cos_hid,
    SIN,
    stride_sin_t,
    stride_sin_hid,
    rope_range_begin: tl.constexpr,
    rope_range_end: tl.constexpr,
    rope_is_neox_style: tl.constexpr,
    model_context_length,
    sink_token_size,
    mask_k,
    sparse_token_size,
    sliding_window_size,
    HID: tl.constexpr,
    BLOCK_TQ: tl.constexpr,
    BLOCK_TK: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    USING_EXTEND: tl.constexpr,
    HAS_FIRST_TOKEN: tl.constexpr,
    EXCLUDE_SLIDING_WINDOW: tl.constexpr,
    NEED_APPLY_ROPE: tl.constexpr,
    EXTEND_BACKEND: tl.constexpr,
):
    tl.static_assert(USING_EXTEND)

    if EXTEND_BACKEND == "self_extend":
        raise Exception()
    elif (
        (EXTEND_BACKEND == "streaming")
        | (EXTEND_BACKEND == "dynamic_extend")
        | (EXTEND_BACKEND == "infllm")
        | (EXTEND_BACKEND == "clamp")
    ):
        pos_tdst_min = tl.min(tl.where(mask_tdst, pos_tdst - 1, 987654321))
        if not NEED_APPLY_ROPE:
            if (
                (pos_tdst_min >= model_context_length) and EXCLUDE_SLIDING_WINDOW
            ) and True:
                assert COS is not None
                assert SIN is not None

                if HAS_FIRST_TOKEN:
                    old_tdst = pos_tdst - 1
                    new_tdst = tl.minimum(
                        old_tdst, sliding_window_size + mask_k + sink_token_size - 1
                    )

                    queries_adjusted = adjust_rope(
                        queries,
                        old_tdst,
                        new_tdst,
                        mask_tdst,
                        idx_hid,
                        COS,
                        stride_cos_t,
                        stride_cos_hid,
                        SIN,
                        stride_sin_t,
                        stride_sin_hid,
                        BLOCK_TQ,
                        HID,
                        idx_hid.shape[0],
                        NEED_APPLY_ROPE,
                        rope_range_begin,
                        rope_range_end,
                        rope_is_neox_style,
                    )

                    keys_adjusted = keys
                else:
                    old_tsrc = idx_tsrc
                    new_tsrc = tl.ravel(
                        (idx_bk * BLOCK_SIZE_K)[:, None]
                        + tl.arange(0, BLOCK_SIZE_K)[None, :]
                    )
                    new_tsrc = tl.maximum(
                        0,
                        new_tsrc
                        + pos_tdst_min
                        - sliding_window_size
                        - sink_token_size
                        - mask_k
                        - BLOCK_TQ
                        + 1,
                    )

                    keys_adjusted = keys.trans(1, 0)
                    keys_adjusted = adjust_rope(
                        keys_adjusted.to(queries.dtype),
                        old_tsrc,
                        new_tsrc,
                        mask_tsrc,
                        idx_hid,
                        COS,
                        stride_cos_t,
                        stride_cos_hid,
                        SIN,
                        stride_sin_t,
                        stride_sin_hid,
                        BLOCK_TK,
                        HID,
                        idx_hid.shape[0],
                        NEED_APPLY_ROPE,
                        rope_range_begin,
                        rope_range_end,
                        rope_is_neox_style,
                    )
                    keys_adjusted = tl.trans(keys_adjusted, 1, 0)

                    queries_adjusted = queries

            else:
                if NEED_APPLY_ROPE:
                    queries = adjust_rope(
                        queries.to(tl.float32),
                        pos_tdst - 1,
                        pos_tdst - 1,
                        mask_tdst,
                        idx_hid,
                        COS,
                        stride_cos_t,
                        stride_cos_hid,
                        SIN,
                        stride_sin_t,
                        stride_sin_hid,
                        BLOCK_TQ,
                        HID,
                        idx_hid.shape[0],
                        True,
                        rope_range_begin,
                        rope_range_end,
                        rope_is_neox_style,
                    ).to(queries.dtype)
                    queries_adjusted = (queries * mask_tdst[:, None]).to(queries.dtype)

                    keys = tl.trans(
                        adjust_rope(
                            tl.trans(keys.to(tl.float32), 1, 0),
                            idx_tsrc,
                            idx_tsrc,
                            mask_tsrc,
                            idx_hid,
                            COS,
                            stride_cos_t,
                            stride_cos_hid,
                            SIN,
                            stride_sin_t,
                            stride_sin_hid,
                            BLOCK_TK,
                            HID,
                            idx_hid.shape[0],
                            True,
                            rope_range_begin,
                            rope_range_end,
                            rope_is_neox_style,
                        ),
                        1,
                        0,
                    ).to(keys.dtype)
                    keys_adjusted = (keys * mask_tsrc[None, :]).to(keys.dtype)

        else:
            tl.static_assert(NEED_APPLY_ROPE)
            tl.static_assert(USING_EXTEND)

            ROPE_DIM = rope_range_end - rope_range_begin

            idx_rope_range = idx_hid - rope_range_begin
            rope_mask = (rope_range_begin <= idx_hid) & (idx_hid < rope_range_end)
            if rope_is_neox_style:
                cos_sin_idx = idx_rope_range % (ROPE_DIM // 2)
                rope_mult = ((idx_rope_range + ROPE_DIM // 2 < ROPE_DIM) * (-2) + 1).to(
                    queries.dtype
                )
            else:
                cos_sin_idx = idx_rope_range // 2
                rope_mult = ((idx_rope_range % 2 == 0) * (-2) + 1).to(queries.dtype)

            if EXCLUDE_SLIDING_WINDOW:
                # NOTE this is seq len
                pos_tdst_max = pos_tdst_min + tl.sum(mask_tdst.to(tl.int32))

                if EXTEND_BACKEND == "streaming":
                    # streaming
                    new_tsrc = idx_rope
                    num_sparse_tokens = (
                        sliding_window_size + sink_token_size + sparse_token_size
                    )
                    if num_sparse_tokens > model_context_length:
                        new_tsrc = new_tsrc - (num_sparse_tokens - model_context_length)
                    # new_tsrc = tl.maximum(
                    #     0,
                    #     new_tsrc
                    #     + pos_tdst_min
                    #     - sliding_window_size
                    #     - sink_token_size
                    #     - mask_k
                    #     + 1,
                    # )
                    new_tsrc = tl.maximum(0, new_tsrc)
                elif EXTEND_BACKEND == "dynamic_extend":
                    # dynamic extend
                    window = model_context_length // 4

                    new_tsrc = tl.where(
                        (idx_tsrc >= (pos_tdst_max - window))
                        | (pos_tdst_max <= model_context_length),
                        idx_tsrc,
                        (
                            (idx_tsrc + window - pos_tdst_min)
                            * (
                                (model_context_length - window)
                                / (pos_tdst_min - window)
                            )
                        ).to(tl.int32)
                        + pos_tdst_min
                        - window,
                    )
                    new_tsrc = tl.maximum(pos_tdst_max - model_context_length, new_tsrc)
                elif EXTEND_BACKEND == "infllm":
                    new_tsrc = tl.ravel(
                        (idx_bk * BLOCK_SIZE_K)[:, None]
                        + tl.arange(0, BLOCK_SIZE_K)[None, :]
                    )
                    new_tsrc = tl.maximum(
                        0, new_tsrc * 0 + pos_tdst_min - sliding_window_size
                    )
                elif EXTEND_BACKEND == "clamp":
                    new_tsrc = idx_tsrc
                    new_tsrc = tl.maximum(
                        new_tsrc,
                        new_tsrc * 0
                        + pos_tdst_min
                        - (model_context_length - mask_tdst.shape[0]),
                    )
                else:
                    raise Exception()
            else:
                if EXTEND_BACKEND == "streaming":
                    new_tsrc = idx_rope
                    num_sparse_tokens = (
                        sliding_window_size + sink_token_size + sparse_token_size
                    )
                    if num_sparse_tokens > model_context_length:
                        new_tsrc = new_tsrc - (num_sparse_tokens - model_context_length)
                    new_tsrc = tl.maximum(0, new_tsrc)
                else:
                    new_tsrc = idx_tsrc

            keys = keys.to(queries.dtype)
            keys_rot = keys_rot.to(queries.dtype)

            cos_new = tl.load(
                COS
                + new_tsrc[None, :].to(tl.int64) * stride_cos_t
                + cos_sin_idx[:, None].to(tl.int64) * stride_cos_hid,
                mask=mask_tsrc[None, :] & rope_mask[:, None],
                other=0.0,
            ).to(keys.dtype)
            sin_new = tl.load(
                SIN
                + new_tsrc[None, :].to(tl.int64) * stride_sin_t
                + cos_sin_idx[:, None].to(tl.int64) * stride_sin_hid,
                mask=mask_tsrc[None, :] & rope_mask[:, None],
                other=0.0,
            ).to(keys.dtype)

            if EXCLUDE_SLIDING_WINDOW:
                if EXTEND_BACKEND == "dynamic_extend":
                    streaming_tsrc = tl.ravel(
                        (idx_bk * BLOCK_SIZE_K)[:, None]
                        + tl.arange(0, BLOCK_SIZE_K)[None, :]
                    )
                    streaming_tsrc = tl.maximum(
                        0,
                        streaming_tsrc
                        + pos_tdst_min
                        - sliding_window_size
                        - sink_token_size
                        - mask_k
                        + 1,
                    )

                    cos_zero = tl.load(
                        COS
                        + streaming_tsrc[None, :].to(tl.int64) * stride_cos_t
                        + cos_sin_idx[:, None].to(tl.int64) * stride_cos_hid,
                        mask=rope_mask[:, None],
                        # mask=mask_tsrc[None, :],
                        other=0.0,
                    ).to(keys.dtype)
                    sin_zero = tl.load(
                        SIN
                        + streaming_tsrc[None, :].to(tl.int64) * stride_sin_t
                        + cos_sin_idx[:, None].to(tl.int64) * stride_sin_hid,
                        mask=rope_mask[:, None],
                        # mask=mask_tsrc[None, :],
                        other=0.0,
                    ).to(keys.dtype)

                    cos_new = (cos_zero * 0.75 + cos_new * 0.25).to(cos_new.dtype)
                    sin_new = (sin_zero * 0.75 + sin_new * 0.25).to(sin_new.dtype)

            keys_rot *= rope_mult[:, None]

            keys_adjusted = tl.where(
                rope_mask[:, None],
                (keys * cos_new + keys_rot * sin_new).to(keys.dtype),
                keys,
            )

            queries_adjusted = queries

    else:
        raise Exception()

    return queries_adjusted, keys_adjusted


@triton.jit
def block_sparse_attention_cuda_step(
    # QKV
    queries_0,
    queries_1,
    keys_0,
    keys_1,
    keys_rot_0,
    keys_rot_1,
    values,
    # indices
    idx_tsrc,
    mask_tsrc,
    idx_tdst,
    mask_tdst,
    # rolling value
    acc,
    l_i,
    m_i,
    # TDST,
    # TSRC,
    sliding_window_size,
    sink_token_size,
    sparse_token_size,
    mask_k,
    EXCLUDE_SLIDING_WINDOW: tl.constexpr,
    HAS_FIRST_TOKEN: tl.constexpr,
    LOGIT_SOFTCAP: tl.constexpr,
    USING_EXTEND: tl.constexpr,
    NEED_APPLY_ROPE: tl.constexpr,
    COS,
    stride_cos_t,
    stride_cos_hid,
    SIN,
    stride_sin_t,
    stride_sin_hid,
    rope_range_begin: tl.constexpr,
    rope_range_end: tl.constexpr,
    rope_is_neox_style: tl.constexpr,
    model_context_length,
    idx_bk,
    pos_tdst,
    idx_hid_q0,
    idx_hid_q1,
    IS_CAUSAL: tl.constexpr,
    HID: tl.constexpr,
    BLOCK_TQ,
    BLOCK_TK,
    BLOCK_SIZE_K: tl.constexpr,
    EXTEND_BACKEND: tl.constexpr = DEFAULT_EXTEND_BACKEND,
    CHUNKED_SW: tl.constexpr = False,
):
    HID_BLOCK_0: tl.constexpr = queries_0.shape[1]
    HID_BLOCK_1: tl.constexpr = queries_1.shape[1] if queries_1 is not None else 0

    if USING_EXTEND:
        if rope_range_begin < HID_BLOCK_0:
            queries_0, keys_0 = apply_rope_to_keys(
                queries_0,
                keys_0,
                keys_rot_0,
                idx_tsrc,
                mask_tsrc,
                mask_tdst,
                pos_tdst,
                idx_bk,
                idx_hid_q0,
                COS,
                stride_cos_t,
                stride_cos_hid,
                SIN,
                stride_sin_t,
                stride_sin_hid,
                rope_range_begin,
                rope_range_end,
                rope_is_neox_style,
                model_context_length,
                sink_token_size,
                mask_k,
                sparse_token_size,
                sliding_window_size,
                HID,
                BLOCK_TQ,
                BLOCK_TK,
                BLOCK_SIZE_K,
                USING_EXTEND,
                HAS_FIRST_TOKEN,
                EXCLUDE_SLIDING_WINDOW,
                NEED_APPLY_ROPE,
                EXTEND_BACKEND,
            )

        if HID_BLOCK_1 > 0:
            tl.static_assert(queries_1.shape[-1] == HID_BLOCK_1)
            queries_1, keys_1 = apply_rope_to_keys(
                queries_1,
                keys_1,
                keys_rot_1,
                idx_tsrc,
                mask_tsrc,
                mask_tdst,
                pos_tdst,
                idx_bk,
                idx_hid_q1,
                COS,
                stride_cos_t,
                stride_cos_hid,
                SIN,
                stride_sin_t,
                stride_sin_hid,
                rope_range_begin,
                rope_range_end,
                rope_is_neox_style,
                model_context_length,
                sink_token_size,
                mask_k,
                sparse_token_size,
                sliding_window_size,
                HID,
                BLOCK_TQ,
                BLOCK_TK,
                BLOCK_SIZE_K,
                USING_EXTEND,
                HAS_FIRST_TOKEN,
                EXCLUDE_SLIDING_WINDOW,
                NEED_APPLY_ROPE,
                EXTEND_BACKEND,
            )

    q_dtype = queries_0.dtype

    cq = tl.sqrt(HID * 1.0) / tl.sqrt(tl.sqrt(HID * 1.0))
    ck = 1 / tl.sqrt(tl.sqrt(HID * 1.0))

    # if q_dtype == tl.float16:
    #     dot_dtype = tl.float8e5
    # elif q_dtype == tl.bfloat16:
    #     dot_dtype = tl.float8e5
    # else:
    #     dot_dtype = q_dtype
    dot_dtype = q_dtype

    qk = tl.dot(
        (queries_0 * cq).to(dot_dtype),
        (keys_0.to(q_dtype) * ck).to(dot_dtype),
        out_dtype=tl.float32,
    ).to(tl.float32)

    if HID_BLOCK_1 > 0:
        qk += tl.dot(
            (queries_1 * cq).to(dot_dtype),
            (keys_1.to(q_dtype) * ck).to(dot_dtype),
            out_dtype=tl.float32,
        ).to(tl.float32)

    if LOGIT_SOFTCAP is not None:
        qk = tl.extra.cuda.libdevice.tanh(qk / LOGIT_SOFTCAP) * LOGIT_SOFTCAP
    qk = qk * 1.44269504

    # if qk_mask == True, then dropped
    if IS_CAUSAL:
        if len(pos_tdst.shape) > 0:
            seq_len = tl.max(pos_tdst)
        else:
            seq_len = pos_tdst

        if EXCLUDE_SLIDING_WINDOW:
            assert not CHUNKED_SW
            # qk_mask = (
            #     ((pos_tdst - 1)[:, None] < idx_tsrc[None, :])
            #     | ((pos_tdst - 1)[:, None] < (idx_tsrc + sliding_window_size)[None, :])
            #     | (~(mask_tdst[:, None] & mask_tsrc[None, :]))
            # )

            qk_mask = ~(mask_tsrc & (idx_tsrc < (seq_len - sliding_window_size)))[
                None, :
            ]
        else:
            # TODO(ainl): we should reduce scanning loop range if CHUNKED_SW is true.
            if not CHUNKED_SW:
                # qk_mask = (
                #     ((pos_tdst - 1)[:, None] < idx_tsrc[None, :])
                #     | (
                #         (pos_tdst - 1)[:, None]
                #         >= (idx_tsrc + sliding_window_size)[None, :]
                #     )
                #     | (~(mask_tdst[:, None] & mask_tsrc[None, :]))
                # )

                qk_mask = (
                    ((pos_tdst - 1)[:, None] < idx_tsrc[None, :])
                    | ~(idx_tsrc[None, :] >= (seq_len - sliding_window_size))
                    | (~(mask_tdst[:, None] & mask_tsrc[None, :]))
                )
            else:
                # qk_mask = (
                #     ((pos_tdst - 1)[:, None] < idx_tsrc[None, :])
                #     | ((pos_tdst - 1)[:, None] >= (idx_tsrc + 1024)[None, :])
                #     | (~(mask_tdst[:, None] & mask_tsrc[None, :]))
                # )
                qk_mask = (
                    ((pos_tdst - 1)[:, None] < idx_tsrc[None, :])
                    # | ((pos_tdst - 1)[:, None] >= (idx_tsrc + sliding_window_size)[None, :])
                    | (~(mask_tdst[:, None] & mask_tsrc[None, :]))
                    # | idx_tsrc[None, :] < ((pos_tdst - 1) - ((pos_tdst - 1) % sliding_window_size))[:, None]
                    | (
                        (
                            idx_tsrc[None, :]
                            < (
                                (pos_tdst - 1)
                                // sliding_window_size
                                * sliding_window_size
                            )[:, None]
                        )
                        # & ((pos_tdst - 1)[:, None] >= (idx_tsrc + 64)[None, :])
                    )
                )
    else:
        qk_mask = ~(mask_tdst[:, None] & mask_tsrc[None, :])

    # [BLOCK_SIZE_Q: tdst, 1: tsrc]
    qk = tl.where(qk_mask, float("-inf"), qk).to(qk.dtype)
    m_ij = tl.maximum(m_i, tl.max(qk, axis=1)[:, None])

    qk = qk - m_ij
    # [BLOCK_SIZE_Q: tdst, BLOCK_BK * BLOCK_SIZE_K: tsrc]
    p = tl.math.exp2(qk)

    p = tl.where(qk_mask, 0, p)

    # [BLOCK_SIZE_Q: tdst, 1: tsrc]
    l_ij = tl.sum(p, axis=1)

    # -- update m_i and l_i
    l_valid = m_ij > -1e50
    alpha = tl.math.exp2(m_i - m_ij)
    l_i = tl.where(
        l_valid,
        (l_i * alpha + l_ij[:, None]).to(l_i.dtype),
        l_i,
    )

    # -- update output accumulator --
    acc = tl.where(
        l_valid,
        acc * alpha.to(acc.dtype)
        + tl.dot(
            p.to(q_dtype),
            values.to(q_dtype),
            out_dtype=tl.float32,
            allow_tf32=True,
        ).to(acc.dtype),
        acc,
    )

    # update m_i and l_i
    m_i = tl.where(l_valid, m_ij.to(m_i.dtype), m_i)

    return acc, l_i, m_i


# def perf_model_block_sparse_attention(**kwargs):
#     block_bk = kwargs['BLOCK_BK']
#     block_k = kwargs['BLOCK_SIZE_K']
#     assert block_k <= 64, 'this will not good idea'
#     if ((block_bk * block_k) <= 64) and ((block_bk * block_k) >= 32):
#         return 0
#     return 999999999 # run might fails


@triton.jit
def apply_rope_to_queries(
    queries,
    pos_tdst,
    rope_tdst,
    idx_hid,
    idx_bsz,
    idx_tdst,
    mask_tdst,
    idx_head,
    Q,
    stride_q_bsz,
    stride_q_tdst,
    stride_q_head,
    stride_q_hid,
    COS,
    stride_cos_t,
    stride_cos_hid,
    SIN,
    stride_sin_t,
    stride_sin_hid,
    rope_range_begin: tl.constexpr,
    rope_range_end: tl.constexpr,
    rope_is_neox_style: tl.constexpr,
):
    ROPE_DIM = rope_range_end - rope_range_begin

    idx_rope_range = idx_hid - rope_range_begin
    rope_mask = (rope_range_begin <= idx_hid) & (idx_hid < rope_range_end)
    if rope_is_neox_style:
        rope_rot_idx = tl.where(
            rope_mask,
            (idx_rope_range + ROPE_DIM // 2) % ROPE_DIM + rope_range_begin,
            idx_hid,
        )
        cos_sin_idx = idx_rope_range % (ROPE_DIM // 2)
        rope_mult = ((idx_rope_range + ROPE_DIM // 2 < ROPE_DIM) * (-2) + 1).to(
            queries.dtype
        )
    else:
        flip = tl.where(idx_rope_range & 1 == 0, 1, -1)
        rope_rot_idx = tl.where(
            rope_mask,
            idx_rope_range + flip + rope_range_begin,
            idx_hid,
        )
        cos_sin_idx = idx_rope_range // 2
        rope_mult = ((idx_rope_range % 2 == 0) * (-2) + 1).to(queries.dtype)

    queries_rot = tl.load(
        Q
        + idx_bsz.to(tl.int64) * stride_q_bsz
        + idx_tdst[:, None].to(tl.int64) * stride_q_tdst
        + idx_head.to(tl.int64) * stride_q_head
        + rope_rot_idx[None, :].to(tl.int64) * stride_q_hid,
        mask=mask_tdst[:, None] & rope_mask[None, :],
        other=0.0,
    )
    if queries_rot.dtype == tl.float8e5:
        queries_rot = queries_rot.to(tl.float16)

    cos_new = tl.load(
        COS
        + rope_tdst[:, None].to(tl.int64) * stride_cos_t
        + cos_sin_idx[None, :].to(tl.int64) * stride_cos_hid,
        mask=mask_tdst[:, None] & rope_mask[None, :],
        other=0.0,
    ).to(queries.dtype)
    sin_new = tl.load(
        SIN
        + rope_tdst[:, None].to(tl.int64) * stride_sin_t
        + cos_sin_idx[None, :].to(tl.int64) * stride_sin_hid,
        mask=mask_tdst[:, None] & rope_mask[None, :],
        other=0.0,
    ).to(queries.dtype)

    queries_rot *= rope_mult[None, :]

    queries = tl.where(
        rope_mask[None, :],
        (queries * cos_new + queries_rot * sin_new).to(queries.dtype),
        queries,
    )

    return queries


def get_block_sparse_attention_configs():
    autotune_disabled = os.getenv("HIP_DISABLE_AUTOTUNE", "1") == "1"
    if autotune_disabled:
        device_name = torch.cuda.get_device_name()
        defaults = {
            "NVIDIA A100-SXM4-80GB": dict(
                num_warps=4,
                num_stages=2,
                maxnreg=256,
            ),
        }.get(device_name, dict(num_warps=4, num_stages=2))
        return [triton.Config({}, **defaults)]
    if os.getenv("HIP_DISABLE_AUTOTUNE_WARNINGS", "0") == "0":
        warnings.warn(
            "Triton autotuning is activated. This should be disabled for faster startup. If you want set HIP_DISABLE_AUTOTUNE=1. Set HIP_DISABLE_AUTOTUNE_WARNINGS=1 to hide this message."
        )

    NUM_WARPS = [4, 8]  # workaround for triton bug
    if triton.__version__ < "3.2.0":
        NUM_WARPS.remove(8)

    configs = []
    # for block_bk in [4, 8, 16, 32]:
    # for block_bk in [16, 32,]:
    for num_warps in NUM_WARPS:
        for num_stages in [
            3,
            4,
            7,
        ]:
            configs.append(
                triton.Config({}, num_warps=num_warps, num_stages=num_stages)
            )
    return configs


@triton.autotune(
    configs=get_block_sparse_attention_configs(),
    key=[
        "BLOCK_SIZE_K",
        "BLOCK_SIZE_Q",
        "HID",
        # "TDST_NEXT_POWER_OF_2",
    ],
    # prune_configs_by={
    #     'perf_model': perf_model_block_sparse_attention,
    #     'top_k': 24,
    # }
)
@triton.jit
def block_sparse_attention_cuda(
    Q,
    stride_q_bsz,
    stride_q_tdst,
    stride_q_head,
    stride_q_hid,
    K,
    stride_k_bsz,
    stride_k_tsrc,
    stride_k_head,
    stride_k_hid,
    V,
    stride_v_bsz,
    stride_v_tsrc,
    stride_v_head,
    stride_v_hid,
    K_DESCALE,
    V_DESCALE,
    POS,
    stride_pos_bsz,
    stride_pos_tdst,
    INDICES,
    stride_indices_b,
    stride_indices_bdst,
    stride_indices_bk,
    KS_START_END,
    stride_ks_start_end_b,
    stride_ks_start_end_bdst,
    stride_ks_start_end_g,
    CONTEXT,
    stride_context_bsz,
    stride_context_tdst,
    stride_context_head,
    stride_context_hid,
    MX,
    NC,
    stride_mx_bsz,
    stride_mx_tdst,
    stride_mx_head,
    HEAD: tl.constexpr,
    BK: tl.constexpr,
    MAX_TDST,
    MAX_TSRC,
    KV_HEAD_REPEAT: tl.constexpr,
    sliding_window_size: tl.constexpr,
    sink_token_size: tl.constexpr,
    LOGIT_SOFTCAP: tl.constexpr,
    USING_EXTEND: tl.constexpr,
    NEED_APPLY_ROPE: tl.constexpr,
    COS,
    stride_cos_t,
    stride_cos_hid,
    SIN,
    stride_sin_t,
    stride_sin_hid,
    rope_range_begin: tl.constexpr,
    rope_range_end: tl.constexpr,
    rope_is_neox_style: tl.constexpr,
    model_context_length,
    # paged attention args template
    USING_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    K_CACHE,
    stride_k_cache_page,
    stride_k_cache_offset,
    stride_k_cache_kv_head,
    stride_k_cache_hid,
    V_CACHE,
    stride_v_cache_page,
    stride_v_cache_offset,
    stride_v_cache_kv_head,
    stride_v_cache_hid,
    BLOCK_TABLE,
    stride_block_table_bsz,
    stride_block_table_page,
    CACHE_SEQ_LENS,
    stride_cache_seq_lens_b,
    USING_OFFLOAD_CACHE: tl.constexpr,
    OFFLOAD_CACHE_KV_PACKED: tl.constexpr,
    GPU_BANK_COUNT: int,
    OFFLOAD_CACHE_UVM_METADATA,
    stride_offload_cache_uvm_metadata_token,
    stride_offload_cache_uvm_metadata_k,
    OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
    stride_offload_cache_gpu_global_metadata_k,
    stride_offload_cache_gpu_global_metadata_pad,
    OFFLOAD_CACHE_GPU_BANK,
    stride_offload_cache_gpu_bank_token,
    stride_offload_cache_gpu_bank_hid,
    OFFLOAD_CACHE_GPU_METADATA,
    stride_offload_cache_gpu_metadata_token,
    stride_offload_cache_gpu_metadata_k,
    OFFLOAD_CACHE_GPU_TABLE,
    stride_offload_cache_gpu_table_head_kv,
    stride_offload_cache_gpu_table_token,
    strdie_offload_cache_gpu_table_k,
    ACCESS_COUNTER,
    stride_access_counter_bsz,
    stride_access_counter_head_kv,
    stride_access_counter_tsrc,
    CACHE_MISS_COUNTER,
    stride_cache_miss_counter_bsz,
    stride_cache_miss_counter_head_kv,
    stride_cache_miss_counter_tsrc,
    TDST_NEXT_POWER_OF_2,
    IS_CAUSAL: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    HID_BLOCK_0: tl.constexpr,
    HID: tl.constexpr,
    HID_BLOCK_V: tl.constexpr,
    HID_V: tl.constexpr,
    # autotuning parameters
    BLOCK_BK: tl.constexpr,
    EXTEND_BACKEND: tl.constexpr,
    UPDATE_CACHE: tl.constexpr,
    CHUNKED_SW: tl.constexpr,
):
    G: tl.constexpr = 1

    pid_bsz = tl.program_id(2).to(tl.int64)
    pid_bdst = tl.program_id(1).to(tl.int64)
    pid_head = tl.program_id(0).to(tl.int64) % HEAD
    pid_v = tl.program_id(0).to(tl.int64) // HEAD
    dim_v_offset = pid_v * HID_BLOCK_V

    idx_bsz = pid_bsz.to(tl.int64)
    idx_head = pid_head
    idx_n = idx_bsz * HEAD + idx_head
    idx_b = idx_n
    idx_g = 0

    idx_bdst = pid_bdst
    if BLOCK_SIZE_Q < 16:
        idx_tdst = BLOCK_SIZE_Q * idx_bdst + tl.arange(0, 16)
        mask_tdst = (idx_tdst < MAX_TDST) & (tl.arange(0, 16) < BLOCK_SIZE_Q)
    else:
        idx_tdst = BLOCK_SIZE_Q * idx_bdst + tl.arange(0, BLOCK_SIZE_Q)
        mask_tdst = idx_tdst < MAX_TDST
    if IS_CAUSAL:
        pos_tdst = tl.load(
            POS
            + idx_bsz.to(tl.int64) * stride_pos_bsz
            + idx_tdst.to(tl.int64) * stride_pos_tdst,
            mask=mask_tdst,
            other=0,
        )
    else:
        pos_tdst = tl.where(
            mask_tdst, tl.full((BLOCK_SIZE_Q,), value=MAX_TSRC, dtype=tl.int64), 0
        )

    ROPE_DIM = rope_range_end - rope_range_begin

    HID_BLOCK_1: tl.constexpr = HID - HID_BLOCK_0

    sparse_token_size: tl.constexpr = BK * BLOCK_SIZE_K

    idx_hid_q0 = tl.arange(0, HID_BLOCK_0)
    rope_mask_0 = (rope_range_begin <= idx_hid_q0) & (idx_hid_q0 < rope_range_end)
    idx_rope_range_q0 = idx_hid_q0 - rope_range_begin
    if rope_is_neox_style:
        rope_rot_idx_0 = tl.where(
            rope_mask_0,
            (idx_rope_range_q0 + ROPE_DIM // 2) % ROPE_DIM + rope_range_begin,
            idx_hid_q0,
        )
    else:
        flip = tl.where(idx_rope_range_q0 % 2 == 0, 1, -1)
        rope_rot_idx_0 = tl.where(
            rope_mask_0,
            idx_rope_range_q0 + flip + rope_range_begin,
            idx_hid_q0,
        )

    if HID_BLOCK_1 > 0:
        idx_hid_q1 = HID_BLOCK_0 + tl.arange(0, HID_BLOCK_1)
        rope_mask_1 = (rope_range_begin <= idx_hid_q1) & (idx_hid_q1 < rope_range_end)
        idx_rope_range_q1 = idx_hid_q1 - rope_range_begin
        if rope_is_neox_style:
            rope_rot_idx_1 = tl.where(
                rope_mask_1,
                (idx_hid_q1 - rope_range_begin + ROPE_DIM // 2) % ROPE_DIM
                + rope_range_begin,
                idx_hid_q1,
            )
        else:
            flip = tl.where(idx_rope_range_q1 % 2 == 0, 1, -1)
            rope_rot_idx_1 = tl.where(
                rope_mask_1,
                idx_rope_range_q1 + flip + rope_range_begin,
                idx_hid_q1,
            )
    else:
        idx_hid_q1 = None
        rope_rot_idx_1 = None

    idx_hid_v = dim_v_offset + tl.arange(0, HID_BLOCK_V)

    if BLOCK_SIZE_Q < 16:
        acc = tl.zeros((16, HID_BLOCK_V), dtype=tl.float32)
        m_i = tl.full((16, 1), -float("inf"), dtype=tl.float32)
        l_i = tl.full((16, 1), 1.0, dtype=tl.float32)
    else:
        acc = tl.zeros((BLOCK_SIZE_Q, HID_BLOCK_V), dtype=tl.float32)
        m_i = tl.full((BLOCK_SIZE_Q, 1), -float("inf"), dtype=tl.float32)
        l_i = tl.full((BLOCK_SIZE_Q, 1), 1.0, dtype=tl.float32)

    if K_DESCALE is not None:
        k_descale = tl.load(
            K_DESCALE +
            idx_bsz.to(tl.int64) * (HEAD // KV_HEAD_REPEAT) +
            (idx_head // KV_HEAD_REPEAT).to(tl.int64),
        )
        v_descale = tl.load(
            V_DESCALE +
            idx_bsz.to(tl.int64) * (HEAD // KV_HEAD_REPEAT) +
            (idx_head // KV_HEAD_REPEAT).to(tl.int64),
        )
    else:
        k_descale = None
        v_descale = None

    range_start = tl.load(
        KS_START_END
        + idx_b.to(tl.int64) * stride_ks_start_end_b
        + idx_bdst.to(tl.int64) * stride_ks_start_end_bdst
        + idx_g.to(tl.int64) * stride_ks_start_end_g
    )
    range_end = tl.load(
        KS_START_END
        + idx_b.to(tl.int64) * stride_ks_start_end_b
        + idx_bdst.to(tl.int64) * stride_ks_start_end_bdst
        + (idx_g + 1).to(tl.int64) * stride_ks_start_end_g
    )
    if BK <= 0:
        range_start = 0
        range_end = 0

    queries_0 = tl.load(
        Q
        + idx_bsz.to(tl.int64) * stride_q_bsz
        + idx_tdst[:, None].to(tl.int64) * stride_q_tdst
        + idx_head.to(tl.int64) * stride_q_head
        + idx_hid_q0[None, :].to(tl.int64) * stride_q_hid,
        mask=mask_tdst[:, None] & (idx_hid_q0[None, :] < HID),
        other=0.0,
    )
    if queries_0.dtype == tl.float8e5:
        queries_0 = queries_0.to(tl.float16)

    if HID_BLOCK_1 > 0:
        queries_1 = tl.load(
            Q
            + idx_bsz.to(tl.int64) * stride_q_bsz
            + idx_tdst[:, None].to(tl.int64) * stride_q_tdst
            + idx_head.to(tl.int64) * stride_q_head
            + idx_hid_q1[None, :].to(tl.int64) * stride_q_hid,
            mask=mask_tdst[:, None] & (idx_hid_q1[None, :] < HID),
            other=0.0,
        )
        if queries_1.dtype == tl.float8e5:
            queries_1 = queries_1.to(tl.float16)
    else:
        queries_1 = None

    if USING_EXTEND and NEED_APPLY_ROPE:
        if EXTEND_BACKEND == "streaming":
            rope_tdst = pos_tdst - 1
            activate_len = sink_token_size + sliding_window_size + BK * BLOCK_SIZE_K
            max_seq_len = tl.max(pos_tdst * mask_tdst)
            rope_tdst = rope_tdst - max_seq_len + activate_len
            rope_tdst = tl.minimum(tl.maximum(0, rope_tdst), model_context_length)
        else:
            rope_tdst = pos_tdst - 1

        if rope_range_begin < HID_BLOCK_0:
            queries_0 = apply_rope_to_queries(
                queries_0,
                pos_tdst,
                rope_tdst,
                idx_hid_q0,
                idx_bsz,
                idx_tdst,
                mask_tdst,
                idx_head,
                Q,
                stride_q_bsz,
                stride_q_tdst,
                stride_q_head,
                stride_q_hid,
                COS,
                stride_cos_t,
                stride_cos_hid,
                SIN,
                stride_sin_t,
                stride_sin_hid,
                rope_range_begin,
                rope_range_end,
                rope_is_neox_style,
            )

        if HID_BLOCK_1 > 0:
            queries_1 = apply_rope_to_queries(
                queries_1,
                pos_tdst,
                rope_tdst,
                idx_hid_q1,
                idx_bsz,
                idx_tdst,
                mask_tdst,
                idx_head,
                Q,
                stride_q_bsz,
                stride_q_tdst,
                stride_q_head,
                stride_q_hid,
                COS,
                stride_cos_t,
                stride_cos_hid,
                SIN,
                stride_sin_t,
                stride_sin_hid,
                rope_range_begin,
                rope_range_end,
                rope_is_neox_style,
            )

    # 6ms
    if (sink_token_size > 0) and True:
        CURR_TSRC = tl.max(pos_tdst)
        for i_tsrc in tl.range(
            0, sink_token_size, BLOCK_BK * BLOCK_SIZE_K, num_stages=1
        ):
            idx_tsrc = i_tsrc + tl.arange(0, BLOCK_BK * BLOCK_SIZE_K)
            mask_tsrc = idx_tsrc < tl.minimum(CURR_TSRC, sink_token_size)

            # idx_n = idx_b * G + idx_group
            keys_0 = load_tokens(
                K,
                stride_k_bsz,
                stride_k_tsrc,
                stride_k_head,
                stride_k_hid,
                USING_PAGES,
                PAGE_SIZE,
                K_CACHE,
                stride_k_cache_page,
                stride_k_cache_offset,
                stride_k_cache_kv_head,
                stride_k_cache_hid,
                BLOCK_TABLE,
                stride_block_table_bsz,
                stride_block_table_page,
                CACHE_SEQ_LENS,
                stride_cache_seq_lens_b,
                USING_OFFLOAD_CACHE,
                OFFLOAD_CACHE_KV_PACKED,
                GPU_BANK_COUNT,
                False,
                OFFLOAD_CACHE_UVM_METADATA,
                stride_offload_cache_uvm_metadata_token,
                stride_offload_cache_uvm_metadata_k,
                OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                stride_offload_cache_gpu_global_metadata_k,
                stride_offload_cache_gpu_global_metadata_pad,
                OFFLOAD_CACHE_GPU_BANK,
                stride_offload_cache_gpu_bank_token,
                stride_offload_cache_gpu_bank_hid,
                OFFLOAD_CACHE_GPU_METADATA,
                stride_offload_cache_gpu_metadata_token,
                stride_offload_cache_gpu_metadata_k,
                OFFLOAD_CACHE_GPU_TABLE,
                stride_offload_cache_gpu_table_head_kv,
                stride_offload_cache_gpu_table_token,
                strdie_offload_cache_gpu_table_k,
                ACCESS_COUNTER,
                stride_access_counter_bsz,
                stride_access_counter_head_kv,
                stride_access_counter_tsrc,
                CACHE_MISS_COUNTER,
                stride_cache_miss_counter_bsz,
                stride_cache_miss_counter_head_kv,
                stride_cache_miss_counter_tsrc,
                idx_bsz,
                idx_tsrc[None, :],
                idx_head // KV_HEAD_REPEAT,
                idx_hid_q0[:, None],
                mask_tsrc[None, :],
                HEAD // KV_HEAD_REPEAT,
                BLOCK_BK * BLOCK_SIZE_K,
                HID_BLOCK_0,
                HID,
                IS_BSA=True,
                UPDATE_CACHE=UPDATE_CACHE,
                V_CACHE=V_CACHE,
                stride_v_cache_page=stride_v_cache_page,
                stride_v_cache_offset=stride_v_cache_offset,
                stride_v_cache_kv_head=stride_v_cache_kv_head,
                stride_v_cache_hid=stride_v_cache_hid,
            )

            if HID_BLOCK_1 > 0:
                keys_1 = load_tokens(
                    K,
                    stride_k_bsz,
                    stride_k_tsrc,
                    stride_k_head,
                    stride_k_hid,
                    USING_PAGES,
                    PAGE_SIZE,
                    K_CACHE,
                    stride_k_cache_page,
                    stride_k_cache_offset,
                    stride_k_cache_kv_head,
                    stride_k_cache_hid,
                    BLOCK_TABLE,
                    stride_block_table_bsz,
                    stride_block_table_page,
                    CACHE_SEQ_LENS,
                    stride_cache_seq_lens_b,
                    USING_OFFLOAD_CACHE,
                    OFFLOAD_CACHE_KV_PACKED,
                    GPU_BANK_COUNT,
                    False,
                    OFFLOAD_CACHE_UVM_METADATA,
                    stride_offload_cache_uvm_metadata_token,
                    stride_offload_cache_uvm_metadata_k,
                    OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                    stride_offload_cache_gpu_global_metadata_k,
                    stride_offload_cache_gpu_global_metadata_pad,
                    OFFLOAD_CACHE_GPU_BANK,
                    stride_offload_cache_gpu_bank_token,
                    stride_offload_cache_gpu_bank_hid,
                    OFFLOAD_CACHE_GPU_METADATA,
                    stride_offload_cache_gpu_metadata_token,
                    stride_offload_cache_gpu_metadata_k,
                    OFFLOAD_CACHE_GPU_TABLE,
                    stride_offload_cache_gpu_table_head_kv,
                    stride_offload_cache_gpu_table_token,
                    strdie_offload_cache_gpu_table_k,
                    ACCESS_COUNTER,
                    stride_access_counter_bsz,
                    stride_access_counter_head_kv,
                    stride_access_counter_tsrc,
                    CACHE_MISS_COUNTER,
                    stride_cache_miss_counter_bsz,
                    stride_cache_miss_counter_head_kv,
                    stride_cache_miss_counter_tsrc,
                    idx_bsz,
                    idx_tsrc[None, :],
                    idx_head // KV_HEAD_REPEAT,
                    idx_hid_q1[:, None],
                    mask_tsrc[None, :],
                    HEAD // KV_HEAD_REPEAT,
                    BLOCK_BK * BLOCK_SIZE_K,
                    HID_BLOCK_1,
                    HID,
                    IS_BSA=True,
                    UPDATE_CACHE=UPDATE_CACHE,
                    V_CACHE=V_CACHE,
                    stride_v_cache_page=stride_v_cache_page,
                    stride_v_cache_offset=stride_v_cache_offset,
                    stride_v_cache_kv_head=stride_v_cache_kv_head,
                    stride_v_cache_hid=stride_v_cache_hid,
                )
            else:
                keys_1 = None

            if USING_EXTEND and NEED_APPLY_ROPE:
                if rope_range_begin < HID_BLOCK_0:
                    keys_rot_0 = load_tokens(
                        K,
                        stride_k_bsz,
                        stride_k_tsrc,
                        stride_k_head,
                        stride_k_hid,
                        USING_PAGES,
                        PAGE_SIZE,
                        K_CACHE,
                        stride_k_cache_page,
                        stride_k_cache_offset,
                        stride_k_cache_kv_head,
                        stride_k_cache_hid,
                        BLOCK_TABLE,
                        stride_block_table_bsz,
                        stride_block_table_page,
                        CACHE_SEQ_LENS,
                        stride_cache_seq_lens_b,
                        USING_OFFLOAD_CACHE,
                        OFFLOAD_CACHE_KV_PACKED,
                        GPU_BANK_COUNT,
                        False,
                        OFFLOAD_CACHE_UVM_METADATA,
                        stride_offload_cache_uvm_metadata_token,
                        stride_offload_cache_uvm_metadata_k,
                        OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                        stride_offload_cache_gpu_global_metadata_k,
                        stride_offload_cache_gpu_global_metadata_pad,
                        OFFLOAD_CACHE_GPU_BANK,
                        stride_offload_cache_gpu_bank_token,
                        stride_offload_cache_gpu_bank_hid,
                        OFFLOAD_CACHE_GPU_METADATA,
                        stride_offload_cache_gpu_metadata_token,
                        stride_offload_cache_gpu_metadata_k,
                        OFFLOAD_CACHE_GPU_TABLE,
                        stride_offload_cache_gpu_table_head_kv,
                        stride_offload_cache_gpu_table_token,
                        strdie_offload_cache_gpu_table_k,
                        ACCESS_COUNTER,
                        stride_access_counter_bsz,
                        stride_access_counter_head_kv,
                        stride_access_counter_tsrc,
                        CACHE_MISS_COUNTER,
                        stride_cache_miss_counter_bsz,
                        stride_cache_miss_counter_head_kv,
                        stride_cache_miss_counter_tsrc,
                        idx_bsz,
                        idx_tsrc[None, :],
                        idx_head // KV_HEAD_REPEAT,
                        rope_rot_idx_0[:, None],
                        mask_tsrc[None, :],
                        HEAD // KV_HEAD_REPEAT,
                        BLOCK_BK * BLOCK_SIZE_K,
                        HID_BLOCK_0,
                        HID,
                        IS_BSA=True,
                        UPDATE_CACHE=UPDATE_CACHE,
                        V_CACHE=V_CACHE,
                        stride_v_cache_page=stride_v_cache_page,
                        stride_v_cache_offset=stride_v_cache_offset,
                        stride_v_cache_kv_head=stride_v_cache_kv_head,
                        stride_v_cache_hid=stride_v_cache_hid,
                    )
                else:
                    keys_rot_0 = None

                if HID_BLOCK_1 > 0:
                    keys_rot_1 = load_tokens(
                        K,
                        stride_k_bsz,
                        stride_k_tsrc,
                        stride_k_head,
                        stride_k_hid,
                        USING_PAGES,
                        PAGE_SIZE,
                        K_CACHE,
                        stride_k_cache_page,
                        stride_k_cache_offset,
                        stride_k_cache_kv_head,
                        stride_k_cache_hid,
                        BLOCK_TABLE,
                        stride_block_table_bsz,
                        stride_block_table_page,
                        CACHE_SEQ_LENS,
                        stride_cache_seq_lens_b,
                        USING_OFFLOAD_CACHE,
                        OFFLOAD_CACHE_KV_PACKED,
                        GPU_BANK_COUNT,
                        False,
                        OFFLOAD_CACHE_UVM_METADATA,
                        stride_offload_cache_uvm_metadata_token,
                        stride_offload_cache_uvm_metadata_k,
                        OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                        stride_offload_cache_gpu_global_metadata_k,
                        stride_offload_cache_gpu_global_metadata_pad,
                        OFFLOAD_CACHE_GPU_BANK,
                        stride_offload_cache_gpu_bank_token,
                        stride_offload_cache_gpu_bank_hid,
                        OFFLOAD_CACHE_GPU_METADATA,
                        stride_offload_cache_gpu_metadata_token,
                        stride_offload_cache_gpu_metadata_k,
                        OFFLOAD_CACHE_GPU_TABLE,
                        stride_offload_cache_gpu_table_head_kv,
                        stride_offload_cache_gpu_table_token,
                        strdie_offload_cache_gpu_table_k,
                        ACCESS_COUNTER,
                        stride_access_counter_bsz,
                        stride_access_counter_head_kv,
                        stride_access_counter_tsrc,
                        CACHE_MISS_COUNTER,
                        stride_cache_miss_counter_bsz,
                        stride_cache_miss_counter_head_kv,
                        stride_cache_miss_counter_tsrc,
                        idx_bsz,
                        idx_tsrc[None, :],
                        idx_head // KV_HEAD_REPEAT,
                        rope_rot_idx_1[:, None],
                        mask_tsrc[None, :],
                        HEAD // KV_HEAD_REPEAT,
                        BLOCK_BK * BLOCK_SIZE_K,
                        HID_BLOCK_1,
                        HID,
                        IS_BSA=True,
                        UPDATE_CACHE=UPDATE_CACHE,
                        V_CACHE=V_CACHE,
                        stride_v_cache_page=stride_v_cache_page,
                        stride_v_cache_offset=stride_v_cache_offset,
                        stride_v_cache_kv_head=stride_v_cache_kv_head,
                        stride_v_cache_hid=stride_v_cache_hid,
                    )
                else:
                    keys_rot_1 = None
            else:
                keys_rot_0 = None
                keys_rot_1 = None

            if k_descale is not None:
                keys_0 *= k_descale
                keys_rot_0 *= k_descale
                if keys_1 is not None:
                    keys_1 *= k_descale
                    keys_rot_1 *= k_descale

            values = load_tokens(
                V,
                stride_v_bsz,
                stride_v_tsrc,
                stride_v_head,
                stride_v_hid,
                USING_PAGES,
                PAGE_SIZE,
                V_CACHE,
                stride_v_cache_page,
                stride_v_cache_offset,
                stride_v_cache_kv_head,
                stride_v_cache_hid,
                BLOCK_TABLE,
                stride_block_table_bsz,
                stride_block_table_page,
                CACHE_SEQ_LENS,
                stride_cache_seq_lens_b,
                USING_OFFLOAD_CACHE,
                OFFLOAD_CACHE_KV_PACKED,
                GPU_BANK_COUNT,
                True,
                OFFLOAD_CACHE_UVM_METADATA,
                stride_offload_cache_uvm_metadata_token,
                stride_offload_cache_uvm_metadata_k,
                OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                stride_offload_cache_gpu_global_metadata_k,
                stride_offload_cache_gpu_global_metadata_pad,
                OFFLOAD_CACHE_GPU_BANK,
                stride_offload_cache_gpu_bank_token,
                stride_offload_cache_gpu_bank_hid,
                OFFLOAD_CACHE_GPU_METADATA,
                stride_offload_cache_gpu_metadata_token,
                stride_offload_cache_gpu_metadata_k,
                OFFLOAD_CACHE_GPU_TABLE,
                stride_offload_cache_gpu_table_head_kv,
                stride_offload_cache_gpu_table_token,
                strdie_offload_cache_gpu_table_k,
                ACCESS_COUNTER,
                stride_access_counter_bsz,
                stride_access_counter_head_kv,
                stride_access_counter_tsrc,
                CACHE_MISS_COUNTER,
                stride_cache_miss_counter_bsz,
                stride_cache_miss_counter_head_kv,
                stride_cache_miss_counter_tsrc,
                idx_bsz,
                idx_tsrc[:, None],
                idx_head // KV_HEAD_REPEAT,
                idx_hid_v[None, :],
                mask_tsrc[:, None],
                HEAD // KV_HEAD_REPEAT,
                BLOCK_BK * BLOCK_SIZE_K,
                HID_BLOCK_V,
                HID_V,
                IS_BSA=True,
                UPDATE_CACHE=UPDATE_CACHE,
                V_CACHE=K_CACHE,
                stride_v_cache_page=stride_k_cache_page,
                stride_v_cache_offset=stride_k_cache_offset,
                stride_v_cache_kv_head=stride_k_cache_kv_head,
                stride_v_cache_hid=stride_k_cache_hid,
            )

            if v_descale is not None:
                value *= v_descale

            acc, l_i, m_i = block_sparse_attention_cuda_step(
                queries_0,
                queries_1,
                keys_0,
                keys_1,
                keys_rot_0,
                keys_rot_1,
                values,
                idx_tsrc,
                mask_tsrc,
                idx_tdst,
                mask_tdst,
                acc,
                l_i,
                m_i,
                sliding_window_size,
                sink_token_size,
                sparse_token_size,
                (range_end - range_start) * BLOCK_SIZE_K,
                True,
                True,
                LOGIT_SOFTCAP,
                USING_EXTEND,
                NEED_APPLY_ROPE,
                COS,
                stride_cos_t,
                stride_cos_hid,
                SIN,
                stride_sin_t,
                stride_sin_hid,
                rope_range_begin,
                rope_range_end,
                rope_is_neox_style,
                model_context_length,
                # idx_rope,
                # tl.arange(0, BLOCK_BK) + i_tsrc // BLOCK_SIZE_K,
                idx_tsrc,
                pos_tdst,
                idx_hid_q0,
                idx_hid_q1,
                IS_CAUSAL,
                HID,
                BLOCK_SIZE_Q,
                BLOCK_BK * BLOCK_SIZE_K,
                BLOCK_SIZE_K,
                EXTEND_BACKEND=EXTEND_BACKEND,
            )

    # 29ms
    if (sliding_window_size > 0) and True:
        CURR_TSRC = tl.max(pos_tdst)
        # CURR_TSRC = (idx_bdst + 1) * BLOCK_SIZE_Q + MAX_TSRC - MAX_TDST
        i_tsrc_range_start = tl.maximum(
            0, CURR_TSRC - sliding_window_size - BLOCK_SIZE_Q
        )
        i_tsrc_range_start = i_tsrc_range_start // BLOCK_SIZE_K * BLOCK_SIZE_K
        i_tsrc_range_start_real = i_tsrc_range_start
        if not CHUNKED_SW:
            i_tsrc_range_start_real = i_tsrc_range_start
        else:
            i_tsrc_range_start_real = tl.maximum(
                i_tsrc_range_start,
                (CURR_TSRC - 1) // sliding_window_size * sliding_window_size
                - BLOCK_SIZE_Q,
            )

        TSRC_RANGE_STEP: tl.constexpr = BLOCK_BK * BLOCK_SIZE_K
        for i_tsrc in tl.range(
            i_tsrc_range_start_real, CURR_TSRC, TSRC_RANGE_STEP, num_stages=1
        ):
            idx_tsrc = i_tsrc + tl.arange(0, BLOCK_BK * BLOCK_SIZE_K)
            mask_tsrc = idx_tsrc < CURR_TSRC

            # idx_n = idx_b * G + idx_group
            keys_0 = load_tokens(
                K,
                stride_k_bsz,
                stride_k_tsrc,
                stride_k_head,
                stride_k_hid,
                USING_PAGES,
                PAGE_SIZE,
                K_CACHE,
                stride_k_cache_page,
                stride_k_cache_offset,
                stride_k_cache_kv_head,
                stride_k_cache_hid,
                BLOCK_TABLE,
                stride_block_table_bsz,
                stride_block_table_page,
                CACHE_SEQ_LENS,
                stride_cache_seq_lens_b,
                USING_OFFLOAD_CACHE,
                OFFLOAD_CACHE_KV_PACKED,
                GPU_BANK_COUNT,
                False,
                OFFLOAD_CACHE_UVM_METADATA,
                stride_offload_cache_uvm_metadata_token,
                stride_offload_cache_uvm_metadata_k,
                OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                stride_offload_cache_gpu_global_metadata_k,
                stride_offload_cache_gpu_global_metadata_pad,
                OFFLOAD_CACHE_GPU_BANK,
                stride_offload_cache_gpu_bank_token,
                stride_offload_cache_gpu_bank_hid,
                OFFLOAD_CACHE_GPU_METADATA,
                stride_offload_cache_gpu_metadata_token,
                stride_offload_cache_gpu_metadata_k,
                OFFLOAD_CACHE_GPU_TABLE,
                stride_offload_cache_gpu_table_head_kv,
                stride_offload_cache_gpu_table_token,
                strdie_offload_cache_gpu_table_k,
                ACCESS_COUNTER,
                stride_access_counter_bsz,
                stride_access_counter_head_kv,
                stride_access_counter_tsrc,
                CACHE_MISS_COUNTER,
                stride_cache_miss_counter_bsz,
                stride_cache_miss_counter_head_kv,
                stride_cache_miss_counter_tsrc,
                idx_bsz,
                idx_tsrc[None, :],
                idx_head // KV_HEAD_REPEAT,
                idx_hid_q0[:, None],
                mask_tsrc[None, :],
                HEAD // KV_HEAD_REPEAT,
                BLOCK_BK * BLOCK_SIZE_K,
                HID_BLOCK_0,
                HID,
                IS_BSA=True,
                UPDATE_CACHE=UPDATE_CACHE,
                V_CACHE=V_CACHE,
                stride_v_cache_page=stride_v_cache_page,
                stride_v_cache_offset=stride_v_cache_offset,
                stride_v_cache_kv_head=stride_v_cache_kv_head,
                stride_v_cache_hid=stride_v_cache_hid,
            )

            if HID_BLOCK_1 > 0:
                keys_1 = load_tokens(
                    K,
                    stride_k_bsz,
                    stride_k_tsrc,
                    stride_k_head,
                    stride_k_hid,
                    USING_PAGES,
                    PAGE_SIZE,
                    K_CACHE,
                    stride_k_cache_page,
                    stride_k_cache_offset,
                    stride_k_cache_kv_head,
                    stride_k_cache_hid,
                    BLOCK_TABLE,
                    stride_block_table_bsz,
                    stride_block_table_page,
                    CACHE_SEQ_LENS,
                    stride_cache_seq_lens_b,
                    USING_OFFLOAD_CACHE,
                    OFFLOAD_CACHE_KV_PACKED,
                    GPU_BANK_COUNT,
                    False,
                    OFFLOAD_CACHE_UVM_METADATA,
                    stride_offload_cache_uvm_metadata_token,
                    stride_offload_cache_uvm_metadata_k,
                    OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                    stride_offload_cache_gpu_global_metadata_k,
                    stride_offload_cache_gpu_global_metadata_pad,
                    OFFLOAD_CACHE_GPU_BANK,
                    stride_offload_cache_gpu_bank_token,
                    stride_offload_cache_gpu_bank_hid,
                    OFFLOAD_CACHE_GPU_METADATA,
                    stride_offload_cache_gpu_metadata_token,
                    stride_offload_cache_gpu_metadata_k,
                    OFFLOAD_CACHE_GPU_TABLE,
                    stride_offload_cache_gpu_table_head_kv,
                    stride_offload_cache_gpu_table_token,
                    strdie_offload_cache_gpu_table_k,
                    ACCESS_COUNTER,
                    stride_access_counter_bsz,
                    stride_access_counter_head_kv,
                    stride_access_counter_tsrc,
                    CACHE_MISS_COUNTER,
                    stride_cache_miss_counter_bsz,
                    stride_cache_miss_counter_head_kv,
                    stride_cache_miss_counter_tsrc,
                    idx_bsz,
                    idx_tsrc[None, :],
                    idx_head // KV_HEAD_REPEAT,
                    idx_hid_q1[:, None],
                    mask_tsrc[None, :],
                    HEAD // KV_HEAD_REPEAT,
                    BLOCK_BK * BLOCK_SIZE_K,
                    HID_BLOCK_1,
                    HID,
                    IS_BSA=True,
                    UPDATE_CACHE=UPDATE_CACHE,
                    V_CACHE=V_CACHE,
                    stride_v_cache_page=stride_v_cache_page,
                    stride_v_cache_offset=stride_v_cache_offset,
                    stride_v_cache_kv_head=stride_v_cache_kv_head,
                    stride_v_cache_hid=stride_v_cache_hid,
                )
            else:
                keys_1 = None

            if USING_EXTEND and NEED_APPLY_ROPE:
                if rope_range_begin < HID_BLOCK_0:
                    keys_rot_0 = load_tokens(
                        K,
                        stride_k_bsz,
                        stride_k_tsrc,
                        stride_k_head,
                        stride_k_hid,
                        USING_PAGES,
                        PAGE_SIZE,
                        K_CACHE,
                        stride_k_cache_page,
                        stride_k_cache_offset,
                        stride_k_cache_kv_head,
                        stride_k_cache_hid,
                        BLOCK_TABLE,
                        stride_block_table_bsz,
                        stride_block_table_page,
                        CACHE_SEQ_LENS,
                        stride_cache_seq_lens_b,
                        USING_OFFLOAD_CACHE,
                        OFFLOAD_CACHE_KV_PACKED,
                        GPU_BANK_COUNT,
                        False,
                        OFFLOAD_CACHE_UVM_METADATA,
                        stride_offload_cache_uvm_metadata_token,
                        stride_offload_cache_uvm_metadata_k,
                        OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                        stride_offload_cache_gpu_global_metadata_k,
                        stride_offload_cache_gpu_global_metadata_pad,
                        OFFLOAD_CACHE_GPU_BANK,
                        stride_offload_cache_gpu_bank_token,
                        stride_offload_cache_gpu_bank_hid,
                        OFFLOAD_CACHE_GPU_METADATA,
                        stride_offload_cache_gpu_metadata_token,
                        stride_offload_cache_gpu_metadata_k,
                        OFFLOAD_CACHE_GPU_TABLE,
                        stride_offload_cache_gpu_table_head_kv,
                        stride_offload_cache_gpu_table_token,
                        strdie_offload_cache_gpu_table_k,
                        ACCESS_COUNTER,
                        stride_access_counter_bsz,
                        stride_access_counter_head_kv,
                        stride_access_counter_tsrc,
                        CACHE_MISS_COUNTER,
                        stride_cache_miss_counter_bsz,
                        stride_cache_miss_counter_head_kv,
                        stride_cache_miss_counter_tsrc,
                        idx_bsz,
                        idx_tsrc[None, :],
                        idx_head // KV_HEAD_REPEAT,
                        rope_rot_idx_0[:, None],
                        mask_tsrc[None, :],
                        HEAD // KV_HEAD_REPEAT,
                        BLOCK_BK * BLOCK_SIZE_K,
                        HID_BLOCK_0,
                        HID,
                        IS_BSA=True,
                        UPDATE_CACHE=UPDATE_CACHE,
                        V_CACHE=V_CACHE,
                        stride_v_cache_page=stride_v_cache_page,
                        stride_v_cache_offset=stride_v_cache_offset,
                        stride_v_cache_kv_head=stride_v_cache_kv_head,
                        stride_v_cache_hid=stride_v_cache_hid,
                    )
                else:
                    keys_rot_0 = None

                if HID_BLOCK_1 > 0:
                    keys_rot_1 = load_tokens(
                        K,
                        stride_k_bsz,
                        stride_k_tsrc,
                        stride_k_head,
                        stride_k_hid,
                        USING_PAGES,
                        PAGE_SIZE,
                        K_CACHE,
                        stride_k_cache_page,
                        stride_k_cache_offset,
                        stride_k_cache_kv_head,
                        stride_k_cache_hid,
                        BLOCK_TABLE,
                        stride_block_table_bsz,
                        stride_block_table_page,
                        CACHE_SEQ_LENS,
                        stride_cache_seq_lens_b,
                        USING_OFFLOAD_CACHE,
                        OFFLOAD_CACHE_KV_PACKED,
                        GPU_BANK_COUNT,
                        False,
                        OFFLOAD_CACHE_UVM_METADATA,
                        stride_offload_cache_uvm_metadata_token,
                        stride_offload_cache_uvm_metadata_k,
                        OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                        stride_offload_cache_gpu_global_metadata_k,
                        stride_offload_cache_gpu_global_metadata_pad,
                        OFFLOAD_CACHE_GPU_BANK,
                        stride_offload_cache_gpu_bank_token,
                        stride_offload_cache_gpu_bank_hid,
                        OFFLOAD_CACHE_GPU_METADATA,
                        stride_offload_cache_gpu_metadata_token,
                        stride_offload_cache_gpu_metadata_k,
                        OFFLOAD_CACHE_GPU_TABLE,
                        stride_offload_cache_gpu_table_head_kv,
                        stride_offload_cache_gpu_table_token,
                        strdie_offload_cache_gpu_table_k,
                        ACCESS_COUNTER,
                        stride_access_counter_bsz,
                        stride_access_counter_head_kv,
                        stride_access_counter_tsrc,
                        CACHE_MISS_COUNTER,
                        stride_cache_miss_counter_bsz,
                        stride_cache_miss_counter_head_kv,
                        stride_cache_miss_counter_tsrc,
                        idx_bsz,
                        idx_tsrc[None, :],
                        idx_head // KV_HEAD_REPEAT,
                        rope_rot_idx_1[:, None],
                        mask_tsrc[None, :],
                        HEAD // KV_HEAD_REPEAT,
                        BLOCK_BK * BLOCK_SIZE_K,
                        HID_BLOCK_1,
                        HID,
                        IS_BSA=True,
                        UPDATE_CACHE=UPDATE_CACHE,
                        V_CACHE=V_CACHE,
                        stride_v_cache_page=stride_v_cache_page,
                        stride_v_cache_offset=stride_v_cache_offset,
                        stride_v_cache_kv_head=stride_v_cache_kv_head,
                        stride_v_cache_hid=stride_v_cache_hid,
                    )
                else:
                    keys_rot_1 = None
            else:
                keys_rot_0 = None
                keys_rot_1 = None

            if k_descale is not None:
                keys_0 *= k_descale
                keys_rot_0 *= k_descale
                if keys_1 is not None:
                    keys_1 *= k_descale
                    keys_rot_1 *= k_descale

            values = load_tokens(
                V,
                stride_v_bsz,
                stride_v_tsrc,
                stride_v_head,
                stride_v_hid,
                USING_PAGES,
                PAGE_SIZE,
                V_CACHE,
                stride_v_cache_page,
                stride_v_cache_offset,
                stride_v_cache_kv_head,
                stride_v_cache_hid,
                BLOCK_TABLE,
                stride_block_table_bsz,
                stride_block_table_page,
                CACHE_SEQ_LENS,
                stride_cache_seq_lens_b,
                USING_OFFLOAD_CACHE,
                OFFLOAD_CACHE_KV_PACKED,
                GPU_BANK_COUNT,
                True,
                OFFLOAD_CACHE_UVM_METADATA,
                stride_offload_cache_uvm_metadata_token,
                stride_offload_cache_uvm_metadata_k,
                OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                stride_offload_cache_gpu_global_metadata_k,
                stride_offload_cache_gpu_global_metadata_pad,
                OFFLOAD_CACHE_GPU_BANK,
                stride_offload_cache_gpu_bank_token,
                stride_offload_cache_gpu_bank_hid,
                OFFLOAD_CACHE_GPU_METADATA,
                stride_offload_cache_gpu_metadata_token,
                stride_offload_cache_gpu_metadata_k,
                OFFLOAD_CACHE_GPU_TABLE,
                stride_offload_cache_gpu_table_head_kv,
                stride_offload_cache_gpu_table_token,
                strdie_offload_cache_gpu_table_k,
                ACCESS_COUNTER,
                stride_access_counter_bsz,
                stride_access_counter_head_kv,
                stride_access_counter_tsrc,
                CACHE_MISS_COUNTER,
                stride_cache_miss_counter_bsz,
                stride_cache_miss_counter_head_kv,
                stride_cache_miss_counter_tsrc,
                idx_bsz,
                idx_tsrc[:, None],
                idx_head // KV_HEAD_REPEAT,
                idx_hid_v[None, :],
                mask_tsrc[:, None],
                HEAD // KV_HEAD_REPEAT,
                BLOCK_BK * BLOCK_SIZE_K,
                HID_BLOCK_V,
                HID_V,
                IS_BSA=True,
                UPDATE_CACHE=UPDATE_CACHE,
                V_CACHE=K_CACHE,
                stride_v_cache_page=stride_k_cache_page,
                stride_v_cache_offset=stride_k_cache_offset,
                stride_v_cache_kv_head=stride_k_cache_kv_head,
                stride_v_cache_hid=stride_k_cache_hid,
            )

            if v_descale is not None:
                value *= v_descale

            acc, l_i, m_i = block_sparse_attention_cuda_step(
                queries_0,
                queries_1,
                keys_0,
                keys_1,
                keys_rot_0,
                keys_rot_1,
                values,
                idx_tsrc,
                mask_tsrc,
                idx_tdst,
                mask_tdst,
                acc,
                l_i,
                m_i,
                sliding_window_size,
                sink_token_size,
                sparse_token_size,
                (range_end - range_start) * BLOCK_SIZE_K,
                False,
                False,
                LOGIT_SOFTCAP,
                USING_EXTEND,
                NEED_APPLY_ROPE,
                COS,
                stride_cos_t,
                stride_cos_hid,
                SIN,
                stride_sin_t,
                stride_sin_hid,
                rope_range_begin,
                rope_range_end,
                rope_is_neox_style,
                model_context_length,
                # tl.arange(0, BLOCK_BK) +\
                #     (range_end - range_start) +\
                #     (sink_token_size // BLOCK_SIZE_K) +\
                #     (i_tsrc-i_tsrc_range_start) // BLOCK_SIZE_K,
                # tl.arange(0, BLOCK_BK)
                # + (i_tsrc - i_tsrc_range_start) // BLOCK_SIZE_K
                # + (
                #     tl.max(pos_tdst * mask_tdst)
                #     - tl.sum(mask_tdst.to(tl.int32))
                #     - sliding_window_size
                # )
                # // BLOCK_SIZE_K,
                idx_tsrc
                - (tl.max(mask_tdst * pos_tdst) - sliding_window_size)
                + sink_token_size
                + BK * BLOCK_SIZE_K,
                pos_tdst,
                idx_hid_q0,
                idx_hid_q1,
                IS_CAUSAL,
                HID,
                BLOCK_SIZE_Q,
                BLOCK_BK * BLOCK_SIZE_K,
                BLOCK_SIZE_K,
                EXTEND_BACKEND=EXTEND_BACKEND,
                CHUNKED_SW=CHUNKED_SW,
            )

    # 60ms
    if (BK > 0) and True:
        for i_bk in tl.range(
            range_start, range_start + (BK * G), BLOCK_BK, num_stages=1
        ):
            idx_bk = i_bk + tl.arange(0, BLOCK_BK)
            mask_bk = (idx_bk < (range_start + BK * G)) & (idx_bk < range_end)

            if i_bk < range_end:
                idx_tsrc_start = tl.load(
                    INDICES
                    + idx_b.to(tl.int64) * stride_indices_b
                    + idx_bdst.to(tl.int64) * stride_indices_bdst
                    + idx_bk.to(tl.int64) * stride_indices_bk,
                    mask=mask_bk,
                )
                idx_tsrc_start = tl.where(mask_bk, idx_tsrc_start, MAX_TSRC * G + 1)
                idx_tsrc = idx_tsrc_start[:, None] + tl.arange(0, BLOCK_SIZE_K)[None, :]
                idx_tsrc = tl.reshape(idx_tsrc, (BLOCK_BK * BLOCK_SIZE_K))
                mask_tsrc_from_bk = mask_bk[:, None] & tl.full(
                    (1, BLOCK_SIZE_K), 1, dtype=tl.int1
                )
                mask_tsrc_from_bk = tl.reshape(
                    mask_tsrc_from_bk, (BLOCK_BK * BLOCK_SIZE_K)
                )
                mask_tsrc = (
                    (idx_tsrc < (MAX_TSRC * (idx_g + 1)))
                    & (idx_tsrc >= (MAX_TSRC * idx_g))
                    & mask_tsrc_from_bk
                )
                idx_tsrc = idx_tsrc % MAX_TSRC
                mask_tsrc = (
                    mask_tsrc
                    & (idx_tsrc < tl.max(pos_tdst))
                    & (idx_tsrc >= sink_token_size)
                )

                keys_0 = load_tokens(
                    K,
                    stride_k_bsz,
                    stride_k_tsrc,
                    stride_k_head,
                    stride_k_hid,
                    USING_PAGES,
                    PAGE_SIZE,
                    K_CACHE,
                    stride_k_cache_page,
                    stride_k_cache_offset,
                    stride_k_cache_kv_head,
                    stride_k_cache_hid,
                    BLOCK_TABLE,
                    stride_block_table_bsz,
                    stride_block_table_page,
                    CACHE_SEQ_LENS,
                    stride_cache_seq_lens_b,
                    USING_OFFLOAD_CACHE,
                    OFFLOAD_CACHE_KV_PACKED,
                    GPU_BANK_COUNT,
                    False,
                    OFFLOAD_CACHE_UVM_METADATA,
                    stride_offload_cache_uvm_metadata_token,
                    stride_offload_cache_uvm_metadata_k,
                    OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                    stride_offload_cache_gpu_global_metadata_k,
                    stride_offload_cache_gpu_global_metadata_pad,
                    OFFLOAD_CACHE_GPU_BANK,
                    stride_offload_cache_gpu_bank_token,
                    stride_offload_cache_gpu_bank_hid,
                    OFFLOAD_CACHE_GPU_METADATA,
                    stride_offload_cache_gpu_metadata_token,
                    stride_offload_cache_gpu_metadata_k,
                    OFFLOAD_CACHE_GPU_TABLE,
                    stride_offload_cache_gpu_table_head_kv,
                    stride_offload_cache_gpu_table_token,
                    strdie_offload_cache_gpu_table_k,
                    ACCESS_COUNTER,
                    stride_access_counter_bsz,
                    stride_access_counter_head_kv,
                    stride_access_counter_tsrc,
                    CACHE_MISS_COUNTER,
                    stride_cache_miss_counter_bsz,
                    stride_cache_miss_counter_head_kv,
                    stride_cache_miss_counter_tsrc,
                    idx_bsz,
                    idx_tsrc[None, :],
                    idx_head // KV_HEAD_REPEAT,
                    idx_hid_q0[:, None],
                    mask_tsrc[None, :],
                    HEAD // KV_HEAD_REPEAT,
                    BLOCK_BK * BLOCK_SIZE_K,
                    HID_BLOCK_0,
                    HID,
                    IS_BSA=True,
                    UPDATE_CACHE=UPDATE_CACHE,
                    V_CACHE=V_CACHE,
                    stride_v_cache_page=stride_v_cache_page,
                    stride_v_cache_offset=stride_v_cache_offset,
                    stride_v_cache_kv_head=stride_v_cache_kv_head,
                    stride_v_cache_hid=stride_v_cache_hid,
                )

                if HID_BLOCK_1 > 0:
                    keys_1 = load_tokens(
                        K,
                        stride_k_bsz,
                        stride_k_tsrc,
                        stride_k_head,
                        stride_k_hid,
                        USING_PAGES,
                        PAGE_SIZE,
                        K_CACHE,
                        stride_k_cache_page,
                        stride_k_cache_offset,
                        stride_k_cache_kv_head,
                        stride_k_cache_hid,
                        BLOCK_TABLE,
                        stride_block_table_bsz,
                        stride_block_table_page,
                        CACHE_SEQ_LENS,
                        stride_cache_seq_lens_b,
                        USING_OFFLOAD_CACHE,
                        OFFLOAD_CACHE_KV_PACKED,
                        GPU_BANK_COUNT,
                        False,
                        OFFLOAD_CACHE_UVM_METADATA,
                        stride_offload_cache_uvm_metadata_token,
                        stride_offload_cache_uvm_metadata_k,
                        OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                        stride_offload_cache_gpu_global_metadata_k,
                        stride_offload_cache_gpu_global_metadata_pad,
                        OFFLOAD_CACHE_GPU_BANK,
                        stride_offload_cache_gpu_bank_token,
                        stride_offload_cache_gpu_bank_hid,
                        OFFLOAD_CACHE_GPU_METADATA,
                        stride_offload_cache_gpu_metadata_token,
                        stride_offload_cache_gpu_metadata_k,
                        OFFLOAD_CACHE_GPU_TABLE,
                        stride_offload_cache_gpu_table_head_kv,
                        stride_offload_cache_gpu_table_token,
                        strdie_offload_cache_gpu_table_k,
                        ACCESS_COUNTER,
                        stride_access_counter_bsz,
                        stride_access_counter_head_kv,
                        stride_access_counter_tsrc,
                        CACHE_MISS_COUNTER,
                        stride_cache_miss_counter_bsz,
                        stride_cache_miss_counter_head_kv,
                        stride_cache_miss_counter_tsrc,
                        idx_bsz,
                        idx_tsrc[None, :],
                        idx_head // KV_HEAD_REPEAT,
                        idx_hid_q1[:, None],
                        mask_tsrc[None, :],
                        HEAD // KV_HEAD_REPEAT,
                        BLOCK_BK * BLOCK_SIZE_K,
                        HID_BLOCK_1,
                        HID,
                        IS_BSA=True,
                        UPDATE_CACHE=UPDATE_CACHE,
                        V_CACHE=V_CACHE,
                        stride_v_cache_page=stride_v_cache_page,
                        stride_v_cache_offset=stride_v_cache_offset,
                        stride_v_cache_kv_head=stride_v_cache_kv_head,
                        stride_v_cache_hid=stride_v_cache_hid,
                    )
                else:
                    keys_1 = None

                if USING_EXTEND and NEED_APPLY_ROPE:
                    if rope_range_begin < HID_BLOCK_0:
                        keys_rot_0 = load_tokens(
                            K,
                            stride_k_bsz,
                            stride_k_tsrc,
                            stride_k_head,
                            stride_k_hid,
                            USING_PAGES,
                            PAGE_SIZE,
                            K_CACHE,
                            stride_k_cache_page,
                            stride_k_cache_offset,
                            stride_k_cache_kv_head,
                            stride_k_cache_hid,
                            BLOCK_TABLE,
                            stride_block_table_bsz,
                            stride_block_table_page,
                            CACHE_SEQ_LENS,
                            stride_cache_seq_lens_b,
                            USING_OFFLOAD_CACHE,
                            OFFLOAD_CACHE_KV_PACKED,
                            GPU_BANK_COUNT,
                            False,
                            OFFLOAD_CACHE_UVM_METADATA,
                            stride_offload_cache_uvm_metadata_token,
                            stride_offload_cache_uvm_metadata_k,
                            OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                            stride_offload_cache_gpu_global_metadata_k,
                            stride_offload_cache_gpu_global_metadata_pad,
                            OFFLOAD_CACHE_GPU_BANK,
                            stride_offload_cache_gpu_bank_token,
                            stride_offload_cache_gpu_bank_hid,
                            OFFLOAD_CACHE_GPU_METADATA,
                            stride_offload_cache_gpu_metadata_token,
                            stride_offload_cache_gpu_metadata_k,
                            OFFLOAD_CACHE_GPU_TABLE,
                            stride_offload_cache_gpu_table_head_kv,
                            stride_offload_cache_gpu_table_token,
                            strdie_offload_cache_gpu_table_k,
                            ACCESS_COUNTER,
                            stride_access_counter_bsz,
                            stride_access_counter_head_kv,
                            stride_access_counter_tsrc,
                            CACHE_MISS_COUNTER,
                            stride_cache_miss_counter_bsz,
                            stride_cache_miss_counter_head_kv,
                            stride_cache_miss_counter_tsrc,
                            idx_bsz,
                            idx_tsrc[None, :],
                            idx_head // KV_HEAD_REPEAT,
                            rope_rot_idx_0[:, None],
                            mask_tsrc[None, :],
                            HEAD // KV_HEAD_REPEAT,
                            BLOCK_BK * BLOCK_SIZE_K,
                            HID_BLOCK_0,
                            HID,
                            IS_BSA=True,
                            UPDATE_CACHE=UPDATE_CACHE,
                            V_CACHE=V_CACHE,
                            stride_v_cache_page=stride_v_cache_page,
                            stride_v_cache_offset=stride_v_cache_offset,
                            stride_v_cache_kv_head=stride_v_cache_kv_head,
                            stride_v_cache_hid=stride_v_cache_hid,
                        )
                    else:
                        keys_rot_0 = None

                    if HID_BLOCK_1 > 0:
                        keys_rot_1 = load_tokens(
                            K,
                            stride_k_bsz,
                            stride_k_tsrc,
                            stride_k_head,
                            stride_k_hid,
                            USING_PAGES,
                            PAGE_SIZE,
                            K_CACHE,
                            stride_k_cache_page,
                            stride_k_cache_offset,
                            stride_k_cache_kv_head,
                            stride_k_cache_hid,
                            BLOCK_TABLE,
                            stride_block_table_bsz,
                            stride_block_table_page,
                            CACHE_SEQ_LENS,
                            stride_cache_seq_lens_b,
                            USING_OFFLOAD_CACHE,
                            OFFLOAD_CACHE_KV_PACKED,
                            GPU_BANK_COUNT,
                            False,
                            OFFLOAD_CACHE_UVM_METADATA,
                            stride_offload_cache_uvm_metadata_token,
                            stride_offload_cache_uvm_metadata_k,
                            OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                            stride_offload_cache_gpu_global_metadata_k,
                            stride_offload_cache_gpu_global_metadata_pad,
                            OFFLOAD_CACHE_GPU_BANK,
                            stride_offload_cache_gpu_bank_token,
                            stride_offload_cache_gpu_bank_hid,
                            OFFLOAD_CACHE_GPU_METADATA,
                            stride_offload_cache_gpu_metadata_token,
                            stride_offload_cache_gpu_metadata_k,
                            OFFLOAD_CACHE_GPU_TABLE,
                            stride_offload_cache_gpu_table_head_kv,
                            stride_offload_cache_gpu_table_token,
                            strdie_offload_cache_gpu_table_k,
                            ACCESS_COUNTER,
                            stride_access_counter_bsz,
                            stride_access_counter_head_kv,
                            stride_access_counter_tsrc,
                            CACHE_MISS_COUNTER,
                            stride_cache_miss_counter_bsz,
                            stride_cache_miss_counter_head_kv,
                            stride_cache_miss_counter_tsrc,
                            idx_bsz,
                            idx_tsrc[None, :],
                            idx_head // KV_HEAD_REPEAT,
                            rope_rot_idx_1[:, None],
                            mask_tsrc[None, :],
                            HEAD // KV_HEAD_REPEAT,
                            BLOCK_BK * BLOCK_SIZE_K,
                            HID_BLOCK_1,
                            HID,
                            IS_BSA=True,
                            UPDATE_CACHE=UPDATE_CACHE,
                            V_CACHE=V_CACHE,
                            stride_v_cache_page=stride_v_cache_page,
                            stride_v_cache_offset=stride_v_cache_offset,
                            stride_v_cache_kv_head=stride_v_cache_kv_head,
                            stride_v_cache_hid=stride_v_cache_hid,
                        )
                    else:
                        keys_rot_1 = None
                else:
                    keys_rot_0 = None
                    keys_rot_1 = None

                if k_descale is not None:
                    keys_0 *= k_descale
                    keys_rot_0 *= k_descale
                    if keys_1 is not None:
                        keys_1 *= k_descale
                        keys_rot_1 *= k_descale

                values = load_tokens(
                    V,
                    stride_v_bsz,
                    stride_v_tsrc,
                    stride_v_head,
                    stride_v_hid,
                    USING_PAGES,
                    PAGE_SIZE,
                    V_CACHE,
                    stride_v_cache_page,
                    stride_v_cache_offset,
                    stride_v_cache_kv_head,
                    stride_v_cache_hid,
                    BLOCK_TABLE,
                    stride_block_table_bsz,
                    stride_block_table_page,
                    CACHE_SEQ_LENS,
                    stride_cache_seq_lens_b,
                    USING_OFFLOAD_CACHE,
                    OFFLOAD_CACHE_KV_PACKED,
                    GPU_BANK_COUNT,
                    True,
                    OFFLOAD_CACHE_UVM_METADATA,
                    stride_offload_cache_uvm_metadata_token,
                    stride_offload_cache_uvm_metadata_k,
                    OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                    stride_offload_cache_gpu_global_metadata_k,
                    stride_offload_cache_gpu_global_metadata_pad,
                    OFFLOAD_CACHE_GPU_BANK,
                    stride_offload_cache_gpu_bank_token,
                    stride_offload_cache_gpu_bank_hid,
                    OFFLOAD_CACHE_GPU_METADATA,
                    stride_offload_cache_gpu_metadata_token,
                    stride_offload_cache_gpu_metadata_k,
                    OFFLOAD_CACHE_GPU_TABLE,
                    stride_offload_cache_gpu_table_head_kv,
                    stride_offload_cache_gpu_table_token,
                    strdie_offload_cache_gpu_table_k,
                    ACCESS_COUNTER,
                    stride_access_counter_bsz,
                    stride_access_counter_head_kv,
                    stride_access_counter_tsrc,
                    CACHE_MISS_COUNTER,
                    stride_cache_miss_counter_bsz,
                    stride_cache_miss_counter_head_kv,
                    stride_cache_miss_counter_tsrc,
                    idx_bsz,
                    idx_tsrc[:, None],
                    idx_head // KV_HEAD_REPEAT,
                    idx_hid_v[None, :],
                    mask_tsrc[:, None],
                    HEAD // KV_HEAD_REPEAT,
                    BLOCK_BK * BLOCK_SIZE_K,
                    HID_BLOCK_V,
                    HID_V,
                    IS_BSA=True,
                    UPDATE_CACHE=UPDATE_CACHE,
                    V_CACHE=K_CACHE,
                    stride_v_cache_page=stride_k_cache_page,
                    stride_v_cache_offset=stride_k_cache_offset,
                    stride_v_cache_kv_head=stride_k_cache_kv_head,
                    stride_v_cache_hid=stride_k_cache_hid,
                )

                if v_descale is not None:
                    value *= v_descale

                acc, l_i, m_i = block_sparse_attention_cuda_step(
                    queries_0,
                    queries_1,
                    keys_0,
                    keys_1,
                    keys_rot_0,
                    keys_rot_1,
                    values,
                    idx_tsrc,
                    mask_tsrc,
                    idx_tdst,
                    mask_tdst,
                    acc,
                    l_i,
                    m_i,
                    sliding_window_size,
                    sink_token_size,
                    sparse_token_size,
                    (range_end - range_start) * BLOCK_SIZE_K,
                    True,
                    False,
                    LOGIT_SOFTCAP,
                    USING_EXTEND,
                    NEED_APPLY_ROPE,
                    COS,
                    stride_cos_t,
                    stride_cos_hid,
                    SIN,
                    stride_sin_t,
                    stride_sin_hid,
                    rope_range_begin,
                    rope_range_end,
                    rope_is_neox_style,
                    model_context_length,
                    tl.reshape(
                        idx_bk[:, None] * BLOCK_SIZE_K
                        + tl.arange(0, BLOCK_SIZE_K)[None, :],
                        BLOCK_SIZE_K * BLOCK_BK,
                    )
                    + sink_token_size,
                    pos_tdst,
                    idx_hid_q0,
                    idx_hid_q1,
                    IS_CAUSAL,
                    HID,
                    BLOCK_SIZE_Q,
                    BLOCK_BK * BLOCK_SIZE_K,
                    BLOCK_SIZE_K,
                    EXTEND_BACKEND=EXTEND_BACKEND,
                )
            else:
                pass

    if MX is not None and NC is not None:
        mx_nc_offsets = (
            idx_bsz.to(tl.int64) * stride_mx_bsz
            + idx_tdst[:, None].to(tl.int64) * stride_mx_tdst
            + idx_head.to(tl.int64) * stride_mx_head
        )

        tl.store(MX + mx_nc_offsets, m_i, mask=mask_tdst[:, None])
        tl.store(NC + mx_nc_offsets, l_i, mask=mask_tdst[:, None])

    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / (tl.where(l_i == 0.0, 1e-20, l_i))

    tl.store(
        CONTEXT
        + idx_bsz.to(tl.int64) * stride_context_bsz
        + idx_tdst[:, None].to(tl.int64) * stride_context_tdst
        + idx_head.to(tl.int64) * stride_context_head
        + idx_hid_v[None, :].to(tl.int64) * stride_context_hid,
        mask=mask_tdst[:, None] & (idx_hid_v < HID_V),
        value=acc.to(CONTEXT.type.element_ty),
        # eviction_policy='evict_first',
        # cache_modifier='.cs', # TODO: uncomment this
        # value = l_i
    )


from .utils import capture


@capture
def block_sparse_attention(
    q: Tensor,
    k: Optional[Tensor],
    v: Optional[Tensor],
    seq_lens: Tensor,
    indices: Tensor,
    ks: Tensor,
    ks_count: Tensor,
    ks_start_end: Tensor,
    args: "HiPAttentionArgs",
    access_counter: Tensor,
    cache_miss_counter: Tensor,
    EXTEND_BACKEND: str = DEFAULT_EXTEND_BACKEND,
    model_context_length: int = 131072,
    extend_context_length: int = 131072,
    offload_update_cache: bool = False,
    return_running_statistics: bool = False,
    k_descale: Tensor = None,
    v_descale: Tensor = None,
):
    BSZ, TDST, HEAD, HID = q.shape
    if k is not None:
        _, TSRC, KV_HEAD, _ = k.shape
        BSRC = cdiv_python(TSRC, args.block_size_k)
        MAX_TSRC = TSRC
        MAX_BSRC = BSRC
        HID_V = v.shape[-1]
    else:
        if args.k_cache is not None:
            NUM_PAGE, PAGE_SIZE, KV_HEAD, _ = args.k_cache.shape
            HID_V = args.v_cache.shape[-1]
        else:
            KV_HEAD = args.offload_cache.k_uvm.bank_cpu.shape[-2]
            HID_V = args.offload_cache.v_uvm.bank_cpu.shape[-1]
        TSRC = None
        BSRC = None
        # MAX_TSRC = NUM_PAGE * PAGE_SIZE
        MAX_TSRC = extend_context_length
        MAX_BSRC = cdiv_python(MAX_TSRC, args.block_size_k)
    HID_V = args.v_hidden_dim if args.v_hidden_dim is not None else HID_V
    N = BSZ * HEAD
    # assert q.shape == k.shape
    BDST = cdiv_python(TDST, args.block_size_q)
    KV_HEAD_REPEAT = HEAD // KV_HEAD
    assert KV_HEAD_REPEAT * KV_HEAD == HEAD

    B = N
    assert B == N
    BK = indices.shape[-1]  # cdiv_python(args.mask_k, args.block_size_k)

    context = torch.empty((BSZ, TDST, HEAD, HID_V), dtype=q.dtype, device=q.device)

    # BLOCK_BK = 64 // block_size_k
    # if block_size_k > 4:
    #     BLOCK_BK = 128 // block_size_k
    # elif block_size_k > 8:
    #     BLOCK_BK = 256 // block_size_k
    # BLOCK_BK = 64 // args.block_size_k

    max_block_size = int(os.getenv("SA_BLOCK_SIZE", "128"))
    BLOCK_BK = max_block_size // args.block_size_k
    BLOCK_BK = max(1, min(max_block_size, BLOCK_BK))
    if "SA_BLOCK_BK" in os.environ:
        BLOCK_BK = int(os.environ["SA_BLOCK_BK"])

    assert BLOCK_BK > 0, BLOCK_BK

    if return_running_statistics:
        MX = torch.zeros((BSZ, TDST, HEAD), dtype=torch.float32, device=q.device)
        NC = torch.zeros((BSZ, TDST, HEAD), dtype=torch.float32, device=q.device)
    else:
        MX = NC = None

    # sliding_window_size = min(sliding_window_size, block_size_k * 16)

    if args.rope_cos is not None:
        assert len(args.rope_cos.stride()) == 2
        assert len(args.rope_sin.stride()) == 2

    assert context.ndim == 4
    if ks_start_end is not None:
        assert ks_start_end.ndim == 3
    if indices is not None:
        assert indices.ndim == 3
    assert q.ndim == 4
    if k is not None:
        assert k.ndim == 4
        assert v.ndim == 4
    elif args.using_paged_cache:
        if args.k_cache is not None:
            assert args.k_cache.ndim == 4
            assert args.v_cache.ndim == 4
        else:
            assert args.offload_cache.k_uvm.bank_cpu.ndim == 3
            assert args.offload_cache.v_uvm.bank_cpu.ndim == 3
    else:
        raise Exception()
    assert seq_lens.ndim == 2

    if args.rope_range[0] == 0 and args.rope_range[1] == HID:
        HID_BLOCK = triton.next_power_of_2(HID)
    else:
        assert triton.next_power_of_2(args.rope_range[0]) == args.rope_range[0]
        assert args.rope_range[1] == HID
        HID_BLOCK = args.rope_range[0]

    HID_BLOCK_V = triton.next_power_of_2(min(HID_V, 256))
    NUM_HID_V_BLOCKS = triton.cdiv(HID_V, HID_BLOCK_V)

    if k_descale is not None:
        k_descale = k_descale.contiguous()
        v_descale = v_descale.contiguous()
        assert k_descale.shape == (BSZ, HEAD // KV_HEAD_REPEAT)
        assert k_descale.shape == v_descale.dtype

    grid = (HEAD * NUM_HID_V_BLOCKS, BDST, BSZ)
    pre_device = torch.get_default_device()
    torch.set_default_device(q.device)

    # print(indices.shape, indices[0, -1], ks_start_end[0, -1])
    # if indices.shape[1] == 1:
    #     input()

    if os.getenv("HIP_VERBOSE", "0") == "1":
        print(
            f"{HEAD=}",
            f"{BK=}",
            f"{KV_HEAD_REPEAT=}",
            f"{args.sliding_window_size=}",
            f"{args.sink_token_size=}",
            f"{args.logit_softcap=}",
            f"{args.using_extend=}",
            f"{args.need_apply_rope=}",
            f"{args.rope_range[0]=}",
            f"{args.rope_range[1]=}",
            f"{args.using_paged_cache=}",
            f"{args.k_cache.shape[1] if args.k_cache is not None else None=}",
            f"{args.is_causal=}",
            f"{args.block_size_q=}",
            f"{args.block_size_k=}",
            f"{HID_BLOCK=}",
            f"{HID=}",
            f"{HID_BLOCK_V=}",
            f"{HID_V=}",
            f"{BLOCK_BK=}",
            f"{EXTEND_BACKEND=}",
            f"{offload_update_cache=}",
            sep=", ",
        )
    block_sparse_attention_cuda[grid](
        q,
        *safe_stride(q, 4),
        k,
        *safe_stride(k, 4),
        v,
        *safe_stride(v, 4),
        k_descale,
        v_descale,
        seq_lens,
        *safe_stride(seq_lens, 2),
        indices,
        *safe_stride(indices, 3),
        ks_start_end,
        *safe_stride(ks_start_end, 3),
        context,
        *safe_stride(context, 4),
        MX,
        NC,
        *safe_stride(MX, 3),
        HEAD,
        BK,
        TDST,
        MAX_TSRC,
        KV_HEAD_REPEAT,
        args.sliding_window_size,
        args.sink_token_size,
        args.logit_softcap,
        *args.args_extend(),
        model_context_length,
        *args.args_paged_kv_cache(),
        *args.args_offload_cache(is_masking=False),
        access_counter,
        *safe_stride(access_counter, 3),
        cache_miss_counter,
        *safe_stride(cache_miss_counter, 3),
        triton.next_power_of_2(TDST),
        args.is_causal,
        args.block_size_q,
        args.block_size_k,
        HID_BLOCK,
        HID,
        HID_BLOCK_V,
        HID_V,
        # 2,
        BLOCK_BK=BLOCK_BK,
        EXTEND_BACKEND=EXTEND_BACKEND,
        UPDATE_CACHE=offload_update_cache,
        CHUNKED_SW=args.using_chunked_sliding_window,
        # num_warps=4,
        # num_stages=2 if not using_extend else 1,
    )
    torch.set_default_device(pre_device)

    if (
        (os.getenv("HIP_CUMSUM", "0") == "1")
        and isinstance(v, Tensor)
        and q.shape[1] > 1
    ):
        v_cumsum = (
            v.cumsum(dim=1)
            / torch.arange(1, v.shape[1] + 1, device=v.device)[None, :, None, None]
        )
        a = torch.arange(1, v.shape[1] + 1, device=v.device)[None, :, None]
        b = (
            ks.repeat_interleave(args.block_size_q, 1)[:, : v.shape[1]]
            .view(BSZ, HEAD, -1)
            .permute(0, 2, 1)
            * args.block_size_k
        )
        scaler = ((a - b) / a).clamp_min(0)[:, :, :, None].pow(2) * 0.05
        context = (
            context * (1 - scaler)
            + v_cumsum.repeat_interleave(HEAD // KV_HEAD, dim=2) * scaler
        )

    if return_running_statistics:
        return context, (MX, NC)
    else:
        return context

```

### `hip_attn/v1_2/attention_metadata.py`

```py
import copy
import os
import warnings
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Dict, List, Literal, Optional

import torch
from torch import Tensor

if TYPE_CHECKING:
    from hip_attn.v1_2.uvm_gpu_cache import HiPOffloadCache


HIP_DEBUG_ALLOW_GATHER_KV_CACHE = (
    os.getenv("HIP_DEBUG_ALLOW_GATHER_KV_CACHE", "0") == "1"
)


def safe_stride(x: Optional[Tensor], ndim: int):
    if x is None:
        return tuple(
            [
                0,
            ]
            * ndim
        )
    else:
        stride = x.stride()
        assert len(stride) == ndim
        return stride


@dataclass
class Stage:
    stage_block_size_q: int
    stage_block_stride_q: int
    stage_chunk_size: int
    stage_k: Optional[int]
    stage_stride: int

    require_realign_index: bool = False
    require_reset_score: bool = False
    require_post_sort: bool = False


@dataclass
class NopStage(Stage):
    require_realign_index: bool = True
    require_reset_score: bool = False
    require_post_sort: bool = True


@dataclass
class EvalScoreStage(Stage):
    block_chunk: int = 64
    stage_extend_backend: Optional[str] = None
    require_reset_score: bool = True
    require_post_sort: bool = True


@dataclass
class ScanStage(Stage):
    stage_extend_backend: Optional[str] = None
    using_landmark: Optional[bool] = None
    require_realign_index: bool = True
    require_reset_score: bool = True
    require_post_sort: bool = True


@dataclass
class EnsembleScoreStage(Stage):
    reduce_method: str = "sum"
    require_reset_score: bool = True
    require_post_sort: bool = True


StatKeys = Literal[
    "unique_access_count",
    "access_count",
    "cache_miss_count",
    "cache_hit_ratio",
]


@dataclass
class HiPAttentionCacheAccessStatistics:
    # [BSZ, HEAD_KV, MAX_TSRC]
    access_counter: Tensor
    # [BSZ, HEAD_KV, MAX_TSRC]
    cache_miss_counter: Tensor

    def compute_statistics(self) -> Dict[
        StatKeys,
        Tensor,
    ]:
        # FIXME: heejun
        if (os.getenv("HIP_DISABLE_COMPUTE_STATISTICS", "1") == "0") and (
            self.access_counter is not None
        ):
            unique_access_count = self.access_counter.clamp(0, 1).sum()
            access_counts = self.access_counter.sum()
            cache_miss_counts = self.cache_miss_counter.sum()
            cache_hit_ratio = 1 - (cache_miss_counts / access_counts)
        else:
            unique_access_count = None
            access_counts = None
            cache_miss_counts = None
            cache_hit_ratio = None

        return {
            "unique_access_count": unique_access_count,
            "access_count": access_counts,
            "cache_miss_count": cache_miss_counts,
            "cache_hit_ratio": cache_hit_ratio,
        }


@dataclass
class HiPAttentionStageInputCache:
    indices_left: Tensor
    indices_right: Tensor
    out_scores: Tensor


@dataclass
class HiPAttentionState:
    # [MAX_NUM_TOKENS, HEAD]
    landmark_scores: torch.Tensor
    # [NUM_STAGES, MAX_NUM_TOKENS // CHUNK_SIZE, K]
    landmark_indices: List[torch.Tensor]

    @classmethod
    def from_args(
        cls, q: torch.Tensor, args: "HiPAttentionArgs", k: Optional[torch.Tensor] = None
    ):
        if k is None:
            assert args.using_paged_cache

        if args.get_k_cache() is not None:
            k_cache = args.get_k_cache()
            num_tokens = k_cache.shape[0] * k_cache.shape[1]
        else:
            num_tokens = k.shape[1]

        num_tokens = max(args.extend_context_length, num_tokens)
        num_tokens = max(int(os.getenv("HIP_DEBUG_MAX_TOKENS", "0")), num_tokens)

        # padding for SGlang
        num_tokens += 1024

        num_heads = q.shape[2]
        landmark_scores = torch.zeros(
            (num_tokens, num_heads), dtype=torch.float32, device=q.device
        )

        return HiPAttentionState(
            landmark_scores=landmark_scores,
            landmark_indices=None,
        )


@dataclass
class HiPAttentionOutputMetadata:
    indices: Optional[Tensor]
    ks: Optional[Tensor]
    ks_count: Optional[Tensor]
    ks_start_end: Optional[Tensor]

    # memory access statistics
    mask_cache_statistics: Optional[HiPAttentionCacheAccessStatistics]
    sa_cache_statistics: Optional[HiPAttentionCacheAccessStatistics]

    # stage caches
    stage_caches: Optional[List[HiPAttentionStageInputCache]]

    state: Optional[HiPAttentionState] = None


@dataclass
class HiPAttentionArgs:
    position_ids: Optional[Tensor] = None

    sink_token_size: int = 256
    sliding_window_size: int = 512
    block_size_k: int = 64  # for optimization this will be BLOCK_CHUNK

    block_size_q: int = 64  # no effect, set automatically
    mask_k: int = 512  # no effect, set automatically

    second_stage_k: int = 2048
    stages: List[Stage] = field(
        default_factory=lambda: [
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=4,
                stage_chunk_size=256,
                stage_k=None,
                stage_stride=1,
            ),
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=4,
                stage_chunk_size=32,
                stage_k=32768,
                stage_stride=1,
            ),
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=1,
                stage_chunk_size=8,
                stage_k=8192,
                stage_stride=1,
            ),
        ]
    )
    model_context_length: int = 131072
    extend_context_length: int = 512 * 1024

    using_landmark: bool = field(
        default_factory=lambda: os.getenv("HIP_DEBUG_LANDMARK_BASED_SCAN_STAGE", "1")
        == "1"
    )
    landmark_stage_k: List[int] = field(default_factory=lambda: [1, 1, 1])

    # kernel args,
    mask_only: bool = False
    block_sparse_block_size_q: Optional[int] = 64
    v_hidden_dim: Optional[int] = None

    scan_early_terminate: int = 1
    stage_early_terminate: int = 1
    scan_extend_backend: str = "relative"
    sa_extend_backend: str = "streaming"
    low_percent: float = 0.0
    low_k_ratio: float = 1.0
    dim_to_lower: Literal["head", "seq"] = "head"
    q_mask: Optional[Tensor] = None
    k_mask: Optional[Tensor] = None
    idx_pca_hid_q: Optional[Tensor] = None
    idx_pca_hid_k: Optional[Tensor] = None

    is_causal: bool = True

    using_extend: bool = False
    need_apply_rope: bool = False
    rope_cos: Optional[Tensor] = None
    rope_sin: Optional[Tensor] = None
    rope_range: Optional[tuple[int, int]] = None
    rope_is_neox_style: Optional[bool] = None

    offload_cache: "Optional[HiPOffloadCache]" = None
    k_cache: Optional[Tensor] = None
    v_cache: Optional[Tensor] = None
    cache_seq_lens: Optional[Tensor] = None
    block_table: Optional[Tensor] = None

    # to support gemma2
    logit_softcap: Optional[float] = None

    online_update_cache: bool = False

    require_cache_statistics: bool = True
    require_stage_caches: bool = True

    disable_flashdecode: bool = False

    sliding_window_indices: Optional[torch.Tensor] = None

    using_chunked_sliding_window: bool = False

    # NOTE: use only for debugging purpose
    layer_id: int = 31

    query_for_landmark: Optional[Tensor] = None
    position_ids_for_landmark: Optional[Tensor] = None

    is_decode: bool = False

    bsa_return_running_statistics: bool = False
    bsa_sliding_window_size: int = -1

    k_descale: Optional[Tensor] = None
    v_descale: Optional[Tensor] = None

    def __post_init__(self):
        if self.rope_cos is not None and self.rope_cos.ndim == 3:
            self.rope_cos = self.rope_cos.view(-1, self.rope_cos.shape[-1])
            self.rope_sin = self.rope_sin.view(-1, self.rope_sin.shape[-1])
        # if self.q_quant is not None:
        #     assert self.q_quant.ndim == 4
        #     assert self.k_quant.ndim == 4
        self.update_flags()

    def update_flags(self):
        if self.logit_softcap == 0:
            self.logit_softcap = None
        self.using_paged_cache = (self.k_cache is not None) or (
            self.offload_cache is not None
        )
        if self.using_paged_cache:
            if self.k_cache is not None:
                self.paged_cache_page_count = self.k_cache.shape[0]
                self.paged_cache_page_size = self.k_cache.shape[1]
            else:
                self.paged_cache_page_count = self.offload_cache.get_page_count()
                self.paged_cache_page_size = 1
            assert self.paged_cache_page_size in (1, 2, 4, 8, 16, 32)
        if self.logit_softcap == 0:
            self.logit_softcap = None

    def clone(self):
        self.update_flags()
        return copy.copy(self)

    def json(self, convert_tensor_to_meta=True):
        from dataclasses import fields

        json = {}
        for field in fields(self):
            json[field.name] = getattr(self, field.name)

        if convert_tensor_to_meta:
            for k, v in json.items():
                if isinstance(v, Tensor):
                    v = f"{v.dtype}{list(v.shape)}@{v.device}.{v.data_ptr():02X}"
                json[k] = v

        return json

    def args_extend(self):
        return (
            self.using_extend,
            self.need_apply_rope,
            *self.args_rope_cos(),
            *self.args_rope_sin(),
            self.rope_range[0],
            self.rope_range[1],
            self.rope_is_neox_style,
        )

    def args_rope_cos(self):
        return (
            self.rope_cos,
            *safe_stride(self.rope_cos, 2),
        )

    def args_rope_sin(self):
        return (
            self.rope_sin,
            *safe_stride(self.rope_sin, 2),
        )

    def args_paged_kv_cache(self, disable_cache: bool = False):
        using_page = self.using_paged_cache

        if disable_cache:
            return (
                False,
                1,
                None,
                0,
                0,
                0,
                0,
                None,
                0,
                0,
                0,
                0,
                None,
                0,
                0,
                None,
                0,
            )

        if self.offload_cache is None:
            if using_page:
                assert self.v_cache is not None
                assert self.k_cache.ndim == self.v_cache.ndim
                assert self.k_cache.ndim == 4
                assert self.block_table is not None
                assert self.block_table.ndim == 2
                assert self.cache_seq_lens is not None
                assert self.cache_seq_lens.ndim == 1
                page_size = self.k_cache.shape[1]
            else:
                page_size = 0

            return (
                using_page,
                page_size,
                self.k_cache,
                *safe_stride(self.k_cache, 4),
                self.v_cache,
                *safe_stride(self.v_cache, 4),
                self.block_table,
                *safe_stride(self.block_table, 2),
                self.cache_seq_lens,
                *safe_stride(self.cache_seq_lens, 1),
            )
        else:
            assert using_page

            k_cache = self.offload_cache.k_uvm.bank_gpu.unsqueeze(1)
            v_cache = self.offload_cache.v_uvm.bank_gpu.unsqueeze(1)

            return (
                True,
                1,
                k_cache,
                *safe_stride(k_cache, 4),
                v_cache,
                *safe_stride(v_cache, 4),
                self.block_table,
                *safe_stride(self.block_table, 2),
                self.cache_seq_lens,
                *safe_stride(self.cache_seq_lens, 1),
            )

    def args_offload_cache(self, is_masking, disable_cache: bool = False):
        if self.offload_cache and (not disable_cache):
            gpu_cache = (
                self.offload_cache.mask_k_cache
                if is_masking
                else self.offload_cache.sa_kv_cache
            )
            is_packed = gpu_cache.kv_packed
            uvm_metadata = self.offload_cache.k_uvm.metadata
            return (
                True,
                is_packed,
                gpu_cache.bank.shape[0],
                uvm_metadata,
                *safe_stride(uvm_metadata, 2),
                gpu_cache.global_metadata,
                *safe_stride(gpu_cache.global_metadata, 2),
                gpu_cache.bank,
                *safe_stride(gpu_cache.bank, 2),
                gpu_cache.metadata,
                *safe_stride(gpu_cache.metadata, 2),
                gpu_cache.table,
                *safe_stride(gpu_cache.table, 3),
            )
        else:
            return (
                False,
                False,
                0,
                None,
                0,
                0,
                None,
                0,
                0,
                None,
                0,
                0,
                None,
                0,
                0,
                None,
                0,
                0,
                0,
            )

    def get_k_cache(self):
        if self.k_cache is not None:
            k_cache = self.k_cache
        elif self.offload_cache is not None:
            k_cache = self.offload_cache.k_uvm.bank_gpu.unsqueeze(1)
        else:
            k_cache = None

        # k_cache: [MAX_TOKENS, 1, HEAD, HID]
        return k_cache

    def get_v_cache(self):
        if self.v_cache is not None:
            v_cache = self.v_cache
        elif self.offload_cache is not None:
            v_cache = self.offload_cache.v_uvm.bank_gpu.unsqueeze(1)
        else:
            v_cache = None

        # v_cache: [MAX_TOKENS, 1, HEAD, HID]
        return v_cache

    def gather_extend_k_from_paged_cache(
        self,
        disable_gqa=False,
        gqa_q: torch.Tensor = None,
        position_ids: torch.Tensor = None,
    ):
        k_cache = self.get_k_cache()
        # self.block_table[BLOCK_TABLE_BSZ, MODEL_SEQ_LEN]
        assert self.block_table is not None
        if position_ids is None:
            position_ids = self.position_ids
        assert position_ids is not None
        assert (
            position_ids.shape[0] == self.block_table.shape[0]
        ), f"{position_ids.shape} == {self.block_table.shape}"
        # k_cache: [T, HEAD, HID]
        k = k_cache[:, 0, :, :][self.block_table.gather(dim=1, index=position_ids)]
        if gqa_q is not None:
            B, T, H, D = gqa_q.shape
            assert k.shape == (B, T, k.shape[2], D), f"{gqa_q.shape} {k.shape}"
        if disable_gqa:
            k = k.repeat_interleave(gqa_q.shape[2] // k.shape[2], dim=2)
        return k

    def gather_k_from_paged_cache(
        self,
        chunk_size: int = 1,
        disable_gqa=False,
        gqa_q: torch.Tensor = None,
        seq_len: int = None,
    ):
        if not HIP_DEBUG_ALLOW_GATHER_KV_CACHE:
            raise Exception(
                "Please set HIP_DEBUG_ALLOW_GATHER_KV_CACHE=1 for allow this behavior"
            )
        else:
            # warnings.warn("For developers: gathering paged cache will occure overhead.")
            pass

        k_cache = self.get_k_cache()
        assert self.block_table is not None

        if seq_len is None:
            seq_len = self.block_table.shape[1]

        is_fp8 = k_cache.dtype in (torch.float8_e5m2, torch.float8_e4m3fn)
        index_dtype = torch.uint8 if is_fp8 else k_cache.dtype

        k = k_cache.view(index_dtype)[:, 0, :, :][
            self.block_table[
                :,
                : seq_len - (seq_len % chunk_size),
            ]
        ].view(k_cache.dtype)
        if disable_gqa:
            k = k.repeat_interleave(gqa_q.shape[2] // k.shape[2], dim=2)
        return k

    def gather_v_from_paged_cache(
        self,
        chunk_size: int = 1,
        disable_gqa: bool = False,
        gqa_q: torch.Tensor = None,
        seq_len: int = None,
    ):
        if not HIP_DEBUG_ALLOW_GATHER_KV_CACHE:
            raise Exception(
                "Please set HIP_DEBUG_ALLOW_GATHER_KV_CACHE=1 for allow this behavior"
            )
        else:
            # warnings.warn("For developers: gathering paged cache will occure overhead.")
            pass

        if self.v_cache is not None:
            assert self.v_cache is not None
            v_cache = self.v_cache
        else:
            v_cache = self.offload_cache.v_uvm.bank_gpu.unsqueeze(1)

        assert self.block_table is not None

        if seq_len is None:
            seq_len = self.block_table.shape[1]

        is_fp8 = v_cache.dtype in (torch.float8_e5m2, torch.float8_e4m3fn)
        index_dtype = torch.uint8 if is_fp8 else v_cache.dtype

        v = v_cache.view(index_dtype)[:, 0, :, :][
            self.block_table[
                :,
                : seq_len - (seq_len % chunk_size),
            ]
        ].view(v_cache.dtype)
        if disable_gqa:
            v = v.repeat_interleave(gqa_q.shape[2] // v.shape[2], dim=2)
        return v

    def pretty(self) -> str:
        json = asdict(self)
        for k, v in json.items():
            if isinstance(v, torch.Tensor):
                json[k] = f"{v.dtype}{list(v.shape)}@{str(v.device)}"
        return str(json)
```

### `hip_attn/v1_2/compute_scores_landmark.py`

```py
import os
from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor

from hip_attn.v1_2.attention_metadata import safe_stride


@triton.jit
def split_half(x: tl.tensor):
    return tl.split(tl.trans(tl.reshape(x, [x.shape[0], 2, x.shape[1] // 2]), 0, 2, 1))


@triton.jit
def merge_half(x: tl.tensor, y: tl.tensor):
    return tl.reshape(
        tl.trans(tl.join(x, y), 0, 2, 1), x.shape[0], x.shape[1] + y.shape[1]
    )


@triton.jit
def de_rope(
    vec: tl.tensor,
    cos: tl.tensor,
    sin: tl.tensor,
):
    c0, ch = split_half(cos.to(tl.float32))
    s0, sh = split_half(sin.to(tl.float32))
    vr0, vrh = split_half(vec.to(tl.float32))

    out0 = (vrh * s0 + vr0 * ch) / (c0 * ch + sh * s0 + 1e-20)
    outh = (out0 * c0 - vr0) / (s0 + 1e-20)

    out = merge_half(out0, outh).to(vec.dtype)
    return out


@triton.jit
def de_rope_load(
    vec: tl.tensor,
    idx_t: tl.tensor,
    mask_t: tl.tensor,
    COS,
    stride_cos_t,
    stride_cos_hid,
    SIN,
    stride_sin_t,
    stride_sin_hid,
):
    cos = tl.load(
        COS
        + idx_t[:, None] * stride_cos_t
        + tl.arange(0, vec.shape[1])[None, :] * stride_cos_hid,
        mask=mask_t[:, None],
        other=0,
    )

    sin = tl.load(
        SIN
        + idx_t[:, None] * stride_sin_t
        + tl.arange(0, vec.shape[1])[None, :] * stride_sin_hid,
        mask=mask_t[:, None],
        other=0,
    )

    return de_rope(vec, cos, sin)


configs = [
    triton.Config(
        {
            "BLOCK_CHUNK": BLOCK_CHUNK,
        },
        num_stages=s,
        num_warps=w,
    )
    for BLOCK_CHUNK in [64, 128, 256]
    for s in [3, 4, 7]
    for w in [4, 8]
    # for BM in [128,]
    # for BN in [64,]
    # for s in [3, ]
    # for w in [4, ]
]


def keep(conf):
    BLOCK_CHUNK = conf.kwargs["BLOCK_CHUNK"]
    return True


@triton.autotune(
    list(filter(keep, configs)),
    key=["HID"]
)
@triton.jit
def _compute_scores_landmark_cuda(
    Q,
    stride_q_bsz,
    stride_q_tdst,
    stride_q_head,
    stride_q_hid,
    K,
    stride_k_bsz,
    stride_k_tsrc,
    stride_k_head_kv,
    stride_k_hid,
    K_CACHE,
    stride_k_cache_t,
    stride_k_cache_page,
    stride_k_cache_head_kv,
    stride_k_cache_hid,
    BLOCK_TABLE,
    stride_block_table_bsz,
    stride_block_table_tsrc,
    POS,
    stride_pos_bsz,
    stride_pos_tdst,
    INDICES_LEFT,
    stride_indices_left_bsz,
    stride_indices_left_bdst,
    stride_indices_left_head,
    stride_indices_left_chunk,
    LANDMARK,
    stride_landmark_bsz,
    stride_landmark_tchunk,
    stride_landmark_head,
    stride_landmark_k,
    SCORES,
    stride_scores_bsz,
    stride_scores_bdst,
    stride_scores_head,
    stride_scores_tchunk,
    COS,
    stride_cos_t,
    stride_cos_hid,
    SIN,
    stride_sin_t,
    stride_sin_hid,
    HEAD_KV: int,
    HEAD: int,
    TDST: int,
    NUM_CHUNKS: int,
    SLIDING_WINDOW_SIZE: int,
    HID: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_STRIDE_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    USING_PAGED_CACHE: tl.constexpr,
    DEROPE: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
):
    BDST = tl.cdiv(TDST, BLOCK_SIZE_Q)

    pid = tl.program_id(0).to(tl.int64)

    idx_head = pid % HEAD
    idx_head_kv = idx_head // (HEAD // HEAD_KV)
    pid = pid // HEAD

    idx_bdst = pid % BDST
    pid = pid // BDST

    idx_bsz = pid

    idx_hid = tl.arange(0, HID)

    Q = Q + idx_bsz * stride_q_bsz + idx_head * stride_q_head
    if K is not None:
        K = K + idx_bsz * stride_k_bsz + idx_head_kv * stride_k_head_kv
    if K_CACHE is not None:
        K_CACHE = (
            K_CACHE
            +
            # 0 * stride_k_cache_page +
            idx_head_kv * stride_k_cache_head_kv
        )
        BLOCK_TABLE = BLOCK_TABLE + idx_bsz * stride_block_table_bsz
    INDICES_LEFT = (
        INDICES_LEFT
        + idx_bsz * stride_indices_left_bsz
        + idx_bdst * stride_indices_left_bdst
        + idx_head * stride_indices_left_head
    )
    LANDMARK = (
        LANDMARK + idx_bsz * stride_landmark_bsz + idx_head * stride_landmark_head
    )
    SCORES = (
        SCORES
        + idx_bsz * stride_scores_bsz
        + idx_bdst * stride_scores_bdst
        + idx_head * stride_scores_head
    )

    idx_tdst = (
        tl.arange(0, BLOCK_SIZE_Q // BLOCK_STRIDE_Q) * BLOCK_STRIDE_Q
        + idx_bdst * BLOCK_SIZE_Q
    )
    mask_tdst = idx_tdst < TDST
    pos_tdst = tl.load(
        POS + idx_bsz * stride_pos_bsz + idx_tdst * stride_pos_tdst,
        mask=mask_tdst,
        other=0,
    )
    pos_tdst_max = tl.max(pos_tdst * mask_tdst)
    seq_len_max = pos_tdst_max + 1 - SLIDING_WINDOW_SIZE

    queries = tl.load(
        Q + idx_tdst[:, None] * stride_q_tdst + idx_hid[None, :] * stride_q_hid,
        mask=mask_tdst[:, None],
        other=0,
    )  # .to(tl.float8e5)

    if DEROPE:
        queries = de_rope_load(
            queries,
            pos_tdst,
            mask_tdst,
            COS,
            stride_cos_t,
            stride_cos_hid,
            SIN,
            stride_sin_t,
            stride_sin_hid,
        )

    for i_chunk in range(0, NUM_CHUNKS, BLOCK_CHUNK):
        idx_chunk = tl.arange(0, BLOCK_CHUNK) + i_chunk
        mask_chunk = idx_chunk < NUM_CHUNKS
        idx_k = tl.arange(0, BLOCK_K)
        idx_tsrc_base = tl.load(
            INDICES_LEFT + idx_chunk * stride_indices_left_chunk, mask=mask_chunk
        )
        idx_tchunk = idx_tsrc_base // CHUNK_SIZE
        idx_tsrc_offset = tl.load(
            LANDMARK
            + idx_tchunk[:, None] * stride_landmark_tchunk
            + idx_k[None, :] * stride_landmark_k,
            mask=mask_chunk[:, None],
        )
        idx_tsrc = idx_tsrc_base[:, None] + idx_tsrc_offset
        mask_tsrc = mask_chunk[:, None] & (idx_tsrc < seq_len_max)
        idx_tsrc = tl.reshape(idx_tsrc, BLOCK_CHUNK * BLOCK_K)
        mask_tsrc = tl.reshape(mask_tsrc, BLOCK_CHUNK * BLOCK_K)

        if seq_len_max >= tl.min(tl.where(mask_tsrc, idx_tsrc, 98765431)):
            if not USING_PAGED_CACHE:
                keys = tl.load(
                    K
                    + idx_tsrc[None, :] * stride_k_tsrc
                    + idx_hid[:, None] * stride_k_hid,
                    mask=mask_tsrc[None, :],
                    other=0.0,
                )  # .to(tl.float8e5)
            else:
                block_index = tl.load(
                    BLOCK_TABLE + idx_tsrc * stride_block_table_tsrc,
                    mask=mask_tsrc,
                    other=0,
                )
                keys = tl.load(
                    K_CACHE
                    + block_index[None, :] * stride_k_cache_t
                    + idx_hid[:, None] * stride_k_cache_hid,
                    mask=mask_tsrc[None, :],
                    other=0.0,
                )

            if keys.dtype == tl.float8e5:
                keys = keys.to(tl.float16)

            if DEROPE:
                keys = tl.trans(
                    de_rope_load(
                        tl.trans(keys, 1, 0),
                        idx_tsrc,
                        mask_tsrc,
                        COS,
                        stride_cos_t,
                        stride_cos_hid,
                        SIN,
                        stride_sin_t,
                        stride_sin_hid,
                    ),
                    1,
                    0,
                )

            scores = tl.dot(
                queries.to(keys.dtype),
                keys,
            )

            scores = tl.where(scores == 0.0, float("-inf"), scores).to(scores.dtype)

            # mask = (
            #     (mask_tdst[:, None] & mask_tsrc[None, :]) &
            #     ((pos_tdst - SLIDING_WINDOW_SIZE)[:, None] >= idx_tsrc[None, :])
            # )
            # scores = tl.where(mask, scores, float('-inf')).to(scores.dtype)
            # scores = tl.where(mask, scores, 0)

            if BLOCK_K > 1:
                scores = tl.reshape(
                    scores, BLOCK_SIZE_Q // BLOCK_STRIDE_Q, BLOCK_CHUNK, BLOCK_K
                )
                scores = tl.max(scores, axis=0)
                scores = tl.max(scores, axis=-1)
            else:
                scores = tl.reshape(scores, BLOCK_SIZE_Q // BLOCK_STRIDE_Q, BLOCK_CHUNK)
                scores = tl.max(scores, axis=0)

            tl.store(
                SCORES + idx_chunk * stride_scores_tchunk,
                value=scores,
                mask=mask_chunk,
            )


from .utils import capture


@capture
def compute_scores_landmark(
    # [BSZ, TDST, HEAD, HID]
    q: Tensor,
    # [BSZ, TSRC, HEAD_KV, HID]
    k: Tensor,
    # [T, 1, HEAD_KV, HID]
    k_cache: Optional[Tensor],
    # [BSZ, MAX_TSRC]
    block_table: Optional[Tensor],
    # [BSZ, TDST]
    position_ids: Tensor,
    # [BSZ, BDST, HEAD, CHUNK_COUNT]
    indices_left: Tensor,
    # [BSZ, TSRC // CHUNK_SIZE, HEAD, K]
    landmarks: Tensor,
    cos: Optional[Tensor],
    sin: Optional[Tensor],
    BLOCK_SIZE_Q: int,
    BLOCK_STRIDE_Q: int,
    CHUNK_SIZE: int,
    SLIDING_WINDOW_SIZE: int,
) -> Tensor:
    # output: [BSZ, BDST, HEAD, CHUNK_COUNT]
    BSZ, TDST, HEAD, HID = q.shape
    BDST = triton.cdiv(TDST, BLOCK_SIZE_Q)
    if k is not None:
        _, TSRC, HEAD_KV, _ = k.shape
        assert k.shape == (BSZ, TSRC, HEAD_KV, HID)
    else:
        assert k_cache is not None
        HEAD_KV = k_cache.shape[-2]
    assert position_ids.shape == (BSZ, TDST)
    K = landmarks.shape[-1]
    assert landmarks.shape == (BSZ, landmarks.shape[1], HEAD, K)
    CHUNK_COUNT = indices_left.shape[-1]
    assert indices_left.shape == (BSZ, BDST, HEAD, CHUNK_COUNT)
    if k_cache is not None:
        assert k_cache.shape[2:] == (HEAD_KV, HID)
        assert k_cache.shape[1] == 1

    BLOCK_K = K
    # BLOCK_CHUNK = int(os.getenv('SA_BLOCK_SIZE_LANDMARK', '128')) // BLOCK_K
    # assert BLOCK_CHUNK > 0

    USING_PAGED_CACHE = k_cache is not None
    DEROPE = False

    scores = torch.full(
        (BSZ, BDST, HEAD, CHUNK_COUNT),
        dtype=torch.float32,
        device=q.device,
        fill_value=float("-inf"),
    )

    grid = lambda kwargs: (BSZ * BDST * HEAD,)
    _compute_scores_landmark_cuda[grid](
        q,
        *safe_stride(q, 4),
        k,
        *safe_stride(k, 4),
        k_cache,
        *safe_stride(k_cache, 4),
        block_table,
        *safe_stride(block_table, 2),
        position_ids,
        *safe_stride(position_ids, 2),
        indices_left,
        *safe_stride(indices_left, 4),
        landmarks,
        *safe_stride(landmarks, 4),
        scores,
        *safe_stride(scores, 4),
        cos,
        *safe_stride(cos, 2),
        sin,
        *safe_stride(sin, 2),
        HEAD_KV,
        HEAD,
        TDST,
        CHUNK_COUNT,
        SLIDING_WINDOW_SIZE,
        HID,
        BLOCK_SIZE_Q,
        BLOCK_STRIDE_Q,
        BLOCK_K,
        CHUNK_SIZE,
        USING_PAGED_CACHE,
        DEROPE,
        # BLOCK_CHUNK,
        # num_warps=4,
        # num_stages=3,
    )

    return scores

```

### `hip_attn/v1_2/compute_v_cos.py`

```py
import triton
import triton.language as tl

from hip_attn.v1_2.uvm_gpu_cache import load_tokens


@triton.jit
def compute_v_cos(
    V,
    stride_v_bsz,
    stride_v_tsrc,
    stride_v_head_kv,
    stride_v_hid,
    INDICES,
    stride_indices_bsz,
    stride_indices_bdst,
    stride_indices_head,
    stride_indices_k,
    POS,
    stride_pos_bsz,
    stride_pos_tdst,
    OUT_SCORES,
    stride_out_scores_bsz,
    stride_out_scores_bdst,
    stride_out_scores_head,
    stride_out_scores_k,
    # paged attention args template
    USING_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    K_CACHE,
    stride_k_cache_page,
    stride_k_cache_offset,
    stride_k_cache_kv_head,
    stride_k_cache_hid,
    V_CACHE,
    stride_v_cache_page,
    stride_v_cache_offset,
    stride_v_cache_kv_head,
    stride_v_cache_hid,
    BLOCK_TABLE,
    stride_block_table_bsz,
    stride_block_table_page,
    CACHE_SEQ_LENS,
    stride_cache_seq_lens_b,
    # offload cache args template
    USING_OFFLOAD_CACHE: tl.constexpr,
    OFFLOAD_CACHE_KV_PACKED: tl.constexpr,
    GPU_BANK_COUNT,
    OFFLOAD_CACHE_UVM_METADATA,
    stride_offload_cache_uvm_metadata_token,
    stride_offload_cache_uvm_metadata_k,
    OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
    stride_offload_cache_gpu_global_metadata_k,
    stride_offload_cache_gpu_global_metadata_pad,
    OFFLOAD_CACHE_GPU_BANK,
    stride_offload_cache_gpu_bank_token,
    stride_offload_cache_gpu_bank_hid,
    OFFLOAD_CACHE_GPU_METADATA,
    stride_offload_cache_gpu_metadata_token,
    stride_offload_cache_gpu_metadata_k,
    OFFLOAD_CACHE_GPU_TABLE,
    stride_offload_cache_gpu_table_head_kv,
    stride_offload_cache_gpu_table_token,
    stride_offload_cache_gpu_table_k,
    ACCESS_COUNTER,
    stride_access_counter_bsz,
    stride_access_counter_head_kv,
    stride_access_counter_tsrc,
    CACHE_MISS_COUNTER,
    stride_cache_miss_counter_bsz,
    stride_cache_miss_counter_head_kv,
    stride_cache_miss_counter_tsrc,
    TDST,
    TSRC,
    HEAD,
    KS,
    HEAD_GROUP: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_STRIDE_Q: tl.constexpr,
    BLOCK_STRIDE_K: tl.constexpr,
    BLOCK_HID: tl.constexpr,
):
    pid = tl.program_id(0)

    idx_head = pid % HEAD
    pid = pid // HEAD

    idx_bk = pid % tl.cdiv(KS, GROUP_K)
    idx_k = idx_bk * GROUP_K + tl.arange(0, GROUP_K)
    mask_k = idx_k < KS
    pid = pid // tl.cdiv(KS, GROUP_K)

    idx_bdst = pid % tl.cdiv(TDST, BLOCK_SIZE_Q)
    idx_tdst = (
        idx_bdst * BLOCK_SIZE_Q
        + tl.arange(0, BLOCK_SIZE_Q // BLOCK_STRIDE_Q) * BLOCK_STRIDE_Q
    )
    mask_tdst = idx_tdst < TDST
    idx_bsz = pid // tl.cdiv(TDST, BLOCK_SIZE_Q)

    idx_hid = tl.arange(0, BLOCK_HID)

    pos_tdst = tl.load(
        POS + idx_bsz * stride_pos_bsz + idx_tdst * stride_pos_tdst,
        mask=mask_tdst,
        other=0,
    )
    mask_tdst = mask_tdst & (pos_tdst < TSRC)
    seq_len = tl.max(pos_tdst)  # + 1

    indices = tl.load(
        INDICES
        + idx_bsz * stride_indices_bsz
        + idx_bdst * stride_indices_bdst
        + idx_head * stride_indices_head
        + idx_k * stride_indices_k,
        mask=mask_k,
        other=seq_len + 2 * BLOCK_SIZE_K,
    )
    indices = indices // BLOCK_SIZE_K * BLOCK_SIZE_K

    idx_tsrc = tl.ravel(indices[:, None] + tl.arange(0, BLOCK_SIZE_K)[None, :])
    mask_tsrc = (idx_tsrc < seq_len) & (idx_tsrc >= 0)

    # values_tdst = tl.load(
    #     V +\
    #         idx_bsz * stride_v_bsz+\
    #         idx_tdst[:, None] * stride_v_tsrc+\
    #         (idx_head // HEAD_GROUP) * stride_v_head_kv +\
    #         idx_hid[None, :] * stride_v_hid,
    #     mask=mask_tdst[:, None],
    #     other=0,
    # )

    # tl.static_assert(not USING_OFFLOAD_CACHE)
    values_tdst = load_tokens(
        V,
        stride_v_bsz,
        stride_v_tsrc,
        stride_v_head_kv,
        stride_v_hid,
        USING_PAGES,
        PAGE_SIZE,
        V_CACHE,
        stride_v_cache_page,
        stride_v_cache_offset,
        stride_v_cache_kv_head,
        stride_v_cache_hid,
        BLOCK_TABLE,
        stride_block_table_bsz,
        stride_block_table_page,
        CACHE_SEQ_LENS,
        stride_cache_seq_lens_b,
        USING_OFFLOAD_CACHE,
        OFFLOAD_CACHE_KV_PACKED,
        GPU_BANK_COUNT,
        True,
        OFFLOAD_CACHE_UVM_METADATA,
        stride_offload_cache_uvm_metadata_token,
        stride_offload_cache_uvm_metadata_k,
        OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
        stride_offload_cache_gpu_global_metadata_k,
        stride_offload_cache_gpu_global_metadata_pad,
        OFFLOAD_CACHE_GPU_BANK,
        stride_offload_cache_gpu_bank_token,
        stride_offload_cache_gpu_bank_hid,
        OFFLOAD_CACHE_GPU_METADATA,
        stride_offload_cache_gpu_metadata_token,
        stride_offload_cache_gpu_metadata_k,
        OFFLOAD_CACHE_GPU_TABLE,
        stride_offload_cache_gpu_table_head_kv,
        stride_offload_cache_gpu_table_token,
        stride_offload_cache_gpu_table_k,
        ACCESS_COUNTER,
        stride_access_counter_bsz,
        stride_access_counter_head_kv,
        stride_access_counter_tsrc,
        CACHE_MISS_COUNTER,
        stride_cache_miss_counter_bsz,
        stride_cache_miss_counter_head_kv,
        stride_cache_miss_counter_tsrc,
        idx_bsz,
        pos_tdst[:, None],
        idx_head // HEAD_GROUP,
        idx_hid[None, :],
        mask_tdst[:, None],
        HEAD // HEAD_GROUP,
        BLOCK_SIZE_Q // BLOCK_STRIDE_Q,
        BLOCK_HID,
    ).to(tl.bfloat16)

    # values_tdst = (
    #     tl.sum(values_tdst, axis=0) /\
    #     tl.sum(mask_tdst.to(tl.int32))
    # )

    # values_tsrc = tl.load(
    #     V +\
    #         idx_bsz * stride_v_bsz +\
    #         idx_tsrc[:, None] * stride_v_tsrc +\
    #         (idx_head // HEAD_GROUP) * stride_v_head_kv +\
    #         idx_hid[None, :] * stride_v_hid,
    #     mask=mask_tsrc[:, None],
    #     other=0,
    # )

    values_tsrc = load_tokens(
        V,
        stride_v_bsz,
        stride_v_tsrc,
        stride_v_head_kv,
        stride_v_hid,
        USING_PAGES,
        PAGE_SIZE,
        V_CACHE,
        stride_v_cache_page,
        stride_v_cache_offset,
        stride_v_cache_kv_head,
        stride_v_cache_hid,
        BLOCK_TABLE,
        stride_block_table_bsz,
        stride_block_table_page,
        CACHE_SEQ_LENS,
        stride_cache_seq_lens_b,
        USING_OFFLOAD_CACHE,
        OFFLOAD_CACHE_KV_PACKED,
        GPU_BANK_COUNT,
        True,
        OFFLOAD_CACHE_UVM_METADATA,
        stride_offload_cache_uvm_metadata_token,
        stride_offload_cache_uvm_metadata_k,
        OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
        stride_offload_cache_gpu_global_metadata_k,
        stride_offload_cache_gpu_global_metadata_pad,
        OFFLOAD_CACHE_GPU_BANK,
        stride_offload_cache_gpu_bank_token,
        stride_offload_cache_gpu_bank_hid,
        OFFLOAD_CACHE_GPU_METADATA,
        stride_offload_cache_gpu_metadata_token,
        stride_offload_cache_gpu_metadata_k,
        OFFLOAD_CACHE_GPU_TABLE,
        stride_offload_cache_gpu_table_head_kv,
        stride_offload_cache_gpu_table_token,
        stride_offload_cache_gpu_table_k,
        ACCESS_COUNTER,
        stride_access_counter_bsz,
        stride_access_counter_head_kv,
        stride_access_counter_tsrc,
        CACHE_MISS_COUNTER,
        stride_cache_miss_counter_bsz,
        stride_cache_miss_counter_head_kv,
        stride_cache_miss_counter_tsrc,
        idx_bsz,
        idx_tsrc[:, None],
        idx_head // HEAD_GROUP,
        idx_hid[None, :],
        mask_tsrc[:, None],
        HEAD // HEAD_GROUP,
        GROUP_K * BLOCK_SIZE_K,
        BLOCK_HID,
    ).to(tl.bfloat16)

    # values_tsrc = (
    #     tl.sum(tl.reshape(values_tsrc, [GROUP_K, BLOCK_SIZE_K, BLOCK_HID]), axis=1) /\
    #     tl.sum(tl.reshape(mask_tsrc.to(tl.int32), [GROUP_K, BLOCK_SIZE_K, 1]), axis=1)
    # )

    values_tdst_norm = tl.sqrt(
        tl.sum(values_tdst.to(tl.float32) * values_tdst.to(tl.float32), axis=-1)
    )
    values_tsrc_norm = tl.sqrt(
        tl.sum(values_tsrc.to(tl.float32) * values_tsrc.to(tl.float32), axis=-1)
    )

    normalized_values_tdst = values_tdst
    normalized_values_tsrc = values_tsrc
    normalized_values_tdst = values_tdst / tl.maximum(values_tdst_norm[:, None], 1e-20)
    normalized_values_tsrc = values_tsrc / tl.maximum(values_tsrc_norm[:, None], 1e-20)

    # -
    # cos_sim_scores = tl.sum(normalized_values_tdst[None, :] * normalized_values_tsrc, axis=-1)
    cos_sim_scores = tl.dot(
        normalized_values_tdst, tl.trans(normalized_values_tsrc, 1, 0)
    )
    # cos_sim_scores = ((cos_sim_scores + 1) * 0.5).to(tl.float32)
    cos_sim_scores = cos_sim_scores  # * cos_sim_scores * cos_sim_scores

    scores = tl.reshape(
        cos_sim_scores, (BLOCK_SIZE_Q // BLOCK_STRIDE_Q, GROUP_K, BLOCK_SIZE_K)
    )
    # scores = tl.reshape(values_tsrc_norm, (GROUP_K, BLOCK_SIZE_K))
    mask_scores = tl.reshape(
        mask_tdst[:, None] & mask_tsrc[None, :],
        (BLOCK_SIZE_Q // BLOCK_STRIDE_Q, GROUP_K, BLOCK_SIZE_K),
    )
    scores = scores * mask_scores

    # reduce-mean
    mask_scores = tl.sum(mask_scores.to(scores.dtype), axis=-1)
    scores = tl.sum(scores, axis=-1)
    mask_scores = tl.sum(mask_scores.to(scores.dtype), axis=0)
    scores = tl.sum(scores, axis=0) / tl.maximum(mask_scores, 1e-20)
    # -

    # scores = tl.sum(values_tdst[None, :] * values_tsrc, axis=1)

    # reduce max
    # scores = tl.max(tl.max(scores, axis=-1), axis=0)

    # norm reduce-mean
    # scores = tl.reshape(values_tsrc_norm, (GROUP_K, BLOCK_SIZE_K))
    # scores = tl.sum(scores, axis=-1) / tl.maximum(tl.sum(tl.reshape(mask_tsrc, (GROUP_K, BLOCK_SIZE_K)), axis=-1), 1e-20)

    # scores = tl.sum(values_tdst[None, :] * values_tsrc)

    tl.store(
        OUT_SCORES
        + idx_bsz * stride_out_scores_bsz
        + idx_bdst * stride_out_scores_bdst
        + idx_head * stride_out_scores_head
        + idx_k * stride_out_scores_k,
        value=scores,
        mask=mask_k,
    )

```

### `hip_attn/v1_2/eval_stage.py`

```py
import triton
import triton.language as tl

from hip_attn.v1_2.scan_stage import load_keys_with_rope


@triton.jit
def calculate_chunk_score(
    Q,
    stride_q_bsz,
    stride_q_tdst,
    stride_q_head,
    stride_q_hid,
    K,
    stride_k_bsz,
    stride_k_tsrc,
    stride_k_head_kv,
    stride_k_hid,
    POS,
    stride_pos_bsz,
    stride_pos_tdst,
    COS,
    stride_cos_t,
    stride_cos_hid,
    SIN,
    stride_sin_t,
    stride_sin_hid,
    # paged attention args template
    USING_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    K_CACHE,
    stride_k_cache_page,
    stride_k_cache_offset,
    stride_k_cache_kv_head,
    stride_k_cache_hid,
    V_CACHE,
    stride_v_cache_page,
    stride_v_cache_offset,
    stride_v_cache_kv_head,
    stride_v_cache_hid,
    BLOCK_TABLE,
    stride_block_table_bsz,
    stride_block_table_page,
    CACHE_SEQ_LENS,
    stride_cache_seq_lens_b,
    # offload cache args template
    USING_OFFLOAD_CACHE: tl.constexpr,
    OFFLOAD_CACHE_METHOD: tl.constexpr,
    OFFLOAD_CACHE_BUDGET: tl.constexpr,
    OFFLOAD_CACHE_KV_HEAD: tl.constexpr,
    OFFLOAD_CACHE_K_TABLES,
    stride_offload_cache_k_tables_n,
    stride_offload_cache_k_tables_t,
    OFFLOAD_CACHE_K_BANKS,
    stride_offload_cache_k_banks_n,
    stride_offload_cache_k_banks_page,
    stride_offload_cache_k_banks_offset,
    stride_offload_cache_k_banks_hid,
    OFFLOAD_CACHE_K_BANK_STATS,
    stride_offload_cache_k_bank_stats_n,
    stride_offload_cache_k_bank_stats_page,
    stride_offload_cache_k_bank_stats_k,
    OFFLOAD_CACHE_COUNTERS,
    stride_offload_cache_counters_n,
    stride_offload_cache_counters_k,
    INDICES_LEFT,
    stride_indices_left_bsz,
    stride_indices_left_bdst,
    stride_indices_left_head,
    stride_indices_left_chunk,
    INDICES_RIGHT,
    stride_indices_right_bsz,
    stride_indices_right_bdst,
    stride_indices_right_head,
    stride_indices_right_chunk,
    OUT_SCORES,
    stride_out_scores_bsz,
    stride_out_scores_bdst,
    stride_out_scores_head,
    stride_out_scores_chunk,
    model_context_length,
    sliding_window_size,
    num_sinks,
    max_chunk_size,
    TDST,
    BDST,
    BDST_SCAN,
    N_HEAD,
    N_CHUNK,
    HEAD_GROUP,
    USING_EXTEND: tl.constexpr,
    NEED_APPLY_ROPE: tl.constexpr,
    EXTEND_BACKEND: tl.constexpr,
    BLOCK_HID: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_STRIDE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_STRIDE_K: tl.constexpr,
    SCAN_STRIDE: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
    REDUCE: tl.constexpr = "max",
):
    pid = tl.program_id(0).to(tl.int64)

    # idx_chunk = pid % N_CHUNK
    # pid = pid // N_CHUNK
    idx_head = pid % N_HEAD
    pid = pid // N_HEAD
    idx_bdst_scan = pid % BDST_SCAN
    pid = pid // BDST_SCAN
    idx_bsz = pid

    tl.static_assert(
        (NEED_APPLY_ROPE and USING_EXTEND) or (not (NEED_APPLY_ROPE or USING_EXTEND))
    )

    idx_tdst = (
        idx_bdst_scan * SCAN_STRIDE * BLOCK_SIZE_Q
        + (BDST * BLOCK_SIZE_Q - BDST_SCAN * SCAN_STRIDE * BLOCK_SIZE_Q)
        + tl.arange(0, BLOCK_SIZE_Q // BLOCK_STRIDE_Q) * BLOCK_STRIDE_Q
    )
    mask_tdst = (idx_tdst < TDST) & (idx_tdst >= 0)
    idx_hid = tl.arange(0, BLOCK_HID)
    mask_hid = idx_hid < BLOCK_HID

    pos_tdst = tl.load(
        POS + idx_bsz * stride_pos_bsz + idx_tdst * stride_pos_tdst,
        mask=mask_tdst,
        other=0,
    )
    pos_tdst_min = tl.min(tl.where(mask_tdst, pos_tdst, 999999999))
    pos_tdst_max = tl.max(pos_tdst)

    # real_pos_tdst_min = idx_bdst * BLOCK_SIZE_Q + TSRC - TDST
    # real_pos_tdst_min = tl.min(tl.where(mask_tdst, pos_tdst, 99999999999))

    # pos_tdst_min = (real_pos_tdst_min - sliding_window_size - num_sinks).to(tl.int32)
    # pos_tdst_min = tl.maximum(pos_tdst_min, 0)

    queries = tl.load(
        Q
        + idx_bsz * stride_q_bsz
        + idx_tdst[:, None] * stride_q_tdst
        + idx_head * stride_q_head
        + idx_hid[None, :] * stride_q_hid,
        mask=mask_tdst[:, None],
        other=0,
    )

    if NEED_APPLY_ROPE and USING_EXTEND:
        if EXTEND_BACKEND == "dynamic_extend":
            new_tdst = pos_tdst
        elif EXTEND_BACKEND == "self_extend":
            new_tdst = pos_tdst
        elif EXTEND_BACKEND == "streaming":
            new_tdst = tl.minimum(pos_tdst, N_CHUNK + sliding_window_size)
        elif EXTEND_BACKEND == "relative":
            new_tdst = pos_tdst * 0 + sliding_window_size
        else:
            raise Exception()

        queries_rot = tl.load(
            Q
            + idx_bsz * stride_q_bsz
            + idx_tdst[:, None] * stride_q_tdst
            + idx_head * stride_q_head
            + ((idx_hid + BLOCK_HID // 2) % BLOCK_HID)[None, :] * stride_q_hid,
            mask=mask_tdst[:, None],
            other=0,
        )

        cos_new = tl.load(
            COS
            + new_tdst[:, None].to(tl.int64) * stride_cos_t
            + (idx_hid % (BLOCK_HID // 2))[None, :] * stride_cos_hid,
            mask=mask_tdst[:, None],
            other=0.0,
        ).to(queries.dtype)
        sin_new = tl.load(
            SIN
            + new_tdst[:, None].to(tl.int64) * stride_sin_t
            + (idx_hid % (BLOCK_HID // 2))[None, :] * stride_sin_hid,
            mask=mask_tdst[:, None],
            other=0.0,
        ).to(queries.dtype)

        queries_rot = queries_rot * (
            ((idx_hid + BLOCK_HID // 2)[None, :] < BLOCK_HID) * (-2) + 1
        ).to(queries_rot.dtype)

        queries = (queries * cos_new + queries_rot * sin_new).to(queries.dtype)

    for idx_chunk_start in range(0, N_CHUNK, BLOCK_CHUNK):
        # for idx_chunk in range(tl.cdiv(pos_tdst_max, max_chunk_size)):
        idx_chunk = tl.arange(0, BLOCK_CHUNK) + idx_chunk_start
        mask_chunk = idx_chunk < N_CHUNK
        idx_tsrc_left = tl.load(
            INDICES_LEFT
            + idx_bsz * stride_indices_left_bsz
            + idx_bdst_scan * stride_indices_left_bdst
            + idx_head * stride_indices_left_head
            + idx_chunk * stride_indices_left_chunk,
            mask=mask_chunk,
            other=987654321,
        ).to(tl.int64)

        idx_tsrc_right = tl.load(
            INDICES_RIGHT
            + idx_bsz * stride_indices_right_bsz
            + idx_bdst_scan * stride_indices_right_bdst
            + idx_head * stride_indices_right_head
            + idx_chunk * stride_indices_right_chunk,
            mask=mask_chunk,
            other=987654321,
        ).to(tl.int64)

        if tl.min(idx_tsrc_left) <= pos_tdst_max:
            idx_tsrc_center = (idx_tsrc_left + idx_tsrc_right) // 2
            idx_tsrc_left = tl.maximum(0, idx_tsrc_center - BLOCK_SIZE_K // 2)
            idx_tsrc = (
                idx_tsrc_left[:, None]
                + tl.arange(0, BLOCK_SIZE_K // BLOCK_STRIDE_K)[None, :] * BLOCK_STRIDE_K
            )
            idx_tsrc = tl.ravel(idx_tsrc)
            mask_tsrc = idx_tsrc <= (tl.max(pos_tdst) - sliding_window_size)

            keys = load_keys_with_rope(
                K,
                stride_k_bsz,
                stride_k_tsrc,
                stride_k_head_kv,
                stride_k_hid,
                COS,
                stride_cos_t,
                stride_cos_hid,
                SIN,
                stride_sin_t,
                stride_sin_hid,
                # paged attention args template
                USING_PAGES,
                PAGE_SIZE,
                K_CACHE,
                stride_k_cache_page,
                stride_k_cache_offset,
                stride_k_cache_kv_head,
                stride_k_cache_hid,
                BLOCK_TABLE,
                stride_block_table_bsz,
                stride_block_table_page,
                CACHE_SEQ_LENS,
                stride_cache_seq_lens_b,
                # offload cache args template
                USING_OFFLOAD_CACHE,
                OFFLOAD_CACHE_METHOD,
                OFFLOAD_CACHE_BUDGET,
                OFFLOAD_CACHE_KV_HEAD,
                OFFLOAD_CACHE_K_TABLES,
                stride_offload_cache_k_tables_n,
                stride_offload_cache_k_tables_t,
                OFFLOAD_CACHE_K_BANKS,
                stride_offload_cache_k_banks_n,
                stride_offload_cache_k_banks_page,
                stride_offload_cache_k_banks_offset,
                stride_offload_cache_k_banks_hid,
                OFFLOAD_CACHE_K_BANK_STATS,
                stride_offload_cache_k_bank_stats_n,
                stride_offload_cache_k_bank_stats_page,
                stride_offload_cache_k_bank_stats_k,
                OFFLOAD_CACHE_COUNTERS,
                stride_offload_cache_counters_n,
                stride_offload_cache_counters_k,
                queries,
                idx_bsz,
                idx_tsrc,
                idx_head // HEAD_GROUP,
                idx_hid,
                idx_chunk,
                mask_tsrc,
                mask_tdst,
                mask_hid,
                pos_tdst_min,
                model_context_length,
                num_sinks,
                USING_EXTEND,
                EXTEND_BACKEND,
                NEED_APPLY_ROPE,
                BLOCK_SIZE_K,
                BLOCK_HID,
                True,
                HEAD // HEAD_GROUP,
                UPDATE_CACHE,
            )

            scores = tl.dot(
                (
                    queries
                    * (tl.sqrt(BLOCK_HID * 1.0) / tl.sqrt(tl.sqrt(BLOCK_HID * 1.0))).to(
                        queries.dtype
                    )
                ).to(queries.dtype),
                (
                    keys.to(queries.dtype)
                    * (1 / tl.sqrt(tl.sqrt(BLOCK_HID * 1.0))).to(queries.dtype)
                ).to(queries.dtype),
                allow_tf32=True,
                out_dtype=tl.float32,
            ).to(queries.dtype)

            if REDUCE == "max":
                scores_reduced = tl.where(
                    mask_tdst[:, None] & mask_tsrc[None, :], scores, -32000.0
                )
                scores_reduced = tl.reshape(
                    scores_reduced,
                    BLOCK_SIZE_Q // BLOCK_STRIDE_Q,
                    BLOCK_CHUNK,
                    BLOCK_SIZE_K // BLOCK_STRIDE_K,
                )
                scores_reduced = tl.max(scores_reduced, axis=0)
                scores_reduced = tl.max(scores_reduced, axis=-1)
            # elif REDUCE == 'mean':
            #     scores_reduced = tl.sum(tl.where(
            #         mask_tdst[:, None] & mask_tsrc[None, :],
            #         scores,
            #         0
            #     )) / tl.sum((mask_tdst[:, None] & mask_tsrc[None, :]).to(tl.int32))
            else:
                raise Exception()

            tl.store(
                OUT_SCORES
                + idx_bsz * stride_out_scores_bsz
                + idx_bdst_scan * stride_out_scores_bdst
                + idx_head * stride_out_scores_head
                + idx_chunk * stride_out_scores_chunk,
                value=scores_reduced,
                mask=mask_chunk,
            )
        else:
            tl.store(
                OUT_SCORES
                + idx_bsz * stride_out_scores_bsz
                + idx_bdst_scan * stride_out_scores_bdst
                + idx_head * stride_out_scores_head
                + idx_chunk * stride_out_scores_chunk,
                value=-32000.0,
                mask=mask_chunk,
            )

```

### `hip_attn/v1_2/landmark_sample.py`

```py
import os
from typing import Optional

import matplotlib.pyplot as plt
import torch
import triton
import triton.language as tl
from torch import Tensor

from .attention_metadata import HiPAttentionArgs, HiPAttentionState, safe_stride
from .utils import capture

if os.getenv("HIP_DISABLE_AUTOTUNE", "0") == "1":
    configs = [
        triton.Config(
            {"BLOCK_TSRC": BLOCK_TSRC, "BLOCK_TDST": BLOCK_TDST},
            num_stages=s,
            num_warps=w,
        )
        for BLOCK_TSRC in [128]
        for BLOCK_TDST in [128]
        for s in [
            3,
        ]
        for w in [
            4,
        ]
    ]
else:
    configs = [
        triton.Config(
            {"BLOCK_TSRC": BLOCK_TSRC, "BLOCK_TDST": BLOCK_TDST},
            num_stages=s,
            num_warps=w,
        )
        for BLOCK_TSRC in [64, 128]
        for BLOCK_TDST in [64, 128]
        for s in [
            1,
            3,
            4,
        ]
        for w in [4, 8]
    ]


def keep(conf):
    BLOCK_TSRC = conf.kwargs["BLOCK_TSRC"]
    BLOCK_TDST = conf.kwargs["BLOCK_TDST"]
    return True


@triton.autotune(list(filter(keep, configs)), key=["HID", "USING_PAGED_CACHE"])
@triton.jit
def _sw_score_sample(
    Q,
    stride_q_bsz,
    stride_q_tdst,
    stride_q_head,
    stride_q_hid,
    K,
    stride_k_bsz,
    stride_k_tsrc,
    stride_k_head_kv,
    stride_k_hid,
    POS,
    stride_pos_bsz,
    stride_pos_tdst,
    USING_PAGED_CACHE: tl.constexpr,
    K_CACHE,
    stride_k_cache_page,
    stride_k_cache_offset,
    stride_k_cache_head_kv,
    stride_k_cache_hid,
    BLOCK_TABLE,
    stride_block_table_bsz,
    stride_block_table_tsrc,
    SCORES,
    stride_scores_bsz,
    stride_scores_head,
    stride_scores_tdst,
    window_size,
    T,
    HEAD,
    HEAD_KV,
    HID: tl.constexpr,
    BLOCK_TDST: tl.constexpr,
    BLOCK_TSRC: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)

    idx_head = pid % HEAD
    idx_head_kv = idx_head // (HEAD // HEAD_KV)
    pid = pid // HEAD

    idx_bsrc = pid % tl.cdiv(T, BLOCK_TSRC)
    pid = pid // tl.cdiv(T, BLOCK_TSRC)

    idx_tsrc_start = idx_bsrc * BLOCK_TSRC
    idx_tsrc = tl.arange(0, BLOCK_TSRC) + idx_tsrc_start
    mask_tsrc = idx_tsrc < T

    idx_bsz = pid
    idx_hid = tl.arange(0, HID)

    pos_tdst_start = tl.load(
        POS + idx_bsz * stride_pos_bsz + idx_tsrc_start * stride_pos_tdst,
    )

    pos_tsrc = tl.arange(0, BLOCK_TSRC) + pos_tdst_start

    if USING_PAGED_CACHE:
        tl.static_assert(USING_PAGED_CACHE)
        idx_page = tl.load(
            BLOCK_TABLE
            + idx_bsz * stride_block_table_bsz
            + pos_tsrc * stride_block_table_tsrc,
            mask=mask_tsrc,
        )
        keys = tl.load(
            K_CACHE
            + idx_page[None, :] * stride_k_cache_page
            + 0 * stride_k_cache_offset
            + idx_head_kv * stride_k_cache_head_kv
            + idx_hid[:, None] * stride_k_cache_hid,
            mask=mask_tsrc[None, :],
            other=0.0,
        )
    else:
        keys = tl.load(
            K
            + idx_bsz * stride_k_bsz
            + pos_tsrc[None, :] * stride_k_tsrc
            + idx_head_kv * stride_k_head_kv
            + idx_hid[:, None] * stride_k_hid,
            mask=mask_tsrc[None, :],
            other=0.0,
        )

    dot_dtype = torch.float16 if Q.dtype.element_ty == tl.float8e5 else Q.dtype.element_ty
    keys = keys.to(dot_dtype)

    acc = tl.zeros((BLOCK_TSRC,), dtype=tl.float32) + 42

    for i_start in range(0, tl.maximum(BLOCK_TDST, BLOCK_TSRC), BLOCK_TDST):
        idx_tdst = idx_tsrc_start + tl.arange(0, BLOCK_TDST) + i_start
        mask_tdst = idx_tdst < T

        pos_tdst = tl.load(
            POS + idx_bsz * stride_pos_bsz + idx_tdst * stride_pos_tdst,
            mask=mask_tdst,
        )

        queries = tl.load(
            Q
            + idx_bsz * stride_q_bsz
            + idx_tdst[:, None] * stride_q_tdst
            + idx_head * stride_q_head
            + idx_hid[None, :] * stride_q_hid,
            mask=mask_tdst[:, None],
            other=0,
        ).to(dot_dtype)

        scores = tl.dot(queries, keys)

        mask = pos_tdst[:, None] >= pos_tsrc[None, :]
        acc = acc + tl.sum(scores * mask, axis=0)

    for i_start in range(
        tl.maximum(BLOCK_TDST, BLOCK_TSRC), window_size + BLOCK_TSRC, BLOCK_TDST
    ):
        idx_tdst = idx_tsrc_start + tl.arange(0, BLOCK_TDST) + i_start
        mask_tdst = idx_tdst < T

        queries = tl.load(
            Q
            + idx_bsz * stride_q_bsz
            + idx_tdst[:, None] * stride_q_tdst
            + idx_head * stride_q_head
            + idx_hid[None, :] * stride_q_hid,
            mask=mask_tdst[:, None],
            other=0,
        ).to(dot_dtype)

        scores = tl.dot(queries, keys)

        acc = acc + tl.sum(scores, axis=0)

    weight = tl.minimum(T - idx_tsrc, window_size)
    acc = acc / weight

    tl.store(
        SCORES
        + idx_bsz * stride_scores_bsz
        + idx_head * stride_scores_head
        + idx_tsrc * stride_scores_tdst,
        mask=mask_tsrc,
        value=acc,
    )


@capture
def landmark_sample(
    q: Tensor,
    k: Optional[Tensor],
    state: Optional[HiPAttentionState],
    args: HiPAttentionArgs,
    BSZ,
    HEAD,
    HEAD_KV,
    BDST,
    DEBUG,
    __logall_index,
):
    landmark_chunk = 512
    landmark_derope = False

    HID = q.shape[-1]

    __fused = True

    if __fused:
        q_for_landmark = (
            args.query_for_landmark if args.query_for_landmark is not None else q
        )
        position_ids_for_landmark = (
            args.position_ids_for_landmark
            if args.position_ids_for_landmark is not None
            else args.position_ids
        )
        TDST = q_for_landmark.shape[1]
        assert q_for_landmark.shape[0] == BSZ
        assert q_for_landmark.shape[2] == HEAD
        assert position_ids_for_landmark.shape[0] == BSZ
        assert position_ids_for_landmark.shape[1] == TDST

        _using_k = (not args.using_paged_cache) and (k is not None)
        _using_paged_k = args.using_paged_cache and (k is None)
        assert _using_k or _using_paged_k, f"todo {_using_k} or {_using_paged_k}"
        assert not landmark_derope, "todo"

        TDST_PADDED = (
            TDST
            if (TDST % landmark_chunk) == 0
            else TDST + (landmark_chunk - TDST % landmark_chunk)
        )

        k_cache = args.get_k_cache()
        if k_cache is not None:
            k_cache = k_cache[..., : q.shape[-1]]

        landmark_scores = torch.full(
            (BSZ, HEAD, TDST_PADDED),
            fill_value=float("-inf"),
            dtype=torch.float32,
            device=q.device,
        )

        # NOTE: TDST should divided by TSRC, because TSRC == TDST here (only handle new chunk)
        grid = lambda kwargs: (BSZ * HEAD * triton.cdiv(TDST, kwargs["BLOCK_TSRC"]),)

        _sw_score_sample[grid](
            q_for_landmark,
            *safe_stride(q_for_landmark, 4),
            k,
            *safe_stride(k, 4),
            position_ids_for_landmark,
            *safe_stride(position_ids_for_landmark, 2),
            args.using_paged_cache,
            k_cache,
            *safe_stride(k_cache, 4),
            args.block_table,
            *safe_stride(args.block_table, 2),
            landmark_scores,
            *safe_stride(landmark_scores, 3),
            landmark_chunk,
            TDST,
            HEAD,
            HEAD_KV,
            HID,
            # BLOCK_TDST,
            # BLOCK_TSRC,
        )

        landmark_scores[:, :, -min(landmark_chunk, 32) :].fill_(0)

        if state is not None:
            if args.block_table is not None:
                q_block_index = args.block_table.gather(
                    dim=1, index=position_ids_for_landmark
                )
            else:
                assert args.position_ids.shape[0] == 1
                q_block_index = args.position_ids[0]
            # sanity_check = q_block_index.amax().item()
            # assert sanity_check < state.landmark_scores.shape[0], f'{sanity_check=} < {state.landmark_scores.shape=}[0]'
            state.landmark_scores[q_block_index] = (
                landmark_scores[:, :, : q_for_landmark.shape[1]]
                .contiguous()
                .permute(0, 2, 1)
            )
            if args.block_table is not None:
                landmark_scores = state.landmark_scores[
                    args.block_table[
                        :,
                        : args.block_table.shape[1]
                        - (args.block_table.shape[1] % landmark_chunk),
                    ]
                ]
            else:
                assert k is not None
                assert k.shape[0] == 1
                landmark_scores = state.landmark_scores[None, : k.shape[1], :]
            landmark_scores = landmark_scores.permute(0, 2, 1)

        # print(q_for_landmark.shape, HEAD, HEAD_KV, TDST, triton.cdiv(TDST, BLOCK_TSRC), position_ids_for_landmark.shape)
        if DEBUG:
            plt.clf()
            plt.plot(
                landmark_scores[
                    0,
                    0,
                ]
                .cpu()
                .numpy()
            )
            plt.savefig("dummy_landmark.png")
    else:

        def pad_seq(t: torch.Tensor):
            if (t.shape[1] % landmark_chunk) == 0:
                return t
            pad = landmark_chunk - t.shape[1] % landmark_chunk
            return torch.nn.functional.pad(t, pad=(0, 0, 0, 0, 0, pad))

        def split_half(x: Tensor):
            HID = x.shape[-1]
            return x[..., : HID // 2], x[..., HID // 2 :]

        def merge_half(x: Tensor, y: Tensor):
            return torch.cat([x, y], dim=-1)

        def de_rope(vec: Tensor, cos: Tensor, sin: Tensor):
            c0, ch = split_half(cos)
            s0, sh = split_half(sin)
            vr0, vrh = split_half(vec)

            out0 = (vrh * s0 + vr0 * ch) / (c0 * ch + sh * s0 + 1e-20)
            outh = (out0 * c0 - vr0) / (s0 + 1e-20)
            out = merge_half(out0, outh)
            return out

        if state is not None:
            q_for_landmark = (
                args.query_for_landmark if args.query_for_landmark is not None else q
            )
            position_ids_for_landmark = (
                args.position_ids_for_landmark
                if args.position_ids_for_landmark is not None
                else args.position_ids
            )
            k_chunk = args.gather_extend_k_from_paged_cache(
                disable_gqa=False,
                gqa_q=q_for_landmark,
                position_ids=position_ids_for_landmark,
            )

            q_tp = pad_seq(q_for_landmark)
            TDST_PADDED = q_tp.shape[1]
            k_tp = pad_seq(k_chunk)
            TSRC_PADDED = k_tp.shape[1]
            assert TDST_PADDED == TSRC_PADDED, f"{TDST_PADDED} == {TSRC_PADDED}"

            if landmark_derope:
                padded_position_ids_for_landmark = pad_seq(
                    position_ids_for_landmark[:, :, None, None]
                )[:, :, 0, 0]
                q_tp = de_rope(
                    q_tp,
                    args.rope_cos[padded_position_ids_for_landmark, :][:, :, None, :],
                    args.rope_sin[padded_position_ids_for_landmark, :][:, :, None, :],
                )
                k_tp = de_rope(
                    k_tp,
                    args.rope_cos[padded_position_ids_for_landmark, :][:, :, None, :],
                    args.rope_sin[padded_position_ids_for_landmark, :][:, :, None, :],
                )

            q_tp = q_tp.permute(0, 2, 1, 3).reshape(
                BSZ, HEAD, TDST_PADDED // landmark_chunk, landmark_chunk, HID
            )
            k_tp = (
                k_tp.permute(0, 2, 3, 1)
                .reshape(
                    BSZ, HEAD_KV, HID, TSRC_PADDED // landmark_chunk, landmark_chunk
                )
                .permute(0, 1, 3, 2, 4)
                .repeat_interleave(dim=1, repeats=HEAD // HEAD_KV)
            )

            landmark_scores = torch.matmul(q_tp, k_tp)  # .to(torch.float32)

            # TODO Need to handle chunked prefill scenario
            idx_t = torch.arange(0, landmark_chunk, device=q_for_landmark.device)
            mask = idx_t[:, None] >= idx_t[None, :]
            landmark_scores = landmark_scores * mask[None, None, None, :, :]
            assert landmark_scores.shape == (
                BSZ,
                HEAD,
                TSRC_PADDED // landmark_chunk,
                landmark_chunk,
                landmark_chunk,
            )
            landmark_scores = (
                landmark_scores.sum(dim=3, dtype=torch.float32)
                / mask.int().sum(dim=0)[None, None, None, :]
            )
            landmark_scores = landmark_scores.view(BSZ, HEAD, TSRC_PADDED)
            landmark_scores[:, :, q_for_landmark.shape[1] :].fill_(float("-inf"))

            q_block_index = args.block_table.gather(
                dim=1, index=position_ids_for_landmark
            )
            # sanity_check = q_block_index.amax().item()
            # assert sanity_check < state.landmark_scores.shape[0], f'{sanity_check=} < {state.landmark_scores.shape=}[0]'
            state.landmark_scores[q_block_index] = (
                landmark_scores[:, :, : q_for_landmark.shape[1]]
                .contiguous()
                .permute(0, 2, 1)
            )
            landmark_scores = state.landmark_scores[
                args.block_table[
                    :,
                    : args.block_table.shape[1]
                    - (args.block_table.shape[1] % landmark_chunk),
                ]
            ]
            landmark_scores = landmark_scores.permute(0, 2, 1)
            # print('landmark score extended', args.layer_id)
        else:
            q_for_landmark = (
                args.query_for_landmark if args.query_for_landmark is not None else q
            )
            position_ids_for_landmark = (
                args.position_ids_for_landmark
                if args.position_ids_for_landmark is not None
                else args.position_ids
            )

            q_tp = pad_seq(q_for_landmark)
            TDST_PADDED = q_tp.shape[1]
            k_tp = pad_seq(k)
            TSRC_PADDED = k_tp.shape[1]
            assert TDST_PADDED == TSRC_PADDED

            if landmark_derope:
                padded_position_ids_for_landmark = pad_seq(
                    position_ids_for_landmark[:, :, None, None]
                )[:, :, 0, 0]
                q_tp = de_rope(
                    q_tp,
                    args.rope_cos[padded_position_ids_for_landmark, :][:, :, None, :],
                    args.rope_sin[padded_position_ids_for_landmark, :][:, :, None, :],
                )
                k_tp = de_rope(
                    k_tp,
                    args.rope_cos[padded_position_ids_for_landmark, :][:, :, None, :],
                    args.rope_sin[padded_position_ids_for_landmark, :][:, :, None, :],
                )

            q_tp = q_tp.permute(0, 2, 1, 3).reshape(
                BSZ, HEAD, TDST_PADDED // landmark_chunk, landmark_chunk, HID
            )
            k_tp = (
                k_tp.permute(0, 2, 3, 1)
                .reshape(
                    BSZ, HEAD_KV, HID, TSRC_PADDED // landmark_chunk, landmark_chunk
                )
                .permute(0, 1, 3, 2, 4)
                .repeat_interleave(dim=1, repeats=HEAD // HEAD_KV)
            )
            # print(q_tp.shape, k_tp.shape)
            landmark_scores = torch.matmul(q_tp, k_tp)  # .to(torch.float32)
            # TODO Need to handle chunked prefill scenario
            # idx_tdst = args.position_ids[0]
            idx_t = torch.arange(0, landmark_chunk, device=q.device)
            mask = idx_t[:, None] >= idx_t[None, :]
            landmark_scores = landmark_scores * mask[None, None, None, :, :]
            assert landmark_scores.shape == (
                BSZ,
                HEAD,
                TSRC_PADDED // landmark_chunk,
                landmark_chunk,
                landmark_chunk,
            )
            landmark_scores = (
                landmark_scores.sum(dim=3) / mask.int().sum(dim=0)[None, None, None, :]
            )
            landmark_scores = landmark_scores.view(BSZ, HEAD, TSRC_PADDED)
            landmark_scores[:, :, k.shape[1] :].fill_(float("-inf"))

    if DEBUG and (BDST > 1):
        os.makedirs("./cache/mask_log", exist_ok=True)
        t = landmark_scores[0, 0, :].cpu().numpy()
        plt.clf()
        plt.plot(t)
        plt.savefig(f"./cache/mask_log/{__logall_index}_landmark_scores.png")

    return landmark_scores
```

### `hip_attn/v1_2/scan_stage.py`

```py
import os
import warnings

import torch
import triton
import triton.language as tl

from hip_attn.utils.rope import adjust_rope
from hip_attn.v1_2.attention_metadata import safe_stride
from hip_attn.v1_2.uvm_gpu_cache import load_tokens


@triton.jit
def load_keys_with_rope(
    K,
    stride_k_bsz,
    stride_k_tsrc,
    stride_k_head_kv,
    stride_k_hid,
    COS,
    stride_cos_t,
    stride_cos_hid,
    SIN,
    stride_sin_t,
    stride_sin_hid,
    # paged attention args template
    USING_PAGES,
    PAGE_SIZE,
    K_CACHE,
    stride_k_cache_page,
    stride_k_cache_offset,
    stride_k_cache_kv_head,
    stride_k_cache_hid,
    BLOCK_TABLE,
    stride_block_table_bsz,
    stride_block_table_page,
    CACHE_SEQ_LENS,
    stride_cache_seq_lens_b,
    USING_OFFLOAD_CACHE: tl.constexpr,
    OFFLOAD_CACHE_KV_PACKED: tl.constexpr,
    GPU_BANK_COUNT,
    OFFLOAD_CACHE_UVM_METADATA,
    stride_offload_cache_uvm_metadata_token,
    stride_offload_cache_uvm_metadata_k,
    OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
    stride_offload_cache_gpu_global_metadata_k,
    stride_offload_cache_gpu_global_metadata_pad,
    OFFLOAD_CACHE_GPU_BANK,
    stride_offload_cache_gpu_bank_token,
    stride_offload_cache_gpu_bank_hid,
    OFFLOAD_CACHE_GPU_METADATA,
    stride_offload_cache_gpu_metadata_token,
    stride_offload_cache_gpu_metadata_k,
    OFFLOAD_CACHE_GPU_TABLE,
    stride_offload_cache_gpu_table_head_kv,
    stride_offload_cache_gpu_table_token,
    strdie_offload_cache_gpu_table_k,
    ACCESS_COUNTER,
    stride_access_counter_bsz,
    stride_access_counter_head_kv,
    stride_access_counter_tsrc,
    CACHE_MISS_COUNTER,
    stride_cache_miss_counter_bsz,
    stride_cache_miss_counter_head_kv,
    stride_cache_miss_counter_tsrc,
    queries_dtype,
    idx_bsz,
    idx_tsrc,
    idx_head_kv,
    idx_hid,
    idx_chunk,
    mask_tsrc_active,
    mask_tdst,
    mask_hid,
    real_pos_tdst_min,
    model_context_length,
    num_sinks,
    USING_EXTEND,
    EXTEND_BACKEND,
    NEED_APPLY_ROPE,
    BLOCK_CHUNK,
    BLOCK_HID: tl.constexpr,
    HID_DIM,
    IS_RIGHT,
    HEAD_KV,
    UPDATE_CACHE,
    rope_range_begin: tl.constexpr,
    rope_range_end: tl.constexpr,
    rope_is_neox_style: tl.constexpr,
):
    keys_left = load_tokens(
        K,
        stride_k_bsz,
        stride_k_tsrc,
        stride_k_head_kv,
        stride_k_hid,
        USING_PAGES,
        PAGE_SIZE,
        K_CACHE,
        stride_k_cache_page,
        stride_k_cache_offset,
        stride_k_cache_kv_head,
        stride_k_cache_hid,
        BLOCK_TABLE,
        stride_block_table_bsz,
        stride_block_table_page,
        CACHE_SEQ_LENS,
        stride_cache_seq_lens_b,
        USING_OFFLOAD_CACHE,
        OFFLOAD_CACHE_KV_PACKED,
        GPU_BANK_COUNT,
        False,
        OFFLOAD_CACHE_UVM_METADATA,
        stride_offload_cache_uvm_metadata_token,
        stride_offload_cache_uvm_metadata_k,
        OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
        stride_offload_cache_gpu_global_metadata_k,
        stride_offload_cache_gpu_global_metadata_pad,
        OFFLOAD_CACHE_GPU_BANK,
        stride_offload_cache_gpu_bank_token,
        stride_offload_cache_gpu_bank_hid,
        OFFLOAD_CACHE_GPU_METADATA,
        stride_offload_cache_gpu_metadata_token,
        stride_offload_cache_gpu_metadata_k,
        OFFLOAD_CACHE_GPU_TABLE,
        stride_offload_cache_gpu_table_head_kv,
        stride_offload_cache_gpu_table_token,
        strdie_offload_cache_gpu_table_k,
        ACCESS_COUNTER,
        stride_access_counter_bsz,
        stride_access_counter_head_kv,
        stride_access_counter_tsrc,
        CACHE_MISS_COUNTER,
        stride_cache_miss_counter_bsz,
        stride_cache_miss_counter_head_kv,
        stride_cache_miss_counter_tsrc,
        idx_bsz,
        idx_tsrc[None, :],
        idx_head_kv,
        idx_hid[:, None],
        mask_tsrc_active[None, :],  # & mask_hid[:, None],
        # mask_tsrc_active[None, :] & mask_hid[:, None],
        HEAD_KV,
        BLOCK_CHUNK,
        BLOCK_HID,
        HID_DIM,
        UPDATE_CACHE=UPDATE_CACHE,
    ).to(queries_dtype)

    if USING_EXTEND:
        ROPE_DIM = rope_range_end - rope_range_begin

        idx_rope_range = idx_hid - rope_range_begin
        rope_mask = (rope_range_begin <= idx_hid) & (idx_hid < rope_range_end)
        if rope_is_neox_style:
            rope_rot_idx = tl.where(
                rope_mask,
                (idx_rope_range + ROPE_DIM // 2) % ROPE_DIM + rope_range_begin,
                idx_hid,
            )
            cos_sin_idx = idx_rope_range % (ROPE_DIM // 2)
            rope_mult = ((idx_rope_range + ROPE_DIM // 2 < ROPE_DIM) * (-2) + 1).to(
                queries_dtype
            )
        else:
            flip = tl.where(idx_rope_range & 1 == 0, 1, -1)
            rope_rot_idx = tl.where(
                rope_mask,
                idx_rope_range + flip + rope_range_begin,
                idx_hid,
            )
            cos_sin_idx = idx_rope_range // 2
            rope_mult = ((idx_rope_range % 2 == 0) * (-2) + 1).to(queries_dtype)

        real_pos_tdst_max = tl.sum(mask_tdst.to(tl.int32)) + real_pos_tdst_min
        tsrc_extend = tl.maximum(0, real_pos_tdst_max - model_context_length)
        if NEED_APPLY_ROPE or (tsrc_extend >= 0):
            old_tsrc = idx_tsrc

            if EXTEND_BACKEND == "dynamic_extend":
                window = model_context_length // 4

                new_tsrc = tl.where(
                    (idx_tsrc >= (real_pos_tdst_max - window))
                    | (real_pos_tdst_max <= model_context_length),
                    idx_tsrc,
                    # idx_tsrc * 0 + real_pos_tdst_max,
                    (
                        (idx_tsrc.to(tl.float32) - (real_pos_tdst_min - window))
                        * (
                            (model_context_length - window)
                            / (real_pos_tdst_min - window)
                        ).to(tl.float32)
                    ).to(tl.int32)
                    + (real_pos_tdst_min - window),
                )
                # new_tsrc = idx_tsrc * 0 + real_pos_tdst_max
                new_tsrc = tl.maximum(
                    real_pos_tdst_max - model_context_length, new_tsrc
                )
            elif EXTEND_BACKEND == "self_extend":
                window = 8192
                group_size = 16

                new_tsrc = tl.where(
                    idx_tsrc >= (real_pos_tdst_max - window),
                    idx_tsrc,
                    tl.where(
                        real_pos_tdst_max <= model_context_length,
                        idx_tsrc,
                        (idx_tsrc - real_pos_tdst_min) // group_size
                        + real_pos_tdst_min,
                    ),
                )
                new_tsrc = tl.maximum(0, new_tsrc)
            elif EXTEND_BACKEND == "relative":
                new_tsrc = idx_chunk * 0
                if IS_RIGHT:
                    new_tsrc += 1
            elif EXTEND_BACKEND == "infllm":
                new_tsrc = idx_chunk * 0
            elif EXTEND_BACKEND == "streaming":
                # streaming
                new_tsrc = idx_chunk
            else:
                raise Exception()

            if not NEED_APPLY_ROPE:
                tl.static_assert(False)
                keys_left = keys_left.trans(1, 0)
                keys_left = adjust_rope(
                    keys_left,
                    old_tsrc,
                    new_tsrc,
                    mask_tsrc_active,
                    idx_hid,
                    COS,
                    stride_cos_t,
                    stride_cos_hid,
                    SIN,
                    stride_sin_t,
                    stride_sin_hid,
                    BLOCK_CHUNK,
                    BLOCK_HID,
                    HID_DIM,
                    NEED_APPLY_ROPE,
                    rope_range_begin,
                    rope_range_end,
                    rope_is_neox_style,
                ).to(keys_left.dtype)
                keys_left = tl.trans(keys_left, 1, 0)
                keys_left = (keys_left * mask_tsrc_active[None, :]).to(keys_left.dtype)
            else:
                keys_left_rot = load_tokens(
                    K,
                    stride_k_bsz,
                    stride_k_tsrc,
                    stride_k_head_kv,
                    stride_k_hid,
                    USING_PAGES,
                    PAGE_SIZE,
                    K_CACHE,
                    stride_k_cache_page,
                    stride_k_cache_offset,
                    stride_k_cache_kv_head,
                    stride_k_cache_hid,
                    BLOCK_TABLE,
                    stride_block_table_bsz,
                    stride_block_table_page,
                    CACHE_SEQ_LENS,
                    stride_cache_seq_lens_b,
                    USING_OFFLOAD_CACHE,
                    OFFLOAD_CACHE_KV_PACKED,
                    GPU_BANK_COUNT,
                    False,
                    OFFLOAD_CACHE_UVM_METADATA,
                    stride_offload_cache_uvm_metadata_token,
                    stride_offload_cache_uvm_metadata_k,
                    OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                    stride_offload_cache_gpu_global_metadata_k,
                    stride_offload_cache_gpu_global_metadata_pad,
                    OFFLOAD_CACHE_GPU_BANK,
                    stride_offload_cache_gpu_bank_token,
                    stride_offload_cache_gpu_bank_hid,
                    OFFLOAD_CACHE_GPU_METADATA,
                    stride_offload_cache_gpu_metadata_token,
                    stride_offload_cache_gpu_metadata_k,
                    OFFLOAD_CACHE_GPU_TABLE,
                    stride_offload_cache_gpu_table_head_kv,
                    stride_offload_cache_gpu_table_token,
                    strdie_offload_cache_gpu_table_k,
                    ACCESS_COUNTER,
                    stride_access_counter_bsz,
                    stride_access_counter_head_kv,
                    stride_access_counter_tsrc,
                    CACHE_MISS_COUNTER,
                    stride_cache_miss_counter_bsz,
                    stride_cache_miss_counter_head_kv,
                    stride_cache_miss_counter_tsrc,
                    idx_bsz,
                    idx_tsrc[None, :],
                    idx_head_kv,
                    rope_rot_idx[:, None],
                    mask_tsrc_active[None, :],
                    HEAD_KV,
                    BLOCK_CHUNK,
                    BLOCK_HID,
                    HID_DIM,
                    # NOTE: in previous load, the fetch should be succesfully done.
                    UPDATE_CACHE=UPDATE_CACHE,
                ).to(queries_dtype)

                # TODO: multiply -right
                # keys_left_rot = tl.where(
                #     (idx_hid + BLOCK_HID // 2)[:, None] < BLOCK_HID,
                #     -keys_left_rot,
                #     keys_left_rot
                # )

                keys_left_rot *= rope_mult[:, None]

                cos_new = tl.load(
                    COS
                    + new_tsrc[None, :].to(tl.int64) * stride_cos_t
                    + cos_sin_idx[:, None] * stride_cos_hid,
                    mask=mask_tsrc_active[None, :] & rope_mask[:, None],
                    other=0.0,
                ).to(keys_left.dtype)
                sin_new = tl.load(
                    SIN
                    + new_tsrc[None, :].to(tl.int64) * stride_sin_t
                    + cos_sin_idx[:, None] * stride_sin_hid,
                    mask=mask_tsrc_active[None, :] & rope_mask[:, None],
                    other=0.0,
                ).to(keys_left.dtype)

                keys_left = tl.where(
                    rope_mask[:, None],
                    keys_left * cos_new + keys_left_rot * sin_new,
                    keys_left,
                )

    return keys_left


@triton.jit
def pool_queries(
    idx_bsz,
    idx_head,
    pos_tdst,
    idx_tdst,
    mask_tdst,
    idx_hid,
    mask_hid,
    Q,
    stride_q_bsz,
    stride_q_tdst,
    stride_q_head,
    stride_q_hid,
    COS,
    stride_cos_t,
    stride_cos_hid,
    SIN,
    stride_sin_t,
    stride_sin_hid,
    rope_range_begin: tl.constexpr,
    rope_range_end: tl.constexpr,
    rope_is_neox_style: tl.constexpr,
    HID_DIM: int,
    TDST: int,
    CHUNK_COUNT: int,
    real_pos_tdst_min: int,
    model_context_length: int,
    sliding_window_size: int,
    USING_EXTEND: tl.constexpr,
    NEED_APPLY_ROPE: tl.constexpr,
    EXTEND_BACKEND: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    HID_BLOCK: tl.constexpr,
    STRIDE_Q: tl.constexpr,
):
    ROPE_DIM = rope_range_end - rope_range_begin

    idx_rope_range = idx_hid - rope_range_begin
    rope_mask = (rope_range_begin <= idx_hid) & (idx_hid < rope_range_end)
    if rope_is_neox_style:
        rope_rot_idx = tl.where(
            rope_mask,
            (idx_rope_range + ROPE_DIM // 2) % ROPE_DIM + rope_range_begin,
            idx_hid,
        )
        cos_sin_idx = idx_rope_range % (ROPE_DIM // 2)
        rope_mult = (idx_rope_range + ROPE_DIM // 2 < ROPE_DIM) * (-2) + 1
    else:
        flip = tl.where(idx_rope_range & 1 == 0, 1, -1)
        rope_rot_idx = tl.where(
            rope_mask,
            idx_rope_range + flip + rope_range_begin,
            idx_hid,
        )
        cos_sin_idx = idx_rope_range // 2
        rope_mult = (idx_rope_range % 2 == 0) * (-2) + 1

    queries_sum = tl.zeros((BLOCK_SIZE_Q // STRIDE_Q, HID_BLOCK), dtype=tl.float32)
    queries_counter = tl.zeros((BLOCK_SIZE_Q // STRIDE_Q,), dtype=tl.int32)
    tl.static_assert(BLOCK_SIZE_Q // STRIDE_Q > 0)

    for i_offset in tl.range(0, STRIDE_Q, num_stages=3):
        idx_tdst_iter = idx_tdst + i_offset
        mask_tdst_iter = mask_tdst & (idx_tdst_iter < TDST)
        queries_iter = tl.load(
            Q
            + idx_bsz * stride_q_bsz
            + idx_tdst_iter[:, None] * stride_q_tdst
            + idx_head * stride_q_head
            + idx_hid[None, :] * stride_q_hid,
            mask=mask_tdst_iter[:, None] & mask_hid[None, :],
            other=0.0,
        )
        if queries_iter.dtype == tl.float8e5:
            queries_iter = queries_iter.to(tl.float16)

        if USING_EXTEND:
            if NEED_APPLY_ROPE or (real_pos_tdst_min >= model_context_length):
                old_tdst = pos_tdst
                if EXTEND_BACKEND == "dynamic_extend":
                    new_tdst = pos_tdst
                elif EXTEND_BACKEND == "self_extend":
                    new_tdst = pos_tdst
                elif EXTEND_BACKEND == "relative":
                    new_tdst = pos_tdst * 0 + 1 + sliding_window_size
                elif EXTEND_BACKEND == "infllm":
                    new_tdst = pos_tdst * 0 + sliding_window_size
                elif EXTEND_BACKEND == "streaming":
                    # streaming
                    new_tdst = tl.minimum(pos_tdst, CHUNK_COUNT + sliding_window_size)
                else:
                    raise Exception()

                if NEED_APPLY_ROPE:
                    queries_rot = tl.load(
                        Q
                        + idx_bsz * stride_q_bsz
                        + idx_tdst_iter[:, None] * stride_q_tdst
                        + idx_head * stride_q_head
                        + rope_rot_idx[None, :] * stride_q_hid,
                        mask=mask_tdst_iter[:, None]
                        & rope_mask[None, :]
                        & mask_hid[None, :],
                        other=0.0,
                    )
                    if queries_rot.dtype == tl.float8e5:
                        queries_rot = queries_rot.to(tl.float16)

                    cos_new = tl.load(
                        COS
                        + new_tdst[:, None].to(tl.int64) * stride_cos_t
                        + cos_sin_idx[None, :] * stride_cos_hid,
                        mask=mask_tdst_iter[:, None]
                        & rope_mask[None, :]
                        & mask_hid[None, :],
                        other=0.0,
                    ).to(queries_iter.dtype)
                    sin_new = tl.load(
                        SIN
                        + new_tdst[:, None].to(tl.int64) * stride_sin_t
                        + cos_sin_idx[None, :] * stride_sin_hid,
                        mask=mask_tdst_iter[:, None]
                        & rope_mask[None, :]
                        & mask_hid[None, :],
                        other=0.0,
                    ).to(queries_iter.dtype)

                    queries_rot *= rope_mult[None, :].to(queries_rot.dtype)

                    queries_iter = tl.where(
                        rope_mask[None, :] & mask_hid[None, :],
                        (queries_iter * cos_new + queries_rot * sin_new).to(
                            queries_iter.dtype
                        ),
                        queries_iter,
                    )
                else:
                    raise Exception()
                    queries_iter = adjust_rope(
                        queries_iter,
                        old_tdst,
                        new_tdst,
                        mask_tdst_iter,
                        idx_hid,
                        COS,
                        stride_cos_t,
                        stride_cos_hid,
                        SIN,
                        stride_sin_t,
                        stride_sin_hid,
                        BLOCK_SIZE_Q // STRIDE_Q,
                        HID_BLOCK,
                        HID_DIM,
                        NEED_APPLY_ROPE,
                        rope_range_begin,
                        rope_range_end,
                        rope_is_neox_style,
                    ).to(queries_iter.dtype)
                    queries_iter = (queries_iter * mask_tdst_iter[:, None]).to(
                        queries_iter.dtype
                    )

        queries_sum += queries_iter
        queries_counter += mask_tdst_iter.to(tl.int32)

    queries = (queries_sum / (queries_counter[:, None] + 1e-12)) * mask_tdst[:, None]
    if Q.dtype.element_ty != tl.float8e5:
        queries = queries.to(Q.dtype.element_ty)
    else:
        queries = queries.to(tl.float16)

    return queries


def get_scan_stage_configs():
    autotune_disabled = os.getenv("HIP_DISABLE_AUTOTUNE", "1") == "1"
    if autotune_disabled:
        device_name = torch.cuda.get_device_name()
        defaults = {
            "NVIDIA A100-SXM4-80GB": dict(
                num_warps=4,
                num_stages=2,
                maxnreg=256,
            ),
        }.get(device_name, dict(num_warps=4, num_stages=2))
        return [triton.Config({}, **defaults)]
    if os.getenv("HIP_DISABLE_AUTOTUNE_WARNINGS", "0") == "0":
        warnings.warn(
            "triton autotuning is activated. this should be disabled for faster startup. if you want set HIP_DISABLE_AUTOTUNE=1"
        )

    NUM_WARPS = [4]  # workaround for triton bug
    if triton.__version__ >= "3.2.0":
        NUM_WARPS.append(8)

    configs = []
    for LOAD_Q_EACH_TIME in [
        False,
    ]:
        for num_warps in NUM_WARPS:
            for num_stages in [1, 2, 3]:
                configs.append(
                    triton.Config(
                        {"LOAD_Q_EACH_TIME": LOAD_Q_EACH_TIME},
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                )
    return configs


@triton.autotune(
    configs=get_scan_stage_configs(),
    key=[
        "BLOCK_SIZE_Q",
        "HID_DIM",
        "USING_PAGES",
    ],
    restore_value=[
        "INDICES_LEFT",
        "INDICES_RIGHT",
    ]
)
@triton.jit
def chunk_controllable_sampling_mask_cuda(
    Q,
    stride_q_bsz,
    stride_q_tdst,
    stride_q_head,
    stride_q_hid,
    K,
    stride_k_bsz,
    stride_k_tsrc,
    stride_k_head_kv,
    stride_k_hid,
    POS,
    stride_pos_bsz,
    stride_pos_tdst,
    # paged attention args template
    USING_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    K_CACHE,
    stride_k_cache_page,
    stride_k_cache_offset,
    stride_k_cache_kv_head,
    stride_k_cache_hid,
    V_CACHE,
    stride_v_cache_page,
    stride_v_cache_offset,
    stride_v_cache_kv_head,
    stride_v_cache_hid,
    BLOCK_TABLE,
    stride_block_table_bsz,
    stride_block_table_page,
    CACHE_SEQ_LENS,
    stride_cache_seq_lens_b,
    USING_OFFLOAD_CACHE: tl.constexpr,
    OFFLOAD_CACHE_KV_PACKED: tl.constexpr,
    GPU_BANK_COUNT,
    OFFLOAD_CACHE_UVM_METADATA,
    stride_offload_cache_uvm_metadata_token,
    stride_offload_cache_uvm_metadata_k,
    OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
    stride_offload_cache_gpu_global_metadata_k,
    stride_offload_cache_gpu_global_metadata_pad,
    OFFLOAD_CACHE_GPU_BANK,
    stride_offload_cache_gpu_bank_token,
    stride_offload_cache_gpu_bank_hid,
    OFFLOAD_CACHE_GPU_METADATA,
    stride_offload_cache_gpu_metadata_token,
    stride_offload_cache_gpu_metadata_k,
    OFFLOAD_CACHE_GPU_TABLE,
    stride_offload_cache_gpu_table_head_kv,
    stride_offload_cache_gpu_table_token,
    strdie_offload_cache_gpu_table_k,
    INDICES_LEFT,
    stride_indices_left_bsz,
    stride_indices_left_bdst,
    stride_indices_left_head,
    stride_indices_left_chunk,
    INDICES_RIGHT,
    stride_indices_right_bsz,
    stride_indices_right_bdst,
    stride_indices_right_head,
    stride_indices_right_chunk,
    OUT_SCORES,
    stride_out_scores_bsz,
    stride_out_scores_bdst,
    stride_out_scores_head,
    stride_out_scores_chunk,
    COS,
    stride_cos_t,
    stride_cos_hid,
    SIN,
    stride_sin_t,
    stride_sin_hid,
    rope_range_begin: tl.constexpr,
    rope_range_end: tl.constexpr,
    rope_is_neox_style: tl.constexpr,
    MASK_ACCESS_COUNTER,
    stride_mask_access_counter_bsz,
    stride_mask_access_counter_head_kv,
    stride_mask_access_counter_tsrc,
    MASK_CACHE_MISS_COUNTER,
    stride_mask_cache_miss_counter_bsz,
    stride_mask_cache_miss_counter_head_kv,
    stride_mask_cache_miss_counter_tsrc,
    CHUNK_COUNT: int,
    MAX_TSRC: int,
    TDST: int,
    HEAD: int,
    sliding_window_size: int,
    num_sinks: int,
    model_context_length: int,
    group_jobs: int,
    total_jobs: int,
    HID_DIM: tl.constexpr,
    HID_BLOCK_0: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr = 32,
    STRIDE_Q: tl.constexpr = 1,
    BLOCK_CHUNK: tl.constexpr = 32,
    HEAD_GROUP: tl.constexpr = 4,
    REDUCE: tl.constexpr = "max",
    USING_EXTEND: tl.constexpr = False,
    EXTEND_BACKEND: tl.constexpr = "relative",
    NEED_APPLY_ROPE: tl.constexpr = False,
    TERMINATE_SIZE: tl.constexpr = 1,
    SCAN_STRIDE: tl.constexpr = 1,
    UPDATE_CACHE: tl.constexpr = True,
    ORACLE_MAXIMUM: tl.constexpr = False,
    LOAD_Q_EACH_TIME: tl.constexpr = False,
    COMPUTE_MLA_ROPE: tl.constexpr = False,
):
    BDST = tl.cdiv(TDST, BLOCK_SIZE_Q)
    BDST_SCAN = tl.cdiv(BDST, SCAN_STRIDE)
    BCHUNK = tl.cdiv(CHUNK_COUNT, BLOCK_CHUNK)

    pid_group = tl.program_id(0).to(tl.int64)

    for i in range(group_jobs):
        pid = pid_group * group_jobs + i
        if pid < total_jobs:
            idx_head = pid % HEAD
            pid = pid // HEAD
            idx_bdst_scan = pid % BDST_SCAN
            pid = pid // BDST_SCAN
            idx_bchunk = pid % BCHUNK
            pid = pid // BCHUNK
            idx_bsz = pid

            # idx_tdst = idx_bdst * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q // STRIDE_Q) * STRIDE_Q
            # mask_tdst = idx_tdst < TDST
            if BLOCK_SIZE_Q // STRIDE_Q < 16:
                idx_tdst = (
                    (BDST - 1)
                    - (BDST_SCAN - 1) * SCAN_STRIDE
                    + idx_bdst_scan * SCAN_STRIDE
                ) * BLOCK_SIZE_Q + tl.arange(0, 16) * STRIDE_Q
                mask_tdst = (
                    (idx_tdst < TDST)
                    & (idx_tdst >= 0)
                    & (tl.arange(0, 16) < (BLOCK_SIZE_Q // STRIDE_Q))
                )
            else:
                idx_tdst = (
                    (BDST - 1)
                    - (BDST_SCAN - 1) * SCAN_STRIDE
                    + idx_bdst_scan * SCAN_STRIDE
                ) * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q // STRIDE_Q) * STRIDE_Q
                mask_tdst = (idx_tdst < TDST) & (idx_tdst >= 0)

            HID_BLOCK_1: tl.constexpr = HID_DIM - HID_BLOCK_0

            idx_hid_q0 = tl.arange(0, HID_BLOCK_0)
            mask_hid_q0 = idx_hid_q0 < HID_DIM

            if HID_BLOCK_1 > 0:
                idx_hid_q1 = HID_BLOCK_0 + tl.arange(0, HID_BLOCK_1)
                mask_hid_q1 = idx_hid_q1 < HID_DIM
            else:
                idx_hid_q1 = None
                mask_hid_q1 = None

            pos_tdst = tl.load(
                POS + idx_bsz * stride_pos_bsz + idx_tdst * stride_pos_tdst,
                mask=mask_tdst,
                other=0,
            )

            # real_pos_tdst_min = idx_bdst * BLOCK_SIZE_Q + TSRC - TDST
            real_pos_tdst_min = tl.min(tl.where(mask_tdst, pos_tdst, 999999999))
            real_pos_tdst_min = tl.where(
                tl.sum(mask_tdst.to(tl.int32)) > 0, real_pos_tdst_min, -1
            )

            if Q.dtype.element_ty != tl.float8e5:
                q_dtype = Q.dtype.element_ty
            else:
                q_dtype = tl.float16
            cq = (tl.sqrt(HID_DIM * 1.0) / tl.sqrt(tl.sqrt(HID_DIM * 1.0))).to(q_dtype)
            ck = (1.0 / tl.sqrt(tl.sqrt(HID_DIM * 1.0))).to(q_dtype)

            if real_pos_tdst_min >= 0:
                pos_tdst_min = (real_pos_tdst_min - sliding_window_size).to(tl.int32)
                pos_tdst_min = tl.maximum(pos_tdst_min, 0)

                idx_chunk = idx_bchunk * BLOCK_CHUNK + tl.arange(0, BLOCK_CHUNK)
                mask_chunk = idx_chunk < CHUNK_COUNT

                idx_tsrc_left = tl.load(
                    INDICES_LEFT
                    + idx_bsz * stride_indices_left_bsz
                    + idx_bdst_scan * stride_indices_left_bdst
                    + idx_head * stride_indices_left_head
                    + idx_chunk * stride_indices_left_chunk,
                    mask=mask_chunk,
                    other=MAX_TSRC,
                ).to(tl.int32)

                idx_tsrc_right = tl.load(
                    INDICES_RIGHT
                    + idx_bsz * stride_indices_right_bsz
                    + idx_bdst_scan * stride_indices_right_bdst
                    + idx_head * stride_indices_right_head
                    + idx_chunk * stride_indices_right_chunk,
                    mask=mask_chunk,
                    other=MAX_TSRC,
                ).to(tl.int32)

                if (real_pos_tdst_min + BLOCK_SIZE_Q * SCAN_STRIDE) >= tl.min(
                    idx_tsrc_left
                ):
                    max_chunk_size = tl.max(idx_tsrc_right - idx_tsrc_left).to(
                        tl.float32
                    )

                    scores = tl.zeros((BLOCK_CHUNK,), dtype=tl.float32) - 32000.0

                    if not LOAD_Q_EACH_TIME:
                        queries_0 = pool_queries(
                            idx_bsz,
                            idx_head,
                            pos_tdst,
                            idx_tdst,
                            mask_tdst,
                            idx_hid_q0,
                            mask_hid_q0,
                            Q,
                            stride_q_bsz,
                            stride_q_tdst,
                            stride_q_head,
                            stride_q_hid,
                            COS,
                            stride_cos_t,
                            stride_cos_hid,
                            SIN,
                            stride_sin_t,
                            stride_sin_hid,
                            rope_range_begin,
                            rope_range_end,
                            rope_is_neox_style,
                            HID_DIM,
                            TDST,
                            CHUNK_COUNT,
                            real_pos_tdst_min,
                            model_context_length,
                            sliding_window_size,
                            USING_EXTEND and (rope_range_begin < HID_BLOCK_0),
                            NEED_APPLY_ROPE and (rope_range_begin < HID_BLOCK_0),
                            EXTEND_BACKEND,
                            BLOCK_SIZE_Q,
                            HID_BLOCK_0,
                            STRIDE_Q,
                        )

                        if HID_BLOCK_1 > 0:
                            queries_1 = pool_queries(
                                idx_bsz,
                                idx_head,
                                pos_tdst,
                                idx_tdst,
                                mask_tdst,
                                idx_hid_q1,
                                mask_hid_q1,
                                Q,
                                stride_q_bsz,
                                stride_q_tdst,
                                stride_q_head,
                                stride_q_hid,
                                COS,
                                stride_cos_t,
                                stride_cos_hid,
                                SIN,
                                stride_sin_t,
                                stride_sin_hid,
                                rope_range_begin,
                                rope_range_end,
                                rope_is_neox_style,
                                HID_DIM,
                                TDST,
                                CHUNK_COUNT,
                                real_pos_tdst_min,
                                model_context_length,
                                sliding_window_size,
                                USING_EXTEND,
                                NEED_APPLY_ROPE,
                                EXTEND_BACKEND,
                                BLOCK_SIZE_Q,
                                HID_BLOCK_1,
                                STRIDE_Q,
                            )
                        else:
                            queries_1 = None

                    matmul_dtype = q_dtype
                    # max_chunk_size
                    # while max_chunk_size >= TERMINATE_SIZE:
                    #     max_chunk_size /= 2.0
                    for _ in tl.range(
                        0,
                        tl.ceil(tl.log2(max_chunk_size / TERMINATE_SIZE)).to(tl.int32),
                        num_stages=1 if USING_EXTEND else 3,
                    ):
                        mask_tsrc_active = (
                            mask_chunk
                            & (idx_tsrc_left < idx_tsrc_right)
                            & (idx_tsrc_left <= pos_tdst_min)
                            & (idx_tsrc_left >= 0)
                        )
                        idx_tsrc_center = (idx_tsrc_left + idx_tsrc_right) // 2

                        assert (
                            not ORACLE_MAXIMUM
                        ), "this is deprecated at fad90fea0d37ba88c04e90f2c5597e6800e97e8f"

                        idx_tsrc = (idx_tsrc_left + idx_tsrc_center) // 2

                        if LOAD_Q_EACH_TIME:
                            queries_0 = pool_queries(
                                idx_bsz,
                                idx_head,
                                pos_tdst,
                                idx_tdst,
                                mask_tdst,
                                idx_hid_q0,
                                mask_hid_q0,
                                Q,
                                stride_q_bsz,
                                stride_q_tdst,
                                stride_q_head,
                                stride_q_hid,
                                COS,
                                stride_cos_t,
                                stride_cos_hid,
                                SIN,
                                stride_sin_t,
                                stride_sin_hid,
                                rope_range_begin,
                                rope_range_end,
                                rope_is_neox_style,
                                HID_DIM,
                                TDST,
                                CHUNK_COUNT,
                                real_pos_tdst_min,
                                model_context_length,
                                sliding_window_size,
                                USING_EXTEND and (rope_range_begin < HID_BLOCK_0),
                                NEED_APPLY_ROPE and (rope_range_begin < HID_BLOCK_0),
                                EXTEND_BACKEND,
                                BLOCK_SIZE_Q,
                                HID_BLOCK_0,
                                STRIDE_Q,
                            )

                        keys_left_0 = load_keys_with_rope(
                            K,
                            stride_k_bsz,
                            stride_k_tsrc,
                            stride_k_head_kv,
                            stride_k_hid,
                            COS,
                            stride_cos_t,
                            stride_cos_hid,
                            SIN,
                            stride_sin_t,
                            stride_sin_hid,
                            # paged attention args template
                            USING_PAGES,
                            PAGE_SIZE,
                            K_CACHE,
                            stride_k_cache_page,
                            stride_k_cache_offset,
                            stride_k_cache_kv_head,
                            stride_k_cache_hid,
                            BLOCK_TABLE,
                            stride_block_table_bsz,
                            stride_block_table_page,
                            CACHE_SEQ_LENS,
                            stride_cache_seq_lens_b,
                            USING_OFFLOAD_CACHE,
                            OFFLOAD_CACHE_KV_PACKED,
                            GPU_BANK_COUNT,
                            OFFLOAD_CACHE_UVM_METADATA,
                            stride_offload_cache_uvm_metadata_token,
                            stride_offload_cache_uvm_metadata_k,
                            OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                            stride_offload_cache_gpu_global_metadata_k,
                            stride_offload_cache_gpu_global_metadata_pad,
                            OFFLOAD_CACHE_GPU_BANK,
                            stride_offload_cache_gpu_bank_token,
                            stride_offload_cache_gpu_bank_hid,
                            OFFLOAD_CACHE_GPU_METADATA,
                            stride_offload_cache_gpu_metadata_token,
                            stride_offload_cache_gpu_metadata_k,
                            OFFLOAD_CACHE_GPU_TABLE,
                            stride_offload_cache_gpu_table_head_kv,
                            stride_offload_cache_gpu_table_token,
                            strdie_offload_cache_gpu_table_k,
                            MASK_ACCESS_COUNTER,
                            stride_mask_access_counter_bsz,
                            stride_mask_access_counter_head_kv,
                            stride_mask_access_counter_tsrc,
                            MASK_CACHE_MISS_COUNTER,
                            stride_mask_cache_miss_counter_bsz,
                            stride_mask_cache_miss_counter_head_kv,
                            stride_mask_cache_miss_counter_tsrc,
                            q_dtype,
                            idx_bsz,
                            idx_tsrc,
                            idx_head // HEAD_GROUP,
                            idx_hid_q0,
                            idx_chunk,
                            mask_tsrc_active,
                            mask_tdst,
                            mask_hid_q0,
                            real_pos_tdst_min,
                            model_context_length,
                            num_sinks,
                            USING_EXTEND and (rope_range_begin < HID_BLOCK_0),
                            EXTEND_BACKEND,
                            NEED_APPLY_ROPE and (rope_range_begin < HID_BLOCK_0),
                            BLOCK_CHUNK,
                            HID_BLOCK_0,
                            HID_DIM,
                            False,
                            HEAD // HEAD_GROUP,
                            UPDATE_CACHE,
                            rope_range_begin,
                            rope_range_end,
                            rope_is_neox_style,
                        )

                        scores_left = tl.dot(
                            (queries_0 * cq).to(matmul_dtype),
                            (keys_left_0.to(q_dtype) * ck).to(matmul_dtype),
                        ).to(scores.dtype)

                        if HID_BLOCK_1 > 0:
                            if LOAD_Q_EACH_TIME:
                                queries_1 = pool_queries(
                                    idx_bsz,
                                    idx_head,
                                    pos_tdst,
                                    idx_tdst,
                                    mask_tdst,
                                    idx_hid_q1,
                                    mask_hid_q1,
                                    Q,
                                    stride_q_bsz,
                                    stride_q_tdst,
                                    stride_q_head,
                                    stride_q_hid,
                                    COS,
                                    stride_cos_t,
                                    stride_cos_hid,
                                    SIN,
                                    stride_sin_t,
                                    stride_sin_hid,
                                    rope_range_begin,
                                    rope_range_end,
                                    rope_is_neox_style,
                                    HID_DIM,
                                    TDST,
                                    CHUNK_COUNT,
                                    real_pos_tdst_min,
                                    model_context_length,
                                    sliding_window_size,
                                    USING_EXTEND,
                                    NEED_APPLY_ROPE,
                                    EXTEND_BACKEND,
                                    BLOCK_SIZE_Q,
                                    HID_BLOCK_1,
                                    STRIDE_Q,
                                )

                            keys_left_1 = load_keys_with_rope(
                                K,
                                stride_k_bsz,
                                stride_k_tsrc,
                                stride_k_head_kv,
                                stride_k_hid,
                                COS,
                                stride_cos_t,
                                stride_cos_hid,
                                SIN,
                                stride_sin_t,
                                stride_sin_hid,
                                # paged attention args template
                                USING_PAGES,
                                PAGE_SIZE,
                                K_CACHE,
                                stride_k_cache_page,
                                stride_k_cache_offset,
                                stride_k_cache_kv_head,
                                stride_k_cache_hid,
                                BLOCK_TABLE,
                                stride_block_table_bsz,
                                stride_block_table_page,
                                CACHE_SEQ_LENS,
                                stride_cache_seq_lens_b,
                                USING_OFFLOAD_CACHE,
                                OFFLOAD_CACHE_KV_PACKED,
                                GPU_BANK_COUNT,
                                OFFLOAD_CACHE_UVM_METADATA,
                                stride_offload_cache_uvm_metadata_token,
                                stride_offload_cache_uvm_metadata_k,
                                OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                                stride_offload_cache_gpu_global_metadata_k,
                                stride_offload_cache_gpu_global_metadata_pad,
                                OFFLOAD_CACHE_GPU_BANK,
                                stride_offload_cache_gpu_bank_token,
                                stride_offload_cache_gpu_bank_hid,
                                OFFLOAD_CACHE_GPU_METADATA,
                                stride_offload_cache_gpu_metadata_token,
                                stride_offload_cache_gpu_metadata_k,
                                OFFLOAD_CACHE_GPU_TABLE,
                                stride_offload_cache_gpu_table_head_kv,
                                stride_offload_cache_gpu_table_token,
                                strdie_offload_cache_gpu_table_k,
                                MASK_ACCESS_COUNTER,
                                stride_mask_access_counter_bsz,
                                stride_mask_access_counter_head_kv,
                                stride_mask_access_counter_tsrc,
                                MASK_CACHE_MISS_COUNTER,
                                stride_mask_cache_miss_counter_bsz,
                                stride_mask_cache_miss_counter_head_kv,
                                stride_mask_cache_miss_counter_tsrc,
                                q_dtype,
                                idx_bsz,
                                idx_tsrc,
                                idx_head // HEAD_GROUP,
                                idx_hid_q1,
                                idx_chunk,
                                mask_tsrc_active,
                                mask_tdst,
                                mask_hid_q1,
                                real_pos_tdst_min,
                                model_context_length,
                                num_sinks,
                                USING_EXTEND,
                                EXTEND_BACKEND,
                                NEED_APPLY_ROPE,
                                BLOCK_CHUNK,
                                HID_BLOCK_1,
                                HID_DIM,
                                False,
                                HEAD // HEAD_GROUP,
                                UPDATE_CACHE,
                                rope_range_begin,
                                rope_range_end,
                                rope_is_neox_style,
                            )

                            if COMPUTE_MLA_ROPE:
                                scores_left += tl.dot(
                                    (queries_1 * cq).to(matmul_dtype),
                                    (keys_left_1.to(q_dtype) * ck).to(matmul_dtype),
                                ).to(scores.dtype)

                        if REDUCE == "max":
                            scores_left = tl.where(
                                mask_tdst[:, None], scores_left, float("-inf")
                            )
                            scores_left = tl.max(scores_left, axis=0).to(
                                scores_left.dtype
                            )
                        elif REDUCE == "mean":
                            scores_left = tl.where(
                                mask_tdst[:, None], scores_left, float("0")
                            )
                            scores_left = tl.sum(scores_left, axis=0).to(
                                scores_left.dtype
                            )
                            scores_left = (
                                scores_left / tl.sum(mask_tdst.to(tl.float32))
                            ).to(scores_left.dtype)
                        else:
                            raise Exception()
                        scores_left = tl.where(
                            mask_tsrc_active, scores_left, float("-inf")
                        ).to(scores_left.dtype)

                        idx_tsrc = (idx_tsrc_center + idx_tsrc_right) // 2

                        if LOAD_Q_EACH_TIME:
                            queries_0 = pool_queries(
                                idx_bsz,
                                idx_head,
                                pos_tdst,
                                idx_tdst,
                                mask_tdst,
                                idx_hid_q0,
                                mask_hid_q0,
                                Q,
                                stride_q_bsz,
                                stride_q_tdst,
                                stride_q_head,
                                stride_q_hid,
                                COS,
                                stride_cos_t,
                                stride_cos_hid,
                                SIN,
                                stride_sin_t,
                                stride_sin_hid,
                                rope_range_begin,
                                rope_range_end,
                                rope_is_neox_style,
                                HID_DIM,
                                TDST,
                                CHUNK_COUNT,
                                real_pos_tdst_min,
                                model_context_length,
                                sliding_window_size,
                                USING_EXTEND and (rope_range_begin < HID_BLOCK_0),
                                NEED_APPLY_ROPE and (rope_range_begin < HID_BLOCK_0),
                                EXTEND_BACKEND,
                                BLOCK_SIZE_Q,
                                HID_BLOCK_0,
                                STRIDE_Q,
                            )

                        keys_right_0 = load_keys_with_rope(
                            K,
                            stride_k_bsz,
                            stride_k_tsrc,
                            stride_k_head_kv,
                            stride_k_hid,
                            COS,
                            stride_cos_t,
                            stride_cos_hid,
                            SIN,
                            stride_sin_t,
                            stride_sin_hid,
                            # paged attention args template
                            USING_PAGES,
                            PAGE_SIZE,
                            K_CACHE,
                            stride_k_cache_page,
                            stride_k_cache_offset,
                            stride_k_cache_kv_head,
                            stride_k_cache_hid,
                            BLOCK_TABLE,
                            stride_block_table_bsz,
                            stride_block_table_page,
                            CACHE_SEQ_LENS,
                            stride_cache_seq_lens_b,
                            USING_OFFLOAD_CACHE,
                            OFFLOAD_CACHE_KV_PACKED,
                            GPU_BANK_COUNT,
                            OFFLOAD_CACHE_UVM_METADATA,
                            stride_offload_cache_uvm_metadata_token,
                            stride_offload_cache_uvm_metadata_k,
                            OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                            stride_offload_cache_gpu_global_metadata_k,
                            stride_offload_cache_gpu_global_metadata_pad,
                            OFFLOAD_CACHE_GPU_BANK,
                            stride_offload_cache_gpu_bank_token,
                            stride_offload_cache_gpu_bank_hid,
                            OFFLOAD_CACHE_GPU_METADATA,
                            stride_offload_cache_gpu_metadata_token,
                            stride_offload_cache_gpu_metadata_k,
                            OFFLOAD_CACHE_GPU_TABLE,
                            stride_offload_cache_gpu_table_head_kv,
                            stride_offload_cache_gpu_table_token,
                            strdie_offload_cache_gpu_table_k,
                            MASK_ACCESS_COUNTER,
                            stride_mask_access_counter_bsz,
                            stride_mask_access_counter_head_kv,
                            stride_mask_access_counter_tsrc,
                            MASK_CACHE_MISS_COUNTER,
                            stride_mask_cache_miss_counter_bsz,
                            stride_mask_cache_miss_counter_head_kv,
                            stride_mask_cache_miss_counter_tsrc,
                            q_dtype,
                            idx_bsz,
                            idx_tsrc,
                            idx_head // HEAD_GROUP,
                            idx_hid_q0,
                            idx_chunk,
                            mask_tsrc_active,
                            mask_tdst,
                            mask_hid_q0,
                            real_pos_tdst_min,
                            model_context_length,
                            num_sinks,
                            USING_EXTEND and (rope_range_begin < HID_BLOCK_0),
                            EXTEND_BACKEND,
                            NEED_APPLY_ROPE and (rope_range_begin < HID_BLOCK_0),
                            BLOCK_CHUNK,
                            HID_BLOCK_0,
                            HID_DIM,
                            True,
                            HEAD // HEAD_GROUP,
                            UPDATE_CACHE,
                            rope_range_begin,
                            rope_range_end,
                            rope_is_neox_style,
                        )

                        scores_right = tl.dot(
                            (queries_0 * cq).to(matmul_dtype),
                            (keys_right_0.to(q_dtype) * ck).to(matmul_dtype),
                        ).to(scores.dtype)

                        if HID_BLOCK_1 > 0:
                            if LOAD_Q_EACH_TIME:
                                queries_1 = pool_queries(
                                    idx_bsz,
                                    idx_head,
                                    pos_tdst,
                                    idx_tdst,
                                    mask_tdst,
                                    idx_hid_q1,
                                    mask_hid_q1,
                                    Q,
                                    stride_q_bsz,
                                    stride_q_tdst,
                                    stride_q_head,
                                    stride_q_hid,
                                    COS,
                                    stride_cos_t,
                                    stride_cos_hid,
                                    SIN,
                                    stride_sin_t,
                                    stride_sin_hid,
                                    rope_range_begin,
                                    rope_range_end,
                                    rope_is_neox_style,
                                    HID_DIM,
                                    TDST,
                                    CHUNK_COUNT,
                                    real_pos_tdst_min,
                                    model_context_length,
                                    sliding_window_size,
                                    USING_EXTEND,
                                    NEED_APPLY_ROPE,
                                    EXTEND_BACKEND,
                                    BLOCK_SIZE_Q,
                                    HID_BLOCK_1,
                                    STRIDE_Q,
                                )

                            keys_right_1 = load_keys_with_rope(
                                K,
                                stride_k_bsz,
                                stride_k_tsrc,
                                stride_k_head_kv,
                                stride_k_hid,
                                COS,
                                stride_cos_t,
                                stride_cos_hid,
                                SIN,
                                stride_sin_t,
                                stride_sin_hid,
                                # paged attention args template
                                USING_PAGES,
                                PAGE_SIZE,
                                K_CACHE,
                                stride_k_cache_page,
                                stride_k_cache_offset,
                                stride_k_cache_kv_head,
                                stride_k_cache_hid,
                                BLOCK_TABLE,
                                stride_block_table_bsz,
                                stride_block_table_page,
                                CACHE_SEQ_LENS,
                                stride_cache_seq_lens_b,
                                USING_OFFLOAD_CACHE,
                                OFFLOAD_CACHE_KV_PACKED,
                                GPU_BANK_COUNT,
                                OFFLOAD_CACHE_UVM_METADATA,
                                stride_offload_cache_uvm_metadata_token,
                                stride_offload_cache_uvm_metadata_k,
                                OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
                                stride_offload_cache_gpu_global_metadata_k,
                                stride_offload_cache_gpu_global_metadata_pad,
                                OFFLOAD_CACHE_GPU_BANK,
                                stride_offload_cache_gpu_bank_token,
                                stride_offload_cache_gpu_bank_hid,
                                OFFLOAD_CACHE_GPU_METADATA,
                                stride_offload_cache_gpu_metadata_token,
                                stride_offload_cache_gpu_metadata_k,
                                OFFLOAD_CACHE_GPU_TABLE,
                                stride_offload_cache_gpu_table_head_kv,
                                stride_offload_cache_gpu_table_token,
                                strdie_offload_cache_gpu_table_k,
                                MASK_ACCESS_COUNTER,
                                stride_mask_access_counter_bsz,
                                stride_mask_access_counter_head_kv,
                                stride_mask_access_counter_tsrc,
                                MASK_CACHE_MISS_COUNTER,
                                stride_mask_cache_miss_counter_bsz,
                                stride_mask_cache_miss_counter_head_kv,
                                stride_mask_cache_miss_counter_tsrc,
                                q_dtype,
                                idx_bsz,
                                idx_tsrc,
                                idx_head // HEAD_GROUP,
                                idx_hid_q1,
                                idx_chunk,
                                mask_tsrc_active,
                                mask_tdst,
                                mask_hid_q1,
                                real_pos_tdst_min,
                                model_context_length,
                                num_sinks,
                                USING_EXTEND,
                                EXTEND_BACKEND,
                                NEED_APPLY_ROPE,
                                BLOCK_CHUNK,
                                HID_BLOCK_1,
                                HID_DIM,
                                True,
                                HEAD // HEAD_GROUP,
                                UPDATE_CACHE,
                                rope_range_begin,
                                rope_range_end,
                                rope_is_neox_style,
                            )

                            if COMPUTE_MLA_ROPE:
                                scores_right += tl.dot(
                                    (queries_1 * cq).to(matmul_dtype),
                                    (keys_right_1.to(q_dtype) * ck).to(matmul_dtype),
                                ).to(scores.dtype)

                        if REDUCE == "max":
                            scores_right = tl.where(
                                mask_tdst[:, None], scores_right, float("-inf")
                            )
                            scores_right = tl.max(scores_right, axis=0).to(
                                scores_right.dtype
                            )
                        elif REDUCE == "mean":
                            scores_right = tl.where(
                                mask_tdst[:, None], scores_right, float("0")
                            )
                            scores_right = tl.sum(scores_right, axis=0).to(
                                scores_right.dtype
                            )
                            scores_right = (
                                scores_right / tl.sum(mask_tdst.to(tl.float32))
                            ).to(scores_right.dtype)
                        else:
                            raise Exception()
                        scores_right = tl.where(
                            mask_tsrc_active, scores_right, float("-inf")
                        ).to(scores_right.dtype)

                        mask_left_win = scores_left > scores_right
                        idx_tsrc_left = tl.where(
                            mask_tsrc_active,
                            tl.where(
                                mask_left_win,
                                idx_tsrc_left,
                                idx_tsrc_center,
                            ),
                            idx_tsrc_left,
                        )

                        idx_tsrc_right = tl.where(
                            mask_tsrc_active,
                            tl.where(
                                mask_left_win,
                                idx_tsrc_center,
                                idx_tsrc_right,
                            ),
                            idx_tsrc_right,
                        )

                        scores = tl.maximum(
                            scores,
                            tl.where(
                                mask_tsrc_active,
                                tl.where(
                                    mask_left_win,
                                    scores_left,
                                    scores_right,
                                ),
                                scores,
                            ),
                        )

                    # idx_tsrc_center = (idx_tsrc_left + idx_tsrc_right) // 2
                    # idx_tsrc_left = idx_tsrc_center - TERMINATE_SIZE // 2
                    # idx_tsrc_right = idx_tsrc_left + TERMINATE_SIZE

                    tl.store(
                        INDICES_LEFT
                        + idx_bsz * stride_indices_left_bsz
                        + idx_bdst_scan * stride_indices_left_bdst
                        + idx_head * stride_indices_left_head
                        + idx_chunk * stride_indices_left_chunk,
                        value=idx_tsrc_left,
                        mask=mask_chunk,
                    )

                    tl.store(
                        INDICES_RIGHT
                        + idx_bsz * stride_indices_right_bsz
                        + idx_bdst_scan * stride_indices_right_bdst
                        + idx_head * stride_indices_right_head
                        + idx_chunk * stride_indices_right_chunk,
                        value=idx_tsrc_right,
                        mask=mask_chunk,
                    )

                    tl.store(
                        OUT_SCORES
                        + idx_bsz * stride_out_scores_bsz
                        + idx_bdst_scan * stride_out_scores_bdst
                        + idx_head * stride_out_scores_head
                        + idx_chunk * stride_out_scores_chunk,
                        value=scores,
                        mask=mask_chunk,
                    )


from hip_attn.v1_2.utils import capture


@capture
def chunk_controllable_sampling_mask(
    args,
    chunk_count,
    BLOCK_CHUNK,
    TDST,
    BLOCK_SIZE_Q,
    STAGE_STRIDE,
    HEAD,
    BSZ,
    q,
    k_mask,
    position_ids,
    indices_left,
    indices_right,
    out_scores,
    mask_access_counter,
    mask_cache_miss_counter,
    MAX_TSRC,
    HID,
    HID_BLOCK,
    stage_block_stride_q,
    HEAD_KV,
    extend_backend,
):
    if not (args.online_update_cache and (args.offload_cache is not None)):
        grid = (
            BSZ
            * triton.cdiv(chunk_count, BLOCK_CHUNK)
            * triton.cdiv(triton.cdiv(TDST, BLOCK_SIZE_Q), STAGE_STRIDE)
            * HEAD,
        )
        njobs = grid[0]
        group_jobs = 1
    else:
        njobs = (
            BSZ
            * triton.cdiv(chunk_count, BLOCK_CHUNK)
            * triton.cdiv(triton.cdiv(TDST, BLOCK_SIZE_Q), STAGE_STRIDE)
            * HEAD
        )
        sm_count = num_streaming_multiprocessor()
        group_jobs = triton.cdiv(njobs, sm_count)
        grid = (min(sm_count, njobs),)

    chunk_controllable_sampling_mask_cuda[grid](
        q,
        *q.stride(),
        k_mask,
        *safe_stride(k_mask, 4),
        position_ids,
        *position_ids.stride(),
        *args.args_paged_kv_cache(disable_cache=k_mask is not None),
        *args.args_offload_cache(True, disable_cache=k_mask is not None),
        indices_left,
        *indices_left.stride(),
        indices_right,
        *indices_right.stride(),
        out_scores,
        *out_scores.stride(),
        args.rope_cos,
        *safe_stride(args.rope_cos, 2),
        args.rope_sin,
        *safe_stride(args.rope_sin, 2),
        args.rope_range[0],
        args.rope_range[1],
        args.rope_is_neox_style,
        mask_access_counter,
        *safe_stride(mask_access_counter, 3),
        mask_cache_miss_counter,
        *safe_stride(mask_cache_miss_counter, 3),
        chunk_count,
        MAX_TSRC,
        q.shape[1],
        HEAD,
        args.sliding_window_size,
        args.sink_token_size,
        # model_context_length if (not scan_extend_backend == 'streaming') else 0,
        args.model_context_length,
        group_jobs,
        njobs,
        HID_DIM=HID,
        HID_BLOCK_0=HID_BLOCK,
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        STRIDE_Q=stage_block_stride_q,
        BLOCK_CHUNK=BLOCK_CHUNK,
        HEAD_GROUP=HEAD // HEAD_KV,
        USING_EXTEND=args.using_extend and (extend_backend != "none"),
        EXTEND_BACKEND=extend_backend,
        NEED_APPLY_ROPE=args.need_apply_rope and (extend_backend != "none"),
        TERMINATE_SIZE=args.stage_early_terminate,
        SCAN_STRIDE=STAGE_STRIDE,
        UPDATE_CACHE=args.online_update_cache,
        ORACLE_MAXIMUM=False,  # NOTE: seems has bug... but why?
        COMPUTE_MLA_ROPE=os.getenv("HIP_DEBUG_SCAN_COMPUTE_MLA_ROPE", "0") == "1",
    )

```

### `hip_attn/v1_2/stage_prologue.py`

```py
import torch
import triton

from .attention_metadata import HiPAttentionArgs, Stage
from .utils import capture


@capture
@torch.compile(dynamic=True)
def stage_prologue(
    q: torch.Tensor,
    indices_left: torch.Tensor,
    indices_right: torch.Tensor,
    out_scores: torch.Tensor,
    stage_k: int,
    stage_chunk_size: int,
    chunk_size: int,
    stage_info: Stage,
    args: HiPAttentionArgs,
    TDST,
    BDST,
    STAGE_STRIDE,
    BLOCK_SIZE_Q,
):
    assert (stage_k % chunk_size) == 0, f"{stage_k} % {chunk_size}"
    indices_left = indices_left[..., : stage_k // chunk_size]
    require_align = stage_info.require_realign_index
    if require_align:
        indices_left = (
            indices_left - args.sink_token_size
        ) // chunk_size * chunk_size + args.sink_token_size
        indices_right = indices_left + chunk_size
    else:
        indices_right = indices_right[..., : stage_k // chunk_size]
    out_scores = out_scores[..., : stage_k // chunk_size]
    # NOTE: revert this
    if stage_info.require_reset_score:
        out_scores.fill_(-32000.0)

    require_sort = BDST > 1
    if require_sort:
        indices_left, t_indices = indices_left.sort(dim=-1)
        indices_right = indices_right.gather(dim=-1, index=t_indices)
        out_scores = out_scores.gather(dim=-1, index=t_indices)

    if BLOCK_SIZE_Q != stage_info.stage_block_size_q:
        assert stage_info.stage_block_size_q > 0
        assert BLOCK_SIZE_Q > stage_info.stage_block_size_q
        assert (BLOCK_SIZE_Q % stage_info.stage_block_size_q) == 0

        num_split = BLOCK_SIZE_Q // stage_info.stage_block_size_q
        BLOCK_SIZE_Q = stage_info.stage_block_size_q
        BDST = triton.cdiv(TDST, BLOCK_SIZE_Q)

        indices_left = indices_left.repeat_interleave(num_split, 1)[
            :, -BDST:
        ].contiguous()
        indices_right = indices_right.repeat_interleave(num_split, 1)[
            :, -BDST:
        ].contiguous()
        out_scores = out_scores.repeat_interleave(num_split, 1)[:, -BDST:].contiguous()

    if STAGE_STRIDE != stage_info.stage_stride:
        assert stage_info.stage_stride < STAGE_STRIDE
        assert STAGE_STRIDE > 0
        indices_left = indices_left.repeat_interleave(
            STAGE_STRIDE // stage_info.stage_stride, 1
        )[:, -BDST:].contiguous()
        indices_right = indices_right.repeat_interleave(
            STAGE_STRIDE // stage_info.stage_stride, 1
        )[:, -BDST:].contiguous()
        out_scores = out_scores.repeat_interleave(
            STAGE_STRIDE // stage_info.stage_stride, 1
        )[:, -BDST:].contiguous()
        STAGE_STRIDE = stage_info.stage_stride

    assert (chunk_size % stage_chunk_size) == 0
    splits = chunk_size // stage_chunk_size
    chunk_sizes = ((indices_right - indices_left).float() / splits).clamp_min_(0)
    indices_left = (
        indices_left[..., None]
        + (
            torch.arange(0, splits, device=q.device)[None, None, None, None, :]
            * chunk_sizes[..., None]
        )
        .floor()
        .long()
    )
    indices_left = indices_left.flatten(-2, -1)
    indices_right = (
        indices_right[..., None]
        - (
            (
                (splits - 1)
                - torch.arange(0, splits, device=q.device)[None, None, None, None, :]
            )
            * chunk_sizes[..., None]
        )
        .floor()
        .long()
    )
    indices_right = indices_right.flatten(-2, -1)
    out_scores = out_scores.repeat_interleave(splits, -1)

    return indices_left, indices_right, out_scores, BLOCK_SIZE_Q, BDST, STAGE_STRIDE

```

### Modules in `hip_attn.v1_2`

From now, I will add codes included in same module.

#### `hip_attn/v1_2/attention_metadata.py`

```py
import copy
import os
import warnings
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Dict, List, Literal, Optional

import torch
from torch import Tensor

if TYPE_CHECKING:
    from hip_attn.v1_2.uvm_gpu_cache import HiPOffloadCache


HIP_DEBUG_ALLOW_GATHER_KV_CACHE = (
    os.getenv("HIP_DEBUG_ALLOW_GATHER_KV_CACHE", "0") == "1"
)


def safe_stride(x: Optional[Tensor], ndim: int):
    if x is None:
        return tuple(
            [
                0,
            ]
            * ndim
        )
    else:
        stride = x.stride()
        assert len(stride) == ndim
        return stride


@dataclass
class Stage:
    stage_block_size_q: int
    stage_block_stride_q: int
    stage_chunk_size: int
    stage_k: Optional[int]
    stage_stride: int

    require_realign_index: bool = False
    require_reset_score: bool = False
    require_post_sort: bool = False


@dataclass
class NopStage(Stage):
    require_realign_index: bool = True
    require_reset_score: bool = False
    require_post_sort: bool = True


@dataclass
class EvalScoreStage(Stage):
    block_chunk: int = 64
    stage_extend_backend: Optional[str] = None
    require_reset_score: bool = True
    require_post_sort: bool = True


@dataclass
class ScanStage(Stage):
    stage_extend_backend: Optional[str] = None
    using_landmark: Optional[bool] = None
    require_realign_index: bool = True
    require_reset_score: bool = True
    require_post_sort: bool = True


@dataclass
class EnsembleScoreStage(Stage):
    reduce_method: str = "sum"
    require_reset_score: bool = True
    require_post_sort: bool = True


StatKeys = Literal[
    "unique_access_count",
    "access_count",
    "cache_miss_count",
    "cache_hit_ratio",
]


@dataclass
class HiPAttentionCacheAccessStatistics:
    # [BSZ, HEAD_KV, MAX_TSRC]
    access_counter: Tensor
    # [BSZ, HEAD_KV, MAX_TSRC]
    cache_miss_counter: Tensor

    def compute_statistics(self) -> Dict[
        StatKeys,
        Tensor,
    ]:
        # FIXME: heejun
        if (os.getenv("HIP_DISABLE_COMPUTE_STATISTICS", "1") == "0") and (
            self.access_counter is not None
        ):
            unique_access_count = self.access_counter.clamp(0, 1).sum()
            access_counts = self.access_counter.sum()
            cache_miss_counts = self.cache_miss_counter.sum()
            cache_hit_ratio = 1 - (cache_miss_counts / access_counts)
        else:
            unique_access_count = None
            access_counts = None
            cache_miss_counts = None
            cache_hit_ratio = None

        return {
            "unique_access_count": unique_access_count,
            "access_count": access_counts,
            "cache_miss_count": cache_miss_counts,
            "cache_hit_ratio": cache_hit_ratio,
        }


@dataclass
class HiPAttentionStageInputCache:
    indices_left: Tensor
    indices_right: Tensor
    out_scores: Tensor


@dataclass
class HiPAttentionState:
    # [MAX_NUM_TOKENS, HEAD]
    landmark_scores: torch.Tensor
    # [NUM_STAGES, MAX_NUM_TOKENS // CHUNK_SIZE, K]
    landmark_indices: List[torch.Tensor]

    @classmethod
    def from_args(
        cls, q: torch.Tensor, args: "HiPAttentionArgs", k: Optional[torch.Tensor] = None
    ):
        if k is None:
            assert args.using_paged_cache

        if args.get_k_cache() is not None:
            k_cache = args.get_k_cache()
            num_tokens = k_cache.shape[0] * k_cache.shape[1]
        else:
            num_tokens = k.shape[1]

        num_tokens = max(args.extend_context_length, num_tokens)
        num_tokens = max(int(os.getenv("HIP_DEBUG_MAX_TOKENS", "0")), num_tokens)

        # padding for SGlang
        num_tokens += 1024

        num_heads = q.shape[2]
        landmark_scores = torch.zeros(
            (num_tokens, num_heads), dtype=torch.float32, device=q.device
        )

        return HiPAttentionState(
            landmark_scores=landmark_scores,
            landmark_indices=None,
        )


@dataclass
class HiPAttentionOutputMetadata:
    indices: Optional[Tensor]
    ks: Optional[Tensor]
    ks_count: Optional[Tensor]
    ks_start_end: Optional[Tensor]

    # memory access statistics
    mask_cache_statistics: Optional[HiPAttentionCacheAccessStatistics]
    sa_cache_statistics: Optional[HiPAttentionCacheAccessStatistics]

    # stage caches
    stage_caches: Optional[List[HiPAttentionStageInputCache]]

    state: Optional[HiPAttentionState] = None


@dataclass
class HiPAttentionArgs:
    position_ids: Optional[Tensor] = None

    sink_token_size: int = 256
    sliding_window_size: int = 512
    block_size_k: int = 64  # for optimization this will be BLOCK_CHUNK

    block_size_q: int = 64  # no effect, set automatically
    mask_k: int = 512  # no effect, set automatically

    second_stage_k: int = 2048
    stages: List[Stage] = field(
        default_factory=lambda: [
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=4,
                stage_chunk_size=256,
                stage_k=None,
                stage_stride=1,
            ),
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=4,
                stage_chunk_size=32,
                stage_k=32768,
                stage_stride=1,
            ),
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=1,
                stage_chunk_size=8,
                stage_k=8192,
                stage_stride=1,
            ),
        ]
    )
    model_context_length: int = 131072
    extend_context_length: int = 512 * 1024

    using_landmark: bool = field(
        default_factory=lambda: os.getenv("HIP_DEBUG_LANDMARK_BASED_SCAN_STAGE", "1")
        == "1"
    )
    landmark_stage_k: List[int] = field(default_factory=lambda: [1, 1, 1])

    # kernel args,
    mask_only: bool = False
    block_sparse_block_size_q: Optional[int] = 64
    v_hidden_dim: Optional[int] = None

    scan_early_terminate: int = 1
    stage_early_terminate: int = 1
    scan_extend_backend: str = "relative"
    sa_extend_backend: str = "streaming"
    low_percent: float = 0.0
    low_k_ratio: float = 1.0
    dim_to_lower: Literal["head", "seq"] = "head"
    q_mask: Optional[Tensor] = None
    k_mask: Optional[Tensor] = None
    idx_pca_hid_q: Optional[Tensor] = None
    idx_pca_hid_k: Optional[Tensor] = None

    is_causal: bool = True

    using_extend: bool = False
    need_apply_rope: bool = False
    rope_cos: Optional[Tensor] = None
    rope_sin: Optional[Tensor] = None
    rope_range: Optional[tuple[int, int]] = None
    rope_is_neox_style: Optional[bool] = None

    offload_cache: "Optional[HiPOffloadCache]" = None
    k_cache: Optional[Tensor] = None
    v_cache: Optional[Tensor] = None
    cache_seq_lens: Optional[Tensor] = None
    block_table: Optional[Tensor] = None

    # to support gemma2
    logit_softcap: Optional[float] = None

    online_update_cache: bool = False

    require_cache_statistics: bool = True
    require_stage_caches: bool = True

    disable_flashdecode: bool = False

    sliding_window_indices: Optional[torch.Tensor] = None

    using_chunked_sliding_window: bool = False

    # NOTE: use only for debugging purpose
    layer_id: int = 31

    query_for_landmark: Optional[Tensor] = None
    position_ids_for_landmark: Optional[Tensor] = None

    is_decode: bool = False

    bsa_return_running_statistics: bool = False
    bsa_sliding_window_size: int = -1

    k_descale: Optional[Tensor] = None
    v_descale: Optional[Tensor] = None

    def __post_init__(self):
        if self.rope_cos is not None and self.rope_cos.ndim == 3:
            self.rope_cos = self.rope_cos.view(-1, self.rope_cos.shape[-1])
            self.rope_sin = self.rope_sin.view(-1, self.rope_sin.shape[-1])
        # if self.q_quant is not None:
        #     assert self.q_quant.ndim == 4
        #     assert self.k_quant.ndim == 4
        self.update_flags()

    def update_flags(self):
        if self.logit_softcap == 0:
            self.logit_softcap = None
        self.using_paged_cache = (self.k_cache is not None) or (
            self.offload_cache is not None
        )
        if self.using_paged_cache:
            if self.k_cache is not None:
                self.paged_cache_page_count = self.k_cache.shape[0]
                self.paged_cache_page_size = self.k_cache.shape[1]
            else:
                self.paged_cache_page_count = self.offload_cache.get_page_count()
                self.paged_cache_page_size = 1
            assert self.paged_cache_page_size in (1, 2, 4, 8, 16, 32)
        if self.logit_softcap == 0:
            self.logit_softcap = None

    def clone(self):
        self.update_flags()
        return copy.copy(self)

    def json(self, convert_tensor_to_meta=True):
        from dataclasses import fields

        json = {}
        for field in fields(self):
            json[field.name] = getattr(self, field.name)

        if convert_tensor_to_meta:
            for k, v in json.items():
                if isinstance(v, Tensor):
                    v = f"{v.dtype}{list(v.shape)}@{v.device}.{v.data_ptr():02X}"
                json[k] = v

        return json

    def args_extend(self):
        return (
            self.using_extend,
            self.need_apply_rope,
            *self.args_rope_cos(),
            *self.args_rope_sin(),
            self.rope_range[0],
            self.rope_range[1],
            self.rope_is_neox_style,
        )

    def args_rope_cos(self):
        return (
            self.rope_cos,
            *safe_stride(self.rope_cos, 2),
        )

    def args_rope_sin(self):
        return (
            self.rope_sin,
            *safe_stride(self.rope_sin, 2),
        )

    def args_paged_kv_cache(self, disable_cache: bool = False):
        using_page = self.using_paged_cache

        if disable_cache:
            return (
                False,
                1,
                None,
                0,
                0,
                0,
                0,
                None,
                0,
                0,
                0,
                0,
                None,
                0,
                0,
                None,
                0,
            )

        if self.offload_cache is None:
            if using_page:
                assert self.v_cache is not None
                assert self.k_cache.ndim == self.v_cache.ndim
                assert self.k_cache.ndim == 4
                assert self.block_table is not None
                assert self.block_table.ndim == 2
                assert self.cache_seq_lens is not None
                assert self.cache_seq_lens.ndim == 1
                page_size = self.k_cache.shape[1]
            else:
                page_size = 0

            return (
                using_page,
                page_size,
                self.k_cache,
                *safe_stride(self.k_cache, 4),
                self.v_cache,
                *safe_stride(self.v_cache, 4),
                self.block_table,
                *safe_stride(self.block_table, 2),
                self.cache_seq_lens,
                *safe_stride(self.cache_seq_lens, 1),
            )
        else:
            assert using_page

            k_cache = self.offload_cache.k_uvm.bank_gpu.unsqueeze(1)
            v_cache = self.offload_cache.v_uvm.bank_gpu.unsqueeze(1)

            return (
                True,
                1,
                k_cache,
                *safe_stride(k_cache, 4),
                v_cache,
                *safe_stride(v_cache, 4),
                self.block_table,
                *safe_stride(self.block_table, 2),
                self.cache_seq_lens,
                *safe_stride(self.cache_seq_lens, 1),
            )

    def args_offload_cache(self, is_masking, disable_cache: bool = False):
        if self.offload_cache and (not disable_cache):
            gpu_cache = (
                self.offload_cache.mask_k_cache
                if is_masking
                else self.offload_cache.sa_kv_cache
            )
            is_packed = gpu_cache.kv_packed
            uvm_metadata = self.offload_cache.k_uvm.metadata
            return (
                True,
                is_packed,
                gpu_cache.bank.shape[0],
                uvm_metadata,
                *safe_stride(uvm_metadata, 2),
                gpu_cache.global_metadata,
                *safe_stride(gpu_cache.global_metadata, 2),
                gpu_cache.bank,
                *safe_stride(gpu_cache.bank, 2),
                gpu_cache.metadata,
                *safe_stride(gpu_cache.metadata, 2),
                gpu_cache.table,
                *safe_stride(gpu_cache.table, 3),
            )
        else:
            return (
                False,
                False,
                0,
                None,
                0,
                0,
                None,
                0,
                0,
                None,
                0,
                0,
                None,
                0,
                0,
                None,
                0,
                0,
                0,
            )

    def get_k_cache(self):
        if self.k_cache is not None:
            k_cache = self.k_cache
        elif self.offload_cache is not None:
            k_cache = self.offload_cache.k_uvm.bank_gpu.unsqueeze(1)
        else:
            k_cache = None

        # k_cache: [MAX_TOKENS, 1, HEAD, HID]
        return k_cache

    def get_v_cache(self):
        if self.v_cache is not None:
            v_cache = self.v_cache
        elif self.offload_cache is not None:
            v_cache = self.offload_cache.v_uvm.bank_gpu.unsqueeze(1)
        else:
            v_cache = None

        # v_cache: [MAX_TOKENS, 1, HEAD, HID]
        return v_cache

    def gather_extend_k_from_paged_cache(
        self,
        disable_gqa=False,
        gqa_q: torch.Tensor = None,
        position_ids: torch.Tensor = None,
    ):
        k_cache = self.get_k_cache()
        # self.block_table[BLOCK_TABLE_BSZ, MODEL_SEQ_LEN]
        assert self.block_table is not None
        if position_ids is None:
            position_ids = self.position_ids
        assert position_ids is not None
        assert (
            position_ids.shape[0] == self.block_table.shape[0]
        ), f"{position_ids.shape} == {self.block_table.shape}"
        # k_cache: [T, HEAD, HID]
        k = k_cache[:, 0, :, :][self.block_table.gather(dim=1, index=position_ids)]
        if gqa_q is not None:
            B, T, H, D = gqa_q.shape
            assert k.shape == (B, T, k.shape[2], D), f"{gqa_q.shape} {k.shape}"
        if disable_gqa:
            k = k.repeat_interleave(gqa_q.shape[2] // k.shape[2], dim=2)
        return k

    def gather_k_from_paged_cache(
        self,
        chunk_size: int = 1,
        disable_gqa=False,
        gqa_q: torch.Tensor = None,
        seq_len: int = None,
    ):
        if not HIP_DEBUG_ALLOW_GATHER_KV_CACHE:
            raise Exception(
                "Please set HIP_DEBUG_ALLOW_GATHER_KV_CACHE=1 for allow this behavior"
            )
        else:
            # warnings.warn("For developers: gathering paged cache will occure overhead.")
            pass

        k_cache = self.get_k_cache()
        assert self.block_table is not None

        if seq_len is None:
            seq_len = self.block_table.shape[1]

        is_fp8 = k_cache.dtype in (torch.float8_e5m2, torch.float8_e4m3fn)
        index_dtype = torch.uint8 if is_fp8 else k_cache.dtype

        k = k_cache.view(index_dtype)[:, 0, :, :][
            self.block_table[
                :,
                : seq_len - (seq_len % chunk_size),
            ]
        ].view(k_cache.dtype)
        if disable_gqa:
            k = k.repeat_interleave(gqa_q.shape[2] // k.shape[2], dim=2)
        return k

    def gather_v_from_paged_cache(
        self,
        chunk_size: int = 1,
        disable_gqa: bool = False,
        gqa_q: torch.Tensor = None,
        seq_len: int = None,
    ):
        if not HIP_DEBUG_ALLOW_GATHER_KV_CACHE:
            raise Exception(
                "Please set HIP_DEBUG_ALLOW_GATHER_KV_CACHE=1 for allow this behavior"
            )
        else:
            # warnings.warn("For developers: gathering paged cache will occure overhead.")
            pass

        if self.v_cache is not None:
            assert self.v_cache is not None
            v_cache = self.v_cache
        else:
            v_cache = self.offload_cache.v_uvm.bank_gpu.unsqueeze(1)

        assert self.block_table is not None

        if seq_len is None:
            seq_len = self.block_table.shape[1]

        is_fp8 = v_cache.dtype in (torch.float8_e5m2, torch.float8_e4m3fn)
        index_dtype = torch.uint8 if is_fp8 else v_cache.dtype

        v = v_cache.view(index_dtype)[:, 0, :, :][
            self.block_table[
                :,
                : seq_len - (seq_len % chunk_size),
            ]
        ].view(v_cache.dtype)
        if disable_gqa:
            v = v.repeat_interleave(gqa_q.shape[2] // v.shape[2], dim=2)
        return v

    def pretty(self) -> str:
        json = asdict(self)
        for k, v in json.items():
            if isinstance(v, torch.Tensor):
                json[k] = f"{v.dtype}{list(v.shape)}@{str(v.device)}"
        return str(json)

```

#### `hip_attn/v1_2/hip_config.py`

```py
import json
import os
import warnings
from dataclasses import InitVar, dataclass, field
from typing import List, Optional, Union

from hip_attn.v1_2.attention_metadata import ScanStage

HIP_CONFIG_PRESET = os.getenv("HIP_CONFIG_PRESET", "default")

HIP_DEBUG_LANDMARK_BASED_SCAN_STAGE = (
    os.getenv("HIP_DEBUG_LANDMARK_BASED_SCAN_STAGE", "1") == "1"
)
HIP_DEBUG_DELTA_EXP = "exp" in os.getenv("HIP_DELTA_ATTENTION_ARGS", "")

if HIP_DEBUG_LANDMARK_BASED_SCAN_STAGE:
    if HIP_DEBUG_DELTA_EXP:
        _DEFAULT_STAGES = [
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=2,
                stage_chunk_size=64,
                stage_k=None,
                stage_stride=1,
            ),
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=2,
                stage_chunk_size=16,
                stage_k=32768,
                stage_stride=1,
            ),
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=1,
                stage_chunk_size=4,
                stage_k=8192,
                stage_stride=1,
            ),
        ]
    else:
        _DEFAULT_STAGES = [
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=4,
                stage_chunk_size=64,
                stage_k=None,
                stage_stride=1,
            ),
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=4,
                stage_chunk_size=16,
                stage_k=32768,
                stage_stride=1,
            ),
            ScanStage(
                stage_block_size_q=64,
                stage_block_stride_q=1,
                stage_chunk_size=4,
                stage_k=8192,
                stage_stride=1,
            ),
        ]
    _DEFAULT_STAGES_DECODE = [
        ScanStage(
            stage_block_size_q=64,
            stage_block_stride_q=4,
            stage_chunk_size=128,
            stage_k=None,
            stage_stride=1,
        ),
        ScanStage(
            stage_block_size_q=64,
            stage_block_stride_q=4,
            stage_chunk_size=32,
            stage_k=32768,
            stage_stride=1,
        ),
        ScanStage(
            stage_block_size_q=64,
            stage_block_stride_q=1,
            stage_chunk_size=8,
            stage_k=8192,
            stage_stride=1,
        ),
    ]
else:
    _DEFAULT_STAGES = [
        ScanStage(
            stage_block_size_q=64,
            stage_block_stride_q=4,
            stage_chunk_size=128,
            stage_k=None,
            stage_stride=1,
        ),
        ScanStage(
            stage_block_size_q=64,
            stage_block_stride_q=4,
            stage_chunk_size=32,
            stage_k=32768,
            stage_stride=1,
        ),
        ScanStage(
            stage_block_size_q=64,
            stage_block_stride_q=1,
            stage_chunk_size=8,
            stage_k=8192,
            stage_stride=1,
        ),
    ]


@dataclass
class HiPAttentionPerLayerConfig:
    second_stage_k: int = 2048
    sliding_window_size: int = 1024
    sliding_window_size_for_masking_step: Optional[List[int]] = None
    sink_token_size: int = 256
    landmark_stage_k: int = field(default_factory=lambda: [1, 1, 1])
    sa_extend_backend: str = "streaming"
    scan_extend_backend: Optional[str] = None
    stages: list[ScanStage] = field(default_factory=lambda: _DEFAULT_STAGES)

    parsed_json: InitVar[Optional[dict]] = None

    def __post_init__(self, parsed_json: Optional[dict]):
        super().__init__()
        if parsed_json is not None:
            if "second_stage_k" in parsed_json:
                self.second_stage_k = parsed_json["second_stage_k"]
                parsed_json.pop("second_stage_k")
            if "sliding_window_size" in parsed_json:
                self.sliding_window_size = parsed_json["sliding_window_size"]
                parsed_json.pop("sliding_window_size")
            if "sliding_window_size_for_masking_step" in parsed_json:
                self.sliding_window_size_for_masking_step = parsed_json[
                    "sliding_window_size_for_masking_step"
                ]
                parsed_json.pop("sliding_window_size_for_masking_step")
            if "sink_token_size" in parsed_json:
                self.sink_token_size = parsed_json["sink_token_size"]
                parsed_json.pop("sink_token_size")
            if "sa_extend_backend" in parsed_json:
                self.sa_extend_backend = parsed_json["sa_extend_backend"]
                parsed_json.pop("sa_extend_backend")
            if "scan_extend_backend" in parsed_json:
                self.scan_extend_backend = parsed_json["scan_extend_backend"]
                parsed_json.pop("scan_extend_backend")
            if "stages" in parsed_json:
                self.stages = [
                    ScanStage(**stage)
                    if len(stage.keys()) > 0 else
                    ScanStage(64, 1, 32, 32768, 1)
                    for stage in parsed_json["stages"]
                ]
                parsed_json.pop("stages")
            if "landmark_stage_k" in parsed_json:
                self.landmark_stage_k = parsed_json["landmark_stage_k"]
                parsed_json.pop("landmark_stage_k")
            if parsed_json:
                raise ValueError(f"Unknown keys in json: {parsed_json.keys()}")


if HIP_CONFIG_PRESET == "default":
    _DEFAULT_LAEYRS = [
        HiPAttentionPerLayerConfig(
            # sliding_window_size = 777, # NOTE: debugging sw
            second_stage_k=4096,
            sa_extend_backend="streaming",
            scan_extend_backend="streaming",
            stages=_DEFAULT_STAGES,
        ),
        HiPAttentionPerLayerConfig(
            # sliding_window_size = 777, # NOTE: debugging sw
            sliding_window_size=1024,
            second_stage_k=2048,
            sa_extend_backend="streaming",
            scan_extend_backend="relative",
            stages=_DEFAULT_STAGES,
        ),
    ]
    if HIP_DEBUG_LANDMARK_BASED_SCAN_STAGE:
        _DEFAULT_LAEYRS = [
            HiPAttentionPerLayerConfig(
                # sliding_window_size = 777, # NOTE: debugging sw
                second_stage_k=4096,
                sa_extend_backend="streaming",
                scan_extend_backend="streaming",
                stages=_DEFAULT_STAGES,
            ),
            HiPAttentionPerLayerConfig(
                # sliding_window_size = 777, # NOTE: debugging sw
                sliding_window_size=1024,
                second_stage_k=2048,
                sa_extend_backend="streaming",
                scan_extend_backend="relative",
                stages=_DEFAULT_STAGES,
            ),
        ]

        _DEFAULT_LAEYRS_DECODE = [
            HiPAttentionPerLayerConfig(
                # sliding_window_size = 777, # NOTE: debugging sw
                second_stage_k=4096,
                sa_extend_backend="streaming",
                scan_extend_backend="streaming",
                stages=_DEFAULT_STAGES_DECODE,
            ),
            HiPAttentionPerLayerConfig(
                # sliding_window_size = 777, # NOTE: debugging sw
                second_stage_k=2048,
                sa_extend_backend="streaming",
                scan_extend_backend="relative",
                stages=_DEFAULT_STAGES_DECODE,
            ),
        ]
    else:
        _DEFAULT_LAEYRS_DECODE = _DEFAULT_LAEYRS
elif HIP_CONFIG_PRESET == "llama4":
    _DEFAULT_LAEYRS = [
        HiPAttentionPerLayerConfig(
            second_stage_k=4096,
            sa_extend_backend="streaming",
            scan_extend_backend="streaming",
        ),
        HiPAttentionPerLayerConfig(
            second_stage_k=2048,
            sa_extend_backend="streaming",
            scan_extend_backend="relative",
        ),
    ]
    _DEFAULT_STAGES_DECODE = [
        ScanStage(
            stage_block_size_q=64,
            stage_block_stride_q=1,
            stage_chunk_size=32,
            stage_k=None,
            stage_stride=1,
        ),
        ScanStage(
            stage_block_size_q=64,
            stage_block_stride_q=1,
            stage_chunk_size=16,
            stage_k=32768,
            stage_stride=1,
        ),
        ScanStage(
            stage_block_size_q=64,
            stage_block_stride_q=1,
            stage_chunk_size=4,
            stage_k=8192,
            stage_stride=1,
        ),
    ]
    _DEFAULT_LAEYRS_DECODE = [
        HiPAttentionPerLayerConfig(
            second_stage_k=4096,
            sa_extend_backend="streaming",
            scan_extend_backend="streaming",
            stages=_DEFAULT_STAGES_DECODE,
        ),
        HiPAttentionPerLayerConfig(
            second_stage_k=2048,
            sa_extend_backend="streaming",
            scan_extend_backend="relative",
            stages=_DEFAULT_STAGES_DECODE,
        ),
    ]
elif HIP_CONFIG_PRESET == "qwen3":
    _DEFAULT_LAEYRS = [
        HiPAttentionPerLayerConfig(
            second_stage_k=4096,
            sa_extend_backend="streaming",
            scan_extend_backend="streaming",
        ),
        HiPAttentionPerLayerConfig(
            second_stage_k=2048,
            sa_extend_backend="streaming",
            scan_extend_backend="relative",
        ),
    ]
    _DEFAULT_STAGES_DECODE_ST = [
        ScanStage(
            stage_block_size_q=64,
            stage_block_stride_q=1,
            stage_chunk_size=128,
            stage_k=None,
            stage_stride=1,
        ),
        ScanStage(
            stage_block_size_q=64,
            stage_block_stride_q=1,
            stage_chunk_size=16,
            stage_k=32768,
            stage_stride=1,
        ),
        ScanStage(
            stage_block_size_q=64,
            stage_block_stride_q=1,
            stage_chunk_size=4,
            stage_k=8192,
            stage_stride=1,
        ),
    ]
    _DEFAULT_STAGES_DECODE_RT = [
        ScanStage(
            stage_block_size_q=64,
            stage_block_stride_q=1,
            stage_chunk_size=32,
            stage_k=None,
            stage_stride=1,
        ),
        ScanStage(
            stage_block_size_q=64,
            stage_block_stride_q=1,
            stage_chunk_size=16,
            stage_k=32768,
            stage_stride=1,
        ),
        ScanStage(
            stage_block_size_q=64,
            stage_block_stride_q=1,
            stage_chunk_size=4,
            stage_k=8192,
            stage_stride=1,
        ),
    ]
    _DEFAULT_LAEYRS_DECODE = [
        HiPAttentionPerLayerConfig(
            sliding_window_size=24576,
            second_stage_k=4096,
            sa_extend_backend="streaming",
            scan_extend_backend="streaming",
            stages=_DEFAULT_STAGES_DECODE_ST,
        ),
        HiPAttentionPerLayerConfig(
            sliding_window_size=24576,
            second_stage_k=2048,
            sa_extend_backend="streaming",
            scan_extend_backend="relative",
            stages=_DEFAULT_STAGES_DECODE_RT,
        ),
    ]
else:
    raise Exception(f"unknown preset `{HIP_CONFIG_PRESET}`")


def try_parse_json(json_or_path: str):
    if json_or_path is None:
        parsed_json = {}
    elif isinstance(json_or_path, dict):
        parsed_json = json_or_path
    elif json_or_path.startswith("{"):
        parsed_json = json.loads(json_or_path)
    else:
        with open(json_or_path, "r") as f:
            parsed_json = json.load(f)
    return parsed_json


@dataclass
class HiPAttentionConfig:
    dense_layers: list[int] = field(
        default_factory=lambda: [
            0,
            1,
            2,
            3,
        ]
    )
    block_sparse_block_size_q: int = 64
    metadata_cache_max_batch_size: int = 32
    mask_refresh_interval: Union[int, List[int]] = field(
        default_factory=lambda: [64, 16, 8]
    )
    using_extend: bool = True
    layers: list[HiPAttentionPerLayerConfig] = field(
        default_factory=lambda: _DEFAULT_LAEYRS_DECODE
    )
    prefill_layers: list[HiPAttentionPerLayerConfig] = field(
        default_factory=lambda: _DEFAULT_LAEYRS
    )

    # deprecated
    apply_v_dot: bool = False
    prefill_always_dense: bool = False
    decode_always_dense: bool = False
    force_dense: bool = False
    prefill_dense_threshold: int = 8192

    json_or_path: InitVar[Optional[str]] = None
    json_override: InitVar[Optional[str]] = None

    def __post_init__(
        self,
        json_or_path: Optional[str],
        json_override: Optional[str],
    ):
        super().__init__()

        parsed_json = try_parse_json(json_or_path)
        parsed_json_override = try_parse_json(json_override)
        parsed_json.update(parsed_json_override)

        if parsed_json is not None:
            if "apply_v_dot" in parsed_json:
                self.apply_v_dot = parsed_json["apply_v_dot"]
                parsed_json.pop("apply_v_dot")
            if "dense_layers" in parsed_json:
                self.dense_layers = parsed_json["dense_layers"]
                parsed_json.pop("dense_layers")
            if "prefill_always_dense" in parsed_json:
                self.prefill_always_dense = parsed_json["prefill_always_dense"]
                parsed_json.pop("prefill_always_dense")
            if "decode_always_dense" in parsed_json:
                self.decode_always_dense = parsed_json["decode_always_dense"]
                parsed_json.pop("decode_always_dense")
            if "force_dense" in parsed_json:
                self.force_dense = parsed_json["force_dense"]
                parsed_json.pop("force_dense")
            if "prefill_dense_threshold" in parsed_json:
                self.prefill_dense_threshold = parsed_json["prefill_dense_threshold"]
                parsed_json.pop("prefill_dense_threshold")
            if "block_sparse_block_size_q" in parsed_json:
                self.block_sparse_block_size_q = parsed_json[
                    "block_sparse_block_size_q"
                ]
                parsed_json.pop("block_sparse_block_size_q")
            if "metadata_cache_max_batch_size" in parsed_json:
                self.metadata_cache_max_batch_size = parsed_json[
                    "metadata_cache_max_batch_size"
                ]
                parsed_json.pop("metadata_cache_max_batch_size")
            if "mask_refresh_interval" in parsed_json:
                assert isinstance(parsed_json["mask_refresh_interval"], (int, list))
                self.mask_refresh_interval = parsed_json["mask_refresh_interval"]
                parsed_json.pop("mask_refresh_interval")
            if "using_extend" in parsed_json:
                self.using_extend = parsed_json["using_extend"]
                parsed_json.pop("using_extend")
            if "layers" in parsed_json:
                if parsed_json["layers"] is None:
                    self.layers = None
                else:
                    self.layers = [
                        HiPAttentionPerLayerConfig(parsed_json=layer)
                        for layer in parsed_json["layers"]
                    ]
                parsed_json.pop("layers")
            if "prefill_layers" in parsed_json:
                if parsed_json["prefill_layers"] is None:
                    self.prefill_layers = None
                else:
                    self.prefill_layers = [
                        HiPAttentionPerLayerConfig(parsed_json=layer)
                        for layer in parsed_json["prefill_layers"]
                    ]
                parsed_json.pop("prefill_layers")

            # FIXME following args are just temporary. need to be removed when features are stabled
            if "__delta_attention_args" in parsed_json:
                given_args = parsed_json["__delta_attention_args"]
                if os.getenv("HIP_DELTA_ATTENTION_ARGS", given_args) != given_args:
                    warnings.warn(
                        "envvar HIP_DELTA_ATTENTION_ARGS is overrided by hip attention args"
                    )
                os.environ["HIP_DELTA_ATTENTION_ARGS"] = given_args
                parsed_json.pop("__delta_attention_args")
            if "__using_dense_prefill" in parsed_json:
                given_args = parsed_json["__using_dense_prefill"]
                if os.getenv("HIP_DEBUG_USING_DENSE_PREFILL", given_args) != given_args:
                    warnings.warn(
                        "envvar HIP_DEBUG_USING_DENSE_PREFILL is overrided by hip attention args"
                    )
                os.environ["HIP_DEBUG_USING_DENSE_PREFILL"] = "1" if given_args else "0"
                parsed_json.pop("__using_dense_prefill")
            if "__head_reduce" in parsed_json:
                given_args = parsed_json["__head_reduce"]
                if os.getenv("HIP_HEAD_REDUCE", given_args) != given_args:
                    warnings.warn(
                        "envvar HIP_HEAD_REDUCE is overrided by hip attention args"
                    )
                assert int(str(given_args)) == given_args
                os.environ["HIP_HEAD_REDUCE"] = str(given_args)
                parsed_json.pop("__head_reduce")
            if "__using_landmark" in parsed_json:
                given_args = parsed_json["__using_landmark"]
                if (
                    os.getenv("HIP_DEBUG_LANDMARK_BASED_SCAN_STAGE", given_args)
                    != given_args
                ):
                    warnings.warn(
                        "envvar HIP_DEBUG_LANDMARK_BASED_SCAN_STAGE is overrided by hip attention args"
                    )
                assert (int("1" if given_args else "0") == 1) == given_args
                os.environ["HIP_DEBUG_LANDMARK_BASED_SCAN_STAGE"] = (
                    "1" if given_args else "0"
                )
                parsed_json.pop("__using_landmark")
            if "__last_dense" in parsed_json:
                given_args = parsed_json["__last_dense"]
                if os.getenv("HIP_DEBUG_LAST_DENSE", given_args) != given_args:
                    warnings.warn(
                        "envvar HIP_DEBUG_LAST_DENSE is overrided by hip attention args"
                    )
                assert int(str(given_args)) == given_args
                os.environ["HIP_DEBUG_LAST_DENSE"] = str(given_args)
                parsed_json.pop("__last_dense")
            if "__seq_thresh_fa3" in parsed_json:
                given_args = parsed_json["__seq_thresh_fa3"]
                if os.getenv("HIP_DEBUG_SEQ_THRESH_FA3", given_args) != given_args:
                    warnings.warn(
                        "envvar HIP_HEAD_REDUCE is overrided by hip attention args"
                    )
                assert int(str(given_args)) == given_args
                os.environ["HIP_DEBUG_SEQ_THRESH_FA3"] = str(given_args)
                os.environ["HIP_DEBUG_ALLOW_GATHER_KV_CACHE"] = "1"
                parsed_json.pop("__seq_thresh_fa3")

            if parsed_json:
                raise ValueError(f"Unknown keys in json: {parsed_json.keys()}")

        if (self.prefill_layers is None) and (self.layers is not None):
            self.prefill_layers = self.layers
        elif (self.prefill_layers is not None) and (self.layers is None):
            self.layers = self.prefill_layers
        elif (self.prefill_layers is None) and (self.layers is None):
            raise Exception("`prefill_layers` or `layers` should be provided")
        else:
            pass  # okay

        num_stages = len(self.layers[0].stages)
        for layer_config in self.layers:
            assert num_stages == len(layer_config.stages)

        if isinstance(self.mask_refresh_interval, int):
            self.mask_refresh_interval = [
                self.mask_refresh_interval,
            ] * num_stages

        assert (
            self.block_sparse_block_size_q
            <= self.layers[-1].stages[-1].stage_block_size_q
        )
        assert (
            self.block_sparse_block_size_q
            <= self.prefill_layers[-1].stages[-1].stage_block_size_q
        )

```

#### `hip_attn/v1_2/hip_memory_pool.py`

```py
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Literal, Optional

import torch
import triton

from hip_attn.v1_2.attention_metadata import (
    HiPAttentionCacheAccessStatistics,
    HiPAttentionOutputMetadata,
    HiPAttentionStageInputCache,
)

if TYPE_CHECKING:
    from hip_attn.v1_2.hip_config import HiPAttentionConfig

logger = logging.getLogger(__name__)

IGNORE_MISS_MATCH = os.getenv("HIP_DEBUG_IGNORE_MISS_MATCH", "0") == "1"


@dataclass
class CachedBuffer:
    buffer: torch.Tensor
    batch_format: Literal["BH", "B,1,H"]
    dtype: torch.dtype

    def get(self, batch_size: int, head_size: int) -> torch.Tensor:
        if self.buffer.shape[0] < batch_size:
            raise ValueError(
                f"Buffer batch size {self.buffer.shape[0]} is smaller than the requested batch size {batch_size}.\n"
                f'Try lowering --cuda-graph-max-bs or raising --hip-attention-config {{"metadata_cache_max_batch_size"}}.'
            )
        if self.batch_format == "BH":
            return self.buffer[: batch_size * head_size].to(self.dtype, copy=False)
        elif self.batch_format == "B,1,H":
            return self.buffer[:batch_size, :, :head_size].to(self.dtype, copy=False)
        else:
            raise Exception()

    def set(self, value: torch.Tensor):
        if self.buffer.shape[0] < value.shape[0]:
            raise ValueError(
                f"Buffer batch size {self.buffer.shape[0]} is smaller than value batch size {value.shape[0]}.\n"
                f'Try lowering --cuda-graph-max-bs or raising --hip-attention-config {{"metadata_cache_max_batch_size"}}.'
            )
        if self.batch_format == "BH":
            if IGNORE_MISS_MATCH:
                if self.buffer[: value.shape[0]].shape != value.shape:
                    return
            self.buffer[: value.shape[0]].copy_(value.to(self.buffer.dtype))
        elif self.batch_format == "B,1,H":
            if IGNORE_MISS_MATCH:
                if (
                    self.buffer[: value.shape[0], :, : value.shape[2]].shape
                    != value.shape
                ):
                    return
            self.buffer[: value.shape[0], :, : value.shape[2]].copy_(
                value.to(self.buffer.dtype)
            )
        else:
            raise Exception()


class HiPMetadataCachePool:
    cache: List[Dict[str, CachedBuffer]]

    def __init__(
        self,
        max_total_num_tokens: int,
        query_head_num: int,
        layer_num: int,
        context_length: int,
        device: str,
        hip_config: "HiPAttentionConfig",
    ):
        self.hip_config = hip_config
        self.layer_num = layer_num
        self.cache = [{} for _ in range(layer_num)]
        self.head_num = query_head_num
        self.max_batch_size = hip_config.metadata_cache_max_batch_size
        self.device = device
        self.allocated_gpu_bytes = 0
        self.layer_configs = {}

        for layer_idx in range(layer_num):
            require_dense = layer_idx in hip_config.dense_layers
            if len(hip_config.layers) == 2:
                layer_config = hip_config.layers[0 if require_dense else 1]
            else:
                layer_config = hip_config.layers[layer_idx]
            self.layer_configs[layer_idx] = layer_config

            additional_tokens = 0
            # if os.getenv("HIP_DEBUG_SNAP_KV", "0") == "1":
            #     additional_tokens += 8192
            if os.getenv("HIP_DEBUG_UNION_HEAD", "0") == "1":
                additional_tokens += layer_config.second_stage_k * (self.head_num - 1)
            if os.getenv("HIP_DIAG_INFO", None) != None:
                additional_tokens += 8192 if require_dense else 4096
            if os.getenv("HIP_DEBUG_ADD_DELAY_WINDOW", "0") == "1":
                additional_tokens += layer_config.second_stage_k * (
                    64 // layer_config.stages[-1].stage_chunk_size
                )

            actual_tokens = layer_config.second_stage_k + additional_tokens
            if actual_tokens != layer_config.second_stage_k:
                print(
                    f"actual attened tokens are {actual_tokens + layer_config.sliding_window_size + layer_config.sink_token_size}"
                )

            n_chunks = triton.cdiv(
                actual_tokens,
                layer_config.stages[-1].stage_chunk_size,
            )

            num_q_blocks = 1
            self.init_buffer(
                layer_idx,
                "indices",
                [num_q_blocks, n_chunks],
                torch.int64,
                store_dtype=torch.uint32,
            )
            self.init_buffer(layer_idx, "ks", [num_q_blocks], torch.int64)
            self.init_buffer(layer_idx, "ks_count", [num_q_blocks, 1], torch.int64)
            self.init_buffer(layer_idx, "ks_start_end", [num_q_blocks, 2], torch.int64)

            self.init_buffer(
                layer_idx, "mask_access_count", [num_q_blocks], torch.int64
            )
            self.init_buffer(
                layer_idx, "mask_unique_access_count", [num_q_blocks], torch.int64
            )
            self.init_buffer(
                layer_idx, "mask_cache_miss_count", [num_q_blocks], torch.int64
            )

            self.init_buffer(layer_idx, "sa_access_count", [num_q_blocks], torch.int64)
            self.init_buffer(
                layer_idx, "sa_unique_access_count", [num_q_blocks], torch.int64
            )
            self.init_buffer(
                layer_idx, "sa_cache_miss_count", [num_q_blocks], torch.int64
            )

            for i_stage, stage in enumerate(layer_config.stages):
                if i_stage > 0:
                    max_context_length = (
                        context_length
                        - layer_config.sliding_window_size
                        - layer_config.sink_token_size
                    )
                    chunk_count = (
                        min(stage.stage_k, max_context_length) // stage.stage_chunk_size
                    )
                    self.init_buffer(
                        layer_idx,
                        f"stage_{i_stage}_indices_left",
                        [chunk_count],
                        torch.int64,
                        "B,1,H",
                        torch.uint32,
                    )
                    self.init_buffer(
                        layer_idx,
                        f"stage_{i_stage}_indices_right",
                        [chunk_count],
                        torch.int64,
                        "B,1,H",
                        torch.uint32,
                    )
                    self.init_buffer(
                        layer_idx,
                        f"stage_{i_stage}_out_scores",
                        [chunk_count],
                        torch.float32,
                        "B,1,H",
                        torch.bfloat16,
                    )

        self.num_delays = int(os.getenv("HIP_DEBUG_DELAYED_STAGE0", "0"))
        self.delayed_first_stage = [[] for _ in range(self.layer_num)]

        self.allocated_gpu_bytes = self.compute_allocated_bytes()
        logger.info(
            f"Allocated HiP metadata cache pool size: {self.allocated_gpu_bytes / 1024 / 1024:.2f} MB"
        )

    def reset_decode_phase(self):
        # This function is called in sglang/srt/model_executor/forward_batch_info.py
        for layer in self.delayed_first_stage:
            layer.clear()

    def compute_allocated_bytes(self):
        t = 0
        for layer_buffer in self.cache:
            for v in layer_buffer.values():
                t += v.buffer.numel() * v.buffer.element_size()
        return t

    def init_buffer(
        self,
        layer_idx: int,
        name: str,
        shape: List[int],
        dtype: torch.dtype,
        batch_format: Literal["BH", "B,1,H"] = "BH",
        store_dtype: Optional[torch.dtype] = None,
    ):
        layer_buffer = self.cache[layer_idx]
        if batch_format == "BH":
            layer_buffer[name] = CachedBuffer(
                buffer=torch.zeros(
                    (self.max_batch_size * self.head_num, *shape),
                    device=self.device,
                    dtype=dtype if store_dtype is None else store_dtype,
                ),
                batch_format=batch_format,
                dtype=dtype,
            )
        elif batch_format == "B,1,H":
            layer_buffer[name] = CachedBuffer(
                buffer=torch.zeros(
                    (self.max_batch_size, 1, self.head_num, *shape),
                    device=self.device,
                    dtype=dtype if store_dtype is None else store_dtype,
                ),
                batch_format=batch_format,
                dtype=dtype,
            )
        else:
            raise Exception()

    def get_buffer(self, layer_idx: int, name: str, batch_size: int):
        if not layer_idx in range(len(self.cache)):
            raise Exception(f"{layer_idx} is not in range({len(self.cache)})")
        if not name in self.cache[layer_idx]:
            raise Exception(f"{name} is not in {self.cache[layer_idx].keys()}")
        return self.cache[layer_idx][name].get(batch_size, self.head_num)

    def set_buffer(self, layer_idx: int, name: str, value: torch.Tensor):
        if not layer_idx in range(len(self.cache)):
            raise Exception(f"{layer_idx} is not in range({len(self.cache)})")
        if not name in self.cache[layer_idx]:
            raise Exception(f"{name} is not in {self.cache[layer_idx].keys()}")
        self.cache[layer_idx][name].set(value)

    def get_hip_metadata_cache(
        self,
        layer_id: int,
        tdst: int,
        batch_size: int,
        cached_stages: Optional[int],
        block_size_q: int = 64,
    ) -> Optional[HiPAttentionOutputMetadata]:
        assert (
            triton.cdiv(tdst // batch_size, block_size_q) == 1
        ), f"triton.cdiv({tdst} // {batch_size}, {block_size_q}) == 1"

        if (cached_stages is None) or (
            cached_stages == len(self.layer_configs[layer_id].stages)
        ):
            return HiPAttentionOutputMetadata(
                indices=self.get_buffer(layer_id, "indices", batch_size),
                ks=self.get_buffer(layer_id, "ks", batch_size),
                ks_count=self.get_buffer(layer_id, "ks_count", batch_size),
                ks_start_end=self.get_buffer(layer_id, "ks_start_end", batch_size),
                mask_cache_statistics=None,
                sa_cache_statistics=None,
                stage_caches=None,
            )
        elif cached_stages == 0:
            # NOTE: reset the cache, let hip attention compute everything from scratch
            return
        else:
            stage_caches = []
            for i_stage in range(cached_stages + 1):
                if i_stage == 0:
                    stage_caches.append(
                        HiPAttentionStageInputCache(
                            indices_left=None,
                            indices_right=None,
                            out_scores=None,
                        )
                    )
                else:
                    stage_caches.append(
                        HiPAttentionStageInputCache(
                            indices_left=self.get_buffer(
                                layer_id, f"stage_{i_stage}_indices_left", batch_size
                            ),
                            indices_right=self.get_buffer(
                                layer_id, f"stage_{i_stage}_indices_right", batch_size
                            ),
                            out_scores=self.get_buffer(
                                layer_id, f"stage_{i_stage}_out_scores", batch_size
                            ),
                        )
                    )
            return HiPAttentionOutputMetadata(
                indices=None,
                ks=None,
                ks_count=None,
                ks_start_end=None,
                mask_cache_statistics=None,
                sa_cache_statistics=None,
                stage_caches=stage_caches,
            )

    def set_hip_metadata_cache(
        self,
        layer_id: int,
        tdst: int,
        batch_size: int,
        metadata: HiPAttentionOutputMetadata,
        block_size_q: int = 64,
        cached_stages: Optional[int] = None,
    ):
        assert triton.cdiv(tdst // batch_size, block_size_q) == 1

        def update_cache_stats(stats: HiPAttentionCacheAccessStatistics, prefix: str):
            if stats is None:
                # access_count = torch.zeros((1,), dtype=torch.int64, device=self.device)
                # unique_access_count = torch.zeros(
                #     (1,), dtype=torch.int64, device=self.device
                # )
                # cache_miss_count = torch.zeros(
                #     (1,), dtype=torch.int64, device=self.device
                # )
                access_count = None
            else:
                computed_statistics = stats.compute_statistics()
                access_count = computed_statistics["access_count"]
                unique_access_count = computed_statistics["unique_access_count"]
                cache_miss_count = computed_statistics["cache_miss_count"]

            if access_count is not None:
                self.set_buffer(
                    layer_id,
                    f"{prefix}_access_count",
                    access_count.view(1, 1).expand(self.max_batch_size, 1),
                )
                self.set_buffer(
                    layer_id,
                    f"{prefix}_unique_access_count",
                    unique_access_count.view(1, 1).expand(self.max_batch_size, 1),
                )
                self.set_buffer(
                    layer_id,
                    f"{prefix}_cache_miss_count",
                    cache_miss_count.view(1, 1).expand(self.max_batch_size, 1),
                )

        update_cache_stats(metadata.sa_cache_statistics, "sa")
        update_cache_stats(metadata.mask_cache_statistics, "mask")

        if (cached_stages is None) or (cached_stages == len(self.layer_configs[layer_id].stages)):
            return

        self.set_buffer(layer_id, "indices", metadata.indices)
        self.set_buffer(layer_id, "ks", metadata.ks)
        self.set_buffer(layer_id, "ks_count", metadata.ks_count)
        self.set_buffer(layer_id, "ks_start_end", metadata.ks_start_end)

        if metadata.stage_caches is not None:
            for i_stage, cache in enumerate(metadata.stage_caches):
                if i_stage == 0:
                    pass
                elif i_stage == 1:
                    if torch.cuda.is_current_stream_capturing() and self.num_delays > 0:
                        raise Exception(
                            "delayed stage is only supported on eager mode for research purpose."
                        )
                    if len(self.delayed_first_stage[layer_id]) == 0:
                        self.set_buffer(
                            layer_id,
                            f"stage_{i_stage}_indices_left",
                            cache.indices_left,
                        )
                        self.set_buffer(
                            layer_id,
                            f"stage_{i_stage}_indices_right",
                            cache.indices_right,
                        )
                        self.set_buffer(
                            layer_id,
                            f"stage_{i_stage}_out_scores",
                            cache.out_scores,
                        )
                    self.delayed_first_stage[layer_id].append(
                        {
                            "indices_left": cache.indices_left,
                            "indices_right": cache.indices_right,
                            "out_scores": cache.out_scores,
                        }
                    )
                    if len(self.delayed_first_stage[layer_id]) > self.num_delays:
                        delayed_state = self.delayed_first_stage[layer_id].pop(0)
                        self.set_buffer(
                            layer_id,
                            f"stage_{i_stage}_indices_left",
                            delayed_state["indices_left"],
                        )
                        self.set_buffer(
                            layer_id,
                            f"stage_{i_stage}_indices_right",
                            delayed_state["indices_right"],
                        )
                        self.set_buffer(
                            layer_id,
                            f"stage_{i_stage}_out_scores",
                            delayed_state["out_scores"],
                        )
                else:
                    self.set_buffer(
                        layer_id,
                        f"stage_{i_stage}_indices_left",
                        cache.indices_left,
                    )
                    self.set_buffer(
                        layer_id,
                        f"stage_{i_stage}_indices_right",
                        cache.indices_right,
                    )
                    self.set_buffer(
                        layer_id, f"stage_{i_stage}_out_scores", cache.out_scores
                    )

    def compute_cache_statistics(self, batch_size: int):
        def compute(prefix):
            total_access = 0
            total_miss = 0
            for idx_layer in range(self.layer_num):
                access_count = self.get_buffer(
                    idx_layer, f"{prefix}_access_count", batch_size
                )
                miss_count = self.get_buffer(
                    idx_layer, f"{prefix}_cache_miss_count", batch_size
                )
                total_access += access_count.sum()
                total_miss += miss_count.sum()
            return {
                f"{prefix}_access": total_access,
                f"{prefix}_miss": total_miss,
                f"{prefix}_hit_ratio": 1 - (total_miss / total_access),
            }

        result = {}
        result.update(compute("sa"))
        result.update(compute("mask"))
        return result

```

#### `hip_attn/v1_2/mask_refresh_interval.py`

```py
from hip_attn.v1_2.hip_config import HiPAttentionConfig


class HiPMaskRefreshState:
    def __init__(self, hip_config: HiPAttentionConfig):
        self.hip_config = hip_config
        self.decode_index = 0

    def update(self):
        metadata_cached_stages = None

        if self.hip_config.mask_refresh_interval is not None:
            require_refresh = False

            for i_stage, refresh_inteval in enumerate(
                self.hip_config.mask_refresh_interval
            ):
                if self.decode_index % refresh_inteval == 0 and not require_refresh:
                    metadata_cached_stages = i_stage
                    require_refresh = True

            if not require_refresh:
                metadata_cached_stages = None

        if self.decode_index == 0:
            metadata_cached_stages = -1

        self.decode_index += 1

        return metadata_cached_stages

```

#### `hip_attn/v1_2/model_offload_cache.py`

```py
import logging
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from hip_attn.v1_2.hip_config import HiPAttentionConfig
from hip_attn.v1_2.uvm_gpu_cache import HiPOffloadCache

logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024


class HiPModelOffloadCache:
    def __init__(
        self,
        max_token_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: torch.device,
        hip_config: HiPAttentionConfig,
        max_mask_cache_token_size: Union[List[Optional[int]], Optional[int]] = None,
        max_sa_cache_token_size: Union[List[Optional[int]], Optional[int]] = None,
        max_mask_cache_factor: Union[List[Optional[float]], Optional[float]] = None,
        max_sa_cache_factor: Union[List[Optional[float]], Optional[float]] = None,
        chunked_attention_size: int = 0,
        irope_offset: int = 0,
        irope_interval: int = 0,
    ):
        from hip_attn.v1_2.uvm_gpu_cache import HiPOffloadCache, format_size_bytes

        assert isinstance(device, torch.device)
        assert device.index is not None

        def repeat_if_not_list(obj):
            if isinstance(obj, (list, tuple)):
                assert len(obj) == layer_num
            else:
                obj = [
                    obj,
                ] * layer_num
            return obj

        max_mask_cache_token_size = repeat_if_not_list(max_mask_cache_token_size)
        max_sa_cache_token_size = repeat_if_not_list(max_sa_cache_token_size)
        max_mask_cache_factor = repeat_if_not_list(max_mask_cache_factor)
        max_sa_cache_factor = repeat_if_not_list(max_sa_cache_factor)

        self.size = max_token_size
        self.dtype = dtype
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.device = device

        self.chunked_attention_size = chunked_attention_size
        self.irope_offset = irope_offset
        self.irope_interval = irope_interval

        # TODO: derive token sizes from size
        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self.online_update_cache = os.getenv("DEBUG_ONLINE", "0") == "1"
        self.layer_buffer = []
        for layer_id in range(layer_num):
            is_dense = layer_id in hip_config.dense_layers
            if len(hip_config.layers) == 2:
                layer_config = hip_config.layers[0 if is_dense else 1]
            else:
                layer_config = hip_config.layers[layer_id]

            if max_mask_cache_token_size[layer_id] is not None:
                cur_max_mask_cache_token_size = (
                    max_mask_cache_token_size[layer_id] * head_num
                )
                if layer_id in hip_config.dense_layers:
                    cur_max_mask_cache_token_size *= 2
            else:
                assert max_mask_cache_factor[layer_id] is not None
                base_mask_cache_tokens = (
                    (max_token_size / layer_config.stages[0].stage_chunk_size)
                    * 2
                    * math.log2(layer_config.stages[0].stage_chunk_size)
                )
                cur_max_mask_cache_token_size = math.ceil(
                    max_mask_cache_factor[layer_id] * base_mask_cache_tokens
                )
            assert isinstance(cur_max_mask_cache_token_size, int)

            if max_sa_cache_token_size[layer_id] is not None:
                cur_max_sa_cache_token_size = (
                    max_sa_cache_token_size[layer_id] * head_num
                )
                if layer_id in hip_config.dense_layers:
                    cur_max_sa_cache_token_size *= 2
            else:
                assert max_sa_cache_factor[layer_id] is not None
                base_sa_cache_tokens = (
                    layer_config.sink_token_size
                    + layer_config.sliding_window_size
                    + layer_config.second_stage_k
                )
                cur_max_sa_cache_token_size = math.ceil(
                    max_sa_cache_factor[layer_id] * base_sa_cache_tokens
                )
            assert isinstance(cur_max_sa_cache_token_size, int)

            self.layer_buffer.append(
                HiPOffloadCache(
                    layer_id=layer_id,
                    max_token_size=max_token_size + 1,
                    max_mask_cache_token_size=min(
                        max_token_size * head_num, cur_max_mask_cache_token_size
                    ),
                    max_sa_cache_token_size=min(
                        max_token_size * head_num, cur_max_sa_cache_token_size
                    ),
                    head_num=head_num,
                    head_dim=head_dim,
                    dtype=dtype,
                    device=device,
                    online_cache_update=self.online_update_cache,
                )
            )

            uvm_allocated_bytes, gpu_allocated_bytes = self._calc_allocated_bytes()
            logger.info(
                f"[{layer_id + 1}/{layer_num}] "
                f"CPU (UVM): {format_size_bytes(uvm_allocated_bytes)} and "
                f"GPU: {format_size_bytes(gpu_allocated_bytes)} are allocated. "
                f"({self.dtype} on {self.device}, "
                f"{tuple(self.layer_buffer[-1].k_uvm.bank_cpu.shape)}, {tuple(self.layer_buffer[-1].mask_k_cache.bank.shape)})"
            )

        # (layer_id, batch_id) -> (K, V, seq_len)
        self.prefetch_threads: Dict[Tuple[int, int], threading.Thread] = {}
        self.prefetched_kv: Dict[Tuple[int, int], Tuple[Tensor, Tensor, int]] = {}

        self.async_set_threads: Set[threading.Thread] = set()

        self.copy_stream = torch.cuda.Stream(self.device)

        self.enable_async = os.getenv("HIP_DISABLE_AYSNC", "0") == "0"

        # uvm_allocated_bytes, gpu_allocated_bytes = self.calc_allocated_bytes()
        # logger.info(
        #     f'Allocated total CPU (UVM) bytes: {format_size_bytes(uvm_allocated_bytes)}, '
        #     f'Allocated total GPU bytes: {format_size_bytes(gpu_allocated_bytes)}, '
        #     f'{self.dtype} on {self.device}'
        # )

        self.require_validation = os.getenv("HIP_OFFLOAD_CACHE_VALIDATION", "0") == "1"
        if self.require_validation:
            self.validation_cache = MHATokenToKVPool(
                max_token_size,
                dtype=dtype,
                head_num=head_num,
                head_dim=head_dim,
                layer_num=layer_num,
                device=self.device,
            )
        else:
            self.validation_cache = None

    def is_online_cache_update_enabled(self):
        return self.online_update_cache

    def get_kv_buffer(
        self,
        layer_id: int,
    ) -> Tuple[HiPOffloadCache, Any]:
        # Use this function for decode, pass this to `k`
        if self.require_validation:
            return self.layer_buffer[layer_id], self.validation_cache.get_kv_buffer(
                layer_id
            )
        return self.layer_buffer[layer_id], None

    def get_fetched_prefix_kv_buffer(
        self,
        layer_id: int,
        batch_id: Optional[int] = None,
        # you need to pass KV for extend
        cache_k: Optional[Tensor] = None,
        cache_v: Optional[Tensor] = None,
        extend_seq_lens: Optional[Tensor] = None,
        extend_seq_lens_cpu: Optional[List[int]] = None,
    ) -> Tuple[
        Union[Tensor, List[Tensor]], Union[Tensor, List[Tensor]], Union[Any, List[Any]]
    ]:

        if batch_id is not None:
            return self._get_fetched_prefix_kv_buffer_single(
                layer_id=layer_id,
                batch_id=batch_id,
                cache_k=cache_k,
                cache_v=cache_v,
            )

        else:
            k_chunks = []
            v_chunks = []
            offloading_metadata_list = []

            start_len = 0
            for idx_batch, seq_len in enumerate(extend_seq_lens_cpu):
                if seq_len > 0:  # Skip empty sequences
                    k_chunk, v_chunk, offloading_metadata = (
                        self._get_fetched_prefix_kv_buffer_single(
                            layer_id,
                            idx_batch,
                            cache_k=cache_k[start_len : start_len + seq_len].unsqueeze(
                                0
                            ),
                            cache_v=cache_v[start_len : start_len + seq_len].unsqueeze(
                                0
                            ),
                        )
                    )
                    k_chunks.append(k_chunk)
                    v_chunks.append(v_chunk)
                    offloading_metadata_list.append(offloading_metadata)

                else:
                    k_chunks.append(None)
                    v_chunks.append(None)
                    offloading_metadata_list.append(None)

                start_len += seq_len

            return k_chunks, v_chunks, offloading_metadata_list

    def _get_fetched_prefix_kv_buffer_single(
        self,
        layer_id: int,
        batch_id: Optional[int] = None,
        # you need to pass KV for extend
        cache_k: Optional[Tensor] = None,
        cache_v: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Any]:
        # return cache_k, cache_v

        # Use this function for prefill
        handle_id = (layer_id, batch_id)
        prefetch_thread = self.prefetch_threads.get(handle_id, None)
        if prefetch_thread is not None:
            while handle_id not in self.prefetched_kv:
                time.sleep(0.0001)
            # print('start join', flush=True)
            # while True:
            #     try:
            #         prefetch_thread.join(timeout=1.0)
            #         print('joined')
            #         break
            #     except TimeoutError:
            #         print('timeout', layer_id, batch_id)
            #     except RuntimeError:
            #         print('runtime error wtf')
            #         raise RuntimeError('deadlock')

        assert handle_id in self.prefetched_kv, "did prefetch successed?"
        k, v, prefix_seq_len, table, copy_event = self.prefetched_kv.pop(handle_id)

        assert isinstance(k, Tensor)
        assert isinstance(v, Tensor)
        assert isinstance(prefix_seq_len, int)
        assert k.shape == v.shape
        assert k.ndim == 4, f"{k.shape}"
        assert k.shape[0] == 1
        assert k.shape[1] >= prefix_seq_len
        assert k.shape[2] == self.head_num
        assert k.shape[3] == self.head_dim
        assert k.dtype == v.dtype
        assert k.dtype == self.dtype
        assert cache_k.ndim == 4
        assert cache_k.shape[0] == 1
        assert cache_k.shape[2] == self.head_num
        assert cache_k.shape[3] == self.head_dim
        assert k.shape[1] == prefix_seq_len + cache_k.shape[1]
        assert k.dtype in [
            torch.float8_e5m2,
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ]

        if copy_event is not None:
            torch.cuda.current_stream().wait_event(copy_event)

        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)
        # if self.dtype not in [torch.float8_e5m2]:
        #     assert cache_k.dtype == self.dtype
        # else:
        #     if cache_k.dtype != self.dtype:
        #         cache_k = cache_k.to(self.dtype)
        #         cache_v = cache_v.to(self.dtype)

        k[:, prefix_seq_len:, :, :] = cache_k
        v[:, prefix_seq_len:, :, :] = cache_v

        if self.require_validation:
            k_valid, v_valid = self.validation_cache.get_kv_buffer(layer_id)

            assert k.dtype == k_valid.dtype

            k_valid_packed = k_valid[table].unsqueeze(0)
            v_valid_packed = v_valid[table].unsqueeze(0)

            k_err = ((k_valid_packed - k) ** 2).sum()
            v_err = ((v_valid_packed - v) ** 2).sum()

            assert k_err < 1e-5, k_err
            assert v_err < 1e-5, v_err

            return k, v, (k_valid, v_valid)
        else:
            return k, v, None

    def set_kv_buffer(
        self,
        layer_id: int,
        table: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        async_copy: bool = False,
        push_to_gpu_cache: bool = False,
    ):
        if self.require_validation:
            self.validation_cache.set_kv_buffer(
                layer_id,
                table,
                cache_k,
                cache_v,
            )

        if not self.enable_async:
            async_copy = False
        # async_copy = False

        # pass async_copy=True when only prefill (eager mode)
        assert (not async_copy) or (
            async_copy and (not torch.cuda.is_current_stream_capturing())
        )

        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)

        if async_copy:
            stream = self.copy_stream

            table_gpu = table.to(torch.int64)

            start_event = torch.cuda.Event()
            start_event.record(torch.cuda.current_stream(self.device))

            def thread_main():
                try:
                    stream.wait_event(start_event)

                    with torch.cuda.stream(stream):
                        table_cpu = table.to("cpu", non_blocking=False)
                        cache_k_cpu = cache_k.to("cpu", non_blocking=False)
                        cache_v_cpu = cache_v.to("cpu", non_blocking=False)

                        self.layer_buffer[layer_id].set_kv_buffer(
                            table=table_cpu,
                            table_gpu=table_gpu,
                            cache_k=cache_k_cpu,
                            cache_v=cache_v_cpu,
                        )
                finally:
                    self.async_set_threads.remove(t)

            t = threading.Thread(target=thread_main, daemon=True)
            self.async_set_threads.add(t)
            t.start()
        else:
            self.layer_buffer[layer_id].set_kv_buffer(
                table=table,
                table_gpu=table,
                cache_k=cache_k,
                cache_v=cache_v,
            )

    def on_model_start(
        self,
        is_prefill,
        batch_size: int,
        req_to_token: Tensor,
        req_pool_indices: Tensor,
        extend_prefix_lens_cpu: np.array,
        extend_seq_lens_cpu: np.array,
    ):
        require_prefetch = is_prefill

        if require_prefetch:
            # FIXME: find better way to detect this.
            is_first_chunk = extend_prefix_lens_cpu[0] == 0
            # FIXME: find better way to detect this.
            is_inter_chunk = extend_seq_lens_cpu[0] in map(lambda x: 2**x, range(0, 20))
            # BUG(heejun): at the last chunk of prefill, prefetch layer sometimes failes... so disable async
            if not (batch_size == 1 and (is_first_chunk or is_inter_chunk)):
                self.onetime_disable = self.enable_async
                self.enable_async = False
            else:
                self.onetime_disable = False
            self._prefetch_layer(
                0,
                batch_size,
                req_to_token,
                req_pool_indices,
                extend_prefix_lens_cpu,
                extend_seq_lens_cpu,
            )
            # self.wait_prefetch_layer(forward_batch, 0)

    def on_model_end(self, is_prefill: bool):
        require_prefetch = is_prefill

        if require_prefetch:
            self._synchronize()
            self.enable_async = self.enable_async or self.onetime_disable
            self.onetime_disable = False

    def on_layer_start(
        self,
        layer_id: int,
        is_prefill: bool,
        batch_size: int,
        req_to_token: Tensor,
        req_pool_indices: Tensor,
        extend_prefix_lens_cpu: np.array,
        extend_seq_lens_cpu: np.array,
    ):
        require_prefetch = is_prefill

        if require_prefetch and (layer_id < (self.layer_num - 1)):
            self._prefetch_layer(
                layer_id + 1,
                batch_size,
                req_to_token,
                req_pool_indices,
                extend_prefix_lens_cpu,
                extend_seq_lens_cpu,
            )

    def on_layer_end(
        self,
        layer_id: int,
        is_prefill: bool,
    ):
        require_prefetch = is_prefill

        if require_prefetch and (layer_id < (self.layer_num - 1)):
            torch.cuda.current_stream(self.device).synchronize()

    def _prefetch_layer(
        self,
        layer_id: int,
        batch_size: int,
        req_to_token: Tensor,
        req_pool_indices: Tensor,
        extend_prefix_lens_cpu: np.array,
        extend_seq_lens_cpu: np.array,
    ):
        if self.chunked_attention_size > 0:
            if ((layer_id + self.irope_offset) % self.irope_interval) == 0:
                window = 0
            else:
                # for chunked attention
                window = (
                    self.chunked_attention_size
                    + np.amax(extend_seq_lens_cpu).item()
                    + 1024
                )
        else:
            window = 0

        for ibatch in range(batch_size):
            curr_req_pool_indices = req_pool_indices[ibatch : ibatch + 1]
            block_table = req_to_token.index_select(dim=0, index=curr_req_pool_indices)[
                0,
                : extend_prefix_lens_cpu[ibatch] + extend_seq_lens_cpu[ibatch],
            ]
            if window > 0:
                pad = max(
                    0,
                    extend_prefix_lens_cpu[ibatch]
                    + extend_seq_lens_cpu[ibatch]
                    - window,
                )
                block_table = block_table[pad:].contiguous()
            else:
                pad = 0
            # print(block_table, block_table.shape)
            self._prefetch_prefix_kv_buffer(
                layer_id=layer_id,
                batch_id=ibatch,
                table=block_table,
                prefix_seq_len=extend_prefix_lens_cpu[ibatch],
                pad=pad,
            )

    def _prefetch_prefix_kv_buffer(
        self,
        layer_id: int,
        batch_id: int,
        table: Tensor,
        prefix_seq_len: int,
        pad: int,
    ) -> threading.Thread:
        # you must call before get fetched prefix
        assert table.ndim == 1

        hip_offload_cache, _ = self.get_kv_buffer(layer_id)

        handle_id = (layer_id, batch_id)
        assert handle_id not in self.prefetch_threads, handle_id
        assert handle_id not in self.prefetched_kv, handle_id

        if self.enable_async:
            stream = self.copy_stream
            current_stream = torch.cuda.current_stream(self.device)

            table = table.to(torch.int64).to("cpu")

            start_event = torch.cuda.Event()
            start_event.record(current_stream)

            # torch.cuda.synchronize()
            def thread_main():
                try:
                    stream.wait_event(start_event)

                    with torch.cuda.stream(stream):
                        k, v = hip_offload_cache.prefetch_prefix_kv_buffer(
                            table=table,
                            device=self.device,
                            pad=pad,
                        )
                        assert k.device == self.device
                        assert v.device == self.device

                    copy_event = torch.cuda.Event()
                    copy_event.record(stream)

                    self.prefetched_kv[handle_id] = (
                        k,
                        v,
                        prefix_seq_len,
                        table,
                        copy_event,
                    )
                except Exception as ex:
                    print(f"{handle_id} thread dead")
                    raise Exception("thread dead") from ex
                finally:
                    self.prefetch_threads.pop(handle_id)

            t = threading.Thread(target=thread_main, daemon=True)
            self.prefetch_threads[handle_id] = t
            t.start()
        else:
            k, v = hip_offload_cache.prefetch_prefix_kv_buffer(
                table=table.to(torch.int64),
                device=self.device,
                pad=pad,
            )
            assert k.device == self.device
            assert v.device == self.device

            self.prefetched_kv[handle_id] = (k, v, prefix_seq_len, table, None)
        return

    def _synchronize(self):
        torch.cuda.synchronize(device=self.device)
        t = time.time()
        # you must call this function when finish prefill, before decode
        while (len(self.prefetch_threads) > 0) or (len(self.async_set_threads) > 0):
            time.sleep(0.001)
        assert len(self.prefetch_threads) == 0
        assert len(self.async_set_threads) == 0
        assert len(self.prefetched_kv) == 0
        elapsed = time.time() - t
        logger.debug(f"Final layer sync took {elapsed * 1024:.4f} ms")

    def _calc_allocated_bytes(self):
        uvm_allocated_bytes = 0
        gpu_allocated_bytes = 0
        for cache in self.layer_buffer:
            uvm_allocated_bytes += cache.k_uvm.allocated_cpu_bytes
            gpu_allocated_bytes += cache.k_uvm.allocated_gpu_bytes
            uvm_allocated_bytes += cache.v_uvm.allocated_cpu_bytes
            gpu_allocated_bytes += cache.v_uvm.allocated_gpu_bytes
            gpu_allocated_bytes += cache.mask_k_cache.allocated_gpu_bytes
            gpu_allocated_bytes += cache.sa_kv_cache.allocated_gpu_bytes
        return uvm_allocated_bytes, gpu_allocated_bytes


# For validation reference
class MHATokenToKVPool:
    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
    ):
        self.size = size
        self.dtype = dtype
        if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
            # NOTE: Store as torch.uint8 because Tensor.index_put is not implemented for torch.float8_e5m2
            self.store_dtype = torch.uint8
        else:
            self.store_dtype = dtype
        self.device = device

        self.head_num = head_num
        self.head_dim = head_dim
        self.layer_num = layer_num
        self._create_buffers()

        k_size, v_size = self.get_kv_size_bytes()
        logger.info(
            f"Reference KV Cache is allocated. K size: {k_size / GB:.2f} GB, V size: {v_size / GB:.2f} GB."
        )

    def _create_buffers(self):
        # [size, head_num, head_dim] for each layer
        # The padded slot 0 is used for writing dummy outputs from padded tokens.
        self.k_buffer = [
            torch.empty(
                (self.size + 1, self.head_num, self.head_dim),
                dtype=self.store_dtype,
                device=self.device,
            )
            for _ in range(self.layer_num)
        ]
        self.v_buffer = [
            torch.empty(
                (self.size + 1, self.head_num, self.head_dim),
                dtype=self.store_dtype,
                device=self.device,
            )
            for _ in range(self.layer_num)
        ]

    def _clear_buffers(self):
        del self.k_buffer
        del self.v_buffer

    def get_kv_size_bytes(self):
        assert hasattr(self, "k_buffer")
        assert hasattr(self, "v_buffer")
        k_size_bytes = 0
        for k_cache in self.k_buffer:
            k_size_bytes += np.prod(k_cache.shape) * k_cache.dtype.itemsize
        v_size_bytes = 0
        for v_cache in self.v_buffer:
            v_size_bytes += np.prod(v_cache.shape) * v_cache.dtype.itemsize
        return k_size_bytes, v_size_bytes

    # Todo: different memory layout
    def get_flat_data(self, indices):
        # prepare a large chunk of contiguous data for efficient transfer
        flatten = torch.stack(
            [
                torch.stack([self.k_buffer[i][indices] for i in range(self.layer_num)]),
                torch.stack([self.v_buffer[i][indices] for i in range(self.layer_num)]),
            ]
        )
        return flatten

    def transfer(self, indices, flat_data):
        # transfer prepared data from host to device
        flat_data = flat_data.to(device=self.device, non_blocking=False)
        k_data, v_data = flat_data[0], flat_data[1]
        for i in range(self.layer_num):
            self.k_buffer[i][indices] = k_data[i]
            self.v_buffer[i][indices] = v_data[i]

    def get_key_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.k_buffer[layer_id].view(self.dtype)
        return self.k_buffer[layer_id]

    def get_value_buffer(self, layer_id: int):
        if self.store_dtype != self.dtype:
            return self.v_buffer[layer_id].view(self.dtype)
        return self.v_buffer[layer_id]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def set_kv_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
    ):
        if cache_k.dtype != self.dtype:
            if k_scale is not None:
                cache_k.div_(k_scale)
            if v_scale is not None:
                cache_v.div_(v_scale)
            cache_k = cache_k.to(self.dtype)
            cache_v = cache_v.to(self.dtype)
        if self.store_dtype != self.dtype:
            self.k_buffer[layer_id][loc] = cache_k.view(self.store_dtype)
            self.v_buffer[layer_id][loc] = cache_v.view(self.store_dtype)
        else:
            self.k_buffer[layer_id][loc] = cache_k
            self.v_buffer[layer_id][loc] = cache_v

```

#### `hip_attn/v1_2/paged_hip.py`

```py
import copy
import os
import warnings
from typing import Any, List, Optional

import torch
import triton
from flash_attn import flash_attn_func
from matplotlib import pyplot as plt
from sgl_kernel.flash_attn import flash_attn_varlen_func as __flash_attn_varlen_func
from sgl_kernel.flash_attn import flash_attn_with_kvcache

from hip_attn.v1_2.utils import capture


@capture
def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    seqused_q=None,
    seqused_k=None,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    sm_margin=0,
    return_softmax_lse=False,
):
    return __flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        softmax_scale=softmax_scale,
        causal=causal,
        qv=qv,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        window_size=window_size,
        softcap=softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
        return_softmax_lse=return_softmax_lse,
    )


from hip_attn.v1_2.attention_extend import (
    dual_stage_quadratic_hip_attention,
    get_block_sparse_backend,
)
from hip_attn.v1_2.attention_metadata import (
    HiPAttentionArgs,
    HiPAttentionOutputMetadata,
    HiPAttentionState,
)
from hip_attn.v1_2.hip_config import HiPAttentionConfig
from hip_attn.v1_2.query_sparse_attention import query_sparse_attention
from hip_attn.v1_2.uvm_gpu_cache import HiPOffloadCache

try:
    import torch.distributed as dist
    from sglang.srt.distributed import (
        get_tensor_model_parallel_rank,
        split_tensor_along_last_dim,
        tensor_model_parallel_all_gather,
        tensor_model_parallel_all_reduce,
    )

    SGLANG_DIST_ACTIVATED = True
except ImportError as ex:
    SGLANG_DIST_ACTIVATED = False


def get_local_rank() -> 0:
    if SGLANG_DIST_ACTIVATED:
        return get_tensor_model_parallel_rank()
    else:
        return 0


_CHECKOUT_COUNTER = 0


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def cuda_graph_capture_configs(hip_config: HiPAttentionConfig):
    num_stages = len(hip_config.layers[0].stages)
    cache_configs = [(None,)]  # (num_stage_cached,)
    for i_stage in range(num_stages):
        cache_configs.append((i_stage,))
    return cache_configs


def forward_paged_hip(
    query: torch.Tensor,
    sm_scale: float,
    batch_size: int,
    k_cache: Optional[torch.Tensor],
    v_cache: Optional[torch.Tensor],
    offload_cache: Optional[HiPOffloadCache],
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    req_to_tokens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    block_table: torch.Tensor,
    rope_cos: Optional[torch.Tensor],
    rope_sin: Optional[torch.Tensor],
    layer_id: int,
    logit_cap: float,
    orig_context_len: int,
    max_context_len: int,
    hip_config: HiPAttentionConfig,
    is_kv_cache_offload_enabled: Optional[bool] = False,
    rope_range: Optional[tuple[int, int]] = None,
    rope_is_neox_style: Optional[bool] = None,
    extend_seq_lens: Optional[torch.Tensor] = None,
    extend_seq_lens_cpu: Optional[List[int]] = None,
    extend_prefix_lens_cpu: Optional[List[int]] = None,
    cached_metadata: Optional[HiPAttentionOutputMetadata] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    online_update_cache: bool = False,
    offloading_metadata: Any = None,
    is_prefill: Optional[bool] = None,
    is_decode: bool = False,
    query_for_mask: Optional[torch.Tensor] = None,
    diag_sliding_window_indices: Optional[torch.Tensor] = None,
    sliding_window_size: Optional[int] = -1,
    sliding_window_sink: Optional[int] = -1,
    using_chunked_sliding_window: bool = False,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, HiPAttentionOutputMetadata]:

    if is_prefill is not None:
        warnings.warn(
            "Deprecated behavior: `is_prefill` is deprecated. Use `is_decode` instead."
        )
        is_decode = not is_prefill

    if v is None:
        # warnings.warn(
        #     "Deprecated behavior: `k` and `v` should be provided in order to precisely know the output size."
        # )

        if v_cache is not None:
            v_hidden_dim = v_cache.shape[-1]
        else:
            assert offload_cache is not None
            v_hidden_dim = offload_cache.v_uvm.bank_cpu.shape[-1]

    else:
        if isinstance(v, list):
            v_hidden_dim = v[0].shape[-1]

        else:
            v_hidden_dim = v.shape[-1]

            if k.ndim == 3 and v.ndim == 3:  # Ignore if paged attn
                assert (
                    k_cache is not None and v_cache is not None
                ) or offload_cache is not None
                k = v = None

    if not is_decode:
        assert extend_seq_lens_cpu is not None

        # Handle jagged inputs
        if is_kv_cache_offload_enabled is None:
            warnings.warn(
                "Deprecated behavior: `is_kv_cache_offload_enabled` must be specified in the future."
            )
            is_kv_cache_offload_enabled = k is not None and v is not None
        if is_kv_cache_offload_enabled:
            assert isinstance(k, list) and isinstance(v, list)
            assert isinstance(offloading_metadata, list)
            offload_cache = k_cache = v_cache = None

        if query.ndim == 4:
            # NOTE FIXME this seems not correct behavior.
            if extend_seq_lens_cpu is not None:
                if len(extend_seq_lens_cpu) != query.shape[0]:
                    assert (len(extend_seq_lens_cpu) % query.shape[0]) == 0
                    n_repeat = len(extend_seq_lens_cpu) // query.shape[0]
                    # query = query.repeat_interleave(n_repeat, 0)
                    # if k is not None:
                    #     k = k.repeat_interleave(n_repeat, 0)
                    # if v is not None:
                    #     v = v.repeat_interleave(n_repeat, 0)
                    extend_seq_lens_cpu = extend_seq_lens_cpu[::n_repeat]
                    extend_prefix_lens_cpu = extend_prefix_lens_cpu[::n_repeat]
            BSZ_TDST = query.shape[0] * query.shape[1]
            HEAD = query.shape[2]
        elif query.ndim == 3:
            BSZ_TDST, HEAD, _ = query.shape
        else:
            raise Exception()

        # Output tensor
        o = torch.empty(
            (BSZ_TDST, HEAD, v_hidden_dim),
            dtype=query.dtype,
            device=query.device,
        )
        metadata_new = []

        if cached_metadata is not None:
            states = cached_metadata.state
            if isinstance(states, list) and (len(states) < len(extend_seq_lens_cpu)):
                assert (len(extend_seq_lens_cpu) % len(states)) == 0
                n_repeat = len(extend_seq_lens_cpu) // len(states)
                new_states = []
                for state in states:
                    for _ in range(n_repeat):
                        new_states.append(copy.deepcopy(state))
                states = new_states

        start_len = 0
        decoding_reqs = []
        decoding_reqs_positions = []

        # NOTE this is required for prefix
        assert extend_prefix_lens_cpu is not None
        assert len(extend_seq_lens_cpu) == len(extend_prefix_lens_cpu)

        for idx_batch, (seq_len, prefix_len) in enumerate(
            zip(extend_seq_lens_cpu, extend_prefix_lens_cpu)
        ):
            if query.ndim == 4:
                seq_len = query.shape[1]

            if seq_len == 0:  # Skip empty sequences
                decoding_reqs.append(idx_batch)
                decoding_reqs_positions.append(start_len)

            else:
                if not is_kv_cache_offload_enabled:
                    k_chunk = v_chunk = None
                    offloading_metadata_curr = None

                else:  # Offloading enabled
                    k_chunk, v_chunk, offloading_metadata_curr = (
                        k[idx_batch],
                        v[idx_batch],
                        offloading_metadata[idx_batch],
                    )
                if k_chunk is None:
                    k_chunk = (
                        (
                            k[idx_batch : idx_batch + 1]
                            if k.ndim == 4
                            else k[start_len : start_len + seq_len]
                        )
                        if k is not None
                        else None
                    )
                if v_chunk is None:
                    v_chunk = (
                        (
                            v[idx_batch : idx_batch + 1]
                            if v.ndim == 4
                            else v[start_len : start_len + seq_len]
                        )
                        if v is not None
                        else None
                    )

                if cached_metadata is not None:
                    if isinstance(states, list):
                        cached_metadata.state = states[idx_batch]

                o_req, metadata_req = _forward_paged_hip_validate(
                    query=(
                        query[start_len : start_len + seq_len]
                        if query.ndim == 3
                        else query[idx_batch : idx_batch + 1]
                    ),
                    sm_scale=sm_scale,
                    batch_size=1,
                    k=k_chunk,
                    v=v_chunk,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    offload_cache=offload_cache,
                    positions=positions[start_len : start_len + seq_len],
                    seq_lens=seq_lens[idx_batch : idx_batch + 1],
                    req_to_tokens=req_to_tokens,
                    req_pool_indices=req_pool_indices[idx_batch : idx_batch + 1],
                    block_table=None,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    rope_range=rope_range,
                    rope_is_neox_style=rope_is_neox_style,
                    layer_id=layer_id,
                    logit_cap=logit_cap,
                    orig_context_len=orig_context_len,
                    max_context_len=max_context_len,
                    max_batch_context_len=seq_len + prefix_len,
                    v_hidden_dim=v_hidden_dim,
                    hip_config=hip_config,
                    is_kv_cache_offload_enabled=is_kv_cache_offload_enabled,
                    cached_metadata=cached_metadata,
                    online_update_cache=online_update_cache,
                    offloading_metadata=offloading_metadata_curr,
                    is_decode=is_decode,
                    query_for_mask=query_for_mask,
                    diag_sliding_window_indices=diag_sliding_window_indices,
                    sliding_window_size=sliding_window_size,
                    sliding_window_sink=sliding_window_sink,
                    using_chunked_sliding_window=using_chunked_sliding_window,
                    k_descale=k_descale,
                    v_descale=v_descale,
                )
                metadata_new.append(metadata_req)

                o[start_len : start_len + seq_len] = o_req

            start_len += seq_len

        assert len(decoding_reqs) == 0

    else:
        o, metadata_new = _forward_paged_hip_validate(
            query=query,
            sm_scale=sm_scale,
            batch_size=batch_size,
            k_cache=k_cache,
            v_cache=v_cache,
            offload_cache=offload_cache,
            positions=positions,
            seq_lens=seq_lens,
            req_to_tokens=req_to_tokens,
            req_pool_indices=req_pool_indices,
            block_table=block_table,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            rope_range=rope_range,
            rope_is_neox_style=rope_is_neox_style,
            layer_id=layer_id,
            logit_cap=logit_cap,
            orig_context_len=orig_context_len,
            max_context_len=max_context_len,
            max_batch_context_len=max_context_len,
            v_hidden_dim=v_hidden_dim,
            hip_config=hip_config,
            is_kv_cache_offload_enabled=is_kv_cache_offload_enabled,
            cached_metadata=cached_metadata,
            k=k,
            v=v,
            online_update_cache=online_update_cache,
            offloading_metadata=offloading_metadata,
            is_decode=is_decode,
            query_for_mask=query_for_mask,
            diag_sliding_window_indices=diag_sliding_window_indices,
            sliding_window_size=sliding_window_size,
            sliding_window_sink=sliding_window_sink,
            using_chunked_sliding_window=using_chunked_sliding_window,
            k_descale=k_descale,
            v_descale=v_descale,
        )

    return o, metadata_new


def _forward_paged_hip_validate(
    query: torch.Tensor,
    sm_scale: float,
    batch_size: int,
    k_cache: Optional[torch.Tensor],
    v_cache: Optional[torch.Tensor],
    offload_cache: Optional[HiPOffloadCache],
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    req_to_tokens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    block_table: torch.Tensor,
    rope_cos: Optional[torch.Tensor],
    rope_sin: Optional[torch.Tensor],
    layer_id: int,
    logit_cap: float,
    orig_context_len: int,
    max_context_len: int,
    max_batch_context_len: int,
    v_hidden_dim: int,
    hip_config: HiPAttentionConfig,
    is_kv_cache_offload_enabled: Optional[bool] = False,
    rope_range: Optional[tuple[int, int]] = None,
    rope_is_neox_style: Optional[bool] = None,
    cached_metadata: Optional[HiPAttentionOutputMetadata] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    online_update_cache: bool = False,
    offloading_metadata: Any = None,
    is_decode: bool = False,
    query_for_mask: Optional[torch.Tensor] = None,
    diag_sliding_window_indices: Optional[torch.Tensor] = None,
    sliding_window_size: Optional[int] = -1,
    sliding_window_sink: Optional[int] = -1,
    using_chunked_sliding_window: bool = False,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, HiPAttentionOutputMetadata]:

    if is_kv_cache_offload_enabled:
        if k is not None and v is not None:
            # BUG: this padding is neccesary to match non offload scenario. why?
            pad_size = max_context_len
            if k.shape[1] != pad_size:
                k_chunk_padded = torch.zeros(
                    (
                        k.shape[0],
                        pad_size,
                        k.shape[2],
                        k.shape[3],
                    ),
                    dtype=k.dtype,
                    device=k.device,
                )
                k_chunk_padded[:, : k.shape[1]] = k
                del k
                v_chunk_padded = torch.zeros(
                    (
                        v.shape[0],
                        pad_size,
                        v.shape[2],
                        v.shape[3],
                    ),
                    dtype=v.dtype,
                    device=v.device,
                )
                v_chunk_padded[:, : v.shape[1]] = v
                del v
                k = k_chunk_padded
                v = v_chunk_padded

    require_validation = offloading_metadata is not None
    if require_validation:
        if not is_decode:
            k_pages, v_pages = offloading_metadata
        else:
            k_cache_valid, v_cache_valid = offloading_metadata

            err_k = sse(offload_cache.k_uvm.bank_gpu, k_cache_valid)
            err_v = sse(offload_cache.v_uvm.bank_gpu, v_cache_valid)

    o, metadata_new = _forward_paged_hip(
        query=query,
        sm_scale=sm_scale,
        batch_size=batch_size,
        k_cache=k_cache,
        v_cache=v_cache,
        offload_cache=offload_cache,
        positions=positions,
        seq_lens=seq_lens,
        req_to_tokens=req_to_tokens,
        req_pool_indices=req_pool_indices,
        block_table=block_table,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        rope_range=rope_range,
        rope_is_neox_style=rope_is_neox_style,
        layer_id=layer_id,
        logit_cap=logit_cap,
        orig_context_len=orig_context_len,
        max_context_len=max_context_len,
        max_batch_context_len=max_batch_context_len,
        v_hidden_dim=v_hidden_dim,
        hip_config=hip_config,
        cached_metadata=cached_metadata,
        k=k,
        v=v,
        online_update_cache=online_update_cache,
        is_decode=is_decode,
        query_for_mask=query_for_mask,
        diag_sliding_window_indices=diag_sliding_window_indices,
        sliding_window_size=sliding_window_size,
        sliding_window_sink=sliding_window_sink,
        using_chunked_sliding_window=using_chunked_sliding_window,
        k_descale=k_descale,
        v_descale=v_descale,
    )

    if require_validation:
        if not is_decode:
            o_req_valid, _ = _forward_paged_hip(
                query=query,
                sm_scale=sm_scale,
                batch_size=batch_size,
                k_cache=k_pages,
                v_cache=v_pages,
                offload_cache=offload_cache,
                positions=positions,
                seq_lens=seq_lens,
                req_to_tokens=req_to_tokens,
                req_pool_indices=req_pool_indices,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                rope_range=rope_range,
                rope_is_neox_style=rope_is_neox_style,
                layer_id=layer_id,
                logit_cap=logit_cap,
                orig_context_len=orig_context_len,
                max_context_len=max_context_len,
                max_batch_context_len=max_batch_context_len,
                v_hidden_dim=v_hidden_dim,
                hip_config=hip_config,
                cached_metadata=cached_metadata,
                k=k,
                v=v,
                online_update_cache=online_update_cache,
                is_decode=is_decode,
                query_for_mask=query_for_mask,
                diag_sliding_window_indices=diag_sliding_window_indices,
                sliding_window_size=sliding_window_size,
                sliding_window_sink=sliding_window_sink,
                using_chunked_sliding_window=using_chunked_sliding_window,
                k_descale=k_descale,
                v_descale=v_descale,
            )

            o_err = ((o - o_req_valid) ** 2).sum()
            assert o_err < 1e-6, o_err

        else:
            o_valid, metadata_valid = _forward_paged_hip(
                query=query,
                sm_scale=sm_scale,
                batch_size=batch_size,
                k_cache=k_cache_valid,
                v_cache=v_cache_valid,
                offload_cache=None,
                positions=positions,
                seq_lens=seq_lens,
                req_to_tokens=req_to_tokens,
                req_pool_indices=req_pool_indices,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                rope_range=rope_range,
                rope_is_neox_style=rope_is_neox_style,
                layer_id=layer_id,
                logit_cap=logit_cap,
                orig_context_len=orig_context_len,
                max_context_len=max_context_len,
                max_batch_context_len=max_batch_context_len,
                v_hidden_dim=v_hidden_dim,
                hip_config=hip_config,
                cached_metadata=cached_metadata,
                k=k,
                v=v,
                online_update_cache=online_update_cache,
                is_decode=is_decode,
                query_for_mask=query_for_mask,
                diag_sliding_window_indices=diag_sliding_window_indices,
                sliding_window_size=sliding_window_size,
                sliding_window_sink=sliding_window_sink,
                using_chunked_sliding_window=using_chunked_sliding_window,
                k_descale=k_descale,
                v_descale=v_descale,
            )

            err_thresh = 1e-7

            o_sse = sse(o, o_valid)
            err_retry = -1
            err_uvm = None
            if o_sse >= err_thresh:
                indices_err = sse(metadata_new.indices, metadata_valid.indices)
                ks_err = sse(metadata_new.ks, metadata_valid.ks)
                ks_count_err = sse(metadata_new.ks_count, metadata_valid.ks_count)
                ks_start_end_err = sse(
                    metadata_new.ks_start_end, metadata_valid.ks_start_end
                )
                if (metadata_valid.stage_caches is not None) and (
                    len(metadata_valid.stage_caches) > 0
                ):
                    stage1_left_err = sse(
                        metadata_new.stage_caches[1].indices_left,
                        metadata_valid.stage_caches[1].indices_left,
                    )
                    stage1_right_err = sse(
                        metadata_new.stage_caches[1].indices_right,
                        metadata_valid.stage_caches[1].indices_right,
                    )
                    stage1_score_err = sse(
                        metadata_new.stage_caches[1].out_scores,
                        metadata_valid.stage_caches[1].out_scores,
                    )
                    stage2_left_err = sse(
                        metadata_new.stage_caches[2].indices_left,
                        metadata_valid.stage_caches[2].indices_left,
                    )
                    stage2_right_err = sse(
                        metadata_new.stage_caches[2].indices_right,
                        metadata_valid.stage_caches[2].indices_right,
                    )
                    stage2_score_err = sse(
                        metadata_new.stage_caches[2].out_scores,
                        metadata_valid.stage_caches[2].out_scores,
                    )
                else:
                    stage1_left_err = stage1_right_err = stage1_score_err = (
                        stage2_left_err
                    ) = stage2_right_err = stage2_score_err = None

                o_uvm, metadata_uvm = _forward_paged_hip(
                    query=query,
                    sm_scale=sm_scale,
                    batch_size=batch_size,
                    k_cache=offload_cache.k_uvm.bank_gpu,
                    v_cache=offload_cache.v_uvm.bank_gpu,
                    offload_cache=None,
                    positions=positions,
                    seq_lens=seq_lens,
                    req_to_tokens=req_to_tokens,
                    req_pool_indices=req_pool_indices,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    rope_range=rope_range,
                    rope_is_neox_style=rope_is_neox_style,
                    layer_id=layer_id,
                    logit_cap=logit_cap,
                    orig_context_len=orig_context_len,
                    max_context_len=max_context_len,
                    max_batch_context_len=max_batch_context_len,
                    v_hidden_dim=v_hidden_dim,
                    hip_config=hip_config,
                    cached_metadata=cached_metadata,
                    k=k,
                    v=v,
                    online_update_cache=online_update_cache,
                    is_decode=is_decode,
                    query_for_mask=query_for_mask,
                    diag_sliding_window_indices=diag_sliding_window_indices,
                    sliding_window_size=sliding_window_size,
                    sliding_window_sink=sliding_window_sink,
                    using_chunked_sliding_window=using_chunked_sliding_window,
                    k_descale=k_descale,
                    v_descale=v_descale,
                )

                offload_cache.sa_kv_cache.flush()
                offload_cache.mask_k_cache.flush()

                o_retry, metadata_retry = _forward_paged_hip(
                    query=query,
                    sm_scale=sm_scale,
                    batch_size=batch_size,
                    k_cache=None,
                    v_cache=None,
                    offload_cache=offload_cache,
                    positions=positions,
                    seq_lens=seq_lens,
                    req_to_tokens=req_to_tokens,
                    req_pool_indices=req_pool_indices,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    rope_range=rope_range,
                    rope_is_neox_style=rope_is_neox_style,
                    layer_id=layer_id,
                    logit_cap=logit_cap,
                    orig_context_len=orig_context_len,
                    max_context_len=max_context_len,
                    max_batch_context_len=max_batch_context_len,
                    v_hidden_dim=v_hidden_dim,
                    hip_config=hip_config,
                    cached_metadata=cached_metadata,
                    k=k,
                    v=v,
                    online_update_cache=online_update_cache,
                    is_decode=is_decode,
                    query_for_mask=query_for_mask,
                    diag_sliding_window_indices=diag_sliding_window_indices,
                    sliding_window_size=sliding_window_size,
                    sliding_window_sink=sliding_window_sink,
                    using_chunked_sliding_window=using_chunked_sliding_window,
                    k_descale=k_descale,
                    v_descale=v_descale,
                )
                err_uvm = sse(o, o_uvm)
                err_retry = sse(o_valid, o_retry)

                print(o)
                print(o_valid)
                print(metadata_new.indices)
                print(metadata_valid.indices)

                assert o_sse < err_thresh, (
                    f"sse={o_sse}\n"
                    f"err_k (uvm_k <=> valid_k) = {err_k}\n"
                    f"err_v (uvm_v <=> valid_v) = {err_v}\n"
                    f"err_retry (o_valid <=> o_retry) = {err_retry}\n"
                    f"err_uvm (o_first <=> o_uvm_retry) = {err_uvm}\n"
                    f"indices_err={indices_err}\n"
                    f"ks_err={ks_err}\n"
                    f"ks_count_err={ks_count_err}\n"
                    f"ks_start_end_err={ks_start_end_err}\n"
                    f"stage1_left_err={stage1_left_err}\n"
                    f"stage1_right_err={stage1_right_err}\n"
                    f"stage1_score_err={stage1_score_err}\n"
                    f"stage2_left_err={stage2_left_err}\n"
                    f"stage2_right_err={stage2_right_err}\n"
                    f"stage2_score_err={stage2_score_err}\n"
                    f"online_update={online_update_cache}\n"
                )

    return o, metadata_new


def sse(a: torch.Tensor, b: torch.Tensor):
    assert a.dtype == b.dtype
    return ((a - b) ** 2).sum().item()


@capture
def _forward_delta_attn(
    query: torch.Tensor,
    sm_scale: float,
    k: torch.Tensor,
    v: torch.Tensor,
    args: HiPAttentionArgs,
    cached_metadata: HiPAttentionOutputMetadata,
    is_decode: bool,
    delta_attention_args_smooth = False,
    delta_attention_args_just_return = False,
    delta_attention_args_window = 0,
    delta_attention_args_diff = 1,
    delta_attention_args_dense_decode = False,
    delta_attention_args_w = 16,
    delta_attention_args_exp = False,
    delta_attention_args_exp_w = 2,
    delta_attention_args_exp_window = 1024,
    delta_attention_args_exp_sink = 128,
    delta_attention_args_iter_corr = False,
    delta_attention_args_adjust_norm_const = False,
    k_descale: torch.Tensor = None,
    v_descale: torch.Tensor = None,
):
    using_dense_prefill = False

    # if (
    #     (is_decode and delta_attention_args_dense_decode)
    #     or (using_dense_prefill and (not is_decode))
    #     or ((query.shape[1] < 256) and (not is_decode))
    # ):
    #     # for dense decode, BUG this is so slow why?
    #     if args.need_apply_rope and args.using_extend:
    #         k_unpack = args.gather_k_from_paged_cache()
    #         v_unpack = args.gather_v_from_paged_cache()

    #         seq_len = args.position_ids.amax().item() + 1

    #         k_unpack = k_unpack[:, :seq_len]
    #         v_unpack = v_unpack[:, :seq_len]

    #         cos = args.rope_cos
    #         sin = args.rope_sin
    #         assert cos.ndim == 2, cos.shape
    #         assert sin.shape == cos.shape, sin.shape

    #         cos = cos.view(1, cos.shape[-2], 1, cos.shape[-1])
    #         sin = sin.view(1, sin.shape[-2], 1, sin.shape[-1])

    #         idx_tsrc = torch.arange(0, k_unpack.shape[1], device=cos.device)
    #         idx_tsrc.clamp_min_(seq_len - args.model_context_length)

    #         k_unpack = (
    #             (k_unpack * cos[:, idx_tsrc, :, :])
    #             + (rotate_half(k_unpack) * sin[:, idx_tsrc, :, :])
    #         ).to(k_unpack.dtype)

    #         query = (
    #             (query * cos[:, args.position_ids.view(-1), :, :])
    #             + (rotate_half(query) * sin[:, args.position_ids.view(-1), :, :])
    #         ).to(query.dtype)

    #         k_unpack = k_unpack[:, :seq_len]
    #         v_unpack = v_unpack[:, :seq_len]

    #         context = flash_attn_func(
    #             query,
    #             k_unpack,
    #             v_unpack,
    #             causal=True,
    #             softmax_scale=sm_scale,
    #         )
    #     else:
    #         assert args.using_paged_cache

    #         k_cache = args.get_k_cache()
    #         v_cache = args.get_v_cache()

    #         q_reshaped = query\
    #             .contiguous()\
    #             .view(-1, query.shape[2], query.shape[3])\
    #             .to(k_cache.dtype)

    #         # print(k_cache.shape, v_cache.shape)

    #         cu_seqlens_q = (
    #             torch.arange(
    #                 0, query.shape[0] + 1, device=query.device, dtype=torch.int32
    #             )
    #             * query.shape[1]
    #         )
    #         cache_seqlens = (args.position_ids[:, -1] + 1).to(torch.int32)
    #         cu_seqlens_k_new = torch.zeros(
    #             (args.position_ids.shape[0] + 1,),
    #             dtype=torch.int32,
    #             device=q_reshaped.device,
    #         )
    #         cu_seqlens_k_new[1:] = cache_seqlens

    #         block_table = args.block_table

    #         context = flash_attn_with_kvcache(
    #             q=q_reshaped,
    #             k_cache=k_cache,
    #             v_cache=v_cache,
    #             page_table=block_table,
    #             cache_seqlens=cache_seqlens,
    #             cu_seqlens_q=cu_seqlens_q,
    #             cu_seqlens_k_new=cu_seqlens_k_new,
    #             # max_seqlen_q=cu_seqlens_q.amax().item(),
    #             max_seqlen_q=args.model_context_length,
    #             causal=True,
    #             softmax_scale=sm_scale,
    #         )

    #     metadata = None
    # else:

    # On prefill
    assert not is_decode
    assert not torch.cuda.is_current_stream_capturing()

    assert isinstance(delta_attention_args_window, int)
    if delta_attention_args_window == 0:
        assert delta_attention_args_window == 0

        # args_new = args.clone()
        # k_flat = args.gather_k_from_paged_cache()
        # v_flat = args.gather_v_from_paged_cache()
        # seq_len = args.position_ids.amax().item() + 1
        # k_flat = k_flat[:, :seq_len].contiguous()
        # v_flat = v_flat[:, :seq_len].contiguous()
        # args_new.k_cache = None
        # args_new.v_cache = None
        # args_new.block_table = None
        # args_new.using_paged_cache = False
        # cached_metadata.state = None

        # context_sparse, metadata = dual_stage_quadratic_hip_attention(
        #     q=(query * sm_scale).to(query.dtype),
        #     k=k_flat,
        #     v=v_flat,
        #     args=args_new,
        #     cached_metadata=cached_metadata,
        # )

        delta_exp = delta_attention_args_exp

        if delta_exp:
            delta_exp_w = delta_attention_args_exp_w
            delta_exp_bk = 16
            delta_exp_k = 0
            delta_exp_window = delta_attention_args_exp_window
            delta_exp_sink = delta_attention_args_exp_sink
            delta_merge_strategy = "delta"  # replace / delta
            if delta_exp_k == 0:
                delta_exp_bk = 64

            bsa_fn = get_block_sparse_backend(args, query)

            BSZ, TDST, HEAD, HID = query.shape

            args_sw = args.clone()
            if args_sw.rope_range is None:
                args_sw.rope_range = (0, HID)
            args_sw.block_size_q = args_sw.block_sparse_block_size_q
            args_sw.block_size_k = delta_exp_bk
            args_sw.second_stage_k = delta_exp_k
            args_sw.sink_token_size = delta_exp_sink
            args_sw.sliding_window_size = delta_exp_window
            args_sw.sliding_window_indices = None

            BDST = triton.cdiv(TDST, args_sw.block_size_q)
            BH = BSZ * HEAD

            if delta_exp_k == 0:
                indices = torch.zeros(
                    (BH, BDST, delta_exp_k // delta_exp_bk),
                    dtype=torch.int64,
                    device=query.device,
                )
                ks = torch.zeros(
                    (BH, BDST), dtype=torch.int64, device=query.device
                )
                ks_count = ks.unsqueeze(-1)
                ks_start_end = torch.zeros(
                    (BH, BDST, 2), dtype=torch.int64, device=query.device
                )
                ks_start_end[:, :, 1:] = ks[:, :, None]
            else:
                indices = torch.rand(
                    (BH, BDST, delta_exp_k // delta_exp_bk), device=query.device
                )
                indices = (
                    indices
                    * args_sw.position_ids[
                        :, :: args_sw.block_size_q
                    ].repeat_interleave(HEAD, dim=0)[:, :, None]
                )
                indices = indices.to(torch.int64) // delta_exp_bk * delta_exp_bk

                indices, _ = indices.sort(dim=-1)
                indices = indices // args_sw.block_size_k * args_sw.block_size_k

                unique_mask = torch.roll(indices, shifts=1, dims=-1) != indices
                indices = torch.where(
                    unique_mask, indices, torch.iinfo(indices.dtype).max
                )
                indices, _ = indices.sort(dim=-1)
                active_mask = indices < (
                    args_sw.position_ids[
                        :, :: args_sw.block_size_q, None
                    ].repeat_interleave(HEAD, 0)
                    + args_sw.block_size_q
                )
                ks = active_mask.int().sum(-1)
                ks_count = ks.unsqueeze(-1)
                ks_start_end = torch.zeros(
                    (ks.shape[0], ks.shape[1], 2),
                    dtype=torch.int32,
                    device=query.device,
                )
                ks_start_end[:, :, -1] = ks

            context_sw = bsa_fn(
                q=(query * sm_scale).to(query.dtype),
                k=k,
                v=v,
                seq_lens=args_sw.position_ids + 1,
                indices=indices,
                ks=ks,
                ks_count=ks_count,
                ks_start_end=ks_start_end,
                access_counter=None,
                cache_miss_counter=None,
                EXTEND_BACKEND=args_sw.sa_extend_backend,
                model_context_length=args_sw.model_context_length,
                extend_context_length=args_sw.extend_context_length,
                offload_update_cache=False,
                args=args_sw,
            )
            context_sw = context_sw.to(query.dtype)

            args_sparse = args.clone()
            query_sparse = query[:, ::delta_exp_w].contiguous()
            args_sparse.position_ids = args.position_ids[
                :, ::delta_exp_w
            ].contiguous()
            args_sparse.query_for_landmark = query
            args_sparse.position_ids_for_landmark = args.position_ids

            # args_new = args_sparse.clone()
            # k_flat = args_sparse.gather_k_from_paged_cache()
            # v_flat = args_sparse.gather_v_from_paged_cache()
            # seq_len = args_sparse.position_ids.amax().item() + 1
            # k_flat = k_flat[:, :seq_len].contiguous()
            # v_flat = v_flat[:, :seq_len].contiguous()
            # args_new.k_cache = None
            # args_new.v_cache = None
            # args_new.block_table = None
            # args_new.using_paged_cache = False
            # cached_metadata.state = None

            # context_sparse, metadata = dual_stage_quadratic_hip_attention(
            #     q=(query_sparse * sm_scale).to(query.dtype),
            #     k=k_flat,
            #     v=v_flat,
            #     args=args_new,
            #     cached_metadata=cached_metadata,
            # )

            context_sparse, metadata = dual_stage_quadratic_hip_attention(
                q=(query_sparse * sm_scale).to(query.dtype),
                k=k,
                v=v,
                args=args_sparse,
                cached_metadata=cached_metadata,
            )
            context_sparse = context_sparse.to(query.dtype)

            if delta_merge_strategy == "delta":
                context_sw_for_sparse = context_sw[:, ::delta_exp_w]
                delta_sparse = context_sparse - context_sw_for_sparse

                delta_sparse = delta_sparse.repeat_interleave(
                    delta_exp_w, dim=1
                )

                if delta_attention_args_smooth:
                    # (exp) linear interpolate diff
                    delta_sparse_shift = torch.roll(
                        delta_sparse, -delta_exp_w, 1
                    )
                    delta_sparse_shift[:, -delta_exp_w:] = delta_sparse[:, -1:]

                    idx = torch.arange(
                        0, delta_sparse.shape[1], device=delta_sparse.device
                    )
                    idx = (idx % delta_exp_w).float() / delta_exp_w
                    delta_sparse = (
                        delta_sparse
                        + (delta_sparse_shift - delta_sparse)
                        * idx[None, :, None, None]
                    )

                context_sparse = (
                    context_sw + delta_sparse[:, : context_sw.shape[1]]
                )
            elif delta_merge_strategy == "replace":
                context_sw[:, ::delta_exp_w] = context_sparse
                context_sparse = context_sw
            else:
                raise Exception()
        else:
            # args_new = args.clone()
            # k_flat = args.gather_k_from_paged_cache()
            # v_flat = args.gather_v_from_paged_cache()
            # seq_len = args.position_ids.amax().item() + 1
            # k_flat = k_flat[:, :seq_len].contiguous()
            # v_flat = v_flat[:, :seq_len].contiguous()
            # args_new.k_cache = None
            # args_new.v_cache = None
            # args_new.block_table = None
            # args_new.using_paged_cache = False
            # cached_metadata.state = None

            # context_sparse, metadata = dual_stage_quadratic_hip_attention(
            #     q=(query * sm_scale).to(query.dtype),
            #     k=k_flat,
            #     v=v_flat,
            #     args=args_new,
            #     cached_metadata=cached_metadata,
            # )

            args.bsa_return_running_statistics = (
                delta_attention_args_adjust_norm_const
            )

            context_sparse, metadata = dual_stage_quadratic_hip_attention(
                q=(query * sm_scale).to(query.dtype),
                k=k,
                v=v,
                args=args,
                cached_metadata=cached_metadata,
            )

            if delta_attention_args_adjust_norm_const:
                context_sparse, (sparse_mx, sparse_nc) = context_sparse

            context_sparse = context_sparse.to(query.dtype)
            context_sparse = context_sparse[
                :, -query.shape[1] :, :, :
            ].contiguous()

            if delta_attention_args_adjust_norm_const:
                sparse_mx = sparse_mx[:, -query.shape[1] :].contiguous()
                sparse_nc = sparse_nc[:, -query.shape[1] :].contiguous()
    else:
        assert delta_attention_args_window > 0
        bsa_fn = get_block_sparse_backend(args, query)

        # dist.barrier()
        # if get_tensor_model_parallel_rank() == 0:
        #     print(bsa_fn, args.using_extend, sliding_window_size, args.using_chunked_sliding_window)

        BSZ, TDST, HEAD, HID = query.shape

        args_sw = args.clone()
        if args_sw.rope_range is None:
            args_sw.rope_range = (0, HID)
        args_sw.block_size_q = args_sw.block_sparse_block_size_q
        args_sw.block_size_k = args_sw.stages[-1].stage_chunk_size
        args_sw.second_stage_k = 0
        # args_sw.sink_token_size = 0 #NOTE: you should inherit this value
        args_sw.sliding_window_size = delta_attention_args_window
        args_sw.sliding_window_indices = None

        BDST = triton.cdiv(TDST, args_sw.block_size_q)
        BH = BSZ * HEAD

        indices = torch.zeros(
            (BH, BDST, 0), dtype=torch.int64, device=query.device
        )
        ks = torch.zeros((BH, BDST), dtype=torch.int64, device=query.device)
        ks_count = ks.unsqueeze(-1)
        ks_start_end = torch.zeros(
            (BH, BDST, 2), dtype=torch.int64, device=query.device
        )

        context_sparse = bsa_fn(
            q=(query * sm_scale).to(query.dtype),
            k=k,
            v=v,
            seq_lens=args_sw.position_ids + 1,
            indices=indices,
            ks=ks,
            ks_count=ks_count,
            ks_start_end=ks_start_end,
            access_counter=None,
            cache_miss_counter=None,
            EXTEND_BACKEND=args_sw.sa_extend_backend,
            model_context_length=args_sw.model_context_length,
            extend_context_length=args_sw.extend_context_length,
            offload_update_cache=False,
            return_running_statistics=delta_attention_args_adjust_norm_const,
            args=args_sw,
        )
        if delta_attention_args_adjust_norm_const:
            context_sparse, (sparse_mx, sparse_nc) = context_sparse

        context_sparse = context_sparse.to(query.dtype)
        context_sparse = context_sparse[:, -query.shape[1] :, :, :].contiguous()
        if delta_attention_args_adjust_norm_const:
            sparse_mx = sparse_mx[:, -query.shape[1] :].contiguous()
            sparse_nc = sparse_nc[:, -query.shape[1] :].contiguous()
        metadata = None

    # until here, we have only calculated sparse attention
    if delta_attention_args_just_return:
        context = context_sparse
    elif delta_attention_args_iter_corr:
        w_size = delta_attention_args_w * 2

        num_queries = query.shape[1]
        num_dense_first = max(128, w_size)
        num_dense_last = num_queries % w_size + max(128, w_size)
        num_sparse = num_queries - num_dense_first - num_dense_last

        # iteratively correction errors

        def perform_correction(
            context_sparse: torch.Tensor,
            context_sparse_raw: torch.Tensor,
            block_start_indices: torch.Tensor,
            block_size: int,
        ):
            assert block_start_indices.ndim == 1
            assert context_sparse.ndim == 4
            assert context_sparse_raw.shape == context_sparse.shape
            assert not (args.need_apply_rope and args.using_extend)

            assert args.using_paged_cache

            if False:
                context_sparse_raw = context_sparse

            query_for_recomp = query[:, block_start_indices, :, :]
            k_cache = args.get_k_cache()
            v_cache = args.get_v_cache()

            assert args.position_ids.shape[0] == 1
            if get_local_rank() == 0:
                # import matplotlib.pyplot as plt
                # plt.clf()
                # plt.hist(block_start_indices.cpu().numpy(), bins=50)
                # plt.xlim(0, args.position_ids.amax().item() + 1)
                # plt.savefig(f'./dummy_indices_hist_{len(block_start_indices)}.png')

                print(
                    "recomp_attn shapes",
                    query_for_recomp.shape,
                    block_start_indices.shape,
                )
            context_dense = (
                query_sparse_attention(
                    query_for_recomp.permute(0, 2, 1, 3).contiguous(),
                    None,
                    None,
                    args.position_ids[:, block_start_indices],
                    sm_scale,
                    k_cache,
                    v_cache,
                    args.block_table,
                    k_descale=k_descale,
                    v_descale=v_descale,
                )
                .permute(0, 2, 1, 3)
                .contiguous()
            )  # type: torch.Tensor
            assert context_dense.shape[-2:] == query.shape[-2:]

            if block_size > 1:
                assert not delta_attention_args_smooth
                block_diff = diff = (
                    context_dense - context_sparse_raw[:, block_start_indices]
                )
                diff = diff.repeat_interleave(block_size, 1)

                token_indices = (
                    block_start_indices[:, None]
                    + torch.arange(0, block_size, device=context_sparse.device)[
                        None, :
                    ]
                )
                token_indices = token_indices.view(-1)

                context_sparse_new = diff + context_sparse_raw[:, token_indices]
                context_sparse.index_copy_(
                    dim=1, index=token_indices, source=context_sparse_new
                )
            else:
                context_sparse.index_copy_(
                    dim=1, index=block_start_indices, source=context_dense
                )
                block_diff = None

            return context_sparse, block_diff

        block_start_indices = torch.arange(
            num_dense_first,
            num_dense_first + num_sparse,
            step=w_size,
            device=query.device,
        )
        assert (num_dense_first % w_size) == 0
        assert ((num_dense_first + num_sparse) % w_size) == 0

        split = 2

        def block_diff_to_score(block_diff: torch.Tensor):
            return (
                block_diff.squeeze(0)
                .norm(dim=-1, keepdim=False)
                .sum(dim=-1, keepdim=False)
            )
            # return block_diff\
            #     .squeeze(0)\
            #     .abs().sum(dim=-1, keepdim=False)\
            #     .sum(dim=-1, keepdim=False)

        context_sparse_raw = context_sparse.clone()

        context_sparse, block_diff = perform_correction(
            context_sparse,
            context_sparse_raw,
            block_start_indices,
            w_size,
        )
        # [T,]
        block_diff_scores_parent, block_diff_indices = block_diff_to_score(
            block_diff
        ).topk(k=block_diff.shape[1] // split, dim=0, sorted=False)
        block_start_indices_parent = block_start_indices[block_diff_indices]
        block_start_indices_parent, tind = block_start_indices_parent.sort()
        block_diff_scores_parent = block_diff_scores_parent[tind]

        depth = 0
        max_iter = 4
        while (w_size // split) > 0 and (depth < max_iter):
            depth += 1
            block_start_indices_child = (
                block_start_indices_parent + w_size // split
            )
            w_size = w_size // split

            # if get_local_rank() == 0:
            #     print(block_diff_scores_parent)

            context_sparse, block_diff = perform_correction(
                context_sparse,
                context_sparse_raw,
                block_start_indices_child,
                w_size,
            )
            if (w_size // split) > 0:
                block_diff_scores_child = block_diff_to_score(block_diff)
                block_diff_scores_parent, next_blocks_location = torch.cat(
                    [block_diff_scores_parent, block_diff_scores_child]
                ).topk(k=block_diff_scores_parent.shape[0] // 2, sorted=False)
                block_start_indices_parent = torch.cat(
                    [block_start_indices_parent, block_start_indices_child]
                )[next_blocks_location]
                block_start_indices_parent, tind = (
                    block_start_indices_parent.sort()
                )
                block_diff_scores_parent = block_diff_scores_parent[tind]

        # fill dense for first and last part
        dense_indices = torch.cat(
            [
                torch.arange(0, num_dense_first, device=query.device),
                torch.arange(
                    num_dense_first + num_sparse,
                    num_queries,
                    device=query.device,
                ),
            ]
        )
        context_sparse, _ = perform_correction(
            context_sparse,
            context_sparse_raw,
            dense_indices,
            1,
        )

        context = context_sparse
    else:
        num_queries = query.shape[1]
        num_last_dense = num_queries % delta_attention_args_w + max(
            128, delta_attention_args_w
        )
        num_last_dense = min(num_queries, num_last_dense)
        num_sparse = num_queries - num_last_dense

        context_sparse = context_sparse[:, :num_sparse]
        if delta_attention_args_adjust_norm_const:
            sparse_mx = sparse_mx[:, :num_sparse]
            sparse_nc = sparse_nc[:, :num_sparse]

        if num_last_dense > 0:
            idx = torch.arange(
                0,
                # delta_attention_args_w - 1,
                num_sparse,
                step=delta_attention_args_w,
                device=query.device,
            )
            rolling_idx = False
            if rolling_idx:
                idx = (
                    idx + (args.layer_id % delta_attention_args_w)
                ).clamp_max(num_sparse - 1)
            # take mean
            # context_sparse_for_diff = context_sparse[:, :num_sparse]
            # context_sparse_for_diff = context_sparse_for_diff.view(
            #     context_sparse_for_diff.shape[0],
            #     num_sparse // delta_attention_args_w,
            #     delta_attention_args_w,
            #     context_sparse_for_diff.shape[2],
            #     context_sparse_for_diff.shape[3],
            # )
            # context_sparse_for_diff = context_sparse_for_diff.mean(dim=2)

            # take first
            if delta_attention_args_adjust_norm_const:
                context_sparse_for_diff = context_sparse[:, idx]
                sparse_mx_for_diff = sparse_mx[:, idx]
                sparse_nc_for_diff = sparse_nc[:, idx]

            idx = torch.cat(
                (
                    idx,
                    torch.arange(num_sparse, num_queries, device=query.device),
                )
            )
            query_for_dense = query[:, idx]

        if args.need_apply_rope and args.using_extend:
            # TODO: using paged attention
            repeated_k = args.gather_k_from_paged_cache(
                disable_gqa=True, gqa_q=query
            )
            repeated_v = args.gather_v_from_paged_cache(
                disable_gqa=True, gqa_q=query
            )  # B, T, H, D
            # assert repeated_k.shape[2] in (1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 64), repeated_k.shape
            assert repeated_k.shape[2] < 128

            seq_len = args.position_ids.amax().item() + 1
            repeated_k = repeated_k[:, :seq_len]
            repeated_v = repeated_v[:, :seq_len]

            query_for_recomp = query_for_dense

            cos = args.rope_cos
            sin = args.rope_sin
            assert cos.ndim == 2, cos.shape
            assert sin.shape == cos.shape, sin.shape

            cos = cos.view(1, cos.shape[-2], 1, cos.shape[-1])
            sin = sin.view(1, sin.shape[-2], 1, sin.shape[-1])

            idx_tsrc = torch.arange(0, repeated_k.shape[1], device=cos.device)
            idx_tsrc.clamp_min_(seq_len - args.model_context_length)

            repeated_k = (
                (repeated_k * cos[:, idx_tsrc, :, :])
                + (rotate_half(repeated_k) * sin[:, idx_tsrc, :, :])
            ).to(repeated_k.dtype)

            query_for_recomp = (
                (
                    query_for_recomp
                    * cos[:, args.position_ids.view(-1)[idx], :, :]
                )
                + (
                    rotate_half(query_for_recomp)
                    * sin[:, args.position_ids.view(-1)[idx], :, :]
                )
            ).to(query_for_recomp.dtype)

            assert args.position_ids.shape[0] == 1
            context_dense = (
                query_sparse_attention(
                    query_for_recomp.permute(0, 2, 1, 3).contiguous(),
                    repeated_k.permute(0, 2, 1, 3).contiguous(),
                    repeated_v.permute(0, 2, 1, 3).contiguous(),
                    # idx.unsqueeze(0),
                    args.position_ids[:, idx],
                    sm_scale,
                    None,
                    None,
                    None,
                    k_descale=k_descale,
                    v_descale=v_descale,
                )
                .permute(0, 2, 1, 3)
                .contiguous()
            )
        else:
            if args.using_paged_cache:
                assert args.using_paged_cache

                query_for_recomp = query_for_dense
                k_cache = args.get_k_cache()
                v_cache = args.get_v_cache()

                assert args.position_ids.shape[0] == 1
                context_dense = query_sparse_attention(
                    query_for_recomp.permute(0, 2, 1, 3).contiguous(),
                    None,
                    None,
                    args.position_ids[:, idx],
                    sm_scale,
                    k_cache,
                    v_cache,
                    args.block_table,
                    return_running_statistics=delta_attention_args_adjust_norm_const,
                    k_descale=k_descale,
                    v_descale=v_descale,
                )
            else:
                assert k is not None
                assert v is not None
                context_dense = query_sparse_attention(
                    query_for_recomp.permute(0, 2, 1, 3).contiguous(),
                    k.permute(0, 2, 1, 3).contiguous(),
                    v.permute(0, 2, 1, 3).contiguous(),
                    args.position_ids[:, idx],
                    sm_scale,
                    None,
                    None,
                    None,
                    return_running_statistics=delta_attention_args_adjust_norm_const,
                    k_descale=k_descale,
                    v_descale=v_descale,
                )

            if delta_attention_args_adjust_norm_const:
                context_dense, (dense_mx, dense_nc) = context_dense
            else:
                dense_mx = dense_nc = None

            if dense_mx is not None:
                dense_mx = dense_mx.permute(0, 2, 1)
                dense_nc = dense_nc.permute(0, 2, 1)
            context_dense = context_dense.permute(0, 2, 1, 3).contiguous()

        if delta_attention_args_diff == 0:
            context = torch.zeros_like(query)
            context[:, :num_sparse] = context_sparse
            context[:, idx] = context_dense
        elif delta_attention_args_adjust_norm_const:
            idx_dense, idx_last = (idx[:-num_last_dense], idx[-num_last_dense:])
            context_dense, last_context_dense = (
                context_dense[:, :-num_last_dense],
                context_dense[:, -num_last_dense:],
            )

            dense_mx = dense_mx[:, :-num_last_dense]
            dense_nc = dense_nc[:, :-num_last_dense]

            # context_sparse_for_diff_norm = context_sparse_for_diff.float().square().sum(dim=-1, keepdim=True).sqrt()
            # context_dense_norm = context_dense.float().square().sum(dim=-1, keepdim=True).sqrt()
            # scale = context_dense_norm / context_sparse_for_diff_norm

            using_jeff = False
            if using_jeff or (not delta_attention_args_adjust_norm_const):
                # redo the normalization constant for the sparse outputs so we calculate the exact delta region
                # ------------------------------
                if delta_attention_args_adjust_norm_const:
                    # denorm, make alpha, scale, renorm so that the difference in the following block is the exact difference
                    # with the correct normalization constant.
                    context_sparse_for_diff = (
                        context_sparse_for_diff
                        * sparse_nc_for_diff[:, :, :, None]
                    )
                    mx = torch.stack(
                        (dense_mx, sparse_mx_for_diff), dim=0
                    ).amax(dim=0)
                    alpha = torch.exp2(sparse_mx_for_diff - mx)
                    context_sparse_for_diff = (
                        context_sparse_for_diff * alpha[:, :, :, None]
                    )
                    sparse_nc_for_diff = sparse_nc_for_diff * alpha
                    context_sparse_for_diff = (
                        context_sparse_for_diff
                        / sparse_nc_for_diff[:, :, :, None]
                    )
                # ------------------------------

                # take difference
                context_diff = (
                    context_dense - context_sparse_for_diff
                )  # * scale

                context_diff = context_diff.repeat_interleave(
                    delta_attention_args_w, dim=1
                )

                if delta_attention_args_smooth:
                    # (exp) linear interpolate diff
                    context_diff_shift = torch.roll(
                        context_diff, -delta_attention_args_w, 1
                    )
                    context_diff_shift[:, -delta_attention_args_w:] = (
                        context_diff[:, -1:]
                    )

                    offset = torch.arange(
                        0, context_diff.shape[1], device=context_diff.device
                    )
                    offset = (
                        offset % delta_attention_args_w
                    ).float() / delta_attention_args_w
                    context_diff = (
                        context_diff
                        + (context_diff_shift - context_diff)
                        * offset[None, :, None, None]
                    )

                # context_sparse_norm = context_sparse.float().square().sum(dim=-1, keepdim=True).sqrt()
                # scale = context_dense_norm.repeat_interleave(delta_attention_args_w, dim=1) / context_sparse_norm

                # ---------------------------------------------------
                # rescale context sparse to include the normalization constant from the delta region H = (T + H) - T
                if delta_attention_args_adjust_norm_const:
                    # get the 'head' normalization constant which is the normalization constant of the non-sparse indices.
                    h_nc = (
                        dense_nc - sparse_nc_for_diff
                    )  # sparse_nc already applied alpha
                    h_nc = h_nc.repeat_interleave(delta_attention_args_w, dim=1)

                    mx_repeat = mx.repeat_interleave(
                        delta_attention_args_w, dim=1
                    )

                    context_sparse = context_sparse * sparse_nc[:, :, :, None]
                    # mx = torch.stack((dense_mx_repeat, sparse_mx)).amax(dim=0)

                    alpha_for_sparse = torch.exp2(sparse_mx - mx_repeat)
                    context_sparse = (
                        context_sparse * alpha_for_sparse[:, :, :, None]
                    )
                    sparse_nc = sparse_nc * alpha_for_sparse

                    # mx = torch.stack((dense_mx_repeat, sparse_mx)).amax(dim=0)
                    # alpha_for_dense = torch.exp2(dense_mx_repeat - mx)
                    # h_nc = h_nc * alpha_for_dense

                    nc = h_nc + sparse_nc
                    context_sparse = context_sparse / nc[:, :, :, None]
                # ---------------------------------------------------

                # context = context_sparse * scale + context_diff
                context = context_sparse + context_diff
            else:
                scale = 1.0
                numerator = context_dense * dense_nc[:, :, :, None]
                denominator = dense_nc[:, :, :, None]

                # if get_local_rank() == 0:
                #     print('-')
                #     print('wrong mx (%)', (sparse_mx_for_diff > dense_mx).float().mean())
                #     print('avg error', ((sparse_mx_for_diff - dense_mx) * (sparse_mx_for_diff > dense_mx)).mean())
                #     print('max error', ((sparse_mx_for_diff - dense_mx) * (sparse_mx_for_diff > dense_mx)).amax())
                #     print('avg sparse mx', sparse_mx_for_diff.mean())

                t_mx = torch.maximum(sparse_mx_for_diff, dense_mx)

                alpha_sparse = torch.exp2(sparse_mx_for_diff - t_mx)[
                    :, :, :, None
                ]

                alpha_dense = torch.exp2(dense_mx - t_mx)[:, :, :, None]
                # alpha_dense = 1

                # this is the delta with denormalized numerator,
                # denominator is equal to H
                delta = numerator * alpha_dense - alpha_sparse * (
                    context_sparse_for_diff * sparse_nc_for_diff[:, :, :, None]
                )
                h_nc = (
                    denominator * alpha_dense
                    - alpha_sparse * sparse_nc_for_diff[:, :, :, None]
                )

                # if get_local_rank() == 0:
                #     print(denominator[0, :, 0])

                # context_diff = context_dense - context_sparse_for_diff
                # context_diff_norm = torch.norm(context_diff, dim=-1, keepdim=True)
                # context_diff_scale = context_diff_norm / context_diff_norm.amax(dim=1, keepdim=True)
                # scale *= context_diff_scale
                # delta *= scale
                # h_nc *= scale

                def _repeat_interleave(t: torch.Tensor):
                    t = t.repeat_interleave(delta_attention_args_w, dim=1)

                    if delta_attention_args_smooth:
                        # (exp) linear interpolate diff
                        t_shift = torch.roll(t, -delta_attention_args_w, 1)
                        t_shift[:, -delta_attention_args_w:] = t[:, -1:]

                        offset = torch.arange(0, t.shape[1], device=t.device)
                        offset = (
                            offset % delta_attention_args_w
                        ).float() / delta_attention_args_w
                        if t.ndim == 4:
                            t = t + (t_shift - t) * offset[None, :, None, None]
                        else:
                            t = t + (t_shift - t) * offset[None, :, None]
                    return t

                delta = _repeat_interleave(delta)
                h_nc = _repeat_interleave(h_nc)
                dense_mx = _repeat_interleave(
                    torch.maximum(sparse_mx_for_diff, dense_mx)
                )
                # sparse_mx = _repeat_interleave(sparse_mx_for_diff)
                # sparse_nc = _repeat_interleave(sparse_nc_for_diff)

                t_mx = torch.maximum(dense_mx, sparse_mx)
                alpha_sparse = torch.exp2(sparse_mx - t_mx)[:, :, :, None]

                alpha_dense = torch.exp2(dense_mx - t_mx)[:, :, :, None]

                # alpha_dense_mask = torch.zeros(dense_mx.size(1), device=dense_mx.device, dtype=torch.bool)
                # alpha_dense_mask = alpha_dense_mask.view(-1, delta_attention_args_w)
                # alpha_dense_mask[:, 0] = 1
                # alpha_dense_mask = alpha_dense_mask.view(-1)[None, :, None, None]
                # alpha_dense = alpha_dense_mask * 1 + ~alpha_dense_mask * alpha_dense_mask

                numerator = alpha_dense * delta + alpha_sparse * (
                    context_sparse * sparse_nc[:, :, :, None]
                )
                denominator = (
                    alpha_dense * h_nc + alpha_sparse * sparse_nc[:, :, :, None]
                )

                context = numerator / denominator

            context = context.to(query.dtype)
            context.index_copy_(dim=1, index=idx_dense, source=context_dense)
            context = torch.cat([context, last_context_dense], dim=1)

            # if get_local_rank() == 0:
            #     print(
            #         'hit', layer_id,
            #         context_diff.shape,
            #         context_sparse.shape,
            #         context_diff.abs().mean().item(),
            #         context_sparse.abs().mean().item()
            #     )
        else:
            from .delta.apply_delta import apply_delta

            context = apply_delta(
                context_dense,
                context_sparse,
                idx,
                num_last_dense,
                delta_attention_args_w,
                delta_attention_args_smooth,
            )

    return context, metadata

@capture
def _forward_fa3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    position_ids: torch.Tensor,
    using_extend: bool,
    need_apply_rope: bool,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_is_neox_style: bool,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor
):
    assert q.ndim == 4
    assert k.ndim == 4
    assert v.ndim == 4

    len_query_for_fa3 = q.shape[1]

    if (using_extend and need_apply_rope) and True:
        # FIXME do better infer method
        use_mla = triton.next_power_of_2(k.shape[-1]) != k.shape[-1]
        if use_mla:
            q = q.clone()
            k = k.clone()

            # FIXME assume DeepSeek
            rope_dim = k.shape[-1] - (triton.next_power_of_2(k.shape[-1]) // 2)
            assert rope_dim == rope_cos.shape[-1]

            from sglang.srt.layers.rotary_embedding import _rotate_gptj, _rotate_neox

            rotate_fn = _rotate_neox if rope_is_neox_style else _rotate_gptj

            query_rot = q[..., -rope_dim:]
            key_rot = k[..., -rope_dim:]

            assert position_ids.shape[0] == 1
            assert not rope_is_neox_style
            cos_q = rope_cos[
                None,
                position_ids[0, :len_query_for_fa3],
                None,
                : rope_dim // 2,
            ].repeat_interleave(2, -1)
            sin_q = rope_sin[
                None,
                position_ids[0, :len_query_for_fa3],
                None,
                : rope_dim // 2,
            ].repeat_interleave(2, -1)
            cos_k = rope_cos[
                None, : key_rot.shape[1], None, : rope_dim // 2
            ].repeat_interleave(2, -1)
            sin_k = rope_sin[
                None, : key_rot.shape[1], None, : rope_dim // 2
            ].repeat_interleave(2, -1)

            query_rot = query_rot * cos_q + rotate_fn(query_rot) * sin_q
            key_rot = key_rot * cos_k + rotate_fn(key_rot) * sin_k

            q[..., -rope_dim:] = query_rot
            k[..., -rope_dim:] = key_rot

            # qqq = query_fa3[0, -4096:, 0]
            # kkk = k_fa3[0, :, 0]
            # scores = qqq @ kkk.T
            # scores = torch.nn.functional.max_pool2d(scores[None, None, ...], kernel_size=31, stride=15, padding=15)[0,0]

            # plt.clf()
            # plt.title(f'{scores.shape=}')
            # plt.imshow(scores.float().cpu().numpy())
            # plt.savefig('./dummy_scores.png')

            # print(rope_dim, args.rope_cos.shape, args.rope_sin.shape, args.rope_is_neox_style)
        else:
            q = q.clone()
            k = k.clone()

            # FIXME assume GQA/MHA
            rope_dim = k.shape[-1]

            from sglang.srt.layers.rotary_embedding import _rotate_gptj, _rotate_neox

            rotate_fn = _rotate_neox if rope_is_neox_style else _rotate_gptj

            query_rot = q
            key_rot = k

            if rope_is_neox_style:
                cos_q = rope_cos[None, position_ids[0, :len_query_for_fa3], None, :]
                sin_q = rope_sin[None, position_ids[0, :len_query_for_fa3], None, :]
                cos_k = rope_cos[None, : key_rot.shape[1], None, :]
                sin_k = rope_sin[None, : key_rot.shape[1], None, :]
            else:
                assert position_ids.shape[0] == 1
                cos_q = rope_cos[
                    None,
                    position_ids[0, :len_query_for_fa3],
                    None,
                    : rope_dim // 2,
                ].repeat_interleave(2, -1)
                sin_q = rope_sin[
                    None,
                    position_ids[0, :len_query_for_fa3],
                    None,
                    : rope_dim // 2,
                ].repeat_interleave(2, -1)
                cos_k = rope_cos[
                    None, : key_rot.shape[1], None, : rope_dim // 2
                ].repeat_interleave(2, -1)
                sin_k = rope_sin[
                    None, : key_rot.shape[1], None, : rope_dim // 2
                ].repeat_interleave(2, -1)

            q = (query_rot * cos_q + rotate_fn(query_rot) * sin_q).to(query_rot.dtype)
            k = (key_rot * cos_k + rotate_fn(key_rot) * sin_k).to(key_rot.dtype)

    tp_q_head, tp_q_dim = q.shape[2:]
    tp_k_head, tp_k_dim = k.shape[2:]
    tp_v_head, tp_v_dim = v.shape[2:]

    # cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32, device=q.device)
    # max_seqlen_q = q_len

    # # Construct metadata for the Key/Value
    # # For a single sequence of length kv_len, cu_seqlens is just [0, kv_len]
    # cu_seqlens_k = torch.tensor([0, kv_len], dtype=torch.int32, device=k.device)
    # max_seqlen_k = kv_len

    cu_seqlens_q = torch.tensor([0, q.shape[1]], dtype=torch.int32, device=q.device)
    max_seqlen_q = q.shape[1]
    cu_seqlens_k = torch.zeros((2,), dtype=torch.int32, device=k.device)
    cu_seqlens_k[1] = position_ids[0, len_query_for_fa3 - 1] + 1
    max_seqlen_k = k.shape[1]

    # print(query_fa3.shape, k_fa3.shape, v_fa3.shape, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, sm_scale)

    context_fa3 = flash_attn_varlen_func(
        q=q.view(-1, tp_q_head, tp_q_dim).contiguous(),
        k=k.view(-1, tp_k_head, tp_k_dim).contiguous(),
        v=v.view(-1, tp_v_head, tp_v_dim).contiguous(),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=sm_scale,
        causal=True,
        return_softmax_lse=False,
        k_descale=k_descale,
        v_descale=v_descale,
    )

    context_fa3 = context_fa3.view(q.shape[:-1] + (v.shape[-1],))

    return context_fa3


@capture
def _forward_partial_fa3(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    rope_is_neox_style: bool,
    cached_metadata: HiPAttentionOutputMetadata,
    is_decode: bool,
    seq_thresh_fa3: int,
    mixing_len: int,
    args: HiPAttentionArgs,
    max_context_len: int,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    inner_function_do_scale: bool,
    inner_function,
):
    query = q

    context_fa3 = None
    metadata = None

    if (not is_decode) and (seq_thresh_fa3 > 0):
        if args.using_paged_cache:
            pass
        else:
            assert k is not None
            max_context_len = min(max_context_len, k.shape[1])
        min_context_len = max(0, max_context_len - query.shape[1])

        len_query_for_fa3 = max(0, min(seq_thresh_fa3, max_context_len) - min_context_len)
        len_query_for_hip = max(0, max_context_len - max(min_context_len, seq_thresh_fa3 - mixing_len))

        # print(max_context_len, min_context_len, seq_thresh_fa3, len_query_for_fa3, len_query_for_hip)

        if len_query_for_fa3 > 0:
            assert not is_decode
            # assert not args.using_extend, "todo"

            if args.using_paged_cache:
                k = args.gather_k_from_paged_cache(seq_len=min(max_context_len, args.model_context_length))
                v = args.gather_v_from_paged_cache(seq_len=min(max_context_len, args.model_context_length))

            query_fa3 = query[:, :len_query_for_fa3].contiguous()
            len_kv = k.shape[1] - (len_query_for_hip - (query.shape[1] - len_query_for_fa3)) # BUG: this should be bug, because this will lose keys for len_for_mix
            k_fa3 = k[:, :len_kv].contiguous()
            v_fa3 = v[:, :len_kv].contiguous()

            is_fp8 = k.dtype in (torch.float8_e5m2, )
            if is_fp8:
                query_fa3 = query_fa3.to(torch.float16)
                k_fa3 = k_fa3.to(torch.float16)
                v_fa3 = v_fa3.to(torch.float16)

            context_fa3 = _forward_fa3(
                q=query_fa3,
                k=k_fa3,
                v=v_fa3,
                sm_scale=sm_scale,
                position_ids=args.position_ids[:, :len_query_for_fa3],
                using_extend=args.using_extend,
                need_apply_rope=args.need_apply_rope,
                rope_cos=args.rope_cos,
                rope_sin=args.rope_sin,
                rope_is_neox_style=rope_is_neox_style,
                k_descale=k_descale,
                v_descale=v_descale,
            )

    if args.using_paged_cache:
        k = v = None

    if context_fa3 is not None:
        if len_query_for_hip > 0:
            args_sparse = args.clone()
            args_sparse.position_ids = args_sparse.position_ids[:, -len_query_for_hip:]
            if args_sparse.q_mask is not None:
                args_sparse.q_mask = args_sparse.q_mask[:, -len_query_for_hip:]
            if args_sparse.query_for_landmark is not None:
                args_sparse.query_for_landmark = args_sparse.query_for_landmark[
                    :, -len_query_for_hip:
                ]

            yarn_scale = float(os.getenv('HIP_DEBUG_YARN_SCALE_HINT', '1'))
            if yarn_scale > 1:
                assert int(yarn_scale) == yarn_scale
                yarn_scale = int(yarn_scale)
                args_sparse.rope_cos = args_sparse.rope_cos[::yarn_scale]
                args_sparse.rope_sin = args_sparse.rope_sin[::yarn_scale]

            context_sparse, metadata = inner_function(
                q=(query[:, -len_query_for_hip:] * (sm_scale if inner_function_do_scale else 1)).to(query.dtype),
                k=k,
                v=v,
                args=args_sparse,
                cached_metadata=cached_metadata,
            )
            if context_sparse.ndim == 3:
                context_sparse = context_sparse.unsqueeze(0)
                assert context_fa3.shape[0] == 1

            # w = 512
            # wt = 16
            # t = context_sparse.shape[1]
            # if t > w:
            #     t_context = context_sparse[:, t % w:]
            #     t_context_mean = t_context.view(-1, t // w, w, t_context.shape[-2], t_context.shape[-1]).mean(2, keepdim=True)
            #     delta = (torch.repeat_interleave(t_context_mean, w//wt, 1) - t_context.view(-1, t // wt, wt, t_context.shape[-2], t_context.shape[-1])).mean(2)
            #     delta = torch.repeat_interleave(delta, wt, 1)
            #     t_context.add_(delta)

            len_for_mix = (len_query_for_hip + len_query_for_fa3) - query.shape[1]

            if len_for_mix > 0:
                context_fa3_mix = context_fa3[:, -len_for_mix:]
                context_sparse_mix = context_sparse[:, :len_for_mix]

                chunk_len = min(context_sparse_mix.shape[1], len_for_mix)
                offset = min_context_len - (seq_thresh_fa3 - mixing_len)
                scale_global = (
                    torch.arange(
                        offset, offset + chunk_len, device=query.device, dtype=torch.float32
                    )
                    / mixing_len
                )

                len_for_spike = min(chunk_len, 32)
                scale = torch.clamp_min(
                    (torch.arange(0, chunk_len, device=query.device, dtype=torch.float32) - (chunk_len - len_for_spike))
                    / len_for_spike, 0
                ) # * (1 - (offset / mixing_len)) + (offset / mixing_len)

                # scale_spike = (
                #     (torch.arange(
                #         offset, offset + chunk_len, device=query.device, dtype=torch.float32
                #     ) % len_for_spike)
                #     / len_for_spike
                # )
                # scale = torch.maximum(scale, scale_spike)

                scale = torch.maximum(scale, scale_global)

                scale = scale[None, :, None, None]
                context_mix = (
                    context_sparse_mix * scale + context_fa3_mix * (1.0 - scale)
                ).to(context_fa3_mix.dtype)

                context = torch.cat(
                    [
                        context_fa3[:, :-len_for_mix],
                        context_mix,
                        context_sparse[:, len_for_mix:],
                    ],
                    dim=1,
                )
            else:
                context = torch.cat([context_fa3, context_sparse], dim=1)
        else:
            context = context_fa3
    else:
        # no fa3
        context, metadata = inner_function(
            q=(query * (sm_scale if inner_function_do_scale else 1)).to(query.dtype),
            k=k,
            v=v,
            args=args,
            cached_metadata=cached_metadata,
        )

    context = context.to(query.dtype)

    return context, metadata


@capture
def _forward_sliding_window(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    args: HiPAttentionArgs,
    sliding_window_size: int,
    sliding_window_sink: int,
):
    query = q
    bsa_fn = get_block_sparse_backend(args, query)

    # dist.barrier()
    # if get_tensor_model_parallel_rank() == 0:
    #     print(bsa_fn, args.using_extend, sliding_window_size, args.using_chunked_sliding_window)

    BSZ, TDST, HEAD, HID = query.shape

    args = args.clone()
    if args.rope_range is None:
        args.rope_range = (0, HID)
    args.block_size_q = args.block_sparse_block_size_q
    args.block_size_k = args.stages[-1].stage_chunk_size
    args.second_stage_k = 0
    args.sink_token_size = sliding_window_sink
    args.sliding_window_size = (
        sliding_window_size if sliding_window_size is not None else 1024
    )
    args.sliding_window_indices = None

    BDST = triton.cdiv(TDST, args.block_size_q)
    BH = BSZ * HEAD

    indices = torch.zeros((BH, BDST, 0), dtype=torch.int64, device=query.device)
    ks = torch.zeros((BH, BDST), dtype=torch.int64, device=query.device)
    ks_count = ks.unsqueeze(-1)
    ks_start_end = torch.zeros((BH, BDST, 2), dtype=torch.int64, device=query.device)

    context = bsa_fn(
        q=query,
        k=k,
        v=v,
        seq_lens=args.position_ids + 1,
        indices=indices,
        ks=ks,
        ks_count=ks_count,
        ks_start_end=ks_start_end,
        access_counter=None,
        cache_miss_counter=None,
        EXTEND_BACKEND=args.sa_extend_backend,
        model_context_length=args.model_context_length,
        extend_context_length=args.extend_context_length,
        offload_update_cache=False,
        args=args,
    )
    context = context.to(query.dtype)

    return context, None


def _forward_paged_hip(
    query: torch.Tensor,
    sm_scale: float,
    batch_size: int,
    k_cache: Optional[torch.Tensor],
    v_cache: Optional[torch.Tensor],
    offload_cache: Optional[HiPOffloadCache],
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
    req_to_tokens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    block_table: torch.Tensor,
    rope_cos: Optional[torch.Tensor],
    rope_sin: Optional[torch.Tensor],
    layer_id: int,
    logit_cap: float,
    orig_context_len: int,
    max_context_len: int,
    max_batch_context_len: int,
    v_hidden_dim: int,
    hip_config: HiPAttentionConfig,
    rope_range: Optional[tuple[int, int]] = None,
    rope_is_neox_style: Optional[bool] = None,
    cached_metadata: Optional[HiPAttentionOutputMetadata] = None,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    online_update_cache: bool = False,
    is_decode: bool = False,
    query_for_mask: Optional[torch.Tensor] = None,
    diag_sliding_window_indices: Optional[torch.Tensor] = None,
    sliding_window_size: Optional[int] = -1,
    sliding_window_sink: Optional[int] = -1,
    using_chunked_sliding_window: bool = False,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, HiPAttentionOutputMetadata]:
    global _CHECKOUT_COUNTER

    if query.ndim == 3:
        N, num_heads, hidden_dims = query.shape
        dst_seq_len = N // batch_size
    else:
        _bsz, dst_seq_len, num_heads, hidden_dims = query.shape
        assert _bsz == batch_size, f"{query.shape=} {batch_size}"
        N = _bsz * dst_seq_len

    is_dense = layer_id in hip_config.dense_layers
    if not is_decode:
        if len(hip_config.prefill_layers) == 2:
            layer_config = hip_config.prefill_layers[0 if is_dense else 1]
        else:
            layer_config = hip_config.prefill_layers[layer_id]
    else:
        # assert dst_seq_len == 1
        if len(hip_config.layers) == 2:
            layer_config = hip_config.layers[0 if is_dense else 1]
        else:
            layer_config = hip_config.layers[layer_id]

    query = query.view(batch_size, dst_seq_len, num_heads, hidden_dims)
    if query_for_mask is not None:
        query_for_mask = query_for_mask.view(batch_size, -1, num_heads, hidden_dims)

    if k_cache is not None:
        if v_cache.ndim == 4:
            N_PAGE, _, num_heads_kv, hidden_dims_v = v_cache.shape
        else:
            assert v_cache.ndim == 3
            N_PAGE, num_heads_kv, hidden_dims_v = v_cache.shape
        assert N_PAGE == k_cache.shape[0], f"{N_PAGE} != {k_cache.shape[0]}"

        k_cache = k_cache.view(N_PAGE, 1, num_heads_kv, k_cache.shape[-1])
        v_cache = v_cache.view(N_PAGE, 1, num_heads_kv, hidden_dims_v)

    # FIXME: this operation is linear during decoding
    if block_table is None:
        block_table = req_to_tokens.index_select(dim=0, index=req_pool_indices)

    BLOCK_TABLE_BSZ, MODEL_SEQ_LEN = block_table.shape
    assert batch_size == BLOCK_TABLE_BSZ

    if k_descale is not None:
        assert k_descale.shape == (batch_size, num_heads_kv)
        assert v_descale.shape == (batch_size, num_heads_kv)

    # NOTE(heejun): the whole point to need to find gemma is large size of hidden size
    if k_cache is not None:
        hidden_size = k_cache.shape[-1]
    elif k is not None:
        hidden_size = k.shape[-1]
    elif offload_cache is not None:
        hidden_size = offload_cache.k_uvm.bank_cpu.shape[-1]
    else:
        raise Exception()
    is_gemma = hidden_size > 128

    # NOTE this is not needed when offload cache is not needed right..?
    require_cache_statistics = False
    if cached_metadata is None:
        require_cache_statistics = offload_cache is not None
    elif cached_metadata.indices is None:
        require_cache_statistics = offload_cache is not None
    elif os.getenv("HIP_DISABLE_COMPUTE_STATISTICS", "1") == "0":
        require_cache_statistics = offload_cache is not None

    if torch.cuda.is_current_stream_capturing():
        assert is_decode

    args = HiPAttentionArgs(
        # k_cache=(
        #     k_cache.view(torch.uint8)
        #     if isinstance(k_cache, torch.Tensor) and k_cache.dtype == torch.float8_e5m2
        #     else k_cache
        # ),
        k_cache=k_cache,
        # v_cache=(
        #     v_cache.view(torch.uint8)
        #     if isinstance(k_cache, torch.Tensor) and v_cache.dtype == torch.float8_e5m2
        #     else v_cache
        # ),
        v_cache=v_cache,
        offload_cache=offload_cache,
        block_table=block_table,
        cache_seq_lens=seq_lens,
        position_ids=positions.view(batch_size, dst_seq_len),
        block_size_k=32 if is_gemma else 64,  # BLOCK_CHUNK
        sliding_window_size=layer_config.sliding_window_size,
        sink_token_size=layer_config.sink_token_size,
        using_extend=hip_config.using_extend,
        need_apply_rope=hip_config.using_extend,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        rope_range=rope_range,
        rope_is_neox_style=rope_is_neox_style,
        logit_softcap=logit_cap if logit_cap != 0.0 else None,
        second_stage_k=layer_config.second_stage_k,
        stages=layer_config.stages,
        model_context_length=orig_context_len,
        extend_context_length=max_context_len,
        block_sparse_block_size_q=hip_config.block_sparse_block_size_q,
        scan_extend_backend=(
            (
                "relative"
                if hip_config.apply_v_dot
                else ("streaming" if is_dense else "relative")
            )
            if layer_config.scan_extend_backend is None
            else layer_config.scan_extend_backend
        ),
        sa_extend_backend=layer_config.sa_extend_backend,
        online_update_cache=online_update_cache,
        require_cache_statistics=require_cache_statistics,
        disable_flashdecode=not is_decode,
        q_mask=(
            (query_for_mask * sm_scale).to(query.dtype)
            if query_for_mask is not None
            else None
        ),
        sliding_window_indices=(
            diag_sliding_window_indices[layer_id]
            if diag_sliding_window_indices is not None
            else None
        ),
        layer_id=layer_id,
        v_hidden_dim=v_hidden_dim,
        using_chunked_sliding_window=using_chunked_sliding_window,
        is_decode=is_decode,
        landmark_stage_k=layer_config.landmark_stage_k,
        k_descale=k_descale,
        v_descale=v_descale,
    )

    using_dense_prefill = os.getenv("HIP_DEBUG_USING_DENSE_PREFILL", "0") == "1"
    if is_decode:
        using_dense_prefill = False
    else:
        using_dense_prefill = using_dense_prefill and is_dense
        # using_dense_prefill = True

    force_dense_decode = os.getenv("HIP_DEBUG_FORCE_DENSE_DECODE", "0") == "1"
    last_dense = int(os.getenv("HIP_DEBUG_LAST_DENSE", "-1"))

    if last_dense > 0:
        last_dense += dst_seq_len % args.block_sparse_block_size_q

    sliding_window_size_for_masking_step = (
        layer_config.sliding_window_size_for_masking_step
    )
    if (
        isinstance(sliding_window_size_for_masking_step, list)
        and (cached_metadata is not None)
        and (cached_metadata.indices is None)
    ):
        larger_sw_size = sliding_window_size_for_masking_step[
            (
                max(0, len(cached_metadata.stage_caches) - 1)
                if cached_metadata.stage_caches is not None
                else 0
            )
        ]
        args.bsa_sliding_window_size = larger_sw_size

    sliding_window_size = os.getenv("HIP_DEBUG_SLLM_WINDOW", sliding_window_size)
    if isinstance(sliding_window_size, str):
        sliding_window_size = int(sliding_window_size)
    sliding_window_sink = int(
        os.getenv("HIP_DEBUG_SLLM_SINK", max(0, sliding_window_sink))
    )
    if args.second_stage_k == 0:
        sliding_window_size = args.sliding_window_size
        sliding_window_sink = args.sink_token_size

    # Plan 1
    # TODO use flash attention under 100K

    # Plan 2
    # TODO use flash attention under 64K
    # TODO use sparse setting under 128K

    seq_thresh_fa3 = int(os.getenv("HIP_DEBUG_SEQ_THRESH_FA3", "0"))
    if seq_thresh_fa3 > args.model_context_length:
        warnings.warn(
            f"Requested FA3 replacement ({seq_thresh_fa3}) is larger than model context length ({args.model_context_length}). "
            "Consider increase YaRN or using other model. "
            "OR You can decrease HIP_DEBUG_SEQ_THRESH_FA3 up to context length, but it will degrade throughput."
        )
        seq_thresh_fa3 = args.model_context_length

    mixing_len = os.getenv("HIP_DEBUG_FA3_MIXING_LEN", "sw" if seq_thresh_fa3 > 0 else "0")
    if mixing_len.lower() == "sw":
        mixing_len = int(
            sliding_window_size * 1.5
            if isinstance(sliding_window_size, int) and (sliding_window_size > 0) else
            args.sliding_window_size * 1.5
        )
    else:
        mixing_len = int(mixing_len)

    if (seq_thresh_fa3 == 0):
        mixing_len = 0

    if os.getenv("HIP_DEBUG_SEQ_THRESH_FA3_INF_DENSE", "0") == "1":
        if layer_id in hip_config.dense_layers:
            seq_thresh_fa3 = query.shape[1]

    # TODO: if delta norm is too high, then just recompute that whole block.
    # TODO: use partial densely decode. delta attention for decode
    # postfix_recompute_dense-window_[size:int]-diff_[1/0]-w_[size:int]
    # example: HIP_DELTA_ATTENTION_ARGS=window_0-diff_1-w_32-sparse_decode-smooth-exp
    delta_attention_args = os.getenv("HIP_DELTA_ATTENTION_ARGS", None)
    using_delta_attention = delta_attention_args is not None

    if using_delta_attention:
        delta_attention_args_smooth = False
        delta_attention_args_just_return = False
        delta_attention_args_window = 0
        delta_attention_args_diff = 1
        delta_attention_args_dense_decode = False
        delta_attention_args_w = 16
        delta_attention_args_exp = False
        delta_attention_args_exp_w = 2
        delta_attention_args_exp_window = 1024
        delta_attention_args_exp_sink = 128
        delta_attention_args_iter_corr = False
        delta_attention_args_adjust_norm_const = False

        for word in delta_attention_args.split("-"):
            word = word.strip()
            if word == "smooth":
                delta_attention_args_smooth = True
            elif word == "exp":
                delta_attention_args_exp = True
            elif word == "JUST_RETURN":
                delta_attention_args_just_return = True
            elif word == "sparse_decode":
                delta_attention_args_dense_decode = False
            elif word == "dense_decode":
                delta_attention_args_dense_decode = True
            elif word == "recompute_dense":
                pass  # backward compat.
            elif word == "iter_corr":
                delta_attention_args_iter_corr = True
            elif word == "adjust_norm_const":
                delta_attention_args_adjust_norm_const = True
            elif word.startswith("window_"):
                delta_attention_args_window = int(word.split("_")[1])
            elif word.startswith("diff_"):
                delta_attention_args_diff = int(word.split("_")[1])
            elif word.startswith("w_"):
                delta_attention_args_w = int(word.split("_")[1])
            elif word.startswith("expw_"):
                delta_attention_args_exp_w = int(word.split("_")[1])
            elif word.startswith("expsink_"):
                delta_attention_args_exp_sink = int(word.split("_")[1])
            elif word.startswith("expwindow_"):
                delta_attention_args_exp_window = int(word.split("_")[1])
            else:
                warnings.warn(f"unknown delta args: {word}")

        # if layer_id in [0,1,2,3,4,5,8,11,14,17,20,23,26,29,30,33,36,39,41,42,43,44,45,46,47]:
        #     delta_attention_args_adjust_norm_const = False

        assert not delta_attention_args_dense_decode, "todo, did not handled in _forward_delta_attn"

        if get_local_rank() == 0:
            info_msg = (
                f"Delta Attention is activated {delta_attention_args_window=} "
                f"{delta_attention_args_diff=} {delta_attention_args_w=} "
                f"{delta_attention_args_just_return=} "
                f"{delta_attention_args_smooth=} "
                f"{delta_attention_args_dense_decode=} "
                f"{delta_attention_args_exp=} "
                f"{delta_attention_args_exp_w=} "
                f"{delta_attention_args_adjust_norm_const=} "
            )
            warnings.warn(info_msg)

        # args.sa_extend_backend = "clamp"

    if isinstance(sliding_window_size, int) and (sliding_window_size > 0):

        def __forward_sliding_window_wrapper(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            args: HiPAttentionArgs,
            cached_metadata: HiPAttentionOutputMetadata,
        ):
            return _forward_sliding_window(
                q=q,
                k=k,
                v=v,
                args=args,
                sliding_window_size=sliding_window_size,
                sliding_window_sink=sliding_window_sink,
            )

        context, metadata = _forward_partial_fa3(
            q=query,
            k=k,
            v=v,
            sm_scale=sm_scale,
            rope_is_neox_style=rope_is_neox_style,
            cached_metadata=cached_metadata,
            is_decode=is_decode,
            seq_thresh_fa3=seq_thresh_fa3,
            mixing_len=mixing_len,
            args=args,
            max_context_len=max_batch_context_len,
            k_descale=k_descale,
            v_descale=v_descale,
            inner_function_do_scale=True,
            inner_function=__forward_sliding_window_wrapper,
        )
    elif using_delta_attention and (
        (not is_decode) # or (is_decode and delta_attention_args_dense_decode)
    ):
        def __forward_delta_attn_wrapper(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            args: HiPAttentionArgs,
            cached_metadata: HiPAttentionOutputMetadata,
        ):
            return _forward_delta_attn(
                q,
                sm_scale,
                k,
                v,
                args=args,
                cached_metadata=cached_metadata,
                is_decode=is_decode,
                delta_attention_args_smooth=delta_attention_args_smooth,
                delta_attention_args_just_return=delta_attention_args_just_return,
                delta_attention_args_window=delta_attention_args_window,
                delta_attention_args_diff=delta_attention_args_diff,
                delta_attention_args_dense_decode=delta_attention_args_dense_decode,
                delta_attention_args_w=delta_attention_args_w,
                delta_attention_args_exp=delta_attention_args_exp,
                delta_attention_args_exp_w=delta_attention_args_exp_w,
                delta_attention_args_exp_window=delta_attention_args_exp_window,
                delta_attention_args_exp_sink=delta_attention_args_exp_sink,
                delta_attention_args_iter_corr=delta_attention_args_iter_corr,
                delta_attention_args_adjust_norm_const=delta_attention_args_adjust_norm_const,
                k_descale=k_descale,
                v_descale=v_descale,
            )

        context, metadata = _forward_partial_fa3(
            q=query,
            k=k,
            v=v,
            sm_scale=sm_scale,
            rope_is_neox_style=rope_is_neox_style,
            cached_metadata=cached_metadata,
            is_decode=is_decode,
            seq_thresh_fa3=seq_thresh_fa3,
            mixing_len=mixing_len,
            args=args,
            max_context_len=max_batch_context_len,
            k_descale=k_descale,
            v_descale=v_descale,
            inner_function_do_scale=False,
            inner_function=__forward_delta_attn_wrapper,
        )
    elif (force_dense_decode and is_decode) or (
        using_dense_prefill and (not is_decode)
    ):
        if is_decode:
            if args.using_extend:
                args_dense = args.clone()
                args_dense.sliding_window_size = 777
                context, metadata = dual_stage_quadratic_hip_attention(
                    (query * sm_scale).to(query.dtype),
                    k,
                    v,
                    args=args_dense,
                    cached_metadata=cached_metadata,
                )
            else:
                bsa_fn = get_block_sparse_backend(args, query)

                BSZ, TDST, HEAD, HID = query.shape

                args_sw = args.clone()
                if args_sw.rope_range is None:
                    args_sw.rope_range = (0, HID)
                args_sw.block_size_q = args_sw.block_sparse_block_size_q
                args_sw.block_size_k = args_sw.stages[-1].stage_chunk_size
                args_sw.second_stage_k = 0
                args_sw.sliding_window_size = args_sw.model_context_length
                args_sw.sliding_window_indices = None

                BDST = triton.cdiv(TDST, args_sw.block_size_q)
                BH = BSZ * HEAD

                indices = torch.zeros(
                    (BH, BDST, 0), dtype=torch.int64, device=query.device
                )
                ks = torch.zeros((BH, BDST), dtype=torch.int64, device=query.device)
                ks_count = ks.unsqueeze(-1)
                ks_start_end = torch.zeros(
                    (BH, BDST, 2), dtype=torch.int64, device=query.device
                )

                context_sparse = bsa_fn(
                    q=(query * sm_scale).to(query.dtype),
                    k=k,
                    v=v,
                    seq_lens=args_sw.position_ids + 1,
                    indices=indices,
                    ks=ks,
                    ks_count=ks_count,
                    ks_start_end=ks_start_end,
                    access_counter=None,
                    cache_miss_counter=None,
                    EXTEND_BACKEND=args_sw.sa_extend_backend,
                    model_context_length=args_sw.model_context_length,
                    extend_context_length=args_sw.extend_context_length,
                    offload_update_cache=False,
                    args=args_sw,
                )
                context_sparse = context_sparse.to(query.dtype)
                context = context_sparse[:, -query.shape[1] :, :, :].contiguous()
                metadata = None
        else:
            # k_unpack = args.gather_k_from_paged_cache()
            # v_unpack = args.gather_v_from_paged_cache()

            # seq_len = args.position_ids.amax().item() + 1

            # k_unpack = k_unpack[:, :seq_len]
            # v_unpack = v_unpack[:, :seq_len]

            # if k_unpack.dtype in [torch.uint8]:
            #     k_unpack = k_unpack.view(torch.float8_e5m2).to(query.dtype)
            #     v_unpack = v_unpack.view(torch.float8_e5m2).to(query.dtype)
            # assert k_unpack.dtype == query.dtype

            # # if layer_id == 0:
            # #     print(seq_len, layer_id, is_decode, force_dense_decode, using_dense_prefill)

            # if args.need_apply_rope and args.using_extend:
            #     cos = args.rope_cos
            #     sin = args.rope_sin
            #     assert cos.ndim == 2, cos.shape
            #     assert sin.shape == cos.shape, sin.shape

            #     cos = cos.view(1, cos.shape[-2], 1, cos.shape[-1])
            #     sin = sin.view(1, sin.shape[-2], 1, sin.shape[-1])

            #     idx_tsrc = torch.arange(0, k_unpack.shape[1], device=cos.device)
            #     idx_tsrc.clamp_min_(seq_len - args.model_context_length)

            #     assert cos.shape[1] >= k_unpack.shape[1], f'{cos.shape=} {k_unpack.shape}'

            #     k_unpack = (
            #         (k_unpack * cos[:, idx_tsrc, :, :])
            #         + (rotate_half(k_unpack) * sin[:, idx_tsrc, :, :])
            #     ).to(k_unpack.dtype)

            #     query = (
            #         (query * cos[:, args.position_ids.view(-1), :, :])
            #         + (rotate_half(query) * sin[:, args.position_ids.view(-1), :, :])
            #     ).to(query.dtype)

            # k_unpack = k_unpack[:, :seq_len]
            # v_unpack = v_unpack[:, :seq_len]

            # context = flash_attn_func(
            #     query,
            #     k_unpack,
            #     v_unpack,
            #     causal=True,
            #     softmax_scale=sm_scale,
            # )

            if args.using_paged_cache:
                if args.need_apply_rope and args.using_extend:
                    k_unpack = args.gather_k_from_paged_cache()
                    v_unpack = args.gather_v_from_paged_cache()

                    seq_len = args.position_ids.amax().item() + 1

                    k_unpack = k_unpack[:, :seq_len]
                    v_unpack = v_unpack[:, :seq_len]

                    cos = args.rope_cos
                    sin = args.rope_sin
                    assert cos.ndim == 2, cos.shape
                    assert sin.shape == cos.shape, sin.shape

                    cos = cos.view(1, cos.shape[-2], 1, cos.shape[-1])
                    sin = sin.view(1, sin.shape[-2], 1, sin.shape[-1])

                    idx_tsrc = torch.arange(0, k_unpack.shape[1], device=cos.device)
                    idx_tsrc.clamp_min_(seq_len - args.model_context_length)

                    k_unpack = (
                        (k_unpack * cos[:, idx_tsrc, :, :])
                        + (rotate_half(k_unpack) * sin[:, idx_tsrc, :, :])
                    ).to(k_unpack.dtype)

                    query = (
                        (query * cos[:, args.position_ids.view(-1), :, :])
                        + (
                            rotate_half(query)
                            * sin[:, args.position_ids.view(-1), :, :]
                        )
                    ).to(query.dtype)

                    k_unpack = k_unpack[:, :seq_len]
                    v_unpack = v_unpack[:, :seq_len]

                    context = flash_attn_func(
                        query,
                        k_unpack,
                        v_unpack,
                        causal=True,
                        softmax_scale=sm_scale,
                    )
                else:
                    assert args.using_paged_cache

                    k_cache = args.get_k_cache()
                    v_cache = args.get_v_cache()

                    q_reshaped = query.contiguous().view(
                        -1, query.shape[2], query.shape[3]
                    )

                    # print(k_cache.shape, v_cache.shape)

                    cu_seqlens_q = (
                        torch.arange(
                            0,
                            query.shape[0] + 1,
                            device=query.device,
                            dtype=torch.int32,
                        )
                        * query.shape[1]
                    )
                    cache_seqlens = (args.position_ids[:, -1] + 1).to(torch.int32)
                    cu_seqlens_k_new = torch.zeros(
                        (args.position_ids.shape[0] + 1,),
                        dtype=torch.int32,
                        device=q_reshaped.device,
                    )
                    cu_seqlens_k_new[1:] = cache_seqlens

                    block_table = args.block_table

                    context = flash_attn_with_kvcache(
                        q=q_reshaped,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        page_table=block_table,
                        cache_seqlens=cache_seqlens,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k_new=cu_seqlens_k_new,
                        # max_seqlen_q=cu_seqlens_q.amax().item(),
                        max_seqlen_q=args.model_context_length,
                        causal=True,
                        softmax_scale=sm_scale,
                    )
            else:
                print(query.shape, v.shape, k.shape)
                assert batch_size
                context = flash_attn_func(
                    query,
                    k,
                    v,
                    causal=True,
                    softmax_scale=sm_scale,
                )

            metadata = None
    elif is_decode or (query.shape[1] < (last_dense * 2)) or (last_dense <= 0):
        context, metadata = _forward_partial_fa3(
            q=query,
            k=k,
            v=v,
            sm_scale=sm_scale,
            rope_is_neox_style=rope_is_neox_style,
            cached_metadata=cached_metadata,
            is_decode=is_decode,
            seq_thresh_fa3=seq_thresh_fa3,
            mixing_len=mixing_len,
            args=args,
            max_context_len=max_batch_context_len,
            k_descale=k_descale,
            v_descale=v_descale,
            inner_function_do_scale=True,
            inner_function=dual_stage_quadratic_hip_attention,
        )
        # context = context[:, -query.shape[1] :, :, :].contiguous()
    else:
        assert not is_decode
        assert last_dense > 0
        assert query_for_mask is None
        position_ids = args.position_ids

        args_sparse = args.clone()
        args_sparse.position_ids = position_ids[:, :]
        context, metadata = dual_stage_quadratic_hip_attention(
            (query[:, :, :, :] * sm_scale).to(query.dtype),
            k,
            v,
            args=args_sparse,
            cached_metadata=cached_metadata,
        )
        context_sparse = context.to(query.dtype)

        args_dense = args.clone()
        args_dense.sliding_window_size = args_dense.model_context_length // 4
        args_dense.position_ids = position_ids[:, -last_dense:]
        args_dense.second_stage_k *= 2
        args_dense.sink_token_size *= 2
        if args_dense.q_mask is not None:
            args_dense.q_mask = args_dense.q_mask[:, -last_dense:, :, :]
        # print(
        #     query.shape,
        #     k.shape if k is not None else None,
        #     v.shape if v is not None else None,
        #     args_dense.sliding_window_size,
        #     args_dense.sink_token_size,
        #     args_dense.second_stage_k
        # )
        last_block = triton.cdiv(last_dense, args_dense.block_sparse_block_size_q)
        metadata.indices = metadata.indices[:, -last_block:]
        metadata.ks = metadata.ks[:, -last_block:]
        metadata.ks_count = metadata.ks_count[:, -last_block:]
        metadata.ks_start_end = metadata.ks_start_end[:, -last_block:]
        context_dense, metadata = dual_stage_quadratic_hip_attention(
            (query[:, -last_dense:, :, :] * sm_scale).to(query.dtype),
            k,
            v,
            args=args_dense,
            cached_metadata=metadata,
        )
        context_dense = context_dense.to(query.dtype)

        context = torch.cat([context_sparse[:, :-last_dense], context_dense], dim=1)
        context = context[:, -query.shape[1] :, :, :].contiguous()

    layers_to_capture = [0, 1, 2, 3, 4, 8, 12, 16, 24, 31]
    NEED_CHECKOUT = os.getenv("HIP_DEBUG_NEED_CHECKOUT", "0") == "1"
    if (
        NEED_CHECKOUT
        and (get_tensor_model_parallel_rank() == 0)
        and (layer_id in layers_to_capture)
    ):
        root = "./saves/sglang_decode"
        if not os.path.exists(root):
            _CHECKOUT_COUNTER = 0
        filename = f"{root}/checkout_sample_{_CHECKOUT_COUNTER}_layer_{layer_id}_is_decode_{1 if is_decode else 0}.pth"
        os.makedirs(root, exist_ok=True)

        if is_decode or (
            (not is_decode)
            and (dst_seq_len not in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
        ):
            torch.save(
                {
                    "q": query,
                    "sm_scale": sm_scale,
                    "k": (
                        k
                        if k is not None
                        else args.gather_k_from_paged_cache(chunk_size=1)
                    ),
                    "v": (
                        v
                        if k is not None
                        else args.gather_v_from_paged_cache(chunk_size=1)
                    ),
                    "block_table": block_table,
                    "cos": rope_cos,
                    "sin": rope_sin,
                    "out": context,
                    "metadata": metadata,
                },
                filename,
            )
            if is_decode and (layer_id == max(layers_to_capture)):
                _CHECKOUT_COUNTER += 1
            print(f"saved {filename}")

    assert context.dtype == query.dtype
    return context.view(N, num_heads, context.shape[-1]), metadata


class PagedHiPStateful:
    def __init__(self):
        # print('stateful init')
        self.states = dict()

    def __call__(
        self,
        **kwargs,
    ):
        layer_id = kwargs.get("layer_id", None)
        is_decode = kwargs.get("is_decode", False)
        state = self.states.get(layer_id, None)

        cached_metadata = kwargs.pop("cached_metadata", None)
        if cached_metadata is None:
            cached_metadata = HiPAttentionOutputMetadata(
                indices=None,
                ks=None,
                ks_count=None,
                ks_start_end=None,
                mask_cache_statistics=None,
                sa_cache_statistics=None,
                stage_caches=None,
                state=None,
            )

        assert isinstance(cached_metadata, HiPAttentionOutputMetadata)
        cached_metadata.state = state

        o, metadata = forward_paged_hip(
            **kwargs,
            cached_metadata=cached_metadata,
        )

        if not is_decode:
            states = None
            if metadata is not None:
                if isinstance(metadata, list):
                    if (not any(map(lambda x: x is None, metadata))) and (
                        not any(map(lambda x: x.state is None, metadata))
                    ):
                        states = [m.state for m in metadata]
                else:
                    if metadata.state is not None:
                        states = metadata.state
            if states is not None:
                self.states[layer_id] = states

        return o, metadata

```

#### `hip_attn/v1_2/query_sparse_attention.py`

```py
"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Credits: OpenAI kernel team

Extra Credits:

* Original flash attention paper (https://arxiv.org/abs/2205.14135)
* Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import os
import warnings
from typing import Callable, Tuple, Union

import numpy as np

# import pytest
import torch
import triton
import triton.language as tl
import triton.tools.experimental_descriptor

from hip_attn.v1_2.attention_metadata import safe_stride
from hip_attn.v1_2.utils import capture

# DEVICE = triton.runtime.driver.active.get_active_torch_device()
DEVICE = "cuda:0"


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    K_block_ptr,
    V_block_ptr,
    mask_idx,
    start_m,
    qk_scale,
    k_descale,
    v_descale,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    N_CTX: tl.constexpr,
    N_KV: tl.constexpr,
    fp8_v: tl.constexpr,
    USING_PAGED_CACHE: tl.constexpr,
    K_CACHE,
    stride_k_cache_t,
    stride_k_cache_page,
    stride_k_cache_hid,
    V_CACHE,
    stride_v_cache_t,
    stride_v_cache_page,
    stride_v_cache_hid,
    BLOCK_TABLE,
    stride_block_table_tsrc,
    lo,
    hi,
    MASKING: tl.constexpr,
):
    # range of values handled by this stage
    # lo, hi = 0, N_KV
    # lo, hi = 0, tl.max(mask_idx) + 1

    if not USING_PAGED_CACHE:
        K_block_ptr = tl.advance(K_block_ptr, (0, lo))
        V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    else:
        idx_hid = tl.arange(0, HEAD_DIM)
        # idx_tsrc = tl.arange(0, BLOCK_N) + lo
        # mask_tsrc = idx_tsrc < hi

    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N, num_stages=1):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        if not USING_PAGED_CACHE:
            k = tl.load(
                K_block_ptr,
                boundary_check=(1,),
                padding_option="zero"
            )
        else:
            idx_tsrc = tl.arange(0, BLOCK_N) + start_n
            mask_tsrc = idx_tsrc < hi

            idx_t = tl.load(
                BLOCK_TABLE + idx_tsrc.to(tl.int64) * stride_block_table_tsrc,
                mask=mask_tsrc,
            ).to(tl.int64)
            k = tl.load(
                K_CACHE
                + idx_t[None, :] * stride_k_cache_t
                + 0 * stride_k_cache_page
                + idx_hid[:, None] * stride_k_cache_hid,
                mask=mask_tsrc[None, :],
                other=0.0,
            )

        if k_descale is not None:
            k *= k_descale

        # qk = tl.dot(q, k)

        q_dtype = q.dtype

        cq = tl.sqrt(HEAD_DIM * 1.0) / tl.sqrt(tl.sqrt(HEAD_DIM * 1.0))
        ck = 1 / tl.sqrt(tl.sqrt(HEAD_DIM * 1.0))

        qk = tl.dot(
            (q * cq).to(q_dtype),
            (k.to(q_dtype) * ck).to(q_dtype),
            out_dtype=tl.float32,
            allow_tf32=True,
        ).to(tl.float32)

        qk = qk * 1.44269504

        if MASKING:
            mask = (mask_idx[:, None]) >= (start_n + offs_n[None, :])
            qk = tl.where(mask, qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]

        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = (l_i * alpha + l_ij).to(l_i.dtype)
        # -- update output accumulator --
        acc = acc * alpha.to(acc.dtype)[:, None]
        # update acc
        if not USING_PAGED_CACHE:
            v = tl.load(
                V_block_ptr,
                boundary_check=(0,),
                padding_option="zero",
            )
        else:
            v = tl.load(
                V_CACHE
                + idx_t[:, None] * stride_v_cache_t
                + 0 * stride_v_cache_page
                + idx_hid[None, :] * stride_v_cache_hid,
                mask=mask_tsrc[:, None],
                other=0.0,
            )

        if v_descale is not None:
            v *= v_descale

        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(v.dtype)

        acc = acc + tl.dot(
            p.to(q_dtype),
            v.to(q_dtype),
            out_dtype=tl.float32,
            allow_tf32=True,
        )
        # update m_i and l_i
        m_i = m_ij
        if not USING_PAGED_CACHE:
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        else:
            # idx_tsrc = idx_tsrc + BLOCK_N
            # mask_tsrc = idx_tsrc < hi
            pass
    return acc, l_i, m_i


# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
if os.getenv("HIP_DISABLE_AUTOTUNE", "0") == "1":
    configs = [
        triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
        for BM in [
            128,
        ]
        for BN in [
            64,
        ]
        for s in [
            3,
        ]
        for w in [
            4,
        ]
    ]
else:
    configs = [
        triton.Config({"BLOCK_M": BM, "BLOCK_N": BN}, num_stages=s, num_warps=w)
        for BM in [64, 128]
        for BN in [32, 64]
        for s in ([1] if is_hip() else [3, 4, 7])
        for w in [4, 8]
        # for BM in [128,]
        # for BN in [64,]
        # for s in [3, ]
        # for w in [4, ]
    ]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


@triton.autotune(list(filter(keep, configs)), key=[
    # "N_CTX",
    # "N_KV",
    "HEAD_DIM",
    "USING_PAGED_CACHE",
])
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    K_DESCALE,
    V_DESCALE,
    sm_scale,
    M,
    MX,
    NC,
    Out,  #
    MaskIdx,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,  #

    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,  #

    stride_vz,
    stride_vh,
    stride_vk,
    stride_vn,  #

    stride_oz,
    stride_oh,
    stride_om,
    stride_on,  #

    stride_mz,
    stride_mm,
    USING_PAGED_CACHE: tl.constexpr,
    HEAD_REPEAT: tl.constexpr,
    K_CACHE,
    stride_k_cache_t,
    stride_k_cache_page,
    stride_k_cache_head_kv,
    stride_k_cache_hid,
    V_CACHE,
    stride_v_cache_t,
    stride_v_cache_page,
    stride_v_cache_head_kv,
    stride_v_cache_hid,
    BLOCK_TABLE,
    stride_block_table_bsz,
    stride_block_table_tsrc,
    RETURN_POOLED_SCORES: tl.constexpr,
    SCORE_POOLING_BQ: tl.constexpr,
    SCORE_POOLING_BK: tl.constexpr,
    SCORES,
    stride_scores_bsz,
    stride_scores_head,
    stride_scores_bdst,
    stride_scores_bsrc,
    ACC,
    stride_acc_bsz,
    stride_acc_head,
    stride_acc_split,
    stride_acc_tdst,
    stride_acc_hid,
    MI,
    stride_mi_bsz,
    stride_mi_head,
    stride_mi_split,
    strdie_mi_tdst,
    LI,
    stride_li_bsz,
    stride_li_head,
    stride_li_split,
    stride_li_tdst,
    Z,
    H,
    N_CTX,
    N_KV,
    HEAD_DIM: tl.constexpr,
    N_SPLIT,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    V_FP8: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1).to(tl.int64)
    off_z = off_hz // H
    off_h = off_hz % H
    q_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    kv_offset = off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh

    idx_split = tl.program_id(2).to(tl.int64)

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    if not USING_PAGED_CACHE:
        v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
        V_block_ptr = tl.make_block_ptr(
            base=V + kv_offset,
            shape=(N_KV, HEAD_DIM),
            strides=(stride_vk, stride_vn),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=v_order,
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + kv_offset,
            shape=(HEAD_DIM, N_KV),
            strides=(stride_kk, stride_kn),
            offsets=(0, 0),
            block_shape=(HEAD_DIM, BLOCK_N),
            order=(0, 1),
        )
    else:
        K_CACHE = K_CACHE + (off_h.to(tl.int64) // HEAD_REPEAT) * stride_k_cache_head_kv
        V_CACHE = V_CACHE + (off_h.to(tl.int64) // HEAD_REPEAT) * stride_v_cache_head_kv
        BLOCK_TABLE = BLOCK_TABLE + off_z.to(tl.int64) * stride_block_table_bsz
    O_block_ptr = tl.make_block_ptr(
        base=Out + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N_CTX
    offs_n = tl.arange(0, BLOCK_N)

    mask_idx = tl.load(
        MaskIdx + off_z.to(tl.int64) * stride_mz + offs_m.to(tl.int64) * stride_mm,
        mask=mask_m,
        other=0.0,
    )
    # initialize pointer to m and l
    m_i = tl.full([BLOCK_M], dtype=tl.float32, value=float("-inf"))
    l_i = tl.full([BLOCK_M], dtype=tl.float32, value=1.0)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)

    if K_DESCALE is not None:
        k_descale = tl.load(
            K_DESCALE + off_z * H + off_h
        )
        v_descale = tl.load(
            V_DESCALE + off_z * H + off_h
        )
    else:
        k_descale = None
        v_descale = None

    # load q: it will stay in SRAM throughout
    q = tl.load(
        Q_block_ptr,
        boundary_check=(0,),
        padding_option="zero",
    )

    if not USING_PAGED_CACHE:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            K_block_ptr,
            V_block_ptr,
            mask_idx,
            start_m,
            qk_scale,
            k_descale,
            v_descale,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            offs_m,
            offs_n,
            N_CTX,
            N_KV,
            V_FP8,
        )
    else:
        lo = 0
        mid = tl.min(tl.where(mask_m, mask_idx, 987654321)) // BLOCK_N * BLOCK_N
        hi = tl.max(mask_idx) + 1

        if N_SPLIT > 1:
            k_chunk_size = tl.cdiv(hi, N_SPLIT)
            start_k = k_chunk_size * idx_split
            end_k = k_chunk_size * (idx_split + 1)

            # (start_k, end_k) (lo, mid)
            if tl.maximum(start_k, lo) < tl.minimum(end_k, mid):
                acc, l_i, m_i = _attn_fwd_inner(
                    acc,
                    l_i,
                    m_i,
                    q,
                    None,
                    None,
                    mask_idx,
                    start_m,
                    qk_scale,
                    k_descale,
                    v_descale,
                    BLOCK_M,
                    HEAD_DIM,
                    BLOCK_N,
                    offs_m,
                    offs_n,
                    N_CTX,
                    N_KV,
                    V_FP8,
                    USING_PAGED_CACHE=USING_PAGED_CACHE,
                    K_CACHE=K_CACHE,
                    stride_k_cache_t=stride_k_cache_t,
                    stride_k_cache_page=stride_k_cache_page,
                    stride_k_cache_hid=stride_k_cache_hid,
                    V_CACHE=V_CACHE,
                    stride_v_cache_t=stride_v_cache_t,
                    stride_v_cache_page=stride_v_cache_page,
                    stride_v_cache_hid=stride_v_cache_hid,
                    BLOCK_TABLE=BLOCK_TABLE,
                    stride_block_table_tsrc=stride_block_table_tsrc,
                    lo=tl.maximum(start_k, lo),
                    hi=tl.minimum(end_k, mid),
                    MASKING=False,
                )
            # (start_k, end_k) (mid, hi)
            if tl.maximum(start_k, mid) < tl.minimum(end_k, hi):
                acc, l_i, m_i = _attn_fwd_inner(
                    acc,
                    l_i,
                    m_i,
                    q,
                    None,
                    None,
                    mask_idx,
                    start_m,
                    qk_scale,
                    k_descale,
                    v_descale,
                    BLOCK_M,
                    HEAD_DIM,
                    BLOCK_N,
                    offs_m,
                    offs_n,
                    N_CTX,
                    N_KV,
                    V_FP8,
                    USING_PAGED_CACHE=USING_PAGED_CACHE,
                    K_CACHE=K_CACHE,
                    stride_k_cache_t=stride_k_cache_t,
                    stride_k_cache_page=stride_k_cache_page,
                    stride_k_cache_hid=stride_k_cache_hid,
                    V_CACHE=V_CACHE,
                    stride_v_cache_t=stride_v_cache_t,
                    stride_v_cache_page=stride_v_cache_page,
                    stride_v_cache_hid=stride_v_cache_hid,
                    BLOCK_TABLE=BLOCK_TABLE,
                    stride_block_table_tsrc=stride_block_table_tsrc,
                    lo=tl.maximum(start_k, mid),
                    hi=tl.minimum(end_k, hi),
                    MASKING=True,
                )
        else:
            acc, l_i, m_i = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                None,
                None,
                mask_idx,
                start_m,
                qk_scale,
                k_descale,
                v_descale,
                BLOCK_M,
                HEAD_DIM,
                BLOCK_N,
                offs_m,
                offs_n,
                N_CTX,
                N_KV,
                V_FP8,
                USING_PAGED_CACHE=USING_PAGED_CACHE,
                K_CACHE=K_CACHE,
                stride_k_cache_t=stride_k_cache_t,
                stride_k_cache_page=stride_k_cache_page,
                stride_k_cache_hid=stride_k_cache_hid,
                V_CACHE=V_CACHE,
                stride_v_cache_t=stride_v_cache_t,
                stride_v_cache_page=stride_v_cache_page,
                stride_v_cache_hid=stride_v_cache_hid,
                BLOCK_TABLE=BLOCK_TABLE,
                stride_block_table_tsrc=stride_block_table_tsrc,
                lo=lo,
                hi=mid,
                MASKING=False,
            )

            acc, l_i, m_i = _attn_fwd_inner(
                acc,
                l_i,
                m_i,
                q,
                None,
                None,
                mask_idx,
                start_m,
                qk_scale,
                k_descale,
                v_descale,
                BLOCK_M,
                HEAD_DIM,
                BLOCK_N,
                offs_m,
                offs_n,
                N_CTX,
                N_KV,
                V_FP8,
                USING_PAGED_CACHE=USING_PAGED_CACHE,
                K_CACHE=K_CACHE,
                stride_k_cache_t=stride_k_cache_t,
                stride_k_cache_page=stride_k_cache_page,
                stride_k_cache_hid=stride_k_cache_hid,
                V_CACHE=V_CACHE,
                stride_v_cache_t=stride_v_cache_t,
                stride_v_cache_page=stride_v_cache_page,
                stride_v_cache_hid=stride_v_cache_hid,
                BLOCK_TABLE=BLOCK_TABLE,
                stride_block_table_tsrc=stride_block_table_tsrc,
                lo=mid,
                hi=hi,
                MASKING=True,
            )

    # epilogue
    if N_SPLIT > 1:
        # checkout acc, l_i, m_i
        tl.store(
            ACC
            + off_z.to(tl.int64) * stride_acc_bsz
            + off_h.to(tl.int64) * stride_acc_head
            + idx_split.to(tl.int64) * stride_acc_split
            + offs_m.to(tl.int64)[:, None] * stride_acc_tdst
            + tl.arange(0, HEAD_DIM).to(tl.int64)[None, :] * stride_acc_hid,
            mask=mask_m[:, None],
            value=acc,
        )
        tl.store(
            MI
            + off_z.to(tl.int64) * stride_mi_bsz
            + off_h.to(tl.int64) * stride_mi_head
            + idx_split.to(tl.int64) * stride_mi_split
            + offs_m.to(tl.int64) * strdie_mi_tdst,
            mask=mask_m,
            value=m_i,
        )
        tl.store(
            LI
            + off_z.to(tl.int64) * stride_li_bsz
            + off_h.to(tl.int64) * stride_li_head
            + idx_split.to(tl.int64) * stride_li_split
            + offs_m.to(tl.int64) * stride_li_tdst,
            mask=mask_m,
            value=l_i,
        )

    if N_SPLIT <= 1:
        if MX is not None:
            m_ptrs = MX + off_hz * N_CTX + offs_m
            tl.store(m_ptrs, m_i, mask=mask_m)

        if NC is not None:
            l_ptrs = NC + off_hz * N_CTX + offs_m
            tl.store(l_ptrs, l_i, mask=mask_m)

        if M is not None:
            m_i += tl.math.log2(l_i)
            m_ptrs = M + off_hz * N_CTX + offs_m
            tl.store(m_ptrs, m_i, mask=mask_m)

        acc = acc / l_i[:, None]
        tl.store(
            O_block_ptr,
            acc.to(Out.type.element_ty),
            boundary_check=(0,),
        )
    else:
        tl.static_assert(M is None)
        tl.static_assert(MX is None)
        tl.static_assert(NC is None)


@triton.jit
def _attn_merge(
    O,
    stride_o_bsz,
    stride_o_head,
    stride_o_tdst,
    stride_o_hid,
    ACC,
    stride_acc_bsz,
    stride_acc_head,
    stride_acc_split,
    stride_acc_tdst,
    stride_acc_hid,
    MI,
    stride_mi_bsz,
    stride_mi_head,
    stride_mi_split,
    stride_mi_tdst,
    LI,
    stride_li_bsz,
    stride_li_head,
    stride_li_split,
    stride_li_tdst,
    TDST,
    HEAD,
    HID: tl.constexpr,
    N_SPLIT,
    BLOCK_TDST: tl.constexpr,
):
    idx_tdst_start = tl.program_id(0).to(tl.int64) * BLOCK_TDST
    idx_tdst = tl.arange(0, BLOCK_TDST) + idx_tdst_start
    mask_tdst = idx_tdst < TDST
    idx_bsz_head = tl.program_id(1).to(tl.int64)
    idx_bsz = idx_bsz_head // HEAD
    idx_head = idx_bsz_head % HEAD
    idx_hid = tl.arange(0, HID)

    ACC = ACC + idx_bsz * stride_acc_bsz + idx_head * stride_acc_head
    MI = MI + idx_bsz * stride_mi_bsz + idx_head * stride_mi_head
    LI = LI + idx_bsz * stride_li_bsz + idx_head * stride_li_head

    m_i = tl.full([BLOCK_TDST], dtype=tl.float32, value=float("-inf"))
    l_i = tl.zeros([BLOCK_TDST], dtype=tl.float32)
    acc = tl.zeros([BLOCK_TDST, HID], dtype=tl.float32)

    for idx_split in range(N_SPLIT):
        m_split = tl.load(
            MI + idx_split * stride_mi_split + idx_tdst * stride_mi_tdst,
            mask=mask_tdst,
        )
        l_split = tl.load(
            LI + idx_split * stride_li_split + idx_tdst * stride_li_tdst,
            mask=mask_tdst,
        )
        acc_split = tl.load(
            ACC
            + idx_split * stride_acc_split
            + idx_tdst[:, None] * stride_acc_tdst
            + idx_hid[None, :] * stride_acc_hid,
            mask=mask_tdst[:, None],
        )

        tv = acc_split / l_split[:, None]
        tlogic = m_split + tl.math.log2(l_split)

        n_e_max = tl.maximum(tlogic, m_i)

        old_scale = tl.math.exp2(m_i - n_e_max)
        exp_logic = tl.math.exp2(tlogic - n_e_max)
        acc = acc * old_scale[:, None] + exp_logic[:, None] * tv

        l_i = l_i * old_scale + exp_logic
        m_i = n_e_max

    acc = acc / l_i[:, None]

    tl.store(
        O
        + idx_bsz * stride_o_bsz
        + idx_head * stride_o_head
        + idx_tdst[:, None] * stride_o_tdst
        + idx_hid[None, :] * stride_o_hid,
        value=acc.to(O.type.element_ty),
        mask=mask_tdst[:, None],
    )


# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.


class _attention(torch.autograd.Function):

    @capture
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_descale: torch.Tensor,
        v_descale: torch.Tensor,
        mask: torch.Tensor,
        sm_scale: float,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_table: torch.Tensor,
        return_running_statistics: bool,
        return_pooled_scores: bool,
        score_pooling_block_size_q: int,
        score_pooling_block_size_k: int,
        score_pooling_max_seq_len: int,
    ):
        q = (q * sm_scale).to(q.dtype)

        USING_PAGED_CACHE = k_cache is not None
        if not USING_PAGED_CACHE:
            HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        else:
            HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k_cache.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        if not USING_PAGED_CACHE:
            HEAD_DIM_V = v.shape[-1]
        else:
            HEAD_DIM_V = v_cache.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 1
        extra_kern_args = {}
        # Tuning fo
        # r AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        N_CTX = q.shape[2]
        N_HEAD = q.shape[1]
        N_BATCH = q.shape[0]
        V_FP8 = (
            v.dtype == torch.float8_e5m2
            if not USING_PAGED_CACHE
            else v_cache.dtype == torch.float8_e5m2
        )

        # NOTE: this is for backward
        # M = torch.empty(
        #     (q.shape[0], q.shape[1], q.shape[2]),
        #     device=q.device,
        #     dtype=torch.float32,
        # )
        NC = MX = M = None
        if return_running_statistics:
            MX = torch.empty(
                (q.shape[0], q.shape[1], q.shape[2]),
                device=q.device,
                dtype=torch.float32,
            )
            NC = torch.empty(
                (q.shape[0], q.shape[1], q.shape[2]),
                device=q.device,
                dtype=torch.float32,
            )

        if return_pooled_scores:
            warnings.warn(
                "Pooled score should not be returned for efficient inference."
            )

            if k is not None:
                MAX_TSRC = k.shape[2]
            else:
                assert score_pooling_max_seq_len is not None
                MAX_TSRC = score_pooling_max_seq_len

            scores = torch.full(
                (
                    q.shape[0],
                    q.shape[1],
                    triton.cdiv(q.shape[2], score_pooling_block_size_q),
                    triton.cdiv(MAX_TSRC, score_pooling_block_size_k),
                ),
                fill_value=float("-inf"),
                dtype=torch.float32,
                device=q.shape,
            )
        else:
            scores = None

        assert (
            q.shape[1] <= 128
        )  # N HEAD should be smaller than 128. this could be adjusted.
        assert len(mask.size()) == 2, "expecting mask to be 2D"

        N_CTX_BLOCK = 128
        N_PROGRAM = triton.cdiv(N_CTX, N_CTX_BLOCK) * N_HEAD * N_BATCH
        N_SM = 256  # TODO make a good solution to get this without init CUDA context on GPU 0
        N_SPLIT = triton.cdiv(N_SM, N_PROGRAM)
        if return_running_statistics:
            if N_SPLIT > 1:
                warnings.warn("N_SPLIT is ignored. this should be fixed")
            N_SPLIT = 1

        if (N_SPLIT > 1) and (os.getenv("HIP_DEBUG_RECOMPUTE_SPLIT", "1") == "1"):
            # N_SPLIT = 1

            grid = lambda args: (
                triton.cdiv(N_CTX, args["BLOCK_M"]),
                N_BATCH * N_HEAD,
                N_SPLIT,
            )

            acc = torch.zeros(
                (N_BATCH, N_HEAD, N_SPLIT, N_CTX, HEAD_DIM_V),
                dtype=torch.float32,
                device=q.device,
            )
            m_i = torch.zeros(
                (N_BATCH, N_HEAD, N_SPLIT, N_CTX), dtype=torch.float32, device=q.device
            )
            l_i = torch.zeros(
                (N_BATCH, N_HEAD, N_SPLIT, N_CTX), dtype=torch.float32, device=q.device
            )

            _attn_fwd[grid](
                q,
                k,
                v,
                k_descale,
                v_descale,
                sm_scale,
                M,
                MX,
                NC,
                o,
                mask,
                *safe_stride(q, 4),
                *safe_stride(k, 4),
                *safe_stride(v, 4),
                *safe_stride(o, 4),
                *safe_stride(mask, 2),
                k_cache is not None,
                (
                    q.shape[1] // k_cache.shape[2]
                    if k_cache is not None
                    else q.shape[1] // k.shape[1]
                ),
                k_cache,
                *safe_stride(k_cache, 4),
                v_cache,
                *safe_stride(v_cache, 4),
                block_table,
                *safe_stride(block_table, 2),
                return_pooled_scores,
                score_pooling_block_size_q,
                score_pooling_block_size_k,
                scores,
                *safe_stride(scores, 4),
                acc,
                *safe_stride(acc, 5),
                m_i,
                *safe_stride(m_i, 4),
                l_i,
                *safe_stride(l_i, 4),
                q.shape[0],
                q.shape[1],
                N_CTX=N_CTX,
                N_KV=(
                    k.shape[2]
                    if not USING_PAGED_CACHE
                    else k_cache.shape[0] * k_cache.shape[1]
                ),
                HEAD_DIM=HEAD_DIM_K,
                N_SPLIT=N_SPLIT,
                V_FP8=V_FP8,
                **extra_kern_args,
            )

            BLOCK_M = 128
            grid = (
                triton.cdiv(N_CTX, BLOCK_M),
                N_BATCH * N_HEAD,
                1,
            )

            _attn_merge[grid](
                o,
                *safe_stride(o, 4),
                acc,
                *safe_stride(acc, 5),
                m_i,
                *safe_stride(m_i, 4),
                l_i,
                *safe_stride(l_i, 4),
                TDST=N_CTX,
                HEAD=N_HEAD,
                HID=HEAD_DIM_V,
                N_SPLIT=N_SPLIT,
                BLOCK_TDST=BLOCK_M,
            )

            # def sanity_check(t: torch.Tensor):
            #     assert t.isnan().nonzero().shape[0] == 0
            #     assert t.isinf().nonzero().shape[0] == 0
            #     return t

            # l_i = sanity_check(l_i)
            # m_i = sanity_check(m_i)
            # acc = sanity_check(acc)

            # # l_i = torch.where(l_i <= (1.0 + 1e-4), l_i + 1e-4, l_i)

            # logits = acc / l_i[:, :, :, :, None]
            # logits = sanity_check(logits)
            # stats = m_i + torch.log2(l_i)
            # stats = sanity_check(stats)

            # e_sum = torch.zeros_like(l_i[:, :, 0, :].contiguous())
            # e_max = torch.full_like(m_i[:, :, 0, :].contiguous(), fill_value=float('-inf'))
            # acc = torch.zeros_like(o, dtype=torch.float32)

            # for i_split in range(N_SPLIT):
            #     tv = logits[:, :, i_split, :, :]
            #     tv = sanity_check(tv)
            #     tlogic = stats[:, :, i_split, :]
            #     tlogic = sanity_check(tlogic)
            #     n_e_max = torch.maximum(tlogic, e_max)
            #     n_e_max = sanity_check(n_e_max)

            #     old_scale = torch.exp2(e_max - n_e_max)
            #     old_scale = sanity_check(old_scale)
            #     exp_logic = torch.exp2(tlogic - n_e_max)
            #     exp_logic = sanity_check(exp_logic)
            #     acc = acc * old_scale[:, :, :, None] + exp_logic[:, :, :, None] * tv
            #     acc = sanity_check(acc)

            #     e_sum = e_sum * old_scale + exp_logic
            #     e_sum = sanity_check(e_sum)
            #     e_max = n_e_max
            #     e_max = sanity_check(e_max)

            # acc = acc / e_sum[:, :, :, None]
            # acc = sanity_check(acc)

            # o = acc.to(o.dtype)
        else:
            grid = lambda args: (
                triton.cdiv(N_CTX, args["BLOCK_M"]),
                N_BATCH * N_HEAD,
                1,
            )

            _attn_fwd[grid](
                q,
                k,
                v,
                k_descale,
                v_descale,
                sm_scale,
                M,
                MX,
                NC,
                o,  #
                mask,
                *safe_stride(q, 4),
                *safe_stride(k, 4),
                *safe_stride(v, 4),
                *safe_stride(o, 4),
                *safe_stride(mask, 2),
                k_cache is not None,
                (
                    q.shape[1] // k_cache.shape[2]
                    if k_cache is not None
                    else q.shape[1] // k.shape[1]
                ),
                k_cache,
                *safe_stride(k_cache, 4),
                v_cache,
                *safe_stride(v_cache, 4),
                block_table,
                *safe_stride(block_table, 2),
                # acc, m_i, l_i
                None,
                *safe_stride(None, 5),
                None,
                *safe_stride(None, 4),
                None,
                *safe_stride(None, 4),
                return_pooled_scores,
                score_pooling_block_size_q,
                score_pooling_block_size_k,
                scores,
                *safe_stride(scores, 4),
                q.shape[0],
                q.shape[1],  #
                N_CTX=N_CTX,  #
                N_KV=(
                    k.shape[2]
                    if not USING_PAGED_CACHE
                    else k_cache.shape[0] * k_cache.shape[1]
                ),
                HEAD_DIM=HEAD_DIM_K,  #
                N_SPLIT=1,
                V_FP8=V_FP8,
                **extra_kern_args,
            )

        if return_running_statistics:
            return o, (MX, NC)
        else:
            return o

    @staticmethod
    def backward(ctx, do):
        raise NotImplementedError("bwd not implemented for recompute kernel")


# for typing wrapper and provide kwargs
def query_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    sm_scale: float,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    return_running_statistics: bool = False,
    return_pooled_scores: bool = False,
    score_pooling_block_size_q: int = 64,
    score_pooling_block_size_k: int = 64,
    score_pooling_max_seq_len: int = None,
    k_descale: torch.Tensor = None,
    v_descale: torch.Tensor = None,
) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    return _attention.apply(
        q,
        k,
        v,
        k_descale,
        v_descale,
        mask,
        sm_scale,
        k_cache,
        v_cache,
        block_table,
        return_running_statistics,
        return_pooled_scores,
        score_pooling_block_size_q,
        score_pooling_block_size_k,
        score_pooling_max_seq_len,
    )

```

#### `hip_attn/v1_2/utils.py`

```py
import dataclasses
import os
from typing import List, Optional

import torch

try:
    from sglang.srt.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        model_parallel_is_initialized,
        tensor_model_parallel_all_gather,
    )

    SGLANG_DIST_AVAILABLE = True

except:
    SGLANG_DIST_AVAILABLE = False


def get_local_rank():
    if SGLANG_DIST_AVAILABLE:
        return (
            get_tensor_model_parallel_rank() if model_parallel_is_initialized() else 0
        )
    else:
        return 0


@dataclasses.dataclass
class CaptureEvents:
    start: torch.cuda.Event
    end: torch.cuda.Event
    handle: "capture"
    _elapsed: Optional[int] = None

    def elapsed(self):
        if self._elapsed is not None:
            return self._elapsed
        else:
            self.end.synchronize()
            self._elapsed = self.start.elapsed_time(self.end)
            return self._elapsed


class capture(object):
    buffers: List[CaptureEvents] = []
    call_depth: int = 0

    @classmethod
    def report(cls):
        last_elapsed_sum = {}
        last_depth = 0
        for depth, event in capture.buffers:
            if depth < last_depth:
                print(
                    "--" * last_depth,
                    f"[level {last_depth}] took {last_elapsed_sum.get(last_depth, 0)} ms",
                    sep="",
                )
                last_elapsed_sum[last_depth] = 0
            last_depth = depth

            elapsed = event.elapsed()

            if not depth in last_elapsed_sum:
                last_elapsed_sum[depth] = 0
            last_elapsed_sum[depth] += elapsed

            print(
                "--" * depth, f"{event.handle.callback} took {elapsed:.2f} ms", sep=""
            )

        if len(capture.buffers) > 0:
            allocated = torch.cuda.memory_allocated()
            print(f"{allocated / 1024 / 1024:.2f} MB allocated")

        capture.buffers.clear()

    @classmethod
    def add_event(cls, depth: int, event: CaptureEvents):
        capture.buffers.append((depth, event))
        while len(capture.buffers) > 32:
            capture.buffers.pop(0)

    def __init__(self, callback):
        self.callback = callback

    def __enter__(self):
        return self

    def __exit__(self, ex_typ, ex_val, traceback):
        return True

    def __call__(self, *args, **kwargs):
        run_benchmark = (
            (not torch.cuda.is_current_stream_capturing())
            and (kwargs["q"].shape[1] > 1 if "q" in kwargs else True)
            and os.getenv("HIP_DEBUG_BENCH", "0") == "1"
            and os.getenv("HIP_DEBUG_CAPTURE_DECORATOR", "1") == "1"
            and (get_local_rank() == 0)
        )

        if run_benchmark:
            start = torch.cuda.Event(True)
            end = torch.cuda.Event(True)

            start.record()

        my_call_depth = capture.call_depth
        capture.call_depth += 1
        ret = self.callback(*args, **kwargs)
        capture.call_depth -= 1

        if run_benchmark:
            end.record()

            capture.add_event(
                my_call_depth, CaptureEvents(handle=self, start=start, end=end)
            )

        return ret

```

#### `hip_attn/v1_2/uvm_gpu_cache.py`

```py
import math
import os
from typing import Optional, Tuple, Union

import cuda
import cuda.bindings.runtime
import torch
import tqdm
import triton
import triton.language as tl
from torch import Tensor

from hip_attn.v1_1.offload_runner.tensor_from_pointer import (
    alloc_managed_tensor,
    sizeof,
)
from hip_attn.v1_2.attention_metadata import (
    HiPAttentionCacheAccessStatistics,
    HiPAttentionOutputMetadata,
)

MAX_INT: tl.constexpr = tl.constexpr(90000000)
MAX_INT_ACQUIRED: tl.constexpr = tl.constexpr(90000001)


def format_size_bytes(tensor: Union[Tensor, Union[float, int]]) -> str:
    if isinstance(tensor, Tensor):
        byte_size = sizeof(tensor)
    elif isinstance(tensor, (int, float)):
        byte_size = tensor
    else:
        raise Exception()

    if byte_size < 1024:
        return f"{byte_size} B"
    elif byte_size < (1024**2):
        return f"{byte_size / 1024:.2f} KB"
    elif byte_size < (1024**3):
        return f"{byte_size / (1024 ** 2):.2f} MB"
    else:
        return f"{byte_size / (1024 ** 3):.2f} GB"


def debug_print(*args):
    print(f'[HiPOffloadKVPoolMHA] {" ".join(map(lambda x: str(x), args))}')


###############################################################################
#                               Data Structure
###############################################################################


def uvm_note_cpu(tensor: Tensor, prefetch: bool = False):
    cuda.bindings.runtime.cudaMemAdvise(
        tensor.data_ptr(),
        tensor.numel() * tensor.element_size(),
        cuda.bindings.runtime.cudaMemoryAdvise.cudaMemAdviseSetPreferredLocation,
        -1,
    )
    cuda.bindings.runtime.cudaMemAdvise(
        tensor.data_ptr(),
        tensor.numel() * tensor.element_size(),
        cuda.bindings.runtime.cudaMemoryAdvise.cudaMemAdviseSetAccessedBy,
        tensor.device.index,
    )
    if prefetch:
        cuda.bindings.runtime.cudaMemPrefetchAsync(
            tensor.data_ptr(), tensor.numel() * tensor.element_size(), -1, 0
        )


class UVMCache:
    bank_cpu: Tensor
    bank_gpu: Tensor
    metadata: Tensor

    def __init__(
        self,
        layer_id: int,
        max_token_size: int,
        head_num: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.layer_id = layer_id
        self.max_token_size = max_token_size
        self.head_num = head_num
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        if self.device.index is None:
            self.device = torch.get_default_device()

        self.bank_cpu, self.bank_gpu = self.alloc_uvm(
            [max_token_size, head_num, head_dim], dtype=self.dtype
        )

        # {
        #     Token Generation: uint32    # Increase one on every overwrite
        # }
        self.metadata = torch.full(
            [max_token_size, 1], dtype=torch.int32, device=device, fill_value=0
        )

        self.allocated_cpu_bytes = sizeof(self.bank_cpu)
        self.allocated_gpu_bytes = sizeof(self.metadata)

        # debug_print(f'UVMCache: bank={format_size_bytes(self.bank_cpu)}, metadata={format_size_bytes(self.metadata)}')

    def alloc_uvm(self, shape, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        t_gpu, t_cpu = alloc_managed_tensor(shape, dtype, self.device)

        uvm_note_cpu(t_gpu)
        t_cpu.fill_(0)

        return t_cpu, t_gpu

    def gather_cpu(self, table: Tensor, pin_memory=False) -> Tensor:
        assert table.ndim == 1
        assert table.device == self.bank_cpu.device

        # print('gather alloc', flush=True)
        t = torch.empty(
            (table.shape[0], self.bank_cpu.shape[1], self.bank_cpu.shape[2]),
            dtype=self.bank_cpu.dtype,
            device="cpu",
            pin_memory=pin_memory,
        )

        view_dtype = torch.uint16
        view_dtype_np = np.uint16
        if self.bank_cpu.dtype in [torch.float32]:
            view_dtype = torch.uint32
            view_dtype_np = np.uint32
        elif self.bank_cpu.dtype in [torch.float16, torch.bfloat16]:
            view_dtype = torch.uint16
            view_dtype_np = np.uint16
        elif self.bank_cpu.dtype in [torch.uint8, torch.float8_e5m2]:
            view_dtype = torch.uint8
            view_dtype_np = np.uint8
        else:
            raise Exception()

        # t = np.empty(
        #     (table.shape[0], self.bank_cpu.shape[1], self.bank_cpu.shape[2]),
        #     dtype=view_dtype_np,
        # )

        # print('gather pin', flush=True)
        # if pin_memory:
        #     t = t.pin_memory()

        # print('gather index_copy', flush=True)
        index_copy(
            self.bank_cpu.view(dtype=view_dtype).numpy(),
            t.view(dtype=view_dtype).numpy(),
            table.numpy(),
            num_thread=os.cpu_count(),
        )
        # print('gather done', flush=True)

        # t = torch.from_numpy(t).view(self.bank_cpu.dtype)
        # print('convert done', flush=True)

        return t


import numba
import numpy as np


@numba.njit(parallel=True)
def index_copy(
    src: np.ndarray, out: np.ndarray, table: np.ndarray, num_thread: int = 32
):
    chunk_size = math.ceil(table.shape[0] / num_thread)
    for ithread in numba.prange(num_thread):
        for i in range(chunk_size):
            t = chunk_size * ithread + i
            if t < table.shape[0]:
                out[t] = src[table[t]]


def pad_to_cacheline(nelem: int, dtype: torch.dtype):
    byte_size = 4
    if dtype in [torch.int32, torch.uint32, torch.float32]:
        byte_size = 4
    elif dtype in [torch.int64, torch.uint64, torch.float64]:
        byte_size = 8
    elif dtype in [torch.int16, torch.uint16, torch.bfloat16, torch.float16]:
        byte_size = 2
    else:
        raise Exception()

    assert nelem > 0

    # in bytes
    cacheline_size = 32

    step = max(1, cacheline_size // byte_size)
    return nelem if (nelem % step) == 0 else (nelem + step - (nelem % step))


class GPUCache:
    global_metadata: Tensor
    bank: Tensor
    metadata: Tensor
    table: Tensor

    def __init__(
        self,
        k_uvm: UVMCache,
        v_uvm: Optional[UVMCache],
        max_cache_token_size: int,
        online_cache_update: bool,
    ):
        self.k_uvm = k_uvm
        self.v_uvm = v_uvm
        self.head_num = self.k_uvm.head_num
        self.head_dim = self.k_uvm.head_dim
        self.dtype = self.k_uvm.dtype
        self.device = self.k_uvm.device
        self.kv_packed = self.v_uvm is not None
        if self.kv_packed:
            assert self.head_num == self.v_uvm.head_num
            self.head_dim += self.v_uvm.head_dim
        self.max_cache_token_size = max_cache_token_size
        self.max_uvm_token_size = self.k_uvm.max_token_size
        if self.kv_packed:
            assert self.max_uvm_token_size == self.v_uvm.max_token_size

        """
        [
            CachelinePadded {
                current_tick: int32,
            }
            CachelinePadded {
                random_seed: int32,
            }
        ]
        """
        self.global_metadata = torch.zeros(
            (2, pad_to_cacheline(1, torch.int32)), dtype=torch.int32, device=self.device
        )

        self.bank = torch.zeros(
            (self.max_cache_token_size, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )

        """
        CachelinePadded {
            [0] Back reference to table: int64,         # initial handshake, store token index of UVM bank
            [1] Reference to UVM Cache: int64,          # MAX_TOKEN, for token generation check
            [2] Token Generation of UVM Cache: int64,   # To check the version of cached token
            [3] Last accessed tick: int64,
            [4] Not accessed duration: int64,           # Increse one every step
            [5] Did taken in this kernel step: int64    # Reset to zero on every step
            [6] Token hash: int64                       # for debug
        }
        """
        self.metadata = torch.zeros(
            (self.max_cache_token_size, pad_to_cacheline(7, torch.int64)),
            dtype=torch.int64,
            device=self.device,
        )
        # self.metadata[:, 0].fill_(0)
        # self.metadata[:, 1].fill_(0)
        # self.metadata[:, 2].fill_(0)
        # self.metadata[:, 3].fill_(0)
        self.metadata[:, 4].fill_(1)
        # self.metadata[:, 5].fill_(0)

        # NOTE: this table is way too large to pad... sorry
        self.table = torch.full(
            (self.head_num, self.max_uvm_token_size, 1),
            dtype=torch.int32,
            device=self.device,
            fill_value=MAX_INT.value,
        )

        self.allocated_gpu_bytes = (
            sizeof(self.global_metadata)
            + sizeof(self.bank)
            + sizeof(self.metadata)
            + sizeof(self.table)
        )

        self.step = 0

        self.online_update_cache = online_cache_update

    def handle_cache_miss(
        self,
        metadata: HiPAttentionOutputMetadata,
        stats: HiPAttentionCacheAccessStatistics,
    ):
        # self._verify_cache()

        # NOTE: increase not accessed timer
        self.metadata[:, 4].add_(1)
        self.metadata[:, 5].fill_(0)

        # if id(stats) == id(metadata.mask_cache_statistics): return
        if stats is None:
            return
        if self.online_update_cache:
            return

        # NOTE: this function should be capturable.
        # NOTE: this function will called only when mask is updated

        uvm_page_count = self.k_uvm.bank_cpu.shape[0]
        gpu_page_count = self.bank.shape[0]

        assert stats.cache_miss_counter.shape[1:] == (
            self.head_num,
            uvm_page_count,
        ), f"{stats.cache_miss_counter.shape[1:]} == [{self.head_num}, {uvm_page_count}]"

        # update LRU recency
        # increase LRU step
        self.global_metadata[0, 0].add_(1)

        if stats.access_counter.shape[0] != 1:
            # assert False
            # NOTE: if paged attention, stats should be single batch.
            accessed = stats.access_counter.sum(0)
        else:
            accessed = stats.access_counter.squeeze(0)

        assert accessed.ndim == 2
        assert accessed.shape == (self.head_num, uvm_page_count)
        assert self.k_uvm.metadata.shape == (uvm_page_count, 1)
        assert self.global_metadata.shape == (
            2,
            pad_to_cacheline(1, self.global_metadata.dtype),
        )
        assert self.metadata.shape == (
            self.bank.shape[0],
            pad_to_cacheline(5, self.metadata.dtype),
        )
        assert self.table.shape == (self.head_num, uvm_page_count, 1)

        BLOCK_SIZE = 128
        grid = (self.head_num * triton.cdiv(uvm_page_count, BLOCK_SIZE),)
        update_recency[grid](
            accessed,
            *accessed.stride(),
            self.k_uvm.metadata,
            *self.k_uvm.metadata.stride(),
            self.global_metadata,
            *self.global_metadata.stride(),
            self.metadata,
            *self.metadata.stride(),
            self.table,
            *self.table.stride(),
            uvm_page_count,
            self.k_uvm.bank_cpu.shape[1],
            BLOCK_SIZE,
            num_warps=4,
        )
        self.step += 1

        # perform LRU
        assert gpu_page_count <= (
            uvm_page_count * self.head_num
        ), f"{gpu_page_count} <= {(uvm_page_count * self.head_num)}"

        # cache_miss = ((stats.cache_miss_counter > 0) * stats.access_counter).sum(0).view(-1)
        if stats.cache_miss_counter.shape[0] == 1:
            cache_miss = stats.cache_miss_counter.view(-1)
        else:
            cache_miss = (
                ((stats.cache_miss_counter > 0) * stats.access_counter).sum(0).view(-1)
            )
        put_mask = cache_miss > 0
        # put_priority_list = cache_miss.argsort(-1, descending=True, stable=False)
        # put_priority_list = put_priority_list[:gpu_page_count]
        put_priority_list = cache_miss.topk(
            k=gpu_page_count, dim=-1, largest=True, sorted=False
        ).indices
        put_mask = put_mask[put_priority_list]

        slot_recency = self.metadata[:, 3]
        evict_priority_list = slot_recency.argsort(
            dim=-1, descending=False, stable=False
        )

        self.write_cache(
            put_list=put_priority_list,
            put_mask=put_mask,
            evict_list=evict_priority_list,
        )

        # NOTE: for debug
        self.verify_cache(put_mask)

    def verify_cache(
        self,
        put_mask: Optional[Tensor] = None,
        force: bool = False,
        max_verification: int = 10000,
    ):
        if (os.getenv("DEBUG_ONLINE_VERIFY", "0") == "0") and (not force):
            return
        if self.k_uvm.layer_id != 3 and (not force):
            return

        torch.cuda.synchronize(device=self.table.device)
        table = self.table.cpu()
        metadata = self.metadata.cpu()
        bank = self.bank.cpu()
        uvm_metadata = self.k_uvm.metadata.cpu()
        uvm_k_bank = self.k_uvm.bank_cpu
        uvm_v_bank = self.v_uvm.bank_cpu if self.kv_packed else None

        total_table_hit = 0
        total_back_ref_hit = 0
        total_uvm_ref_hit = 0
        total_token_gen_hit = 0
        total_hash_hit = 0
        total_cache_hit = 0
        for idx_head in range(table.shape[0]):
            for idx_page in tqdm.tqdm(
                range(min(table.shape[1], max_verification)),
                dynamic_ncols=True,
                leave=False,
            ):
                target_slot = table[idx_head, idx_page].item()
                if target_slot < MAX_INT:
                    total_table_hit += 1
                    (
                        back_ref,
                        ref_to_uvm,
                        token_gen,
                        last_tick,
                        sleep_tick,
                        is_touched,
                        token_hash,
                    ) = metadata[target_slot, :7]
                    if (back_ref // table.shape[0]) == idx_page:
                        total_back_ref_hit += 1
                        if ref_to_uvm < MAX_INT:
                            total_uvm_ref_hit += 1
                            if uvm_metadata[ref_to_uvm, 0] == token_gen:
                                total_token_gen_hit += 1
                                gpu_value = bank[target_slot]
                                if not self.kv_packed:
                                    cpu_value = uvm_k_bank[idx_page, idx_head]
                                else:
                                    cpu_value = torch.cat(
                                        [
                                            uvm_k_bank[idx_page, idx_head],
                                            uvm_v_bank[idx_page, idx_head],
                                        ],
                                        dim=0,
                                    )
                                gpu_hash = (
                                    gpu_value.view(torch.uint16)
                                    .to(torch.uint32)
                                    .sum()
                                    .item()
                                )
                                cpu_hash = (
                                    cpu_value.view(torch.uint16)
                                    .to(torch.uint32)
                                    .sum()
                                    .item()
                                )
                                mse = ((gpu_value - cpu_value) ** 2).mean().item()
                                check_pass = (mse < 1e-4) or True
                                if not check_pass:
                                    error_location = (
                                        uvm_k_bank.view(torch.uint16)
                                        .to(torch.uint32)
                                        .sum(dim=-1)
                                        == token_hash
                                    ).nonzero()
                                    error_cpu_location = (
                                        uvm_k_bank.view(torch.uint16)
                                        .to(torch.uint32)
                                        .sum(dim=-1)
                                        == gpu_hash
                                    ).nonzero()
                                    error_msg = f"""
cache_hit={total_cache_hit}, token_gen={total_token_gen_hit}, ref_to_uvm={total_uvm_ref_hit}, back_ref={total_back_ref_hit}
GPU = {gpu_value} ({gpu_value.shape})
-----
UVM = {cpu_value} ({cpu_value.shape})
token_hash={token_hash}, gpu_hash={gpu_hash}, cpu_hash={cpu_hash}, token_found={error_location}, uvm_found={error_cpu_location}
head={idx_head}, page={idx_page}, slot={target_slot}, backref={back_ref}, uvmref={ref_to_uvm}, gen={token_gen}, is_touched={is_touched}, mse={mse}"""
                                    assert False, error_msg
                                total_cache_hit += 1
                                if gpu_hash == cpu_hash:
                                    total_hash_hit += 1

        if put_mask is not None:
            print("lastly put", put_mask.sum().item())
        tqdm.tqdm.write(
            f"verified kv_packed={self.kv_packed}, cache_hit={total_cache_hit}, token_gen={total_token_gen_hit}, ref_to_uvm={total_uvm_ref_hit}, back_ref={total_back_ref_hit}, hash_hit={total_hash_hit}"
        )

    def write_cache(
        self,
        put_list: Tensor,
        put_mask: Tensor,
        evict_list: Tensor,
    ):
        assert put_list.shape == put_mask.shape
        assert evict_list.shape == put_list.shape

        BLOCK_SIZE = 128

        qsize = put_list.shape[0]

        grid = (triton.cdiv(qsize, BLOCK_SIZE),)
        write_cache[grid](
            put_list,
            *put_list.stride(),
            put_mask,
            *put_mask.stride(),
            evict_list,
            *evict_list.stride(),
            self.bank,
            *self.bank.stride(),
            self.metadata,
            *self.metadata.stride(),
            self.table,
            *self.table.stride(),
            self.k_uvm.metadata,
            *self.k_uvm.metadata.stride(),
            self.k_uvm.bank_gpu,
            *self.k_uvm.bank_gpu.stride(),
            self.v_uvm.bank_gpu if self.kv_packed else None,
            *(self.v_uvm.bank_gpu.stride() if self.kv_packed else (0, 0, 0)),
            self.global_metadata,
            *self.global_metadata.stride(),
            qsize,
            self.k_uvm.bank_gpu.shape[0],
            self.k_uvm.bank_gpu.shape[1],
            self.kv_packed,
            BLOCK_SIZE,
            self.k_uvm.bank_gpu.shape[-1],
        )

    def on_set_kv_buffer(
        self,
        table: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        # NOTE: cache_k, and cache_v might not on GPU (during prefill)
        # if table is allocated to valid slots, copy tensors to bank
        # if slot is not valid, unlink the table
        # NOTE: but for temporary, just unlink table always

        assert table.device == self.device

        self.table[:, :, 0].index_fill_(
            dim=1, index=table.to(torch.int64), value=MAX_INT.value
        )

    def flush(self):
        self.table[:, :, 0].fill_(MAX_INT.value)


class HiPOffloadCache:
    def __init__(
        self,
        layer_id: int,
        max_token_size: int,
        max_mask_cache_token_size: int,
        max_sa_cache_token_size: int,
        head_num: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        online_cache_update: bool,
    ):
        self.k_uvm = UVMCache(
            layer_id=layer_id,
            max_token_size=max_token_size,
            head_num=head_num,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )

        self.v_uvm = UVMCache(
            layer_id=layer_id,
            max_token_size=max_token_size,
            head_num=head_num,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )

        self.mask_k_cache = GPUCache(
            k_uvm=self.k_uvm,
            v_uvm=None,
            max_cache_token_size=max_mask_cache_token_size,
            online_cache_update=online_cache_update,
        )

        self.sa_kv_cache = GPUCache(
            k_uvm=self.k_uvm,
            v_uvm=self.v_uvm,
            max_cache_token_size=max_sa_cache_token_size,
            online_cache_update=online_cache_update,
        )

    def get_page_count(self):
        assert self.k_uvm.bank_cpu.shape == self.v_uvm.bank_cpu.shape
        return self.k_uvm.bank_cpu.shape[0]

    def prefetch_prefix_kv_buffer(
        self,
        table: Tensor,
        device: torch.device,
        pad: int,
    ) -> Tuple[Tensor, Tensor]:
        if table.device != torch.device("cpu"):
            table = table.to("cpu", non_blocking=False)

        k = self.k_uvm.gather_cpu(table, pin_memory=True)
        v = self.v_uvm.gather_cpu(table, pin_memory=True)

        k = k.to(device, non_blocking=True).unsqueeze(0)
        v = v.to(device, non_blocking=True).unsqueeze(0)

        if pad > 0:
            k = torch.nn.functional.pad(
                k, pad=(0, 0, 0, 0, pad, 0), mode="constant", value=0
            ).to(k.dtype)
            v = torch.nn.functional.pad(
                v, pad=(0, 0, 0, 0, pad, 0), mode="constant", value=0
            ).to(v.dtype)

        return k, v

    def set_kv_buffer(
        self,
        table: torch.Tensor,
        table_gpu: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        cache_device = cache_k.device
        assert table.device == cache_device
        assert cache_v.device == cache_device

        if cache_device == torch.device("cpu"):
            # self.k_uvm.bank_cpu[table] = cache_k
            # self.v_uvm.bank_cpu[table] = cache_v
            if cache_k.dtype in [torch.float16, torch.bfloat16]:
                view_dtype = torch.uint16
            elif cache_k.dtype in [torch.float32]:
                view_dtype = torch.uint32
            elif cache_k.dtype in [torch.uint8, torch.float8_e5m2]:
                view_dtype = torch.uint8
            else:
                raise Exception(f"not supported dtype {cache_k.dtype}")

            set_kv_buffer_(
                self.k_uvm.bank_cpu.view(view_dtype).numpy(),
                cache_k.view(view_dtype).numpy(),
                table.numpy(),
                os.cpu_count(),
            )

            set_kv_buffer_(
                self.v_uvm.bank_cpu.view(view_dtype).numpy(),
                cache_v.view(view_dtype).numpy(),
                table.numpy(),
                os.cpu_count(),
            )
        else:
            assert cache_device == self.k_uvm.device
            self.k_uvm.bank_gpu[table] = cache_k
            self.v_uvm.bank_gpu[table] = cache_v

        self.mask_k_cache.on_set_kv_buffer(
            table=table_gpu,
            cache_k=cache_k,
            cache_v=cache_v,
        )
        self.sa_kv_cache.on_set_kv_buffer(
            table=table_gpu,
            cache_k=cache_k,
            cache_v=cache_v,
        )

        self.k_uvm.metadata.index_put_(
            indices=(table,),
            values=torch.index_select(self.k_uvm.metadata, index=table_gpu, dim=0) + 1,
        )
        self.v_uvm.metadata.index_put_(
            indices=(table,),
            values=torch.index_select(self.v_uvm.metadata, index=table_gpu, dim=0) + 1,
        )

    def handle_cache_miss(self, metadata: HiPAttentionOutputMetadata):
        if metadata.mask_cache_statistics is not None:
            self.mask_k_cache.handle_cache_miss(
                metadata=metadata, stats=metadata.mask_cache_statistics
            )
            self.sa_kv_cache.handle_cache_miss(
                metadata=metadata, stats=metadata.sa_cache_statistics
            )
        else:
            self.mask_k_cache.handle_cache_miss(metadata=metadata, stats=None)
            self.sa_kv_cache.handle_cache_miss(metadata=metadata, stats=None)


###############################################################################
#                               Kernel Function
###############################################################################


@numba.njit(parallel=True, fastmath=True)
def set_kv_buffer_(
    bank: np.ndarray, cache: np.ndarray, table: np.ndarray, num_threads: int
):
    assert table.shape[0] == cache.shape[0]

    chunk_size = math.ceil(table.shape[0] / num_threads)
    for ithread in numba.prange(num_threads):
        for i in range(chunk_size):
            t = ithread * chunk_size + i
            if t < table.shape[0]:
                bank[table[t]] = cache[t]


@triton.jit
def validate_bank_metadata_slots(
    UVM_METADATA,
    stride_uvm_metadata_token,
    stride_uvm_metadata_k,
    METADATA,
    stride_metadata_slot,
    stride_metadata_k,
    idx_slot,
    idx_page,  # this is optional. if given, check backref
    cache_hit,
    HEAD_KV,
):
    cache_hit = (idx_slot < MAX_INT) & cache_hit

    back_ref = tl.load(
        METADATA + idx_slot.to(tl.int64) * stride_metadata_slot + 0 * stride_metadata_k,
        mask=cache_hit,
    )

    if idx_page is not None:
        cache_hit = (
            ((back_ref // HEAD_KV) == idx_page) & (back_ref < MAX_INT) & cache_hit
        )
    else:
        cache_hit = (back_ref < MAX_INT) & cache_hit

    ref_to_uvm = tl.load(
        METADATA + idx_slot.to(tl.int64) * stride_metadata_slot + 1 * stride_metadata_k,
        mask=cache_hit,
    ).to(tl.int64)
    cache_hit = (ref_to_uvm < MAX_INT) & cache_hit

    uvm_token_gen = tl.load(
        UVM_METADATA
        + ref_to_uvm.to(tl.int64) * stride_uvm_metadata_token
        + 0 * stride_uvm_metadata_k,
        mask=cache_hit,
    )
    cache_token_gen = tl.load(
        METADATA + idx_slot.to(tl.int64) * stride_metadata_slot + 2 * stride_metadata_k,
        mask=cache_hit,
    )
    cache_hit = (
        (uvm_token_gen < MAX_INT) & (uvm_token_gen == cache_token_gen) & cache_hit
    )

    tl.static_assert(cache_hit.dtype == tl.int1)

    return cache_hit


@triton.jit
def load_tokens(
    K,
    stride_k_bsz,
    stride_k_tsrc,
    stride_k_head,
    stride_k_hid,
    # paged attention args template
    USING_PAGES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    K_CACHE,
    stride_k_cache_page,
    stride_k_cache_offset,
    stride_k_cache_kv_head,
    stride_k_cache_hid,
    BLOCK_TABLE,
    stride_block_table_bsz,
    stride_block_table_page,
    CACHE_SEQ_LENS,
    stride_cache_seq_lens_b,
    USING_OFFLOAD_CACHE: tl.constexpr,
    OFFLOAD_CACHE_KV_PACKED: tl.constexpr,
    GPU_BANK_COUNT: int,
    OFFLOAD_CACHE_LOAD_VALUE: tl.constexpr,
    OFFLOAD_CACHE_UVM_METADATA,
    stride_offload_cache_uvm_metadata_token,
    stride_offload_cache_uvm_metadata_k,
    OFFLOAD_CACHE_GPU_GLOBAL_METADATA,
    stride_offload_cache_gpu_global_metadata_k,
    stride_offload_cache_gpu_global_metadata_pad,
    OFFLOAD_CACHE_GPU_BANK,
    stride_offload_cache_gpu_bank_token,
    stride_offload_cache_gpu_bank_hid,
    OFFLOAD_CACHE_GPU_METADATA,
    stride_offload_cache_gpu_metadata_token,
    stride_offload_cache_gpu_metadata_k,
    OFFLOAD_CACHE_GPU_TABLE,
    stride_offload_cache_gpu_table_head_kv,
    stride_offload_cache_gpu_table_token,
    strdie_offload_cache_gpu_table_k,
    ACCESS_COUNTER,
    stride_access_counter_bsz,
    stride_access_counter_head_kv,
    stride_access_counter_tsrc,
    CACHE_MISS_COUNTER,
    stride_cache_miss_counter_bsz,
    stride_cache_miss_counter_head_kv,
    stride_cache_miss_counter_tsrc,
    idx_bsz,
    idx_tsrc,
    idx_kv_head,
    idx_hid,
    mask_keys,
    HEAD_KV: int,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_HID: tl.constexpr,
    HID_DIM,
    IS_BSA: tl.constexpr = False,
    UPDATE_CACHE: tl.constexpr = False,
    V_CACHE=None,
    stride_v_cache_page=None,
    stride_v_cache_offset=None,
    stride_v_cache_kv_head=None,
    stride_v_cache_hid=None,
):
    # DEBUG: to load nothing
    # mask_keys = mask_keys & False

    # tl.static_print(OFFLOAD_CACHE_METHOD)

    if not USING_PAGES:
        tl.static_assert(not USING_OFFLOAD_CACHE)

        if ACCESS_COUNTER is not None:
            # tl.atomic_add(
            #     ACCESS_COUNTER +\
            #         idx_bsz.to(tl.int64) * stride_access_counter_bsz +\
            #         idx_kv_head * stride_access_counter_head_kv +\
            #         idx_tsrc * stride_access_counter_tsrc,
            #     mask=mask_keys,
            #     val=1
            # )

            tl.store(
                ACCESS_COUNTER
                + idx_bsz.to(tl.int64) * stride_access_counter_bsz
                + idx_kv_head * stride_access_counter_head_kv
                + idx_tsrc * stride_access_counter_tsrc,
                mask=mask_keys,
                value=1,
            )

        keys = tl.load(
            K
            + idx_bsz.to(tl.int64) * stride_k_bsz
            + idx_tsrc.to(tl.int64) * stride_k_tsrc
            + idx_kv_head.to(tl.int64) * stride_k_head
            + idx_hid.to(tl.int64) * stride_k_hid,
            mask=mask_keys & (idx_hid < HID_DIM),
            other=0.0,
            cache_modifier=".cg",
            # cache_modifier='.cs', # TODO: uncomment this
        )
    else:
        seq_len = tl.load(
            CACHE_SEQ_LENS + idx_bsz.to(tl.int64) * stride_cache_seq_lens_b,
        )
        mask_tsrc = (idx_tsrc >= 0) & (idx_tsrc < seq_len)
        ptrs = (
            BLOCK_TABLE
            + idx_bsz.to(tl.int64) * stride_block_table_bsz
            + (idx_tsrc // PAGE_SIZE).to(tl.int64) * stride_block_table_page
        )
        idx_page = tl.load(
            ptrs,
            mask=mask_tsrc,
            other=0,
            cache_modifier=".cg",
        ).to(tl.int64)
        offset_page = idx_tsrc % PAGE_SIZE

        if ACCESS_COUNTER is not None:
            # tl.atomic_add(
            #     ACCESS_COUNTER +\
            #         idx_bsz.to(tl.int64) * stride_access_counter_bsz +\
            #         idx_kv_head * stride_access_counter_head_kv +\
            #         idx_page * stride_access_counter_tsrc,
            #     mask=mask_keys,
            #     val=1
            # )

            tl.store(
                ACCESS_COUNTER
                + idx_bsz.to(tl.int64) * stride_access_counter_bsz
                + idx_kv_head.to(tl.int64) * stride_access_counter_head_kv
                + idx_page.to(tl.int64) * stride_access_counter_tsrc,
                mask=mask_keys,
                value=1,
            )

        original_mask_keys = mask_keys

        if USING_OFFLOAD_CACHE:
            tl.static_assert(PAGE_SIZE == 1)

            idx_slots = tl.load(
                OFFLOAD_CACHE_GPU_TABLE
                + idx_page.to(tl.int64) * stride_offload_cache_gpu_table_token
                + idx_kv_head * stride_offload_cache_gpu_table_head_kv
                + 0 * strdie_offload_cache_gpu_table_k,
                mask=mask_keys,
            )
            idx_slot_has_reference_to_bank = (idx_slots < MAX_INT) & mask_keys
            idx_slots = idx_slots * idx_slot_has_reference_to_bank

            ALWAYS_VALIDATE_LINK: tl.constexpr = False  # not UPDATE_CACHE

            if ALWAYS_VALIDATE_LINK:
                validated_slots = validate_bank_metadata_slots(
                    OFFLOAD_CACHE_UVM_METADATA,
                    stride_offload_cache_uvm_metadata_token,
                    stride_offload_cache_uvm_metadata_k,
                    OFFLOAD_CACHE_GPU_METADATA,
                    stride_offload_cache_gpu_metadata_token,
                    stride_offload_cache_gpu_metadata_k,
                    idx_slots,
                    idx_page,
                    idx_slot_has_reference_to_bank,
                    HEAD_KV,
                )

                mask_slot_cache_hit = validated_slots & idx_slot_has_reference_to_bank
            else:
                mask_slot_cache_hit = idx_slot_has_reference_to_bank

            # if OFFLOAD_CACHE_LOAD_VALUE:
            #     mask_slot_cache_hit = mask_slot_cache_hit & False

            idx_hid_cached = idx_hid
            if OFFLOAD_CACHE_LOAD_VALUE:
                idx_hid_cached += BLOCK_HID
            keys_cached = tl.load(
                OFFLOAD_CACHE_GPU_BANK
                + idx_slots.to(tl.int64) * stride_offload_cache_gpu_bank_token
                + idx_hid_cached * stride_offload_cache_gpu_bank_hid,
                mask=mask_slot_cache_hit,
                other=0.0,
            )
            if keys_cached.dtype == tl.uint8:
                keys_cached = keys_cached.to(tl.float8e5, bitcast=True).to(tl.bfloat16)
            if keys_cached.dtype == tl.float8e5:
                keys_cached = keys_cached.to(tl.bfloat16)

            if UPDATE_CACHE:
                idx_slots_verify = tl.load(
                    OFFLOAD_CACHE_GPU_TABLE
                    + idx_kv_head.to(tl.int64) * stride_offload_cache_gpu_table_head_kv
                    + idx_page * stride_offload_cache_gpu_table_token
                    + 0 * strdie_offload_cache_gpu_table_k,
                    mask=mask_slot_cache_hit,
                )
                mask_slot_cache_hit = (
                    (idx_slots_verify < MAX_INT)
                    & (idx_slots == idx_slots_verify)
                    & mask_slot_cache_hit
                )

            if mask_slot_cache_hit.shape[0] == 1:
                keys_cached_hash = tl.sum(
                    keys_cached.to(tl.uint16, bitcast=True), axis=0, keep_dims=True
                ).to(tl.uint64)
            elif mask_slot_cache_hit.shape[1] == 1:
                keys_cached_hash = tl.sum(
                    keys_cached.to(tl.uint16, bitcast=True), axis=1, keep_dims=True
                ).to(tl.uint64)
            else:
                raise Exception()
            tl.debug_barrier()

            tl.inline_asm_elementwise(
                "MEMBAR.SC.GPU;", "=r", [], dtype=tl.int32, is_pure=True, pack=1
            )

            truth_hash = tl.load(
                OFFLOAD_CACHE_GPU_METADATA
                + idx_slots.to(tl.int64) * stride_offload_cache_gpu_metadata_token
                + 6 * stride_offload_cache_gpu_metadata_k,
                mask=mask_slot_cache_hit,
            ).to(tl.uint64)
            hash_mask = tl.full((1,), value=1, dtype=tl.uint64)
            hash_mask = (hash_mask << 32) - 1
            if OFFLOAD_CACHE_LOAD_VALUE:
                truth_hash = (truth_hash >> 32) & hash_mask
            else:
                truth_hash = truth_hash & hash_mask
            tl.debug_barrier()
            if UPDATE_CACHE:
                mask_slot_cache_hit = (
                    truth_hash == (keys_cached_hash & hash_mask)
                ) & mask_slot_cache_hit

            tl.static_assert(mask_slot_cache_hit.dtype == tl.int1)

            mask_keys_cache_miss = mask_keys & (~mask_slot_cache_hit)
            mask_slot_cache_hit = mask_keys & (~mask_keys_cache_miss)
            mask_keys = mask_keys_cache_miss

            tl.store(
                OFFLOAD_CACHE_GPU_METADATA
                + idx_slots.to(tl.int64) * stride_offload_cache_gpu_metadata_token
                + 4 * stride_offload_cache_gpu_metadata_k,
                value=0,
                mask=mask_slot_cache_hit,
            )
            # tl.store(
            #     CACHE_MISS_COUNTER +\
            #         idx_bsz.to(tl.int64) * stride_cache_miss_counter_bsz +\
            #         idx_kv_head * stride_cache_miss_counter_head_kv +\
            #         idx_page * stride_cache_miss_counter_tsrc,
            #     mask=mask_slot_cache_hit,
            #     value=0,
            # )

        idx_page_load = idx_page

        keys = tl.load(
            K_CACHE
            + idx_page_load.to(tl.int64) * stride_k_cache_page
            + offset_page.to(tl.int64) * stride_k_cache_offset
            + idx_kv_head.to(tl.int64) * stride_k_cache_kv_head
            + idx_hid.to(tl.int64) * stride_k_cache_hid,
            mask=mask_keys & (idx_hid < HID_DIM),
            other=0.0,
            cache_modifier=".cg",
        )
        if keys.dtype == tl.uint8:
            keys = keys.to(tl.float8e5, bitcast=True).to(tl.bfloat16)
        if keys.dtype == tl.float8e5:
            keys = keys.to(tl.bfloat16)

        if USING_OFFLOAD_CACHE:
            keys = tl.where(
                mask_keys,
                keys,
                keys_cached,
            )

            if CACHE_MISS_COUNTER is not None:
                if UPDATE_CACHE:
                    tl.debug_barrier()
                    tl.store(
                        CACHE_MISS_COUNTER
                        + idx_bsz.to(tl.int64) * stride_cache_miss_counter_bsz
                        + idx_kv_head.to(tl.int64) * stride_cache_miss_counter_head_kv
                        + idx_page.to(tl.int64) * stride_cache_miss_counter_tsrc,
                        mask=mask_keys_cache_miss,
                        value=1,
                    )
                    # mask_victim_slots = (cache_miss_counter != 1) & mask_keys_cache_miss
                    mask_victim_slots = mask_keys_cache_miss  # NOTE: init value if cache miss counter is ignored
                    # table is protected by cache miss counter
                    previous_table = tl.atomic_xchg(
                        OFFLOAD_CACHE_GPU_TABLE
                        + idx_page.to(tl.int64) * stride_offload_cache_gpu_table_token
                        + idx_kv_head * stride_offload_cache_gpu_table_head_kv
                        + 0 * strdie_offload_cache_gpu_table_k,
                        val=MAX_INT_ACQUIRED,
                        mask=mask_victim_slots,
                    )
                    mask_victim_slots = (
                        previous_table != MAX_INT_ACQUIRED
                    ) & mask_victim_slots
                    mask_table_acquired = mask_victim_slots

                    tl.debug_barrier()

                    idx_victim_slots = tl.zeros_like(idx_slots).to(tl.int64) + MAX_INT
                    max_not_accessed_time = tl.zeros_like(idx_slots).to(tl.int64) - 1
                    victim_slot_not_acquired = mask_victim_slots
                    seed = tl.atomic_add(
                        OFFLOAD_CACHE_GPU_GLOBAL_METADATA
                        + 1 * stride_offload_cache_gpu_global_metadata_k
                        + 0 * stride_offload_cache_gpu_global_metadata_pad,
                        val=1,
                    )
                    for i in range(10):
                        pid = (
                            tl.program_id(0).to(tl.int64)
                            * tl.num_programs(1)
                            * tl.num_programs(2)
                            + tl.program_id(1) * tl.num_programs(2)
                            + tl.program_id(2)
                        )
                        if original_mask_keys.shape[0] == 1:
                            idx_randint = tl.randint(
                                seed,
                                tl.arange(0, BLOCK_SIZE_K)[None, :] + i * BLOCK_SIZE_K,
                                10,
                            ).to(tl.int64) & ((1 << 30) - 1)
                        else:
                            idx_randint = tl.randint(
                                seed,
                                tl.arange(0, BLOCK_SIZE_K)[:, None] + i * BLOCK_SIZE_K,
                                10,
                            ).to(tl.int64) & ((1 << 30) - 1)
                        # if IS_BSA:
                        #     idx_victim_slots_try = idx_randint % (8192 * HEAD_KV)
                        # else:
                        #     idx_victim_slots_try = idx_randint % (8192 * HEAD_KV)
                        # idx_victim_slots_try = idx_victim_slots_try * 128 + tl.extra.cuda.smid()
                        idx_victim_slots_try = idx_randint
                        # idx_victim_slots_try = idx_randint * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
                        idx_victim_slots_try = idx_victim_slots_try % GPU_BANK_COUNT
                        # if IS_BSA:
                        #     idx_victim_slots_try = idx_victim_slots_try % (10000 * HEAD_KV)
                        # else:
                        #     idx_victim_slots_try = idx_victim_slots_try % (32000 * HEAD_KV)

                        acquired = victim_slot_not_acquired

                        not_accessed_time = tl.load(
                            OFFLOAD_CACHE_GPU_METADATA
                            + idx_victim_slots_try.to(tl.int64)
                            * stride_offload_cache_gpu_metadata_token
                            + 4 * stride_offload_cache_gpu_metadata_k,
                            mask=mask_victim_slots,
                        )
                        new_old_slot = (
                            not_accessed_time > max_not_accessed_time
                        ) & mask_victim_slots
                        # if already acquired, release it
                        tl.atomic_xchg(
                            OFFLOAD_CACHE_GPU_METADATA
                            + idx_victim_slots.to(tl.int64)
                            * stride_offload_cache_gpu_metadata_token
                            + 5 * stride_offload_cache_gpu_metadata_k,
                            val=0,
                            mask=new_old_slot & (~victim_slot_not_acquired),
                        )
                        tl.atomic_xchg(
                            OFFLOAD_CACHE_GPU_METADATA
                            + idx_victim_slots.to(tl.int64)
                            * stride_offload_cache_gpu_metadata_token
                            + 0 * stride_offload_cache_gpu_metadata_k,
                            val=MAX_INT,
                            mask=new_old_slot & (~victim_slot_not_acquired),
                        )
                        max_not_accessed_time = tl.maximum(
                            max_not_accessed_time, not_accessed_time
                        )
                        victim_slot_not_acquired = (
                            victim_slot_not_acquired | new_old_slot
                        )
                        acquired = victim_slot_not_acquired & (not_accessed_time > 0)

                        # check already written or not
                        previous_state = tl.atomic_xchg(
                            OFFLOAD_CACHE_GPU_METADATA
                            + idx_victim_slots_try.to(tl.int64)
                            * stride_offload_cache_gpu_metadata_token
                            + 5 * stride_offload_cache_gpu_metadata_k,
                            val=1,  # NOTE: this should be MAX_INT_1, but just for temporary.
                            mask=acquired,
                        )
                        acquired = (previous_state != 1) & acquired

                        # check acquired or not
                        previous_state = tl.atomic_xchg(
                            OFFLOAD_CACHE_GPU_METADATA
                            + idx_victim_slots_try.to(tl.int64)
                            * stride_offload_cache_gpu_metadata_token
                            + 0 * stride_offload_cache_gpu_metadata_k,
                            val=MAX_INT_ACQUIRED,  # NOTE: this should be MAX_INT_1, but just for temporary.
                            mask=acquired,
                        )
                        acquired = (previous_state != MAX_INT_ACQUIRED) & acquired

                        previously_acquired = (previous_state < MAX_INT) & acquired
                        previous_idx_page = previous_state // HEAD_KV
                        previous_idx_head_kv = previous_state % HEAD_KV
                        tl.atomic_xchg(
                            OFFLOAD_CACHE_GPU_TABLE
                            + previous_idx_page.to(tl.int64)
                            * stride_offload_cache_gpu_table_token
                            + previous_idx_head_kv
                            * stride_offload_cache_gpu_table_head_kv
                            + 0 * strdie_offload_cache_gpu_table_k,
                            val=MAX_INT,
                            mask=previously_acquired,
                        )

                        idx_victim_slots = tl.where(
                            acquired,
                            idx_victim_slots_try,
                            idx_victim_slots,
                        )

                        victim_slot_not_acquired = (
                            ~acquired
                        ) & victim_slot_not_acquired
                        tl.debug_barrier()
                    tl.debug_barrier()
                    # mask_victim_slots = mask_victim_slots & (idx_victim_slots < MAX_INT)
                    mask_victim_slots = (
                        mask_victim_slots
                        & (~victim_slot_not_acquired)
                        & mask_keys_cache_miss
                        & (idx_victim_slots != MAX_INT)
                        & original_mask_keys
                    )
                    # if not IS_BSA:
                    #     mask_victim_slots = mask_victim_slots & (~victim_slot_not_acquired) & (idx_victim_slots < (32000 * HEAD_KV)) & (idx_victim_slots > -1)
                    # else:
                    #     mask_victim_slots = mask_victim_slots & (~victim_slot_not_acquired) & (idx_victim_slots < (10000 * HEAD_KV)) & (idx_victim_slots > -1)
                    idx_victim_slots = idx_victim_slots * mask_victim_slots
                    tl.debug_barrier()

                    tl.store(
                        OFFLOAD_CACHE_GPU_BANK
                        + idx_victim_slots.to(tl.int64)
                        * stride_offload_cache_gpu_bank_token
                        + idx_hid_cached * stride_offload_cache_gpu_bank_hid,
                        value=keys,
                        mask=mask_victim_slots,
                    )

                    # take token hash for debug
                    if mask_victim_slots.shape[0] == 1:
                        keys_hash = tl.sum(
                            keys.to(tl.uint16, bitcast=True), axis=0, keep_dims=True
                        ).to(tl.uint64)
                    elif mask_victim_slots.shape[1] == 1:
                        keys_hash = tl.sum(
                            keys.to(tl.uint16, bitcast=True), axis=1, keep_dims=True
                        ).to(tl.uint64)
                    else:
                        raise Exception()

                    if IS_BSA:
                        values = tl.load(
                            V_CACHE
                            + idx_page.to(tl.int64) * stride_v_cache_page
                            + offset_page.to(tl.int64) * stride_v_cache_offset
                            + idx_kv_head.to(tl.int64) * stride_v_cache_kv_head
                            + idx_hid.to(tl.int64) * stride_v_cache_hid,
                            mask=mask_victim_slots,
                            other=0.0,
                        )
                        if values.dtype == tl.uint8:
                            values = values.to(tl.float8e5, bitcast=True).to(
                                tl.bfloat16
                            )
                        if values.dtype == tl.float8e5:
                            values = values.to(tl.bfloat16)
                        tl.store(
                            OFFLOAD_CACHE_GPU_BANK
                            + idx_victim_slots.to(tl.int64)
                            * stride_offload_cache_gpu_bank_token  # idx_hid * stride_offload_cache_gpu_bank_hid,
                            + ((idx_hid_cached + BLOCK_HID) % (BLOCK_HID * 2))
                            * stride_offload_cache_gpu_bank_hid,
                            value=values,
                            mask=mask_victim_slots,
                        )

                        if mask_victim_slots.shape[0] == 1:
                            values_hash = tl.sum(
                                values.to(tl.uint16, bitcast=True),
                                axis=0,
                                keep_dims=True,
                            ).to(tl.uint64)
                        elif mask_victim_slots.shape[1] == 1:
                            values_hash = tl.sum(
                                values.to(tl.uint16, bitcast=True),
                                axis=1,
                                keep_dims=True,
                            ).to(tl.uint64)
                        else:
                            raise Exception()
                        if not OFFLOAD_CACHE_LOAD_VALUE:
                            keys_hash = keys_hash | (values_hash << 32)
                        else:
                            keys_hash = (keys_hash << 32) | values_hash

                    tl.store(
                        OFFLOAD_CACHE_GPU_METADATA
                        + idx_victim_slots.to(tl.int64)
                        * stride_offload_cache_gpu_metadata_token
                        + 6 * stride_offload_cache_gpu_metadata_k,
                        value=keys_hash,
                        mask=mask_victim_slots,
                    )

                    uvm_token_gen = tl.load(
                        OFFLOAD_CACHE_UVM_METADATA
                        + idx_page.to(tl.int64)
                        * stride_offload_cache_uvm_metadata_token
                        + 0 * stride_offload_cache_uvm_metadata_k,
                        mask=mask_victim_slots,
                    )

                    tl.store(
                        OFFLOAD_CACHE_GPU_METADATA
                        + idx_victim_slots.to(tl.int64)
                        * stride_offload_cache_gpu_metadata_token
                        + 1 * stride_offload_cache_gpu_metadata_k,
                        value=idx_page,
                        mask=mask_victim_slots,
                    )
                    tl.store(
                        OFFLOAD_CACHE_GPU_METADATA
                        + idx_victim_slots.to(tl.int64)
                        * stride_offload_cache_gpu_metadata_token
                        + 2 * stride_offload_cache_gpu_metadata_k,
                        value=uvm_token_gen,
                        mask=mask_victim_slots,
                    )
                    tl.store(
                        OFFLOAD_CACHE_GPU_METADATA
                        + idx_victim_slots.to(tl.int64)
                        * stride_offload_cache_gpu_metadata_token
                        + 4 * stride_offload_cache_gpu_metadata_k,
                        value=0,
                        mask=mask_victim_slots,
                    )

                    tl.debug_barrier()

                    # core.inline_asm_elementwise("mov.u32 $0, %smid;", "=r", [], dtype=core.int32, is_pure=True, pack=1,
                    #                    _builder=_builder)
                    tl.inline_asm_elementwise(
                        "MEMBAR.SC.GPU;", "=r", [], dtype=tl.int32, is_pure=True, pack=1
                    )

                    # release slot
                    tl.atomic_xchg(
                        OFFLOAD_CACHE_GPU_METADATA
                        + idx_victim_slots.to(tl.int64)
                        * stride_offload_cache_gpu_metadata_token
                        + 0 * stride_offload_cache_gpu_metadata_k,
                        val=idx_page * HEAD_KV + idx_kv_head,
                        mask=mask_victim_slots,
                    )
                    # release table
                    table_slots = tl.where(
                        mask_table_acquired & (~mask_victim_slots),
                        MAX_INT,
                        idx_victim_slots,
                    )
                    tl.atomic_xchg(
                        OFFLOAD_CACHE_GPU_TABLE
                        + idx_page.to(tl.int64) * stride_offload_cache_gpu_table_token
                        + idx_kv_head * stride_offload_cache_gpu_table_head_kv
                        + 0 * strdie_offload_cache_gpu_table_k,
                        val=table_slots,
                        mask=mask_table_acquired,
                    )
                else:
                    tl.store(
                        CACHE_MISS_COUNTER
                        + idx_bsz.to(tl.int64) * stride_cache_miss_counter_bsz
                        + idx_kv_head * stride_cache_miss_counter_head_kv
                        + idx_page * stride_cache_miss_counter_tsrc,
                        mask=mask_keys_cache_miss,
                        value=1,
                    )
    if keys.dtype == tl.uint8:
        keys = keys.to(tl.float8e5, bitcast=True).to(tl.float16)
    if keys.dtype == tl.float8e5:
        keys = keys.to(tl.float16)

    return keys


def update_recency_pytorch(
    accessed_ptr: Tensor,
    uvm_metadata: Tensor,
    global_metadata: Tensor,
    metadata: Tensor,
    table: Tensor,
    head_num: int,
    uvm_page_count: int,
):
    for idx_head_kv in range(head_num):
        for idx_token in tqdm.tqdm(
            range(uvm_page_count), dynamic_ncols=True, leave=False
        ):
            current_tick = global_metadata[0, 0]

            accessed = accessed_ptr[idx_head_kv, idx_token] > 0
            cache_hit = True & accessed
            if not cache_hit:
                continue

            idx_table = table[idx_head_kv, idx_token]
            cache_hit = (idx_table != MAX_INT) & cache_hit
            if not cache_hit:
                continue

            back_ref = metadata[idx_table, 0]
            cache_hit = (back_ref == idx_token) & cache_hit
            if not cache_hit:
                continue

            ref_to_uvm = metadata[idx_table, 1]
            cache_hit = (ref_to_uvm != MAX_INT) & cache_hit
            if not cache_hit:
                continue

            uvm_token_gen = uvm_metadata[ref_to_uvm, 0]
            cache_token_gen = metadata[idx_table, 2]
            cache_hit = (
                (uvm_token_gen != MAX_INT)
                & (uvm_token_gen == cache_token_gen)
                & cache_hit
            )
            if not cache_hit:
                continue

            metadata[idx_table, 3] = current_tick.to(metadata.dtype)


@triton.jit
def update_recency(
    ACCESSED,
    stride_accessed_head_kv,
    stride_accessed_token,
    UVM_METADATA,
    stride_uvm_metadata_token,
    stride_uvm_metadata_k,
    GLOBAL_METADTA,
    stride_global_metadata_k,
    stride_global_metadata_pad,
    METADATA,
    stride_metadata_slot,
    stride_metadata_k,
    TABLE,
    stride_table_head_kv,
    stride_table_token,
    stride_table_k,
    page_count: int,
    HEAD_KV: int,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)

    idx_block = pid % tl.cdiv(page_count, BLOCK_SIZE)
    idx_head_kv = pid // tl.cdiv(page_count, BLOCK_SIZE)

    idx_token = idx_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_token = idx_token < page_count

    current_tick = tl.load(
        GLOBAL_METADTA + 0 * stride_global_metadata_k + 0 * stride_global_metadata_pad
    )

    # TODO: merge with load tokens, verify cache
    accessed = (
        tl.load(
            ACCESSED
            + idx_head_kv.to(tl.int64) * stride_accessed_head_kv
            + idx_token * stride_accessed_token,
            mask=mask_token,
            other=0,
        )
        > 0
    )
    cache_hit = mask_token & accessed

    table = tl.load(
        TABLE
        + idx_head_kv.to(tl.int64) * stride_table_head_kv
        + idx_token * stride_table_token
        + 0 * stride_table_k,
        mask=cache_hit,
        other=MAX_INT,
    ).to(tl.int64)
    cache_hit = (table < MAX_INT) & cache_hit

    ALWAYS_VALIDATE_LINK: tl.constexpr = False  # True

    if ALWAYS_VALIDATE_LINK:
        validated_cache_hit = validate_bank_metadata_slots(
            UVM_METADATA,
            stride_uvm_metadata_token,
            stride_uvm_metadata_k,
            METADATA,
            stride_metadata_slot,
            stride_metadata_k,
            table,
            idx_token,
            cache_hit,
            HEAD_KV,
        )
        cache_hit = cache_hit & validated_cache_hit

    tl.store(
        METADATA + table.to(tl.int64) * stride_metadata_slot + 3 * stride_metadata_k,
        mask=cache_hit,
        value=current_tick,
    )


@triton.jit
def write_cache(
    PUT,
    stride_put_t,
    MASK,
    stride_mask_t,
    EVICT,
    stride_evict_t,
    BANK,
    stride_bank_t,
    stride_bank_hid,
    METADATA,
    stride_metadata_t,
    stride_metadata_k,
    TABLE,
    stride_table_head_kv,
    stride_table_t,
    stride_table_k,
    UVM_METADATA,
    stride_uvm_metadata_t,
    stride_uvm_metadata_k,
    UVM_K_BANK,
    stride_uvm_k_bank_t,
    stride_uvm_k_bank_head_kv,
    stride_uvm_k_bank_hid,
    UVM_V_BANK,
    stride_uvm_v_bank_t,
    stride_uvm_v_bank_head_kv,
    stride_uvm_v_bank_hid,
    GLOBAL_METADATA,
    stride_global_metadata_t,
    stride_global_metadata_k,
    qsize: int,
    page_count: int,
    HEAD_KV: int,
    KV_PACKED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_HID: tl.constexpr,
):
    pid = tl.program_id(0)
    idx_queue = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_queue = idx_queue < qsize

    put_list = tl.load(
        PUT + idx_queue.to(tl.int64) * stride_put_t,
        mask=mask_queue,
    )
    idx_page = put_list % page_count
    idx_head_kv = put_list // page_count

    mask_put = (
        tl.load(MASK + idx_queue.to(tl.int64) * stride_mask_t, mask=mask_queue, other=0)
        != 0
    )
    idx_evict = tl.load(EVICT + idx_queue.to(tl.int64) * stride_evict_t, mask=mask_put)

    # check still it is cache miss
    idx_slot = tl.load(
        TABLE
        + idx_head_kv.to(tl.int64) * stride_table_head_kv
        + idx_page * stride_table_t
        + 0 * stride_table_k,
        mask=mask_put,
        other=MAX_INT,
    )
    is_valid_slot = validate_bank_metadata_slots(
        UVM_METADATA,
        stride_uvm_metadata_k,
        stride_uvm_metadata_k,
        METADATA,
        stride_metadata_t,
        stride_metadata_k,
        idx_slot,
        idx_page,
        mask_put,
        HEAD_KV,
    )
    mask_put = mask_put & (~is_valid_slot)

    # unlink bank <-> table
    victim_table_entry = tl.load(
        METADATA + idx_evict.to(tl.int64) * stride_metadata_t + 0 * stride_metadata_k,
        mask=mask_put,
    )
    tl.store(
        TABLE
        + (victim_table_entry.to(tl.int64) % HEAD_KV) * stride_table_head_kv
        + (victim_table_entry // HEAD_KV) * stride_table_t
        + 0 * stride_table_k,
        value=MAX_INT,
        mask=mask_put & (victim_table_entry < MAX_INT),
    )

    # setup metadata
    tl.store(
        METADATA + idx_evict.to(tl.int64) * stride_metadata_t + 0 * stride_metadata_k,
        mask=mask_put,
        value=idx_page * HEAD_KV + idx_head_kv,
    )
    tl.store(
        METADATA + idx_evict.to(tl.int64) * stride_metadata_t + 1 * stride_metadata_k,
        mask=mask_put,
        value=idx_page,
    )
    token_gen = tl.load(
        UVM_METADATA
        + idx_page.to(tl.int64) * stride_uvm_metadata_t
        + 0 * stride_uvm_metadata_k,
        mask=mask_put,
    )
    tl.store(
        METADATA + idx_evict.to(tl.int64) * stride_metadata_t + 2 * stride_metadata_k,
        mask=mask_put,
        value=token_gen,
    )
    current_tick = tl.load(
        GLOBAL_METADATA + 0 * stride_global_metadata_t + 0 * stride_global_metadata_k,
    )
    tl.store(
        METADATA + idx_evict.to(tl.int64) * stride_metadata_t + 3 * stride_metadata_k,
        mask=mask_put,
        value=current_tick,
    )

    # setup table
    tl.store(
        TABLE
        + idx_page.to(tl.int64) * stride_table_t
        + idx_head_kv * stride_table_head_kv
        + 0 * stride_table_k,
        mask=mask_put,
        value=idx_evict,
    )

    # copy values
    idx_hid = tl.arange(0, BLOCK_HID)

    keys = tl.load(
        UVM_K_BANK
        + idx_page[:, None].to(tl.int64) * stride_uvm_k_bank_t
        + idx_head_kv[:, None] * stride_uvm_k_bank_head_kv
        + idx_hid[None, :] * stride_uvm_k_bank_hid,
        mask=mask_put[:, None],
    )
    tl.store(
        BANK
        + idx_evict[:, None].to(tl.int64) * stride_bank_t
        + idx_hid[None, :] * stride_bank_hid,
        mask=mask_put[:, None],
        value=keys,
    )

    if KV_PACKED:
        values = tl.load(
            UVM_V_BANK
            + idx_page[:, None].to(tl.int64) * stride_uvm_v_bank_t
            + idx_head_kv[:, None] * stride_uvm_v_bank_head_kv
            + idx_hid[None, :] * stride_uvm_v_bank_hid,
            mask=mask_put[:, None],
        )
        tl.store(
            BANK
            + idx_evict[:, None].to(tl.int64) * stride_bank_t
            + (idx_hid + BLOCK_HID)[None, :] * stride_bank_hid,
            mask=mask_put[:, None],
            value=values,
        )

```

### Codes in SGLang

In this section, I will provide the code related to SGLang which is LLM serving framework.

#### `sglang/srt/layers/attention/hip_attention.py`

```py
from __future__ import annotations

import os

"""
HiP Attention Backend for SGLang
https://arxiv.org/pdf/2406.09827
"""

import logging
import time
from typing import TYPE_CHECKING, Optional, Union

import torch
import triton

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.mem_cache.hip_offload_kv_pool_mha import MHATokenToHiPOffloadKVPool

if TYPE_CHECKING:
    from hip_attn.v1_2 import HiPAttentionConfig
    from sglang.srt.speculative.spec_info import SpecInfo

    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput

logger = logging.getLogger(__name__)

try:
    from sglang.srt.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        model_parallel_is_initialized,
        tensor_model_parallel_all_gather,
    )

    SGLANG_DIST_AVAILABLE = True
except:
    SGLANG_DIST_AVAILABLE = False


def get_local_rank():
    if SGLANG_DIST_AVAILABLE:
        return (
            get_tensor_model_parallel_rank() if model_parallel_is_initialized() else 0
        )
    else:
        return 0


from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
)


class HiPAttentionBackend(AttentionBackend):

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        speculative_step_id=0,
        topk=0,
        speculative_num_steps=0,
    ):
        super().__init__()

        from hip_attn.v1_2.paged_hip import PagedHiPStateful

        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA
        self.page_size = model_runner.page_size
        assert self.page_size == 1

        self.forward_paged_hip = PagedHiPStateful()

        self.hip_config: HiPAttentionConfig = (
            model_runner.server_args.hip_attention_config
        )
        self.is_kv_cache_offload_enabled = (
            model_runner.server_args.enable_hip_kv_cache_offload
        )

        self.max_context_len = model_runner.model_config.context_len

        self.tp_rank = model_runner.tp_rank

        self.attention_chunk_size = model_runner.attention_chunk_size

        self.flashattention_backend = FlashAttentionBackend(
            model_runner=model_runner,
            skip_prefill=skip_prefill,
            speculative_step_id=speculative_step_id,
            topk=topk,
            speculative_num_steps=speculative_num_steps,
        )

        self._last_tick = time.time()

        self._block_table: torch.Tensor = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self._block_table = forward_batch.req_to_token_pool.req_to_token.index_select(
            dim=0, index=forward_batch.req_pool_indices
        )

        if forward_batch.forward_mode.is_decode():
            self.flashattention_backend.init_forward_metadata(forward_batch=forward_batch)

    def init_cuda_graph_state(self, max_bs: int):
        self.flashattention_backend.init_cuda_graph_state(
            max_bs=max_bs,
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInfo],
    ):
        self.flashattention_backend.init_forward_metadata_capture_cuda_graph(
            bs=bs,
            num_tokens=num_tokens,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            encoder_lens=encoder_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
        out_cache_loc: torch.Tensor = None,
    ):
        self.flashattention_backend.init_forward_metadata_replay_cuda_graph(
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_sum=seq_lens_sum,
            encoder_lens=encoder_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=out_cache_loc,
        )

    def get_cuda_graph_seq_len_fill_value(self):
        assert self.flashattention_backend.get_cuda_graph_seq_len_fill_value() == 0
        return 0

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ):
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        using_chunked_sw = False
        sw_size = layer.sliding_window_size
        if layer.use_irope:
            using_chunked_sw = True
            sw_size = self.attention_chunk_size

        using_dense_prefill = os.getenv("HIP_DEBUG_USING_DENSE_PREFILL", "0") == "1"
        using_dense_prefill = using_dense_prefill and (
            layer.layer_id in self.hip_config.dense_layers
        )

        force_dense_decode = os.getenv("HIP_DEBUG_FORCE_DENSE_DECODE", "0") == "1"

        delta_attention_args = os.getenv("HIP_DELTA_ATTENTION_ARGS", "")
        delta_dense_decode = any(
            ["dense_decode" == key for key in delta_attention_args.split("-")]
        )

        is_decode = False
        need_dense_prefill = using_chunked_sw or using_dense_prefill
        need_dense_decode = using_chunked_sw or delta_dense_decode

        run_benchmark = (
            (not torch.cuda.is_current_stream_capturing())
            and os.getenv("HIP_DEBUG_BENCH", "0") == "1"
            and (get_local_rank() == 0)
        )

        if run_benchmark:
            start_event = torch.cuda.Event(True)
            end_event = torch.cuda.Event(True)
            start_event.record()

        if (need_dense_prefill and (not is_decode)) or False:
            return self.flashattention_backend.forward_extend(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
                save_kv_cache=save_kv_cache,
                # For multi-head latent attention
                q_rope=q_rope,
                k_rope=k_rope,
            )
        else:
            if not self.is_kv_cache_offload_enabled:
                if k is not None:
                    assert v is not None
                    if save_kv_cache:
                        if not self.use_mla:
                            forward_batch.token_to_kv_pool.set_kv_buffer(
                                layer, cache_loc, k, v
                            )
                        else:
                            forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                                layer,
                                cache_loc,
                                k,
                                k_rope,
                            )

                if not self.use_mla:
                    k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                        layer.layer_id
                    )
                    k_chunk = k.reshape(-1, layer.tp_k_head_num, layer.head_dim)
                    v_chunk = v.reshape(-1, layer.tp_v_head_num, layer.v_head_dim)
                else:
                    kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(
                        layer.layer_id
                    )

                offload_cache = None
                offloading_metadata = None

            else:  # Offloading enabled
                assert not self.use_mla
                assert isinstance(
                    forward_batch.token_to_kv_pool, MHATokenToHiPOffloadKVPool
                )
                if k is not None:
                    assert v is not None
                    if save_kv_cache:
                        if not self.use_mla:
                            forward_batch.token_to_kv_pool.set_kv_buffer(
                                layer,
                                cache_loc,
                                k,
                                v,
                                async_copy=True,
                                push_to_gpu_cache=False,
                            )
                        else:
                            raise Exception()

                k_cache = v_cache = offload_cache = None
                k_chunk, v_chunk, offloading_metadata = (
                    forward_batch.token_to_kv_pool.get_fetched_prefix_kv_buffer(
                        layer_id=layer.layer_id,
                        extend_seq_lens=forward_batch.extend_seq_lens,
                        extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
                        cache_k=k,
                        cache_v=v,
                    )
                )

            # use_cascade_attn = (
            #     forward_batch.forward_mode.is_target_verify() and self.topk > 1
            # )
            use_cascade_attn = False

            if not self.use_mla:
                q_reshaped = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)

                o, _ = self.forward_paged_hip(
                    query=q_reshaped,
                    sm_scale=layer.scaling,
                    batch_size=forward_batch.batch_size,
                    k=k_chunk,
                    v=v_chunk,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    offload_cache=offload_cache,
                    positions=forward_batch.positions,
                    seq_lens=forward_batch.seq_lens,
                    req_to_tokens=forward_batch.req_to_token_pool.req_to_token,
                    req_pool_indices=forward_batch.req_pool_indices,
                    block_table=self._block_table,
                    rope_cos=layer.rope_cos,
                    rope_sin=layer.rope_sin,
                    rope_range=layer.rope_range,
                    rope_is_neox_style=layer.rope_is_neox_style,
                    layer_id=layer.layer_id,
                    logit_cap=layer.logit_cap,
                    orig_context_len=layer.orig_context_len,
                    max_context_len=self.max_context_len,
                    extend_seq_lens=forward_batch.extend_seq_lens,
                    extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
                    extend_prefix_lens_cpu=forward_batch.extend_prefix_lens_cpu,
                    hip_config=self.hip_config,
                    is_kv_cache_offload_enabled=self.is_kv_cache_offload_enabled,
                    online_update_cache=(
                        forward_batch.token_to_kv_pool.is_online_cache_update_enabled()
                        if self.is_kv_cache_offload_enabled
                        else None
                    ),
                    is_decode=False,
                    offloading_metadata=offloading_metadata,
                    sliding_window_size=sw_size,
                    using_chunked_sliding_window=using_chunked_sw,
                )
            else:
                if (
                    # not global_server_args_dict["disable_chunked_prefix_cache"]
                    # and forward_batch.attn_attend_prefix_cache is not None
                    # and not forward_batch.forward_mode.is_target_verify()
                    # and not forward_batch.forward_mode.is_draft_extend()
                    not global_server_args_dict["disable_chunked_prefix_cache"]
                    # and forward_batch.attn_attend_prefix_cache is not None
                    and forward_batch.forward_mode.is_extend()
                    and not forward_batch.forward_mode.is_target_verify()
                    and not forward_batch.forward_mode.is_draft_extend()
                ):
                    # Do multi-head attention with chunked prefix cache

                    assert q.shape[0] == 1, f"{q.shape=}"
                    k_reshaped = k.reshape(1, -1, layer.tp_k_head_num, layer.head_dim)
                    v_reshaped = v.reshape(1, -1, layer.tp_v_head_num, layer.v_head_dim)

                    assert not use_cascade_attn

                    o, metadata = self.forward_paged_hip(
                        query=q,
                        sm_scale=layer.scaling,
                        batch_size=forward_batch.batch_size,
                        k=k_reshaped,
                        v=v_reshaped,
                        k_cache=None,
                        v_cache=None,
                        offload_cache=offload_cache,
                        positions=forward_batch.positions,
                        seq_lens=forward_batch.seq_lens,
                        req_to_tokens=forward_batch.req_to_token_pool.req_to_token,
                        req_pool_indices=forward_batch.req_pool_indices,
                        block_table=self._block_table,
                        rope_cos=layer.rope_cos,
                        rope_sin=layer.rope_sin,
                        rope_range=layer.rope_range,
                        rope_is_neox_style=layer.rope_is_neox_style,
                        layer_id=layer.layer_id,
                        logit_cap=layer.logit_cap,
                        orig_context_len=layer.orig_context_len,
                        max_context_len=self.max_context_len,
                        extend_seq_lens=forward_batch.extend_seq_lens,
                        extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
                        extend_prefix_lens_cpu=forward_batch.extend_prefix_lens_cpu,
                        hip_config=self.hip_config,
                        is_kv_cache_offload_enabled=self.is_kv_cache_offload_enabled,
                        cached_metadata=None,
                        online_update_cache=(
                            forward_batch.token_to_kv_pool.is_online_cache_update_enabled()
                            if self.is_kv_cache_offload_enabled
                            else None
                        ),
                        is_decode=False,
                        offloading_metadata=offloading_metadata,
                        sliding_window_size=sw_size,
                        using_chunked_sliding_window=using_chunked_sw,
                    )
                else:
                    # Do absorbed multi-latent attention

                    require_metadata_checkout = False
                    if forward_batch.forward_mode.is_target_verify():
                        # NOTE: this condition will be graph captured.
                        metadata = forward_batch.hip_metadata_cache_pool.get_hip_metadata_cache(
                            layer.layer_id,
                            q.shape[0],
                            forward_batch.batch_size,
                            forward_batch.hip_metadata_cached_stages,
                            block_size_q=self.hip_config.block_sparse_block_size_q,
                        )
                        require_metadata_checkout = True
                    else:
                        metadata = None

                    kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(
                        layer.layer_id
                    )
                    nope_dim = triton.next_power_of_2(kv_cache.shape[-1]) // 2
                    rope_dim = kv_cache.shape[-1] - nope_dim
                    # print(q.shape, kv_cache.shape, nope_dim, rope_dim)

                    kv_head = kv_cache.shape[-2]
                    q_head = q.shape[-2]

                    k_rope = kv_cache[..., nope_dim:]
                    c_kv = kv_cache[..., :nope_dim]
                    # k_rope_cache = k_rope.view(
                    #     -1,
                    #     self.page_size,
                    #     layer.tp_k_head_num,
                    #     layer.head_dim - layer.v_head_dim,
                    # )
                    c_kv_cache = c_kv.view(-1, self.page_size, kv_head, nope_dim)
                    if q_rope is not None:
                        q_nope = q.view(-1, q_head, nope_dim)
                        q_rope = q_rope.view(-1, q_head, rope_dim)
                    else:
                        q_all = q.contiguous().view(-1, q_head, nope_dim + rope_dim)
                        q_nope = q_all[:, :, :nope_dim]
                        q_rope = q_all[:, :, nope_dim:]

                    assert q_nope.shape[-1] == layer.rope_range[0]
                    assert (q_rope.shape[-1] + q_nope.shape[-1]) == layer.rope_range[1]
                    q_merged = torch.cat([q_nope, q_rope], dim=-1)
                    # TODO FIXME
                    # k_cache = torch.cat([c_kv_cache, k_rope_cache], dim=-1)
                    k_cache = kv_cache
                    v_cache = c_kv_cache

                    if forward_batch.forward_mode.is_draft_extend():
                        sw_size = 512
                        sw_sink = 128
                    else:
                        sw_sink = -1

                    # print(q_merged.shape, k_cache.shape, v_cache.shape, sw_sink, sw_size)

                    o, metadata = self.forward_paged_hip(
                        query=q_merged,
                        sm_scale=layer.scaling,
                        batch_size=forward_batch.batch_size,
                        k=None,
                        v=None,
                        k_cache=k_cache,
                        v_cache=v_cache,
                        offload_cache=offload_cache,
                        positions=forward_batch.positions,
                        seq_lens=forward_batch.seq_lens,
                        req_to_tokens=forward_batch.req_to_token_pool.req_to_token,
                        req_pool_indices=forward_batch.req_pool_indices,
                        block_table=self._block_table,
                        rope_cos=layer.rope_cos,
                        rope_sin=layer.rope_sin,
                        rope_range=layer.rope_range,
                        rope_is_neox_style=layer.rope_is_neox_style,
                        layer_id=layer.layer_id,
                        logit_cap=layer.logit_cap,
                        orig_context_len=layer.orig_context_len,
                        max_context_len=self.max_context_len,
                        hip_config=self.hip_config,
                        is_kv_cache_offload_enabled=self.is_kv_cache_offload_enabled,
                        cached_metadata=metadata,
                        online_update_cache=(
                            forward_batch.token_to_kv_pool.is_online_cache_update_enabled()
                            if self.is_kv_cache_offload_enabled
                            else None
                        ),
                        is_decode=True,
                        offloading_metadata=offloading_metadata,
                        sliding_window_size=sw_size,
                        sliding_window_sink=sw_sink,
                        using_chunked_sliding_window=using_chunked_sw,
                    )

                    if require_metadata_checkout and (metadata is not None):
                        forward_batch.hip_metadata_cache_pool.set_hip_metadata_cache(
                            layer_id=layer.layer_id,
                            tdst=q.shape[0],
                            batch_size=forward_batch.batch_size,
                            metadata=metadata,
                            block_size_q=self.hip_config.block_sparse_block_size_q,
                            cached_stages=forward_batch.hip_metadata_cached_stages,
                        )

                        if self.is_kv_cache_offload_enabled:
                            offload_cache.handle_cache_miss(metadata)

        if run_benchmark:
            from hip_attn.v1_2.utils import capture

            end_event.record()
            end_event.synchronize()

            elapsed = start_event.elapsed_time(end_event)
            elapsed_layer = (time.time() - self._last_tick) * 1000
            self._last_tick = time.time()
            capture.report()
            print(
                f"[hip] layer {layer.layer_id} took {elapsed:.2f} ms (from last tick: {elapsed_layer:.2f} ms)"
            )

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
    ):

        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        using_chunked_sw = False
        sw_size = layer.sliding_window_size
        if layer.use_irope:
            using_chunked_sw = True
            sw_size = self.attention_chunk_size

        using_dense_prefill = os.getenv("HIP_DEBUG_USING_DENSE_PREFILL", "0") == "1"
        using_dense_prefill = using_dense_prefill and (
            layer.layer_id in self.hip_config.dense_layers
        )

        force_dense_decode = os.getenv("HIP_DEBUG_FORCE_DENSE_DECODE", "0") == "1"

        delta_attention_args = os.getenv("HIP_DELTA_ATTENTION_ARGS", "")
        delta_dense_decode = any(
            ["dense_decode" == key for key in delta_attention_args.split("-")]
        )

        is_decode = False
        need_dense_prefill = using_chunked_sw or using_dense_prefill
        need_dense_decode = using_chunked_sw or delta_dense_decode or force_dense_decode

        if need_dense_decode or False:
            o = self.flashattention_backend.forward_decode(
                q=q,
                k=k,
                v=v,
                layer=layer,
                forward_batch=forward_batch,
                save_kv_cache=save_kv_cache,
                # For multi-head latent attention
                q_rope=q_rope,
                k_rope=k_rope,
            )
        else:
            if forward_batch.hip_metadata_cache_pool is not None:
                metadata = forward_batch.hip_metadata_cache_pool.get_hip_metadata_cache(
                    layer.layer_id,
                    q.shape[0],
                    forward_batch.batch_size,
                    forward_batch.hip_metadata_cached_stages,
                    block_size_q=self.hip_config.block_sparse_block_size_q,
                )
            else:
                metadata = None

            if not self.is_kv_cache_offload_enabled:
                if k is not None:
                    assert v is not None
                    if save_kv_cache:
                        if not self.use_mla:
                            forward_batch.token_to_kv_pool.set_kv_buffer(
                                layer, cache_loc, k, v
                            )
                        else:
                            forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                                layer,
                                cache_loc,
                                k,
                                k_rope,
                            )
                if not self.use_mla:
                    k_cache, v_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                        layer.layer_id
                    )
                else:
                    kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(
                        layer.layer_id
                    )

                offload_cache = offloading_metadata = None
            else:  # Offloading enabled
                assert isinstance(
                    forward_batch.token_to_kv_pool, MHATokenToHiPOffloadKVPool
                )
                if k is not None:
                    assert v is not None
                    if save_kv_cache:
                        if not self.use_mla:
                            forward_batch.token_to_kv_pool.set_kv_buffer(
                                layer,
                                cache_loc,
                                k,
                                v,
                                async_copy=False,
                                push_to_gpu_cache=True,
                            )
                        else:
                            raise Exception()

                k_cache = v_cache = None
                offload_cache, offloading_metadata = (
                    forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id)
                )

            if not self.use_mla:
                k_descale = v_descale = None
                if k_cache is not None:
                    if k_cache.dtype not in [
                        torch.float32,
                        torch.float16,
                        torch.bfloat16,
                    ]:
                        assert k_cache.dtype in (torch.float8_e5m2, )
                        if layer.k_scale is not None:
                            descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
                            k_descale = layer.k_scale.expand(descale_shape)
                            v_descale = layer.v_scale.expand(descale_shape)
                            q = q.to(k_cache.dtype)
                        # assert layer.k_scale is not None, "fp8 scale should be handled"

                q_reshaped = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)
                k_reshaped = k.reshape(-1, layer.tp_k_head_num, layer.head_dim)
                v_reshaped = v.reshape(-1, layer.tp_v_head_num, layer.v_head_dim)

                o, metadata = self.forward_paged_hip(
                    query=q_reshaped,
                    sm_scale=layer.scaling,
                    batch_size=forward_batch.batch_size,
                    k=k_reshaped,
                    v=v_reshaped,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    offload_cache=offload_cache,
                    positions=forward_batch.positions,
                    seq_lens=forward_batch.seq_lens,
                    req_to_tokens=forward_batch.req_to_token_pool.req_to_token,
                    req_pool_indices=forward_batch.req_pool_indices,
                    block_table=self._block_table,
                    rope_cos=layer.rope_cos,
                    rope_sin=layer.rope_sin,
                    rope_range=layer.rope_range,
                    rope_is_neox_style=layer.rope_is_neox_style,
                    layer_id=layer.layer_id,
                    logit_cap=layer.logit_cap,
                    orig_context_len=layer.orig_context_len,
                    max_context_len=self.max_context_len,
                    hip_config=self.hip_config,
                    is_kv_cache_offload_enabled=self.is_kv_cache_offload_enabled,
                    cached_metadata=metadata,
                    online_update_cache=(
                        forward_batch.token_to_kv_pool.is_online_cache_update_enabled()
                        if self.is_kv_cache_offload_enabled
                        else None
                    ),
                    is_decode=True,
                    offloading_metadata=offloading_metadata,
                    sliding_window_size=sw_size,
                    using_chunked_sliding_window=using_chunked_sw,
                    k_descale=k_descale,
                    v_descale=v_descale,
                )
            else:
                if k_cache is not None:
                    if k_cache.dtype not in [
                        torch.float32,
                        torch.float16,
                        torch.bfloat16,
                    ]:
                        assert k_cache.dtype in (torch.float8_e5m2, )
                        assert layer.k_scale is not None, "fp8 scale should be handled"
                # print(q.shape, k.shape, q_rope.shape, k_rope.shape)
                # torch.Size([1, 16, 512]) torch.Size([1, 1, 512]) torch.Size([1, 16, 64]) torch.Size([1, 1, 64])

                k_rope = kv_cache[:, :, layer.v_head_dim :]
                c_kv = kv_cache[:, :, : layer.v_head_dim]
                k_rope_cache = k_rope.view(
                    -1,
                    self.page_size,
                    layer.tp_k_head_num,
                    layer.head_dim - layer.v_head_dim,
                )
                c_kv_cache = c_kv.view(
                    -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                )

                if q_rope is not None:
                    q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                    q_rope = q_rope.view(
                        -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                    )
                else:
                    q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
                    q_nope = q_all[:, :, : layer.v_head_dim]
                    q_rope = q_all[:, :, layer.v_head_dim :]
                max_seqlen_q = (
                    self.flashattention_backend.forward_metadata.max_seq_len_q
                )

                # print(q_rope.shape, k_rope_cache.shape, c_kv_cache.shape, q_nope.shape)
                # torch.Size([1, 16, 64]) torch.Size([320001, 1, 1, 64]) torch.Size([320001, 1, 1, 512]) torch.Size([1, 16, 512])

                assert q_nope.shape[-1] == layer.rope_range[0]
                assert (q_rope.shape[-1] + q_nope.shape[-1]) == layer.rope_range[1]
                q_merged = torch.cat([q_nope, q_rope], dim=-1)
                # TODO FIXME
                # k_cache = torch.cat([c_kv_cache, k_rope_cache], dim=-1)
                k_cache = kv_cache
                v_cache = c_kv_cache

                o, metadata = self.forward_paged_hip(
                    query=q_merged,
                    sm_scale=layer.scaling,
                    batch_size=forward_batch.batch_size,
                    k=None,
                    v=None,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    offload_cache=offload_cache,
                    positions=forward_batch.positions,
                    seq_lens=forward_batch.seq_lens,
                    req_to_tokens=forward_batch.req_to_token_pool.req_to_token,
                    req_pool_indices=forward_batch.req_pool_indices,
                    block_table=self._block_table,
                    rope_cos=layer.rope_cos,
                    rope_sin=layer.rope_sin,
                    rope_range=layer.rope_range,
                    rope_is_neox_style=layer.rope_is_neox_style,
                    layer_id=layer.layer_id,
                    logit_cap=layer.logit_cap,
                    orig_context_len=layer.orig_context_len,
                    max_context_len=self.max_context_len,
                    hip_config=self.hip_config,
                    is_kv_cache_offload_enabled=self.is_kv_cache_offload_enabled,
                    cached_metadata=metadata,
                    online_update_cache=(
                        forward_batch.token_to_kv_pool.is_online_cache_update_enabled()
                        if self.is_kv_cache_offload_enabled
                        else None
                    ),
                    is_decode=True,
                    offloading_metadata=offloading_metadata,
                    sliding_window_size=sw_size,
                    using_chunked_sliding_window=using_chunked_sw,
                )

            if (metadata is not None) and (
                forward_batch.hip_metadata_cache_pool is not None
            ):
                forward_batch.hip_metadata_cache_pool.set_hip_metadata_cache(
                    layer_id=layer.layer_id,
                    tdst=q.shape[0],
                    batch_size=forward_batch.batch_size,
                    metadata=metadata,
                    block_size_q=self.hip_config.block_sparse_block_size_q,
                    cached_stages=forward_batch.hip_metadata_cached_stages,
                )

                if self.is_kv_cache_offload_enabled:
                    offload_cache.handle_cache_miss(metadata)

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)


class HiPAttentionMultiStepBackend:

    def __init__(
        self, model_runner: ModelRunner, topk: int, speculative_num_steps: int
    ):
        self.model_runner = model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends = []
        for i in range(self.speculative_num_steps):
            self.attn_backends.append(
                HiPAttentionBackend(
                    model_runner,
                    speculative_step_id=i,
                    topk=self.topk,
                    speculative_num_steps=self.speculative_num_steps,
                )
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(max_bs)

    def init_forward_metadata_capture_cuda_graph(
        self,
        forward_batch: ForwardBatch,
    ):
        assert forward_batch.spec_info is not None
        assert isinstance(forward_batch.spec_info, EagleDraftInput)

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=forward_batch.encoder_lens,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        assert forward_batch.spec_info is not None
        assert isinstance(forward_batch.spec_info, EagleDraftInput)

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_sum,
                encoder_lens=forward_batch.encoder_lens,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
                seq_lens_cpu=forward_batch.seq_lens_cpu,
                out_cache_loc=forward_batch.out_cache_loc,
            )

```

#### `sglang/srt/server_args.py`

```py
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The arguments of the server."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import random
import tempfile
from typing import TYPE_CHECKING, List, Literal, Optional

from sglang.srt.hf_transformers_utils import check_gguf_file, get_config
from sglang.srt.reasoning_parser import ReasoningParser
from sglang.srt.utils import (
    configure_ipv6,
    get_device,
    get_device_memory_capacity,
    is_flashinfer_available,
    is_hip,
    is_port_available,
    is_remote_url,
    is_valid_ipv6_address,
    nullable_str,
)

if TYPE_CHECKING:
    from hip_attn.v1_2 import HiPAttentionConfig

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ServerArgs:
    # Model and tokenizer
    model_path: str
    tokenizer_path: Optional[str] = None
    tokenizer_mode: str = "auto"
    skip_tokenizer_init: bool = False
    enable_tokenizer_batch_encode: bool = False
    load_format: str = "auto"
    trust_remote_code: bool = False
    dtype: str = "auto"
    kv_cache_dtype: str = "auto"
    quantization: Optional[str] = None
    quantization_param_path: Optional[str] = None
    context_length: Optional[int] = None
    device: Optional[str] = None
    served_model_name: Optional[str] = None
    chat_template: Optional[str] = None
    completion_template: Optional[str] = None
    is_embedding: bool = False
    revision: Optional[str] = None

    # Port for the HTTP server
    host: str = "127.0.0.1"
    port: int = 30000

    # Memory and scheduling
    mem_fraction_static: Optional[float] = None
    max_running_requests: Optional[int] = None
    max_total_tokens: Optional[int] = None
    chunked_prefill_size: Optional[int] = None
    max_prefill_tokens: int = 16384
    schedule_policy: str = "fcfs"
    schedule_conservativeness: float = 1.0
    cpu_offload_gb: int = 0
    page_size: int = 1

    # Other runtime options
    tp_size: int = 1
    pp_size: int = 1
    max_micro_batch_size: Optional[int] = None
    stream_interval: int = 1
    stream_output: bool = False
    random_seed: Optional[int] = None
    constrained_json_whitespace_pattern: Optional[str] = None
    watchdog_timeout: float = 300
    dist_timeout: Optional[int] = None  # timeout for torch.distributed
    download_dir: Optional[str] = None
    base_gpu_id: int = 0
    gpu_id_step: int = 1

    # Logging
    log_level: str = "info"
    log_level_http: Optional[str] = None
    log_requests: bool = False
    log_requests_level: int = 0
    show_time_cost: bool = False
    enable_metrics: bool = False
    decode_log_interval: int = 40

    # API related
    api_key: Optional[str] = None
    file_storage_path: str = "sglang_storage"
    enable_cache_report: bool = False
    reasoning_parser: Optional[str] = None

    # Data parallelism
    dp_size: int = 1
    load_balance_method: str = "round_robin"

    # Expert parallelism
    ep_size: int = 1

    # Multi-node distributed serving
    dist_init_addr: Optional[str] = None
    nnodes: int = 1
    node_rank: int = 0

    # Model override args in JSON
    json_model_override_args: str = "{}"

    # HiP Attention
    enable_hip_attention: bool = False
    hip_attention_config: Optional[HiPAttentionConfig] = None

    # HiP Attention Offload
    enable_hip_kv_cache_offload: bool = False
    # On-GPU cache size for sparse top-k mask estimation, in tokens
    hip_max_mask_cache_factor: float = 1.2
    # If the size is not None, we override hip_max_mask_cache_factor for precise control of cache size.
    hip_max_mask_cache_size: Optional[int] = None
    # On-GPU cache size for sparse attention, in tokens
    hip_max_sa_cache_factor: int = 1.2
    # If the size is not None, we override hip_max_sa_cache_factor for precise control of cache size.
    hip_max_sa_cache_size: Optional[int] = None

    # LoRA
    lora_paths: Optional[List[str]] = None
    max_loras_per_batch: int = 8
    lora_backend: str = "triton"

    # Kernel backend
    attention_backend: Optional[str] = None
    sampling_backend: Optional[str] = None
    grammar_backend: Optional[str] = None

    # Speculative decoding
    speculative_algorithm: Optional[str] = None
    speculative_draft_model_path: Optional[str] = None
    speculative_num_steps: Optional[int] = None
    speculative_eagle_topk: Optional[int] = None
    speculative_num_draft_tokens: Optional[int] = None
    speculative_accept_threshold_single: float = 1.0
    speculative_accept_threshold_acc: float = 1.0
    speculative_token_map: Optional[str] = None

    # Double Sparsity
    enable_double_sparsity: bool = False
    ds_channel_config_path: Optional[str] = None
    ds_heavy_channel_num: int = 32
    ds_heavy_token_num: int = 256
    ds_heavy_channel_type: str = "qk"
    ds_sparse_decode_threshold: int = 4096

    # Optimization/debug options
    disable_radix_cache: bool = False
    disable_cuda_graph: bool = False
    disable_cuda_graph_padding: bool = False
    enable_nccl_nvls: bool = False
    disable_outlines_disk_cache: bool = False
    disable_custom_all_reduce: bool = False
    enable_multimodal: Optional[bool] = None
    disable_overlap_schedule: bool = False
    enable_mixed_chunk: bool = False
    enable_dp_attention: bool = False
    enable_ep_moe: bool = False
    enable_deepep_moe: bool = False
    deepep_mode: Optional[Literal["auto", "normal", "low_latency"]] = "auto"
    enable_torch_compile: bool = False
    torch_compile_max_bs: int = 32
    cuda_graph_max_bs: Optional[int] = None
    cuda_graph_bs: Optional[List[int]] = None
    torchao_config: str = ""
    enable_nan_detection: bool = False
    enable_p2p_check: bool = False
    triton_attention_reduce_in_fp32: bool = False
    triton_attention_num_kv_splits: int = 8
    num_continuous_decode_steps: int = 1
    delete_ckpt_after_loading: bool = False
    enable_memory_saver: bool = False
    allow_auto_truncate: bool = False
    enable_custom_logit_processor: bool = False
    tool_call_parser: Optional[str] = None
    enable_hierarchical_cache: bool = False
    hicache_ratio: float = 2.0
    hicache_size: int = 0
    hicache_write_policy: str = "write_through_selective"
    flashinfer_mla_disable_ragged: bool = False
    warmups: Optional[str] = None
    moe_dense_tp_size: Optional[int] = None
    n_share_experts_fusion: int = 0
    disable_chunked_prefix_cache: bool = False
    disable_fast_image_processor: bool = False
    mm_attention_backend: Optional[str] = None

    # Debug tensor dumps
    debug_tensor_dump_output_folder: Optional[str] = None
    debug_tensor_dump_input_file: Optional[str] = None
    debug_tensor_dump_inject: bool = False

    # For PD disaggregation: can be "null" (not disaggregated), "prefill" (prefill-only), or "decode" (decode-only)
    disaggregation_mode: str = "null"
    disaggregation_bootstrap_port: int = 8998
    disaggregation_transfer_backend: str = "mooncake"
    disaggregation_ib_device: Optional[str] = None
    pdlb_url: Optional[str] = None

    def __post_init__(self):
        # Expert parallelism
        if self.enable_ep_moe:
            self.ep_size = self.tp_size
            logger.warning(
                f"EP MoE is enabled. The expert parallel size is adjusted to be the same as the tensor parallel size[{self.tp_size}]."
            )

        # Set missing default values
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path

        if self.device is None:
            self.device = get_device()

        if self.served_model_name is None:
            self.served_model_name = self.model_path

        if self.random_seed is None:
            self.random_seed = random.randint(0, 1 << 30)

        gpu_mem = get_device_memory_capacity(self.device)

        # Set mem fraction static, which depends on the tensor parallelism size
        if self.mem_fraction_static is None:
            parallel_size = self.tp_size * self.pp_size
            if gpu_mem <= 81920:
                if parallel_size >= 16:
                    self.mem_fraction_static = 0.79
                elif parallel_size >= 8:
                    self.mem_fraction_static = 0.81
                elif parallel_size >= 4:
                    self.mem_fraction_static = 0.85
                elif parallel_size >= 2:
                    self.mem_fraction_static = 0.87
                else:
                    self.mem_fraction_static = 0.88
            else:
                self.mem_fraction_static = 0.88
            if gpu_mem > 96 * 1024:
                mem_fraction = self.mem_fraction_static
                self.mem_fraction_static = min(
                    mem_fraction + 48 * 1024 * (1 - mem_fraction) / gpu_mem,
                    (gpu_mem - 1024 * 18)
                    / gpu_mem,  # 15 GB + additional 3GB for cuda graph
                )

        # Set chunked prefill size, which depends on the gpu memory capacity
        if self.chunked_prefill_size is None:
            if gpu_mem is not None and gpu_mem < 25_000:
                self.chunked_prefill_size = 2048
            elif self.disaggregation_mode != "null":
                self.chunked_prefill_size = 16384
            else:
                self.chunked_prefill_size = 8192
        assert self.chunked_prefill_size % self.page_size == 0

        assert self.moe_dense_tp_size in {
            1,
            None,
        }, "moe_dense_tp_size only support 1 and None currently"

        if self.attention_backend == "flashmla":
            logger.warning(
                "FlashMLA only supports a page_size of 64, change page_size to 64."
            )
            self.page_size = 64

        if self.attention_backend == "cutlass_mla":
            logger.warning(
                "Cutlass MLA only supports a page_size of 128, change page_size to 128."
            )
            self.page_size = 128

        # Set cuda graph max batch size
        if self.cuda_graph_max_bs is None:
            # Based on detailed statistics, when serving TP1/TP2 models on lower-end GPUs with HBM<25G, you can either disable cuda graph or set `cuda_graph_max_bs` to a very small value to reduce the memory overhead of creating cuda graphs, with almost no impact on performance. However, when serving models with TP4 or TP8, we need to enable cuda graph to maintain high performance. In this case, we can set `cuda_graph_max_bs` to 80 (half of the default value 160) to reduce the memory overhead of creating cuda graphs. Looking at the logs from TP4 serving of qwen2-72b, a value of 80 is sufficient and can reduce the memory overhead of creating cuda graphs on lower-end GPUs compared to the original 160, avoiding OOM issues.
            if gpu_mem is not None and gpu_mem < 25_000:
                if self.tp_size < 4:
                    self.cuda_graph_max_bs = 8
                else:
                    self.cuda_graph_max_bs = 80

        # Set kernel backends for hpu device
        if self.device == "hpu":
            self.attention_backend = "torch_native"
            self.sampling_backend = "pytorch"

        # Set kernel backends
        if self.sampling_backend is None:
            self.sampling_backend = (
                "flashinfer" if is_flashinfer_available() else "pytorch"
            )

        if self.attention_backend == "torch_native":
            logger.warning(
                "Cuda graph is disabled because of using torch native attention backend"
            )
            self.disable_cuda_graph = True

        # Choose grammar backend
        if self.grammar_backend is None:
            self.grammar_backend = "xgrammar"

        # Data parallelism attention
        if self.enable_dp_attention:
            self.schedule_conservativeness = self.schedule_conservativeness * 0.3
            assert (
                self.dp_size > 1
            ), "Please set a dp-size > 1. You can use 1 < dp-size <= tp-size "
            assert self.tp_size % self.dp_size == 0
            self.chunked_prefill_size = self.chunked_prefill_size // self.dp_size
            logger.warning(
                f"DP attention is enabled. The chunked prefill size is adjusted to {self.chunked_prefill_size} to avoid MoE kernel issues. "
            )

        # DeepEP MoE
        self.enable_sp_layernorm = False
        if self.enable_deepep_moe:
            if self.deepep_mode == "auto":
                assert (
                    not self.enable_dp_attention
                ), "DeepEP MoE `auto` mode is not supported with DP Attention."
            if self.deepep_mode == "normal":
                logger.warning("Cuda graph is disabled because deepep_mode=`normal`")
                self.disable_cuda_graph = True
            self.ep_size = self.tp_size
            self.enable_sp_layernorm = (
                self.dp_size < self.tp_size if self.enable_dp_attention else True
            )
            logger.warning(
                f"DeepEP MoE is enabled. The expert parallel size is adjusted to be the same as the tensor parallel size[{self.tp_size}]."
            )

        # Speculative Decoding
        if self.speculative_algorithm == "NEXTN":
            # NEXTN shares the same implementation of EAGLE
            self.speculative_algorithm = "EAGLE"

        if self.speculative_algorithm in ("EAGLE", "EAGLE3"):
            if self.max_running_requests is None:
                self.max_running_requests = 48
            self.disable_overlap_schedule = True
            logger.warning(
                "Overlap scheduler is disabled because of using "
                "eagle speculative decoding."
            )

            model_arch = get_model_arch(self)

            # Auto set draft_model_path DeepSeek-V3/R1
            if model_arch == "DeepseekV3ForCausalLM":
                if self.speculative_draft_model_path is None:
                    self.speculative_draft_model_path = self.model_path
                else:
                    logger.warning(
                        "DeepSeek MTP does not require setting speculative_draft_model_path."
                    )

            # Auto choose parameters
            if self.speculative_num_steps is None:
                assert (
                    self.speculative_eagle_topk is None
                    and self.speculative_num_draft_tokens is None
                )
                (
                    self.speculative_num_steps,
                    self.speculative_eagle_topk,
                    self.speculative_num_draft_tokens,
                ) = auto_choose_speculative_params(model_arch)

            if self.page_size > 1 and self.speculative_eagle_topk > 1:
                self.speculative_eagle_topk = 1
                logger.warning(
                    "speculative_eagle_topk is adjusted to 1 when page_size > 1"
                )

            if (
                self.speculative_eagle_topk == 1
                and self.speculative_num_draft_tokens != self.speculative_num_steps + 1
            ):
                logger.warning(
                    "speculative_num_draft_tokens is adjusted to speculative_num_steps + 1 when speculative_eagle_topk == 1"
                )
                self.speculative_num_draft_tokens = self.speculative_num_steps + 1

            # The token generated from the verify step is counted.
            # If sepculative_num_steps >= speculative_num_draft_tokens, the additional tokens will definitely be discarded.
            # assert self.speculative_num_steps < self.speculative_num_draft_tokens

        # GGUF
        if (
            self.load_format == "auto" or self.load_format == "gguf"
        ) and check_gguf_file(self.model_path):
            self.quantization = self.load_format = "gguf"

        if is_remote_url(self.model_path):
            self.load_format = "remote"

        # AMD-specific Triton attention KV splits default number
        if is_hip():
            self.triton_attention_num_kv_splits = 16

        # PD disaggregation
        if self.disaggregation_mode == "prefill":
            self.disable_cuda_graph = True
            logger.warning("Cuda graph is disabled for prefill server")
        elif self.disaggregation_mode == "decode":
            self.disable_radix_cache = True
            logger.warning("KV cache is forced as chunk cache for decode server")

        os.environ["SGLANG_ENABLE_TORCH_COMPILE"] = (
            "1" if self.enable_torch_compile else "0"
        )
        # Set env var before grammar backends init
        os.environ["SGLANG_DISABLE_OUTLINES_DISK_CACHE"] = (
            "1" if self.disable_outlines_disk_cache else "0"
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        # Model and port args
        parser.add_argument(
            "--model-path",
            type=str,
            help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
            required=True,
        )
        parser.add_argument(
            "--tokenizer-path",
            type=str,
            default=ServerArgs.tokenizer_path,
            help="The path of the tokenizer.",
        )
        parser.add_argument(
            "--host", type=str, default=ServerArgs.host, help="The host of the server."
        )
        parser.add_argument(
            "--port", type=int, default=ServerArgs.port, help="The port of the server."
        )
        parser.add_argument(
            "--tokenizer-mode",
            type=str,
            default=ServerArgs.tokenizer_mode,
            choices=["auto", "slow"],
            help="Tokenizer mode. 'auto' will use the fast "
            "tokenizer if available, and 'slow' will "
            "always use the slow tokenizer.",
        )
        parser.add_argument(
            "--skip-tokenizer-init",
            action="store_true",
            help="If set, skip init tokenizer and pass input_ids in generate request.",
        )
        parser.add_argument(
            "--enable-tokenizer-batch-encode",
            action="store_true",
            help="Enable batch tokenization for improved performance when processing multiple text inputs. Do not use with image inputs, pre-tokenized input_ids, or input_embeds.",
        )
        parser.add_argument(
            "--load-format",
            type=str,
            default=ServerArgs.load_format,
            choices=[
                "auto",
                "pt",
                "safetensors",
                "npcache",
                "dummy",
                "sharded_state",
                "gguf",
                "bitsandbytes",
                "layered",
                "remote",
            ],
            help="The format of the model weights to load. "
            '"auto" will try to load the weights in the safetensors format '
            "and fall back to the pytorch bin format if safetensors format "
            "is not available. "
            '"pt" will load the weights in the pytorch bin format. '
            '"safetensors" will load the weights in the safetensors format. '
            '"npcache" will load the weights in pytorch format and store '
            "a numpy cache to speed up the loading. "
            '"dummy" will initialize the weights with random values, '
            "which is mainly for profiling."
            '"gguf" will load the weights in the gguf format. '
            '"bitsandbytes" will load the weights using bitsandbytes '
            "quantization."
            '"layered" loads weights layer by layer so that one can quantize a '
            "layer before loading another to make the peak memory envelope "
            "smaller.",
        )
        parser.add_argument(
            "--trust-remote-code",
            action="store_true",
            help="Whether or not to allow for custom models defined on the Hub in their own modeling files.",
        )
        parser.add_argument(
            "--dtype",
            type=str,
            default=ServerArgs.dtype,
            choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
            help="Data type for model weights and activations.\n\n"
            '* "auto" will use FP16 precision for FP32 and FP16 models, and '
            "BF16 precision for BF16 models.\n"
            '* "half" for FP16. Recommended for AWQ quantization.\n'
            '* "float16" is the same as "half".\n'
            '* "bfloat16" for a balance between precision and range.\n'
            '* "float" is shorthand for FP32 precision.\n'
            '* "float32" for FP32 precision.',
        )
        parser.add_argument(
            "--kv-cache-dtype",
            type=str,
            default=ServerArgs.kv_cache_dtype,
            choices=["auto", "fp8_e5m2", "fp8_e4m3"],
            help='Data type for kv cache storage. "auto" will use model data type. "fp8_e5m2" and "fp8_e4m3" is supported for CUDA 11.8+.',
        )
        parser.add_argument(
            "--quantization",
            type=str,
            default=ServerArgs.quantization,
            choices=[
                "awq",
                "fp8",
                "gptq",
                "marlin",
                "gptq_marlin",
                "awq_marlin",
                "bitsandbytes",
                "gguf",
                "modelopt",
                "modelopt_fp4",
                "w8a8_int8",
                "w8a8_fp8",
                "moe_wna16",
            ],
            help="The quantization method.",
        )
        parser.add_argument(
            "--quantization-param-path",
            type=nullable_str,
            default=None,
            help="Path to the JSON file containing the KV cache "
            "scaling factors. This should generally be supplied, when "
            "KV cache dtype is FP8. Otherwise, KV cache scaling factors "
            "default to 1.0, which may cause accuracy issues. ",
        )
        parser.add_argument(
            "--context-length",
            type=int,
            default=ServerArgs.context_length,
            help="The model's maximum context length. Defaults to None (will use the value from the model's config.json instead).",
        )
        parser.add_argument(
            "--device",
            type=str,
            default=ServerArgs.device,
            help="The device to use ('cuda', 'xpu', 'hpu', 'npu', 'cpu'). Defaults to auto-detection if not specified.",
        )
        parser.add_argument(
            "--served-model-name",
            type=str,
            default=ServerArgs.served_model_name,
            help="Override the model name returned by the v1/models endpoint in OpenAI API server.",
        )
        parser.add_argument(
            "--chat-template",
            type=str,
            default=ServerArgs.chat_template,
            help="The buliltin chat template name or the path of the chat template file. This is only used for OpenAI-compatible API server.",
        )
        parser.add_argument(
            "--completion-template",
            type=str,
            default=ServerArgs.completion_template,
            help="The buliltin completion template name or the path of the completion template file. This is only used for OpenAI-compatible API server. only for code completion currently.",
        )
        parser.add_argument(
            "--is-embedding",
            action="store_true",
            help="Whether to use a CausalLM as an embedding model.",
        )
        parser.add_argument(
            "--revision",
            type=str,
            default=None,
            help="The specific model version to use. It can be a branch "
            "name, a tag name, or a commit id. If unspecified, will use "
            "the default version.",
        )

        # Memory and scheduling
        parser.add_argument(
            "--mem-fraction-static",
            type=float,
            default=ServerArgs.mem_fraction_static,
            help="The fraction of the memory used for static allocation (model weights and KV cache memory pool). Use a smaller value if you see out-of-memory errors.",
        )
        parser.add_argument(
            "--max-running-requests",
            type=int,
            default=ServerArgs.max_running_requests,
            help="The maximum number of running requests.",
        )
        parser.add_argument(
            "--max-total-tokens",
            type=int,
            default=ServerArgs.max_total_tokens,
            help="The maximum number of tokens in the memory pool. If not specified, it will be automatically calculated based on the memory usage fraction. "
            "This option is typically used for development and debugging purposes.",
        )
        parser.add_argument(
            "--chunked-prefill-size",
            type=int,
            default=ServerArgs.chunked_prefill_size,
            help="The maximum number of tokens in a chunk for the chunked prefill. Setting this to -1 means disabling chunked prefill.",
        )
        parser.add_argument(
            "--max-prefill-tokens",
            type=int,
            default=ServerArgs.max_prefill_tokens,
            help="The maximum number of tokens in a prefill batch. The real bound will be the maximum of this value and the model's maximum context length.",
        )
        parser.add_argument(
            "--schedule-policy",
            type=str,
            default=ServerArgs.schedule_policy,
            choices=["lpm", "random", "fcfs", "dfs-weight"],
            help="The scheduling policy of the requests.",
        )
        parser.add_argument(
            "--schedule-conservativeness",
            type=float,
            default=ServerArgs.schedule_conservativeness,
            help="How conservative the schedule policy is. A larger value means more conservative scheduling. Use a larger value if you see requests being retracted frequently.",
        )
        parser.add_argument(
            "--cpu-offload-gb",
            type=int,
            default=ServerArgs.cpu_offload_gb,
            help="How many GBs of RAM to reserve for CPU offloading.",
        )
        parser.add_argument(
            "--page-size",
            type=int,
            default=ServerArgs.page_size,
            help="The number of tokens in a page.",
        )

        # Other runtime options
        parser.add_argument(
            "--tensor-parallel-size",
            "--tp-size",
            type=int,
            default=ServerArgs.tp_size,
            help="The tensor parallelism size.",
        )
        parser.add_argument(
            "--pipeline-parallel-size",
            "--pp-size",
            type=int,
            default=ServerArgs.pp_size,
            help="The pipeline parallelism size.",
        )
        parser.add_argument(
            "--max-micro-batch-size",
            type=int,
            default=ServerArgs.max_micro_batch_size,
            help="The maximum micro batch size in pipeline parallelism.",
        )
        parser.add_argument(
            "--stream-interval",
            type=int,
            default=ServerArgs.stream_interval,
            help="The interval (or buffer size) for streaming in terms of the token length. A smaller value makes streaming smoother, while a larger value makes the throughput higher",
        )
        parser.add_argument(
            "--stream-output",
            action="store_true",
            help="Whether to output as a sequence of disjoint segments.",
        )
        parser.add_argument(
            "--random-seed",
            type=int,
            default=ServerArgs.random_seed,
            help="The random seed.",
        )
        parser.add_argument(
            "--constrained-json-whitespace-pattern",
            type=str,
            default=ServerArgs.constrained_json_whitespace_pattern,
            help=r"Regex pattern for syntactic whitespaces allowed in JSON constrained output. For example, to allow the model generate consecutive whitespaces, set the pattern to [\n\t ]*",
        )
        parser.add_argument(
            "--watchdog-timeout",
            type=float,
            default=ServerArgs.watchdog_timeout,
            help="Set watchdog timeout in seconds. If a forward batch takes longer than this, the server will crash to prevent hanging.",
        )
        parser.add_argument(
            "--dist-timeout",
            type=int,
            default=ServerArgs.dist_timeout,
            help="Set timeout for torch.distributed initialization.",
        )
        parser.add_argument(
            "--download-dir",
            type=str,
            default=ServerArgs.download_dir,
            help="Model download directory for huggingface.",
        )
        parser.add_argument(
            "--base-gpu-id",
            type=int,
            default=ServerArgs.base_gpu_id,
            help="The base GPU ID to start allocating GPUs from. Useful when running multiple instances on the same machine.",
        )
        parser.add_argument(
            "--gpu-id-step",
            type=int,
            default=ServerArgs.gpu_id_step,
            help="The delta between consecutive GPU IDs that are used. For example, setting it to 2 will use GPU 0,2,4,...",
        )

        # Logging
        parser.add_argument(
            "--log-level",
            type=str,
            default=ServerArgs.log_level,
            help="The logging level of all loggers.",
        )
        parser.add_argument(
            "--log-level-http",
            type=str,
            default=ServerArgs.log_level_http,
            help="The logging level of HTTP server. If not set, reuse --log-level by default.",
        )
        parser.add_argument(
            "--log-requests",
            action="store_true",
            help="Log metadata, inputs, outputs of all requests. The verbosity is decided by --log-requests-level",
        )
        parser.add_argument(
            "--log-requests-level",
            type=int,
            default=0,
            help="0: Log metadata. 1. Log metadata and partial input/output. 2. Log every input/output.",
            choices=[0, 1, 2],
        )
        parser.add_argument(
            "--show-time-cost",
            action="store_true",
            help="Show time cost of custom marks.",
        )
        parser.add_argument(
            "--enable-metrics",
            action="store_true",
            help="Enable log prometheus metrics.",
        )
        parser.add_argument(
            "--decode-log-interval",
            type=int,
            default=ServerArgs.decode_log_interval,
            help="The log interval of decode batch.",
        )

        # API related
        parser.add_argument(
            "--api-key",
            type=str,
            default=ServerArgs.api_key,
            help="Set API key of the server. It is also used in the OpenAI API compatible server.",
        )
        parser.add_argument(
            "--file-storage-path",
            type=str,
            default=ServerArgs.file_storage_path,
            help="The path of the file storage in backend.",
        )
        parser.add_argument(
            "--enable-cache-report",
            action="store_true",
            help="Return number of cached tokens in usage.prompt_tokens_details for each openai request.",
        )
        parser.add_argument(
            "--reasoning-parser",
            type=str,
            choices=list(ReasoningParser.DetectorMap.keys()),
            default=ServerArgs.reasoning_parser,
            help=f"Specify the parser for reasoning models, supported parsers are: {list(ReasoningParser.DetectorMap.keys())}.",
        )

        # Data parallelism
        parser.add_argument(
            "--data-parallel-size",
            "--dp-size",
            type=int,
            default=ServerArgs.dp_size,
            help="The data parallelism size.",
        )
        parser.add_argument(
            "--load-balance-method",
            type=str,
            default=ServerArgs.load_balance_method,
            help="The load balancing strategy for data parallelism.",
            choices=[
                "round_robin",
                "shortest_queue",
            ],
        )

        # Expert parallelism
        parser.add_argument(
            "--expert-parallel-size",
            "--ep-size",
            type=int,
            default=ServerArgs.ep_size,
            help="The expert parallelism size.",
        )

        # Multi-node distributed serving
        parser.add_argument(
            "--dist-init-addr",
            "--nccl-init-addr",  # For backward compatbility. This will be removed in the future.
            type=str,
            help="The host address for initializing distributed backend (e.g., `192.168.0.2:25000`).",
        )
        parser.add_argument(
            "--nnodes", type=int, default=ServerArgs.nnodes, help="The number of nodes."
        )
        parser.add_argument(
            "--node-rank", type=int, default=ServerArgs.node_rank, help="The node rank."
        )

        # Model override args
        parser.add_argument(
            "--json-model-override-args",
            type=str,
            help="A dictionary in JSON string format used to override default model configurations.",
            default=ServerArgs.json_model_override_args,
        )

        # HiP Attention
        parser.add_argument(
            "--enable-hip-attention",
            action="store_true",
            help="Enable HiP attention. This flag is not compatible with other sparse attention flags (e.g., double sparsity).",
        )
        parser.add_argument(
            "--hip-attention-config",
            "--hip-attention-config-path",
            type=str,
            default=ServerArgs.hip_attention_config,
            help="Path to the HiP attention config file, or the json in string format.",
        )
        parser.add_argument(
            "--hip-attention-config-override-json",
            type=str,
            default=None,
            help="JSON string to override imported HiP Attention configs.",
        )

        # HiP Attention Offload
        parser.add_argument(
            "--enable-hip-kv-cache-offload",
            action="store_true",
            help="Enable HiP KV cache offloading. This option should be set with --enable-hip-attention.",
        )
        parser.add_argument(
            "--hip-max-mask-cache-factor",
            type=float,
            default=ServerArgs.hip_max_mask_cache_factor,
            help=(
                "On-GPU cache size factor for HiP sparse top-k mask estimation kernels. "
                "A cache of size proportional to this value will be allocated on the GPU. "
                "This will be a major determining factor for mask-refreshing decoding step latency."
            ),
        )
        parser.add_argument(
            "--hip-max-mask-cache-size",
            type=int,
            default=ServerArgs.hip_max_mask_cache_size,
            help=(
                "On-GPU cache size for HiP sparse top-k mask estimation kernels. "
                "Overrides --hip-max-sa-cache-factor. Only use this for precise control of the cache size."
            ),
        )
        parser.add_argument(
            "--hip-max-sa-cache-factor",
            type=float,
            default=ServerArgs.hip_max_sa_cache_factor,
            help=(
                "On-GPU cache size factor for HiP sparse attention kernels, in tokens per layer. "
                "A cache of size proportional to this value will be allocated on the GPU`. "
                "This will be a major determining factor for mask-cached decoding step latency."
            ),
        )
        parser.add_argument(
            "--hip-max-sa-cache-size",
            type=int,
            default=ServerArgs.hip_max_sa_cache_size,
            help=(
                "On-GPU cache size for HiP sparse attention kernels, in tokens per layer. "
                "Overrides --hip-max-sa-cache-factor. Only use this for precise control of the cache size."
            ),
        )

        # LoRA
        parser.add_argument(
            "--lora-paths",
            type=str,
            nargs="*",
            default=None,
            action=LoRAPathAction,
            help="The list of LoRA adapters. You can provide a list of either path in str or renamed path in the format {name}={path}.",
        )
        parser.add_argument(
            "--max-loras-per-batch",
            type=int,
            default=8,
            help="Maximum number of adapters for a running batch, include base-only request.",
        )
        parser.add_argument(
            "--lora-backend",
            type=str,
            default="triton",
            help="Choose the kernel backend for multi-LoRA serving.",
        )

        # Kernel backend
        parser.add_argument(
            "--attention-backend",
            type=str,
            choices=[
                "flashinfer",
                "triton",
                "torch_native",
                "fa3",
                "flashmla",
                "cutlass_mla",
                "hip_attention",
            ],
            default=ServerArgs.attention_backend,
            help="Choose the kernels for attention layers.",
        )
        parser.add_argument(
            "--sampling-backend",
            type=str,
            choices=["flashinfer", "pytorch"],
            default=ServerArgs.sampling_backend,
            help="Choose the kernels for sampling layers.",
        )
        parser.add_argument(
            "--grammar-backend",
            type=str,
            choices=["xgrammar", "outlines", "llguidance", "none"],
            default=ServerArgs.grammar_backend,
            help="Choose the backend for grammar-guided decoding.",
        )
        parser.add_argument(
            "--enable-flashinfer-mla",
            action=DeprecatedAction,
            help="--enable-flashinfer-mla is deprecated. Please use '--attention-backend flashinfer' instead.",
        )
        parser.add_argument(
            "--enable-flashmla",
            action=DeprecatedAction,
            help="--enable-flashmla is deprecated. Please use '--attention-backend flashmla' instead.",
        )
        parser.add_argument(
            "--flashinfer-mla-disable-ragged",
            action="store_true",
            help="Not using ragged prefill wrapper when running flashinfer mla",
        )

        # Speculative decoding
        parser.add_argument(
            "--speculative-algorithm",
            type=str,
            choices=["EAGLE", "EAGLE3", "NEXTN"],
            help="Speculative algorithm.",
        )
        parser.add_argument(
            "--speculative-draft-model-path",
            type=str,
            help="The path of the draft model weights. This can be a local folder or a Hugging Face repo ID.",
        )
        parser.add_argument(
            "--speculative-num-steps",
            type=int,
            help="The number of steps sampled from draft model in Speculative Decoding.",
            default=ServerArgs.speculative_num_steps,
        )
        parser.add_argument(
            "--speculative-eagle-topk",
            type=int,
            help="The number of tokens sampled from the draft model in eagle2 each step.",
            default=ServerArgs.speculative_eagle_topk,
        )
        parser.add_argument(
            "--speculative-num-draft-tokens",
            type=int,
            help="The number of tokens sampled from the draft model in Speculative Decoding.",
            default=ServerArgs.speculative_num_draft_tokens,
        )
        parser.add_argument(
            "--speculative-accept-threshold-single",
            type=float,
            help="Accept a draft token if its probability in the target model is greater than this threshold.",
            default=ServerArgs.speculative_accept_threshold_single,
        )
        parser.add_argument(
            "--speculative-accept-threshold-acc",
            type=float,
            help="The accept probability of a draft token is raised from its target probability p to min(1, p / threshold_acc).",
            default=ServerArgs.speculative_accept_threshold_acc,
        )
        parser.add_argument(
            "--speculative-token-map",
            type=str,
            help="The path of the draft model's small vocab table.",
            default=ServerArgs.speculative_token_map,
        )

        # Double Sparsity
        parser.add_argument(
            "--enable-double-sparsity",
            action="store_true",
            help="Enable double sparsity attention",
        )
        parser.add_argument(
            "--ds-channel-config-path",
            type=str,
            default=ServerArgs.ds_channel_config_path,
            help="The path of the double sparsity channel config",
        )
        parser.add_argument(
            "--ds-heavy-channel-num",
            type=int,
            default=ServerArgs.ds_heavy_channel_num,
            help="The number of heavy channels in double sparsity attention",
        )
        parser.add_argument(
            "--ds-heavy-token-num",
            type=int,
            default=ServerArgs.ds_heavy_token_num,
            help="The number of heavy tokens in double sparsity attention",
        )
        parser.add_argument(
            "--ds-heavy-channel-type",
            type=str,
            default=ServerArgs.ds_heavy_channel_type,
            help="The type of heavy channels in double sparsity attention",
        )
        parser.add_argument(
            "--ds-sparse-decode-threshold",
            type=int,
            default=ServerArgs.ds_sparse_decode_threshold,
            help="The type of heavy channels in double sparsity attention",
        )

        # Optimization/debug options
        parser.add_argument(
            "--disable-radix-cache",
            action="store_true",
            help="Disable RadixAttention for prefix caching.",
        )
        parser.add_argument(
            "--disable-cuda-graph",
            action="store_true",
            help="Disable cuda graph.",
        )
        parser.add_argument(
            "--disable-cuda-graph-padding",
            action="store_true",
            help="Disable cuda graph when padding is needed. Still uses cuda graph when padding is not needed.",
        )
        parser.add_argument(
            "--enable-nccl-nvls",
            action="store_true",
            help="Enable NCCL NVLS for prefill heavy requests when available.",
        )
        parser.add_argument(
            "--disable-outlines-disk-cache",
            action="store_true",
            help="Disable disk cache of outlines to avoid possible crashes related to file system or high concurrency.",
        )
        parser.add_argument(
            "--disable-custom-all-reduce",
            action="store_true",
            help="Disable the custom all-reduce kernel and fall back to NCCL.",
        )
        parser.add_argument(
            "--enable-multimodal",
            default=ServerArgs.enable_multimodal,
            action="store_true",
            help="Enable the multimodal functionality for the served model. If the model being served is not multimodal, nothing will happen",
        )
        parser.add_argument(
            "--disable-overlap-schedule",
            action="store_true",
            help="Disable the overlap scheduler, which overlaps the CPU scheduler with GPU model worker.",
        )
        parser.add_argument(
            "--enable-mixed-chunk",
            action="store_true",
            help="Enabling mixing prefill and decode in a batch when using chunked prefill.",
        )
        parser.add_argument(
            "--enable-dp-attention",
            action="store_true",
            help="Enabling data parallelism for attention and tensor parallelism for FFN. The dp size should be equal to the tp size. Currently only DeepSeek-V2 is supported.",
        )
        parser.add_argument(
            "--enable-ep-moe",
            action="store_true",
            help="Enabling expert parallelism for moe. The ep size is equal to the tp size.",
        )
        parser.add_argument(
            "--enable-torch-compile",
            action="store_true",
            help="Optimize the model with torch.compile. Experimental feature.",
        )
        parser.add_argument(
            "--torch-compile-max-bs",
            type=int,
            default=ServerArgs.torch_compile_max_bs,
            help="Set the maximum batch size when using torch compile.",
        )
        parser.add_argument(
            "--cuda-graph-max-bs",
            type=int,
            default=ServerArgs.cuda_graph_max_bs,
            help="Set the maximum batch size for cuda graph.",
        )
        parser.add_argument(
            "--cuda-graph-bs",
            type=int,
            nargs="+",
            help="Set the list of batch sizes for cuda graph.",
        )
        parser.add_argument(
            "--torchao-config",
            type=str,
            default=ServerArgs.torchao_config,
            help="Optimize the model with torchao. Experimental feature. Current choices are: int8dq, int8wo, int4wo-<group_size>, fp8wo, fp8dq-per_tensor, fp8dq-per_row",
        )
        parser.add_argument(
            "--enable-nan-detection",
            action="store_true",
            help="Enable the NaN detection for debugging purposes.",
        )
        parser.add_argument(
            "--enable-p2p-check",
            action="store_true",
            help="Enable P2P check for GPU access, otherwise the p2p access is allowed by default.",
        )
        parser.add_argument(
            "--triton-attention-reduce-in-fp32",
            action="store_true",
            help="Cast the intermidiate attention results to fp32 to avoid possible crashes related to fp16."
            "This only affects Triton attention kernels.",
        )
        parser.add_argument(
            "--triton-attention-num-kv-splits",
            type=int,
            default=ServerArgs.triton_attention_num_kv_splits,
            help="The number of KV splits in flash decoding Triton kernel. Larger value is better in longer context scenarios. The default value is 8.",
        )
        parser.add_argument(
            "--num-continuous-decode-steps",
            type=int,
            default=ServerArgs.num_continuous_decode_steps,
            help="Run multiple continuous decoding steps to reduce scheduling overhead. "
            "This can potentially increase throughput but may also increase time-to-first-token latency. "
            "The default value is 1, meaning only run one decoding step at a time.",
        )
        parser.add_argument(
            "--delete-ckpt-after-loading",
            action="store_true",
            help="Delete the model checkpoint after loading the model.",
        )
        parser.add_argument(
            "--enable-memory-saver",
            action="store_true",
            help="Allow saving memory using release_memory_occupation and resume_memory_occupation",
        )
        parser.add_argument(
            "--allow-auto-truncate",
            action="store_true",
            help="Allow automatically truncating requests that exceed the maximum input length instead of returning an error.",
        )
        parser.add_argument(
            "--enable-custom-logit-processor",
            action="store_true",
            help="Enable users to pass custom logit processors to the server (disabled by default for security)",
        )
        parser.add_argument(
            "--tool-call-parser",
            type=str,
            choices=["qwen25", "mistral", "llama3", "deepseekv3", "pythonic"],
            default=ServerArgs.tool_call_parser,
            help="Specify the parser for handling tool-call interactions. Options include: 'qwen25', 'mistral', 'llama3', 'deepseekv3', and 'pythonic'.",
        )
        parser.add_argument(
            "--enable-hierarchical-cache",
            action="store_true",
            help="Enable hierarchical cache",
        )
        parser.add_argument(
            "--hicache-ratio",
            type=float,
            default=ServerArgs.hicache_ratio,
            help="The ratio of the size of host KV cache memory pool to the size of device pool.",
        )
        parser.add_argument(
            "--hicache-size",
            type=int,
            default=ServerArgs.hicache_size,
            help="The size of host KV cache memory pool in gigabytes, which will override the hicache_ratio if set.",
        )
        parser.add_argument(
            "--hicache-write-policy",
            type=str,
            choices=["write_back", "write_through", "write_through_selective"],
            default=ServerArgs.hicache_write_policy,
            help="The write policy of hierarchical cache.",
        )
        parser.add_argument(
            "--enable-deepep-moe",
            action="store_true",
            help="Enabling DeepEP MoE implementation for EP MoE.",
        )
        parser.add_argument(
            "--moe-dense-tp-size",
            type=int,
            default=ServerArgs.moe_dense_tp_size,
            help="TP size for MoE dense MLP layers. This flag is useful when, with large TP size, there are errors caused by weights in MLP layers having dimension smaller than the min dimension GEMM supports.",
        )
        parser.add_argument(
            "--deepep-mode",
            type=str,
            choices=["normal", "low_latency", "auto"],
            default="auto",
            help="Select the mode when enable DeepEP MoE, could be `normal`, `low_latency` or `auto`. Default is `auto`, which means `low_latency` for decode batch and `normal` for prefill batch.",
        )

        parser.add_argument(
            "--n-share-experts-fusion",
            type=int,
            default=0,
            help="The number of shared_experts need to be replicated to fuse with normal experts in deepseek v3/r1, "
            "set it to tp_size can get best optimized performace.",
        )
        parser.add_argument(
            "--disable-chunked-prefix-cache",
            action="store_true",
            help="Disable chunked prefix cache feature for deepseek, which should save overhead for short sequences.",
        )
        parser.add_argument(
            "--disable-fast-image-processor",
            action="store_true",
            help="Adopt base image processor instead of fast image processor.",
        )

        # Server warmups
        parser.add_argument(
            "--warmups",
            type=str,
            required=False,
            help="Specify custom warmup functions (csv) to run before server starts eg. --warmups=warmup_name1,warmup_name2 "
            "will run the functions `warmup_name1` and `warmup_name2` specified in warmup.py before the server starts listening for requests",
        )

        # Debug tensor dumps
        parser.add_argument(
            "--debug-tensor-dump-output-folder",
            type=str,
            default=ServerArgs.debug_tensor_dump_output_folder,
            help="The output folder for dumping tensors.",
        )
        parser.add_argument(
            "--debug-tensor-dump-input-file",
            type=str,
            default=ServerArgs.debug_tensor_dump_input_file,
            help="The input filename for dumping tensors",
        )
        parser.add_argument(
            "--debug-tensor-dump-inject",
            type=str,
            default=ServerArgs.debug_tensor_dump_inject,
            help="Inject the outputs from jax as the input of every layer.",
        )

        # Disaggregation
        parser.add_argument(
            "--disaggregation-mode",
            type=str,
            default="null",
            choices=["null", "prefill", "decode"],
            help='Only used for PD disaggregation. "prefill" for prefill-only server, and "decode" for decode-only server. If not specified, it is not PD disaggregated',
        )
        parser.add_argument(
            "--disaggregation-bootstrap-port",
            type=int,
            default=ServerArgs.disaggregation_bootstrap_port,
            help="Bootstrap server port on the prefill server. Default is 8998.",
        )
        parser.add_argument(
            "--disaggregation-transfer-backend",
            type=str,
            default=ServerArgs.disaggregation_transfer_backend,
            choices=["mooncake", "nixl"],
            help="The backend for disaggregation transfer. Default is mooncake.",
        )
        parser.add_argument(
            "--disaggregation-ib-device",
            type=str,
            default=ServerArgs.disaggregation_ib_device,
            help="The InfiniBand devices for disaggregation transfer, accepts single device (e.g., --disaggregation-ib-device mlx5_0) "
            "or multiple comma-separated devices (e.g., --disaggregation-ib-device mlx5_0,mlx5_1). "
            "Default is None, which triggers automatic device detection when mooncake backend is enabled.",
        )
        parser.add_argument(
            "--pdlb-url",
            type=str,
            default=None,
            help="The URL of the PD disaggregation load balancer. If set, the prefill/decode server will register with the load balancer.",
        )

        parser.add_argument(
            "--mm-attention-backend",
            type=str,
            choices=["sdpa", "fa3", "triton_attn"],
            default=ServerArgs.mm_attention_backend,
            help="Set multimodal attention backend.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        args.tp_size = args.tensor_parallel_size
        args.pp_size = args.pipeline_parallel_size
        args.dp_size = args.data_parallel_size
        args.ep_size = args.expert_parallel_size

        if args.attention_backend == "hip_attention":
            args.enable_hip_attention = True

        if args.enable_hip_attention:
            from hip_attn.v1_2 import HiPAttentionConfig

            json_or_path = args.hip_attention_config

            args.hip_attention_config = HiPAttentionConfig(
                json_or_path=json_or_path,
                json_override=args.hip_attention_config_override_json,
            )
            if args.attention_backend != 'hip_attention':
                logger.info(
                    f"attention_backend changed {args.attention_backend} -> hip_attention"
                )
            args.attention_backend = "hip_attention"
        else:
            args.hip_attention_config = None

        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})

    def url(self):
        if is_valid_ipv6_address(self.host):
            return f"http://[{self.host}]:{self.port}"
        else:
            return f"http://{self.host}:{self.port}"

    def check_server_args(self):
        assert (
            self.tp_size * self.pp_size
        ) % self.nnodes == 0, "tp_size must be divisible by number of nodes"

        # FIXME pp constraints
        if self.pp_size > 1:
            logger.warning(f"Turn off overlap scheule for pipeline parallelism.")
            self.disable_overlap_schedule = True
            assert (
                self.disable_overlap_schedule
                and self.speculative_algorithm is None
                and not self.enable_mixed_chunk
            ), "Pipeline parallelism is not compatible with overlap schedule, speculative decoding, mixed chunked prefill."

        assert not (
            self.dp_size > 1 and self.nnodes != 1 and not self.enable_dp_attention
        ), "multi-node data parallel is not supported unless dp attention!"
        assert (
            self.max_loras_per_batch > 0
            # FIXME
            and (self.lora_paths is None or self.disable_radix_cache)
        ), "compatibility of lora and cuda graph and radix attention is in progress"
        assert self.base_gpu_id >= 0, "base_gpu_id must be non-negative"
        assert self.gpu_id_step >= 1, "gpu_id_step must be positive"

        if isinstance(self.lora_paths, list):
            lora_paths = self.lora_paths
            self.lora_paths = {}
            for lora_path in lora_paths:
                if "=" in lora_path:
                    name, path = lora_path.split("=", 1)
                    self.lora_paths[name] = path
                else:
                    self.lora_paths[lora_path] = lora_path


def prepare_server_args(argv: List[str]) -> ServerArgs:
    """
    Prepare the server arguments from the command line arguments.

    Args:
        args: The command line arguments. Typically, it should be `sys.argv[1:]`
            to ensure compatibility with `parse_args` when no arguments are passed.

    Returns:
        The server arguments.
    """
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    raw_args = parser.parse_args(argv)
    server_args = ServerArgs.from_cli_args(raw_args)
    return server_args


ZMQ_TCP_PORT_DELTA = 233


@dataclasses.dataclass
class PortArgs:
    # The ipc filename for tokenizer to receive inputs from detokenizer (zmq)
    tokenizer_ipc_name: str
    # The ipc filename for scheduler (rank 0) to receive inputs from tokenizer (zmq)
    scheduler_input_ipc_name: str
    # The ipc filename for detokenizer to receive inputs from scheduler (zmq)
    detokenizer_ipc_name: str

    # The port for nccl initialization (torch.dist)
    nccl_port: int

    # The ipc filename for rpc call between Engine and Scheduler
    rpc_ipc_name: str

    @staticmethod
    def init_new(server_args, dp_rank: Optional[int] = None) -> "PortArgs":
        port = server_args.port + random.randint(100, 1000)
        while True:
            if is_port_available(port):
                break
            if port < 60000:
                port += 42
            else:
                port -= 43

        if not server_args.enable_dp_attention:
            # Normal case, use IPC within a single node
            return PortArgs(
                tokenizer_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
                scheduler_input_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
                detokenizer_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
                nccl_port=port,
                rpc_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            )
        else:
            # DP attention. Use TCP + port to handle both single-node and multi-node.
            if server_args.nnodes == 1 and server_args.dist_init_addr is None:
                dist_init_addr = ("127.0.0.1", server_args.port + ZMQ_TCP_PORT_DELTA)
            elif server_args.dist_init_addr.startswith("["):  # ipv6 address
                port_num, host = configure_ipv6(server_args.dist_init_addr)
                dist_init_addr = (host, str(port_num))
            else:
                dist_init_addr = server_args.dist_init_addr.split(":")

            assert (
                len(dist_init_addr) == 2
            ), "please provide --dist-init-addr as host:port of head node"

            dist_init_host, dist_init_port = dist_init_addr
            port_base = int(dist_init_port) + 1
            if dp_rank is None:
                scheduler_input_port = (
                    port_base + 3
                )  # TokenizerManager to DataParallelController
            else:
                scheduler_input_port = port_base + 3 + 1 + dp_rank

            return PortArgs(
                tokenizer_ipc_name=f"tcp://{dist_init_host}:{port_base}",
                scheduler_input_ipc_name=f"tcp://{dist_init_host}:{scheduler_input_port}",
                detokenizer_ipc_name=f"tcp://{dist_init_host}:{port_base + 1}",
                nccl_port=port,
                rpc_ipc_name=f"tcp://{dist_init_host}:{port_base + 2}",
            )


class LoRAPathAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, {})
        for lora_path in values:
            if "=" in lora_path:
                name, path = lora_path.split("=", 1)
                getattr(namespace, self.dest)[name] = path
            else:
                getattr(namespace, self.dest)[lora_path] = lora_path


class DeprecatedAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=0, **kwargs):
        super(DeprecatedAction, self).__init__(
            option_strings, dest, nargs=nargs, **kwargs
        )

    def __call__(self, parser, namespace, values, option_string=None):
        raise ValueError(self.help)


def get_model_arch(args: ServerArgs):
    hf_config = get_config(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
        model_override_args=json.loads(args.json_model_override_args),
    )
    return hf_config.architectures[0]


def auto_choose_speculative_params(arch: str):
    """
    Automatically choose the parameters for speculative decoding.

    You can tune them on your own models and prompts with scripts/playground/bench_speculative.py
    """
    if arch in ["LlamaForCausalLM"]:
        # The default value for llama
        return (5, 4, 8)
    elif arch in ["DeepseekV3ForCausalLM", "DeepseekV2ForCausalLM"]:
        # The default value for deepseek
        return (5, 4, 8)
    elif arch in ["Grok1ForCausalLM", "Grok1VForCausalLM"]:
        return (5, 4, 8)
    else:
        # The default value for all other models
        return (5, 4, 8)

```

# Now, it is your turn.

Now wait for next question. **Do not say or answer anything.**
