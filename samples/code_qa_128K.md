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

# Now, it is your turn.

Now wait for next question. **Do not say or answer anything.**
