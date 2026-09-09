## geng2025granularity

Learning temporal granularity with quadruplet networks for temporal knowledge graph completion

Temporal Knowledge Graphs (TKGs) capture the dynamic nature of real-world facts by incorporating temporal dimensions that reflect their evolving states. These variations add complexity to the task of knowledge graph completion. Introducing temporal granularity can make the representation of facts more precise. In this paper, we propose Learning Temporal Granularity with Quadruplet Networks (LTGQ), which addresses the inherent heterogeneity of TKGs by embedding entities, relations, and timestamps into distinct specialized spaces. This differentiation enables a finer-grained capture of semantic information across the temporal knowledge graph. Specifically, LTGQ incorporates triaffine transformations to model high-order interactions between the elements of quadruples, such as entities, relations, and timestamps, in TKGs. Simultaneously, it leverages Dynamic Convolutional Neural Networks (DCNNs) to extract representations of latent spaces across different temporal granularities. By achieving more robust alignment between facts and their respective temporal contexts, LTGQ effectively improves the accuracy of temporal knowledge graph completion. The proposed model was validated on five public datasets, demonstrating significant improvements in TKG completion tasks, thereby confirming the effectiveness of our approach.

DOI: 10.1038/s41598-025-00446-z

## zhu2025multigran

Multi-Granularity History Graph Network for temporal knowledge graph reasoning

No abstract returned by OpenAlex.

DOI: 10.1016/j.datak.2025.102496

## dao2025hgct

HGCT: Enhancing temporal knowledge graph reasoning through extrapolated historical fact extraction

No abstract returned by OpenAlex.

DOI: 10.1016/j.knosys.2025.113358

## chen2025dualview

Dual-view temporal knowledge graph reasoning

No abstract returned by OpenAlex.

DOI: 10.1016/j.knosys.2025.114330

## pan2025validity

Leveraging temporal validity of rules via LLMs for enhanced temporal knowledge graph reasoning

No abstract returned by OpenAlex.

DOI: 10.1016/j.knosys.2025.114094

## fernando2025filtering

Memory-Efficient Information Filtering in Contrastive Learning for Temporal Knowledge Graph Reasoning

Temporal knowledge graphs (TKGs) have emerged as a critical component in modern artificial intelligence systems, enabling machines to reason over dynamic information. However, existing methods for TKG reasoning use massive amounts of information and computational resources, failing to capture minimal essential temporal knowledge with their dynamics while eliminating irrelevant or noisy information for a more precise reasoning process. To this end, we introduce the Forest Fire Contrastive Approach (FFCA), a contrastive learning architecture based on forest fire sampling that presents a preferential attachment mechanism for the extrapolation of TKG. This allows high-degree nodes to attract new connections, improving the pipeline's predictive capability while keeping it compact, leading to an efficient learning and inference process. This approach introduces a degree-biased burn probability that gathers a minimal but highly correlated subgraph relevant to the query as the global view. Simultaneously, a sufficient number of the most recent snapshots were gathered as the local view, preserving task-relevant graph information and removing noise while reducing the computational complexity and memory requirements, improving the sustainability of TKG reasoning models. Experiments on three publicly available benchmark datasets widely used for TKG extrapolation tasks demonstrate that the proposed approach achieves competitive predictive performance to current state-of-the-art methods while demonstrating substantial improvements in memory efficiency, a critical consideration for scalability to large-scale temporal knowledge graph datasets.

DOI: 10.1109/ICKG66886.2025.00018

## ji2025stse

STSE: Spatio-temporal state embedding for knowledge graph completion

The explicit integration of temporal and geospatial features into knowledge graphs enables more precise characterization of knowledge dynamics across temporal and geographic dimensions, while simultaneously amplifying the complexity of inferring missing facts in spatio-temporal knowledge graphs (STKGs). To address these dual challenges in Spatio-Temporal Knowledge Graph Completion (STKGC), we present STSE (Spatio-Temporal State Embedder), an innovative embedding framework that systematically coordinates relational semantics with spatial-temporal continuum modeling. Our technical contributions manifest through three key innovations: (1) A novel descriptive form that enhances practicality by distinguishing head/tail entity locations in tuples; (2) A geometry-aware embedding space that dynamically fuses spatial grids and temporal slices through spatio-temporal states, enabling robust reasoning on heterogeneous graphs; (3) An enhanced encoder that captures complex spatio-temporal contextual relationships between entities using modified Transformer and ResNet architectures. Experimental validation across three benchmark datasets (YAGO11k-ST, Wikidata12k-ST, ICEWS05-ST) demonstrates STSE's superiority, achieving 3.8 % Mean Reciprocal Rank ( MRR ) improvement in time prediction and 7.1 % Hits@10 enhancement in spatial location reasoning compared to state-of-the-art baselines. This methodological breakthrough establishes three implementable principles for STKGC: (1) geometric consistency constraints during fusion operations, (2) spatio-temporal difference capturing across heterogeneous relationships, (3) cross-domain applications ranging from epidemiological spread prediction to urban mobility pattern mining. © 2025 Elsevier Science. All rights reserved

DOI: 10.1016/j.knosys.2025.113469

## guo2025mamba

Few-shot temporal knowledge graph completion based on query-adaptive Mamba-enhanced temporal relation learning

No abstract returned by OpenAlex.

DOI: 10.1016/j.knosys.2025.114394

## liu2025uncertkg

Uncertainty modeling for inductive knowledge graph embedding

No abstract returned by OpenAlex.

DOI: 10.1016/j.neunet.2024.107103

## li2025vicinal

Vicinal Label Supervision for Reliable Aleatoric and Epistemic Uncertainty Estimation

No abstract returned by OpenAlex.

DOI: 10.52202/085713-3523

## dai2025mcnet

MCNet: Monotonic Calibration Networks for Expressive Uncertainty Calibration in Online Advertising

In online advertising, uncertainty calibration aims to adjust a ranking model's probability predictions to better approximate the true likelihood of an event, e.g., a click or a conversion. However, existing calibration approaches may lack the ability to effectively model complex nonlinear relations, consider context features, and achieve balanced performance across different data subsets. To tackle these challenges, we introduce a novel model called Monotonic Calibration Networks, featuring three key designs: a monotonic calibration function (MCF), an order-preserving regularizer, and a field-balance regularizer. The nonlinear MCF is capable of naturally modeling and universally approximating the intricate relations between uncalibrated predictions and the posterior probabilities, thus being much more expressive than existing methods. MCF can also integrate context features using a flexible model architecture, thereby achieving context awareness. The order-preserving and field-balance regularizers promote the monotonic relationship between adjacent bins and the balanced calibration performance on data subsets, respectively. Experimental results on both public and industrial datasets demonstrate the superior performance of our method in generating well-calibrated probability predictions.

DOI: 10.1145/3696410.3714802

## wang2025nonexchangeable

Non-exchangeable Conformal Prediction for Temporal Graph Neural Networks

Conformal prediction for graph neural networks (GNNs) offers a promising framework for quantifying uncertainty, enhancing GNN reliability in high-stakes applications. However, existing methods predominantly focus on static graphs, neglecting the evolving nature of real-world graphs. Temporal dependencies in graph structure, node attributes, and ground truth labels violate the fundamental exchangeability assumption of standard conformal prediction methods, limiting their applicability. To address these challenges, in this paper, we introduce NCPNET, a novel end-to-end conformal prediction framework tailored for temporal graphs. Our approach extends conformal prediction to dynamic settings, mitigating statistical coverage violations induced by temporal dependencies. To achieve this, we propose a diffusion-based non-conformity score that captures both topological and temporal uncertainties within evolving networks. Additionally, we develop an efficiency-aware optimization algorithm that improves the conformal prediction process, enhancing computational efficiency and reducing coverage violations. Extensive experiments on diverse real-world temporal graphs, including WIKI, REDDIT, DBLP, and IBM Anti-Money Laundering dataset, demonstrate NCPNET's capability to ensure guaranteed coverage in temporal graphs, achieving up to a 31% reduction in prediction set size on the WIKI dataset, significantly improving efficiency compared to state-of-the-art methods. Our data and code are available at https://github.com/ODYSSEYWT/NCPNET.

DOI: 10.1145/3711896.3737064

## shi2025uqconformal

Reliable Uncertainty Quantification in Machine Learning via Conformal Prediction

Deploying machine learning (ML) models in high-stakes domains such as healthcare and autonomous systems requires reliable uncertainty quantification (UQ) to ensure safe and accurate decision-making. Conformal prediction (CP) offers a robust, distribution-agnostic framework for UQ, providing valid prediction sets that guarantee a specified coverage probability. However, existing CP methods are often limited by assumptions that are violated in real-world scenarios, such as non-i.i.d. data, and by a lack of integration with modern machine learning workflows, particularly in large generative models. This research aims to address these limitations by advancing CP techniques to operate effectively in non-i.i.d. settings, improving predictive efficiency without sacrificing theoretical guarantees, and integrating CP directly into model training processes. These developments will enhance the practical applicability of CP for a wide range of ML tasks, enabling more reliable and interpretable models in high-stakes applications.

DOI: 10.1609/aaai.v39i28.35227

## sun2025uncertaintyupdate

An efficient uncertainty measure with dynamic update mechanisms

No abstract returned by OpenAlex.

DOI: 10.1016/j.knosys.2025.113572

## yuan2025kguq

KG-UQ: Knowledge Graph-Based Uncertainty Quantification for Long Text in Large Language Models

With the commercialization of large language models (LLMs) and their integration into daily life, addressing their susceptibility to hallucinations-unfactual information in generated outputs-has become an urgent priority. Existing uncertainty quantification (UQ) methods often rely on access to LLMs' internal states, which is unavailable for closed-source models like GPTs, or are primarily designed for short text. Current research on long text typically evaluates sentences individually, overlooking smaller semantic units that better capture the text's complexity. Recognizing the potential of knowledge graphs (KGs) to extract structured relationships from unstructured text, we propose KG-UQ, a UQ method leveraging KGs to address the semantic intricacies of long text. Our approach involves constructing KGs from long-text outputs and utilizing their embeddings to estimate uncertainties. Through our analysis, we demonstrate that knowledge graphs are an effective tool for decomposing long text into fundamental statements. However, we also highlight the increased uncertainty introduced during KG construction, stemming from inherent challenges in accurately capturing all semantic information.

DOI: 10.1145/3701716.3717660

## zhao2025stgraph

Uncertainty Quantification of Conformalized Spatio-Temporal Graph Convolutional Network on Forecasting

Accurately forecasting spatio-temporal dynamics in high-resolution and sparsely populated datasets remains a fundamental challenge in many real-world applications such as urban mobility and environmental monitoring. Traditional deterministic machine learning models often rely on point estimates, which inherently lack the ability to capture and quantify predictive uncertainty—particularly in settings where data sparsity and topological complexity prevail. While existing approaches to spatio-temporal prediction have made progress in modeling data distributions or estimating model uncertainty, they typically rely on strong statistical assumptions and fail to provide comprehensive, distribution-free uncertainty quantification. In this work, we introduce a novel hybrid framework that integrates Conformal Prediction (CP) with Spatio-Temporal Graph Convolutional Networks (STGCNs) to address these limitations. Our method builds upon pretrained STGCN models by incorporating a conformal calibration procedure, which constructs a calibration set tailored to the graph's temporal and topological structure under the assumption of sequential exchangeability. This enables the model to produce valid, data-driven prediction intervals without retraining, offering statistically rigorous uncertainty estimates alongside point predictions. The proposed CP-STGCN framework dynamically adjusts its predictive intervals based on the variability observed in the calibration set, ensuring robust coverage across multiple prediction horizons and spatial resolutions. We empirically evaluate our approach on large-scale ride-sharing and taxi demand datasets spanning diverse urban regions and demonstrate that CP-STGCN significantly improves uncertainty quantification. The results show consistent improvements in interval validity, sharpness, and overall predictive accuracy, confirming the efficacy of our method in capturing both epistemic and aleatoric uncertainties in complex spatio-temporal environments.

DOI: 10.1109/ICDM65498.2025.00184

## chen2025gnnuq

Uncertainty quantification with graph neural networks for efficient molecular design

Optimizing molecular design across expansive chemical spaces presents unique challenges, especially in maintaining predictive accuracy under domain shifts. This study integrates uncertainty quantification (UQ), directed message passing neural networks (D-MPNNs), and genetic algorithms (GAs) to address these challenges. We systematically evaluate whether UQ-enhanced D-MPNNs can effectively optimize broad, open-ended chemical spaces and identify the most effective implementation strategies. Using benchmarks from the Tartarus and GuacaMol platforms, our results show that UQ integration via probabilistic improvement optimization (PIO) enhances optimization success in most cases, supporting more reliable exploration of chemically diverse regions. In multi-objective tasks, PIO proves especially advantageous, balancing competing objectives and outperforming uncertainty-agnostic approaches. This work provides practical guidelines for integrating UQ in computational-aided molecular design (CAMD).

DOI: 10.1038/s41467-025-58503-0

## szabadvary2025reject

Classification with reject option: Distribution-free error guarantees via conformal prediction

Machine learning (ML) models always make a prediction, even when they are likely to be wrong. This causes problems in practical applications, as we do not know if we should trust a prediction. ML with reject option addresses this issue by abstaining from making a prediction if it is likely to be incorrect. In this work, we formalise the approach to ML with reject option in binary classification, deriving theoretical guarantees on the resulting error rate. This is achieved through conformal prediction (CP), which produce prediction sets with distribution-free validity guarantees. In binary classification, CP can output prediction sets containing exactly one, two or no labels. By accepting only the singleton predictions, we turn CP into a binary classifier with reject option. Here, CP is formally put in the framework of predicting with reject option. We state and prove the resulting error rate, and give finite sample estimates. Numerical examples provide illustrations of derived error rate through several different conformal prediction settings, ranging from full conformal prediction to offline batch inductive conformal prediction. The former has a direct link to sharp validity guarantees, whereas the latter is more fuzzy in terms of validity guarantees but can be used in practice. Error-reject curves illustrate the trade-off between error rate and reject rate, and can serve to aid a user to set an acceptable error rate or reject rate in practice.

DOI: 10.1016/j.mlwa.2025.100664

## jin2025memorywalk

Memory-based walk-enhanced dynamic graph neural network for temporal graph representation learning

No abstract returned by OpenAlex.

DOI: 10.1016/j.neucom.2025.129759

## guo2025dynamicmeta

Dynamic meta-graph convolutional recurrent network for heterogeneous spatiotemporal graph forecasting

No abstract returned by OpenAlex.

DOI: 10.1016/j.neunet.2024.106805

## vu2025tcrosse

TCrossE: Cross-space interaction of bicomplex and quaternion embeddings for temporal knowledge graph completion

No abstract returned by OpenAlex.

DOI: 10.1016/j.knosys.2025.113321

## yu2025gmve

GMVE: Graph-Mamba variational encoder for interpretable remaining useful life prediction with uncertainty quantification

No abstract returned by OpenAlex.

DOI: 10.1016/j.knosys.2025.114217

## xu2026householder

Temporal householder transformation embedding for temporal knowledge graph completion

No abstract returned by OpenAlex.

DOI: 10.1016/j.knosys.2026.115406

## zhu2026splices

Fact splices and entity aggregation networks for sparse temporal knowledge graph completion

No abstract returned by OpenAlex.

DOI: 10.1016/j.knosys.2026.115387

## peng2026icpe

ICPE-STKG Reasoning: Sparse Temporal Knowledge Graph Reasoning via Inverse-Causal Prior Experience Completion

No abstract returned by OpenAlex.

DOI: 10.1016/j.knosys.2026.116268

## zhang2026dstag

DSTAG: A Semantic Tag-Enhanced Dual-Graph Convolutional Network for Temporal Knowledge Graph Completion

Temporal Knowledge Graph Completion (TKGC) aims to predict missing entities or relations based on historical facts, thereby facilitating the understanding of dynamic system evolution and supporting downstream reasoning tasks. However, existing methods predominantly focus on modeling sequential and structural dependencies, often overlooking the rich semantic information embedded in entities and relations, as well as the higher-order interactions among them, which limits their ability to handle complex, evolving scenarios effectively. To address these limitations, we propose DSTAG, a novel TKGC approach based on a semantic tag-enhanced dual-graph convolutional network. Our method leverages large language models to generate contextualized semantic multi-tags for both entities and relations (e.g., ''political event,'' ''economic activity''), thereby enriching their semantic representations. Furthermore, we introduce a semantic tag representation mechanism that captures higher-order dependencies during the aggregation and propagation of semantic tag information across graphs. DSTAG adopts a dual-graph convolutional network architecture, where the relation graph convolution extracts semantic features between temporal relationships and injects this information into the entity graph convolution, enabling joint modeling of entities and relations. We evaluate DSTAG on three widely used TKG benchmarks: ICEWS14, ICEWS18, and ICEWS05-15. Experimental results show that DSTAG achieves substantial MRR improvements over state-of-the-art baselines by 8.64%, 9.81% and 4.56%, respectively.

DOI: 10.1145/3774904.3792165

## ye2026dualhistory

Dual History Enhancement with Hybrid Hypergraph-Graph Networks for Temporal Knowledge Graph Reasoning

Temporal Knowledge Graph (TKG) reasoning seeks to predict future events by analyzing historical data, where the effective leverage of both local and global historical facts proves crucial. Existing approaches employ graph neural networks (GNNs) and recurrent neural networks (RNNs) for local evolution patterns, complemented by statistical methods to enhance attention to global facts, demonstrating efficient predictive capabilities. However, traditional GNNs, constrained by their low-order neighborhood aggregation design, inherently fail to model potential high-order dependencies among facts. Furthermore, existing global history modeling approaches may introduce irrelevant historical information that interferes with prediction tasks. To address these limitations, we propose a Dual History-aware HyperGraph Network for TKG reasoning, namely DHHGN. Specifically, for local history modeling, we design a hybrid hypergraph-graph joint recurrent convolution module that simultaneously captures low-order neighborhood information and high-order interaction patterns among entities, employing a gating mechanism to adaptively blend their contributions. For global history modeling, we propose a dual history enhancement module that amplifies attention on pivotal historical facts while ensuring holistic integration of all historical contexts. Extensive experiments on four public benchmarks validate that DualHist-HGN consistently outperforms existing state-of-the-art methods across TKG reasoning tasks.

DOI: 10.1145/3774904.3792310

## wan2026tgcallm

TGCA-LLM: Time-Aware Graph-Text Contrastive Alignment for Enhancing LLMs in Temporal Knowledge Graph Completion

Temporal Knowledge Graph Completion (TKGC) aims to infer missing facts by modeling historical events and latent temporal dependencies in Temporal Knowledge Graphs (TKGs). Recently, TKGC methods that integrate graph embeddings into Large Language Models (LLMs) have shown great promise by leveraging the structural information of TKGs together with the powerful reasoning capabilities of LLMs. However, these embedding-based methods are limited by suboptimal graph representations due to noise and long-tail issues in real-world scenarios, and insufficient cross-modal alignment between graph and language, hindering LLMs' ability to fully capture the temporal and structural information of TKGs. To address these issues, we propose TGCA-LLM, a novel embedding-based framework for TKGC. Specifically, TGCA-LLM first employs time-aware contrastive learning to align fact texts with graph structures in the temporal dimension, generating robust graph embeddings and establishing initial cross-modal alignment. Then, through a two-stage tuning process, it enables LLMs to gradually acquire structural and temporal knowledge from graph embeddings while enhancing their cross-modal reasoning capabilities in TKGC. Extensive experiments on three widely used real-world benchmarks demonstrate that TGCA-LLM outperforms state-of-the-art (SOTA) baselines by at least 8.7% MRR, highlighting its effectiveness.

DOI: 10.1609/aaai.v40i18.38612

## lee2026tirano

TiRano: Tensorized Relation-aware Temporal Reasoning for Accurate Knowledge Graph Completion

Given a partially observed Temporal Knowledge Graph (TKG), how can we accurately predict missing entities? Unlike static knowledge graphs, TKGs encode facts within temporal contexts, requiring models to reason over both graph structure and time. However, existing TKGC approaches often sample neighbors solely based on temporal proximity, introducing irrelevant context and noise. Moreover, many methods compress snapshots into latent representations and rely on global sequence encoders for temporal modeling, losing edge-level structure and localized relation-specific patterns.

DOI: 10.1145/3770855.3817784

## wang2026globalinteraction

Temporal knowledge graph completion via global structural representation and deep interaction

No abstract returned by OpenAlex.

DOI: 10.1016/j.ins.2026.123139

## zhang2026logicalpaths

A context-aware temporal knowledge graph completion method based on logical paths

No abstract returned by OpenAlex.

DOI: 10.1016/j.neucom.2026.133049

## fan2026editing

Bridging graph structure and knowledge-guided editing for interpretable temporal knowledge graph reasoning

No abstract returned by OpenAlex.

DOI: 10.1016/j.neunet.2026.108811

## bi2026globallocal

Global–local evolution modeling with cyclic patterns for temporal knowledge graph reasoning

No abstract returned by OpenAlex.

DOI: 10.1016/j.patcog.2025.111828

## wang2026multidim

Temporal knowledge graph reasoning based on multidimensional information interaction and dynamic frequency awareness

No abstract returned by OpenAlex.

DOI: 10.1016/j.neucom.2026.133496

## yu2026dependency

Exploring semantic dependency for reasoning over temporal knowledge graph

No abstract returned by OpenAlex.

DOI: 10.1016/j.engappai.2026.114845

## li2026calendar

CALENDAR+: in-context contrastive learning for temporal knowledge graph reasoning

Temporal Knowledge Graph (TKG) reasoning aims to infer future events from historical facts. Recent advances in large language models (LLMs) have shown that in-context learning can effectively enhance temporal reasoning. While existing approaches over-rely on historical information and overlook crucial non-historical factors, which CALENDAR addresses. However, CALENDAR overlooks event recency and relies heavily on global principles, which leads to inaccuracies in TKG reasoning. To address this limitation, we propose CALENDAR+ (i.e., in-context C ontr A stive L earning t E mporal k N owle D ge gr A ph R easoning), a novel approach that integrates contrastive demonstrations to improve in-context reasoning. In CALENDAR+, we propose a demonstration candidate generation with high-order information method, which generates demonstration candidates from both historical and non-historical information. Moreover, we devise a time-aware contrastive importance based demonstration selection method to emphasize the most informative examples across time. Furthermore, we design a global–local chain-of-history based demonstration format which provides explicit negative principles that guide the model to avoid over-reliance on global and local histories. Extensive experiments show that CALENDAR+ achieves consistent improvements of over 1% across multiple TKG datasets, including Hits@10 of 60.10% on ICEWS14, 53.30% on ICEWS18, and 69.95% on ICEWS05-15, with an MRR gain of 5.67% over the strongest baseline.

DOI: 10.1007/s40747-026-02237-z

## li2026entropyrobust

Entropy-regularized multimodal fusion for robust and explainable knowledge graph completion

No abstract returned by OpenAlex.

DOI: 10.1007/s10618-026-01198-8

## moon2026sharpness

How Sharp and Bias-Robust is a Model? Dual Evaluation Perspectives on Knowledge Graph Completion

Knowledge graph completion (KGC) aims to predict missing facts from the observed KG. While a number of KGC models have been studied, the evaluation of KGC still remain underexplored. In this paper, we observe that existing metrics overlook two key perspectives for KGC evaluation: (A1) predictive sharpness -- the degree of strictness in evaluating an individual prediction, and (A2) popularity-bias robustness -- the ability to predict low-popularity entities. Toward reflecting both perspectives, we propose a novel evaluation framework (PROBE), which consists of a rank transformer (RT) estimating the score of each prediction based on a required level of predictive sharpness and a rank aggregator (RA) aggregating all the scores in a popularity-aware manner. Experiments on real-world KGs reveal that existing metrics tend to over- or under-estimate the accuracy of KGC models, whereas PROBE yields a comprehensive understanding of KGC models and reliable evaluation results.

DOI: 10.1145/3773966.3779401
