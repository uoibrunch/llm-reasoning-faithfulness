# Explanation Consistency as a Proxy for Faithfulness in LLMs

## 📌 Overview
This repository contains selected methodology scripts, data generation pipelines, and analysis code from my Diploma Thesis: **Explanation Consistency as a Signal for Faithfulness in LLMs**. 

Directly proving the faithfulness of a Large Language Model's internal reasoning chain is notoriously difficult. This research investigates whether explanation stability across meaning-preserving prompt perturbations can serve as a scalable, unsupervised proxy for causal faithfulness. The study spans symbolic logic, arithmetic, and open-domain reasoning to distinguish true semantic understanding from superficial model mimicry (sycophancy).

## 🛠️ Tech Stack & Evaluated Models
* **Languages & Tools:** Python, PyTorch, Hugging Face Transformers, Sentence Transformers, Pandas, Scikit-Learn
* **Evaluated Models:** Llama-3-8B-Instruct, Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.2, Mistral-7B-Instruct-v0.3, Qwen-2.5-7B-Instruct, Gemini 2.5 Flash
* **Benchmarks Evaluated:** GSM8K (Arithmetic), StrategyQA (Multi-hop Factual), BBH (Boolean Expressions), e-SNLI (Natural Language Inference)

## 📂 Repository Structure
This repository highlights the core engineering and analytical components of the research:
* `/notebooks/`: Contains representative data generation and evaluation pipelines, including:
    * Fault-tolerant batched inference scripts for Llama, Mistral, and Qwen models.
    * Mechanistic interpretability scripts featuring layer-wise activation patching.
    * Difference-in-Differences (DiD) causal ablation testing.
* `/data/`: Contains sample datasets showcasing the generated LLM rationale clusters, semantic similarity matrices, and accuracy/plausibility metrics.
* `paper.pdf`: The full thesis document detailing comprehensive results, ablation studies, and mathematical formulations.

## 🔬 Core Methodology

### 1. Prompt Perturbation & Explanation Consistency Score (ECS)
For each benchmark problem, 3 meaning-preserving perturbations (lexical, syntactic, pragmatic/contextual) were generated to create a "problem cluster." Models were prompted for Zero-Shot Chain-of-Thought (CoT) rationales. The **Explanation Consistency Score (ECS)** was defined as the mean cosine similarity of the rationales within each cluster.

### 2. Forensic Analysis (Organic vs. Synthetic Consistency)
To prove that high ECS was not merely a byproduct of rote memorization, the pipeline includes forensic structural checks:
* **Lexical Overlap & POS Stiffness:** Measured syntactic rigidity to differentiate between *Organic Stability* (adaptive, semantic reasoning) and *Synthetic Stability* (rigid grammatical template collapse, particularly observed in models over-optimizing for consistency).

### 3. Behavioral Faithfulness Evaluation (Black-Box Causal Testing)
Faithfulness was operationalized via two orthogonal causal ablation studies:
* **Input Ablation Framework (DiD):** Utilized a Difference-in-Differences approach ("Ghost Twins") to measure the Necessity and Sufficiency of cited evidence, strictly controlling for the noise of structural/syntactic damage caused by token masking.
* **Counterfactual Rationale Editing:** Evaluated Rationale Gain and Counterfactual Robustness by forcing the model to condition on its own sanitized rationale (or an intentionally corrupted rationale) using forced-choice log-probability scoring.

## ⚙️ Pipeline Engineering & Model Handling
To scale the execution of these benchmarks across hundreds of samples using open-weight models on constrained hardware environments (e.g., NVIDIA T4 GPUs), the codebase implements several enterprise-grade engineering patterns:
* **Quantized Memory Optimization:** Models are executed using 4-bit NF4 quantization (`BitsAndBytesConfig`) with double quantization enabled, minimizing memory footprint while maintaining compute stability.
* **Fault-Tolerant & Resumable Loops:** Ingestion loops feature dynamic checkpoint scanning. If a generation session is interrupted, the loop auto-recovers completed primary keys, copies partial records, and resumes seamlessly.
* **Token-Level Loop Safeguards:** Custom chat formatting structures are paired with deterministic text generation parameterization and explicit system-level termination configurations (e.g., `<|eot_id|>, <|im_end|>`) to entirely mitigate generation hallucination or token-cloning bugs.
* **Execution Boundary Enforcements:** Causal hidden-state patching routines are protected by native context execution boundaries using UNIX system signals (`signal.SIGALRM`), ensuring edge-case processing stalls are automatically logged and bypassed without crashing active threads.

## 📊 Key Findings
* **The "Valley of Confusion":** Identified a universal, non-monotonic structural feature where intermediate ECS values consistently correlate with a sharp drop in both accuracy and plausibility. This indicates a transient phase where models lock into spurious heuristics (The "Bad Rule" Trap) before converging on semantic alignment.
* **Capability & Quality Proxy:** High organic consistency strongly predicts both exact-match accuracy and reference-free logical plausibility across multiple architectures.
* **Causal Grounding:** DiD ablation and counterfactual editing confirmed that high-ECS reasoning clusters exert genuine causal leverage over the model's final prediction, successfully acting as a measurable proxy for faithfulness.

## 🚀 Future Work: Diversity-Aware Consistency Training
To transition ECS from a passive evaluation metric to an active training signal, the thesis proposes a Constrained Consistency Training objective. This prevents Goodharting (where models exploit the metric via structural mode collapse) by combining a contrastive consistency objective with an entropy regularizer:

L_total = L_task + λ_1 * L_Contrastive + λ_2 * L_Entropy

L_Contrastive = -log( exp(sim(R(x), R(x')) / τ) / [exp(sim(R(x), R(x')) / τ) + Σ exp(sim(R(x), R(y)) / τ)] )
