---
layout: page
permalink: /call-for-papers/
title: Call for Papers
description:
nav: true
nav_order: 1
---

We welcome researchers working on foundation models for structured data to submit their latest original research work to the ICML 2026 workshop on **Foundation Models for Structured Data**.

## Key Information

- Submission link: TBD
- Submission deadline: TBD (~May 01, 2026 (11:59 pm AoE))
- Acceptance notification: May 15, 2026 (anytime before 11:59 pm AoE)
- Camera ready deadline: TBD
- Page Limit: short papers with up to 4 pages; unlimited references and appendices
- Misc: double-blind, non-archival, poster presentations for accepted papers, orals for top rated papers

## Submission Instructions

Submissions should take the form of a short paper of up to **4 pages**. Additional pages containing references and appendices are allowed but the reviewers are _not obliged_ to refer to the appendices when reviewing the paper. Submissions should be made on **OpenReview** (link TBD) in a single `.pdf` file using the [ICML 2026 LaTeX style template](https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip). The impact statement is not required for the submission to the workshop.

The review process is _double-blind_, so please ensure that your submission is properly anonymized. Papers that exceed the page limit or have not been properly anonymized will be desk-rejected without review. Please note that there is no rebuttal phase and the final decisions will be made based solely on the submission and the reviews. Rejected and withdrawn submissions will not be made public.

All accepted submissions will be accompanied by a poster presentation. A number of selected submissions will be invited for lightning and oral talks.

**Dual submission policy**: This workshop is **non-archival**; even though all accepted papers will be available on OpenReview and this website, there are no formally-published proceedings. If a paper is currently under review at another venue, it can still be submitted to this workshop. If a paper has previously appeared in a journal, workshop, or conference, it should be reasonably extended in order to be accepted at this workshop. Parallel submission of papers under review at ICML 2026 is allowed.

## Scope and Topics

We focus on foundation-model approaches for inherently structured objectives on tabular and time-series data, such as classification, regression, forecasting, and structured generation. LLM/agent-based submissions are in scope only if they evaluate on predictive tabular/time-series objectives with strong classical baselines, and report cost/latency and numerical reliability checks. Out of scope are contributions about general-purpose foundation model architectures without a clear structured-data focus, and NLP-centric tasks over tables (e.g., table question answering / semantic parsing).

We invite submissions related to the following topics:

- **Building Foundation Models for (Multimodal) Structured Data**: Following the trends in language and vision domains, recent developments in foundation models in tabular and time series capable of zero-shot inference and/or in-context learning on unseen data have challenged the conventional in-domain training and prediction paradigm. Furthermore, models pretrained in one modality (such as tabular) can also demonstrate promising predictive performance in related domains by transforming the input data into a compatible format. We aim to assess progress in this area and the challenges of developing such models, including novel architectures and insights into scaling. A particular focus in 2026 is on multimodal structured foundation models, which integrate tabular and time-series data with complementary modalities such as text and images, enabling richer supervision, improved generalization, and new capabilities beyond unimodal structured learning.
- **Datasets and Synthetic Data Generation Methods**: The amount of high-quality structured data available in the public domain is limited for developing pretrained models, especially when compared with the amount of data available for other domains, such as language and vision. Structured foundation models that rely on real-world data for pre-training are often constrained by the number of available public datasets, or are limited to pre-training on relatively small-scale datasets from Wikipedia and GitHub. To address this limitation, recent work on foundation models for structured data has focused on developing high-fidelity synthetic data generation schemes and has included this data in their training corpus. This workshop welcomes contributions of high-quality large-scale (multimodal) datasets and synthetic data generation methods for training structured foundation models.
- **Benchmarks**: While efforts have been made to develop unified benchmarks for tabular and time series tasks, new efforts are required to evaluate structured foundation models comprehensively along different dimensions, such as different data characteristics, inference throughput, memory usage, scalability, and data memorization. Because structured-data corpora are small and reused, contamination and memorization can dominate reported gains; we explicitly encourage protocols and benchmarks that measure and mitigate contamination.
- **Alternative Paradigms (LLMs and Agents)**: Beyond purpose-built foundation models for structured data, large language models (LLMs) and agentic systems have shown emerging promise on inherently structured tasks. We invite LLM/agent-based work when evaluation centers on predictive tabular/time-series objectives and includes careful baselines, cost/latency reporting, and numerical reliability checks. Structured settings introduce alternative scaling dimensions beyond data and parameters such as schema diversity, cross-domain transfer, and inference-time compute through planning and tool use (with agents effectively scaling compute at inference). This raises a central open question: Are purpose-built structured foundation models fundamentally more sample- or compute-efficient than general-purpose LLM-based approaches for structured objectives? We welcome work that examines these scaling trade-offs, alongside challenges in interpretability, efficiency, and numerical reliability.
- **Applications of Foundation Models for Structured Data**: Foundation models for structured data can transform industries from climate modeling and fraud detection to supply chain optimization and health monitoring. Real-world deployment requires addressing challenges such as domain adaptation, model reliability, and data privacy. This workshop seeks contributions (1) showcasing novel applications in real-world structured data domains, (2) overcoming challenges such as scaling and inference throughput, and (3) demonstrating domain-specific innovation such as domain-specialized foundation models. We also welcome discussions on ethical considerations, fairness, and bias mitigation to ensure these technologies benefit a broad range of users and applications.

We also explicitly encourage submissions that study scaling across datasets, model size, and compute; and multimodality for structured foundation models (e.g., image-tabular modeling in medical diagnosis; text-time-series modeling for ECG interpretation).

## Contact

If you have questions about this workshop or are not sure if your paper's topic is suitable for submission, please feel free to contact the organizers at [icml-structured-foundation-workshop@googlegroups.com](mailto:icml-structured-foundation-workshop@googlegroups.com).
