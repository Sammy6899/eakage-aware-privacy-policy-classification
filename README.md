# 🔒 🧠 📊 Leakage-Aware Privacy-Practice Classification: A Comparative Study of Classical and Neural Models under Policy-Disjoint Evaluation

> **Course Project:** CSE440 — Natural Language Processing / Machine Learning  
> **Affiliation:** Department of Computer Science and Engineering, BRAC University  
> **Authors:** Nafiz Ahmed Nafi, MD. Amirul Islam Sadat, Priom Halder, Samiha Tasnim Orthi  
> **Dataset Benchmark:** OPP-115 Corpus & Real-world Bangladeshi Corporate Privacy Policies

---

## 📌 Abstract

Privacy policies are long, legally complex documents that users routinely accept without reading. While automated data-practice classification can bridge this transparency gap, benchmark corpora such as OPP-115 contain substantial verbatim boilerplate reused across unrelated organizations. 

This project demonstrates that **conventional row-level stratified splits cause severe data leakage**, placing near-identical boilerplate across training and test sets (95.2% clause overlap). This inflates test macro-F1 scores by up to **+0.123** and distorts true model rankings. Under an honest, **policy-disjoint evaluation** (entire policies held out; 6.0% residual overlap), BERT Base achieves the highest macro-F1 (**0.706**), with Naive Bayes closely matching it (**0.701**). 

In Phase 2, the checkpointed policy-disjoint BERT Base model is transferred zero-shot to **3,120 sentences scraped from 26 real Bangladeshi company privacy policies**. Per-category softmax confidence is introduced as a label-free diagnostic to differentiate genuine sectoral differences (*Data Security*, +6.9pp, 0.740 conf) from out-of-distribution domain shift (*User Choice/Control*, -4.8pp, 0.550 conf).

---

## 🔬 Key Research Questions & Contributions

* **RQ1 & RQ2 (Evaluation Leakage):** Quantified how row-level splits falsely inflate performance across 10 distinct models spanning Classical, Recurrent, and Transformer architectures.
* **RQ3 (Generalization):** Benchmarked model performance under honest policy-disjoint splits, identifying pre-trained contextual representations as the most resilient to boilerplate leakage.
* **RQ4 (Cross-Jurisdictional Transfer):** Evaluated domain transfer from US-centric policies to the emerging digital ecosystem in Bangladesh across 10 major corporate sectors.

---

## 📊 Experimental Results

### 1. Model Performance: Leaky (Random) vs. Honest (Policy-Disjoint)

| Architecture Family | Model | Representation | Leaky Split Macro-F1 | Policy-Disjoint Macro-F1 | Evaluation Gap ($\Delta$ F1) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Transformer** | **BERT Base (Partial Fine-tune)**[cite: 7] | Subword Tokenizer[cite: 7] | 0.721[cite: 7] | **0.706** | **+0.015** |
| **Classical** | Naive Bayes ($\alpha=0.05$) | TF-IDF (Unigram/Bigram) | 0.726 | 0.701 | +0.025 |
| **Classical** | Logistic Regression | TF-IDF | **0.753** | 0.692 | +0.061 |
| **Classical** | Random Forest | TF-IDF | 0.721 | 0.652 | +0.069 |
| **Recurrent** | Bidirectional LSTM | Learned Embedding | 0.684 | 0.608 | +0.076 |
| **Recurrent** | Bidirectional GRU | Learned Embedding | 0.700 | 0.586 | +0.114 |
| **Recurrent** | Bidirectional SimpleRNN | Learned Embedding | 0.612 | 0.489 | +0.123 |
| **Recurrent** | GRU | Learned Embedding | 0.519 | 0.475 | +0.044 |
| **Recurrent** | LSTM | Learned Embedding | 0.549 | 0.442 | +0.107 |
| **Recurrent** | SimpleRNN | Learned Embedding | 0.172 | 0.131 | +0.041 |

> **Key Finding:** Random Forest exhibited the highest raw accuracy (0.780) under the leaky split, but dropped significantly under honest macro-F1, exposing the "accuracy trap" caused by majority-class boilerplate memorization.

---

### 2. External Validation: Bangladeshi Corporate Privacy Policies (Phase 2)

Evaluated on 3,120 extracted sentences across 26 usable websites representing 10 sectors (Banking, MFS, Telecom, E-commerce, Healthcare, etc.)[cite: 7]:

| Practice Category                  | OPP-115 Ground Truth Share | Bangladesh Predicted Share |    Delta   | Mean Softmax Confidence | Diagnostic Interpretation                                                                        |
| :--------------------------------- | :------------------------: | :------------------------: | :--------: | :---------------------: | :----------------------------------------------------------------------------------------------- |
| **First Party Collection/Use**     |           46.97%           |           48.78%           |   +1.81%   |          0.723          | Preserved plurality baseline                                                                     |
| **Third Party Sharing/Collection** |           27.76%           |           21.70%           |   -6.06%   |          0.725          | Consistent distribution                                                                          |
| **Data Security**                  |            4.03%           |           10.93%           | **+6.90%** |        **0.740**        | **Likely Real:** High-confidence sectoral emphasis (Fintech/Banking)                             |
| **User Choice/Control**            |            9.66%           |            4.87%           | **-4.79%** |        **0.550**        | **Likely Domain Shift:** Lowest confidence category; model struggles with local consent phrasing |
| **Data Retention**                 |            1.93%           |            4.10%           |   +2.17%   |          0.612          | Infrequent class in source corpus                                                                |
| **Policy Change**                  |            2.63%           |            4.39%           |   +1.76%   |          0.810          | Distinctive formulaic vocabulary                                                                 |
| **User Access, Edit & Deletion**   |            3.97%           |            2.60%           |   -1.37%   |          0.673          | Underrepresented self-serve controls                                                             |
| **Int'l & Specific Audiences**     |            3.05%           |            2.63%           |   -0.42%   |          0.640          | Preserved low-prevalence tail                                                                    |


---

## 🛠️ Repository Pipeline & Setup

### Installation
```bash
git clone [https://github.com/](https://github.com/)<your-username>/leakage-aware-privacy-policy-classification.git
cd leakage-aware-privacy-policy-classification
pip install -r requirements.txt

Reproducing the Experiments
# 1. Exploratory Data Analysis & Splitting:
python -m src.splits --data_path data/raw/privacy_policy_dataset.csv

# 2. Train Classical and Recurrent Grids:
# Executes 60 hyperparameter tuning runs across both random and policy-disjoint schemes

# 3. Fine-Tune & Checkpoint Policy-Disjoint BERT:
python -m src.models.transformer --mode train --freeze_layers 10 --lr 3e-5 --batch_size 32

# 4. Bangladeshi Corporate Transfer Inference:
python -m src.inference_bd --checkpoint checkpoints/bert_policy_disjoint --input_urls data/raw/bd_company_index_100.csv

## ⚠️ Documented Limitations
Padding without Length Masking: Recurrent architectures evaluated padded tokens through sequence caps (48 tokens), contributing to higher variance.

Scraper Attrition: 26% of targeted Bangladeshi corporate policies failed collection due to JavaScript single-page application (SPA) rendering, representing a slight selection bias.

Tier 1 Unsupervised Transfer: Bangladeshi external evaluation is descriptive; no manual gold-standard ground truth exists for Bangladeshi policies in this iteration.

---

## 👤 Author & Acknowledgments

- **Developer:** Samiha Tasnim Orthi, Nafiz Ahmed Nafi, Amirul Islam Sadat, Priom Halder
- **Course:** CSE440 - Natural Language Processing II (NLP)

---

## 📄 License & Attribution
The OPP-115 corpus is credited to Wilson et al. (2016). Project completed as part of undergraduate coursework at BRAC University.
