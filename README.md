# **Multi-Label ICD Code Classification using MLP**

[**Competition Link**](https://www.kaggle.com/competitions/da5401-2024-ml-challenge)

---

## **Overview**

This project is part of the [**DA5401 2024 ML Challenge**](https://www.kaggle.com/competitions/da5401-2024-ml-challenge).
The task is to **predict multiple ICD codes for a patient based on pre-computed embeddings**. Each sample can have **multiple labels**, making it a **multi-label classification problem**.

Our solution uses a **Multi-Layer Perceptron (MLP)** model trained on PCA-reduced embeddings with a custom metric focusing on **macro F2 score** for better recall.

---

## **Dataset**

* **Input Files**:

  * `embeddings_1.npy`, `embeddings_2.npy` → Pre-computed feature embeddings.
  * `icd_codes_1.txt`, `icd_codes_2.txt` → ICD code labels for each embedding set.
  * `test_data.npy` → Test set for prediction.
* **Labels**: Multiple ICD codes per sample, separated by `;`.

**Label Processing**:

* Extract all unique ICD codes.
* Encode labels using **multi-hot encoding**.

---

## **Preprocessing Pipeline**

1. **Concatenate embeddings** → `X = embeddings_1 + embeddings_2`
2. **Multi-hot encode labels** → Shape: `(n_samples, n_classes)`
3. **Standardize features** using `StandardScaler`.
4. **Dimensionality Reduction**:

   * Apply **PCA** to retain **95% variance**.
5. **Train-Validation Split** → `test_size=0.0001` (small due to data constraints).

---

## **Model Architecture**

The model is a **deep neural network (MLP)** built with TensorFlow/Keras:

| Layer            | Details                                               |
| ---------------- | ----------------------------------------------------- |
| Input            | PCA components                                        |
| Dense (Hidden 1) | 1024 units, activation = **SELU**                     |
| Dropout          | 0.4                                                   |
| Dense (Hidden 2) | 1024 units, activation = **SELU**                     |
| Dropout          | 0.4                                                   |
| Output           | Units = number of ICD codes, activation = **sigmoid** |

**Loss Function**: `binary_crossentropy`
**Optimizer**: `Adam`
**Metrics**:

* Accuracy
* Custom `avg_f2_macro` (Macro F2 Score)

---

## **Custom Components**

* **Custom Macro F2 Metric**: Calculates the macro-averaged F2 score (β=2) inside Keras.
* **Custom Learning Rate Scheduler**:

  * Reduces learning rate by a factor of `0.4` after `3` epochs without improvement.
  * Minimum LR: `1e-6`.

---

## **Training**

* **Epochs**: 100
* **Batch Size**: 1024
* **Validation Set**: Extremely small due to constraints.
* **Callbacks**:

  * `ModelCheckpoint` → Save best model based on **F2 score**.
  * Custom **Learning Rate Scheduler**.

---

## **Evaluation**

On the validation set:

* **Accuracy**
* **Macro F2 Score (β=2)** → prioritizes recall for rare ICD codes.

---

## **Inference & Submission**

* Apply **scaling + PCA** on test data.
* Predict probabilities → Apply **threshold = 0.5**.
* Convert to multi-label string format (`label1;label2;...`).
* Save as `submissions23.csv`:

  ```csv
  id,labels
  1,I10;E11
  2,J45
  ...
  ```

---

## **Dependencies**

Install the following:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn wandb kerastuner
```

---

## **How to Run**

1. Download competition data from [Kaggle](https://www.kaggle.com/competitions/da5401-2024-ml-challenge).
2. Place the following files in the working directory:

   * `embeddings_1.npy`, `embeddings_2.npy`
   * `icd_codes_1.txt`, `icd_codes_2.txt`
   * `test_data.npy`
3. Run:

   ```bash
   python train_icd_classifier.py
   ```
4. Final submission will be saved as:

   ```
   submissions23.csv
   ```

---

## **Performance Metric**

The model is optimized for **Macro F2 Score** to ensure **high recall for multiple ICD codes**.

---

## **Future Improvements**

✔ Use **Transformer-based models** for embeddings instead of static vectors.
✔ Implement **threshold tuning per class** for better performance.
✔ Explore **ensemble methods** combining MLP with tree-based models.
✔ Use **Bayesian Optimization** for hyperparameter tuning.

---

✅ **Competition Link**: [https://www.kaggle.com/competitions/da5401-2024-ml-challenge](https://www.kaggle.com/competitions/da5401-2024-ml-challenge)


