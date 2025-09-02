# **Multi-Label ICD Code Classification using MLP**

## **Overview**

This project addresses a **multi-label classification problem** where the task is to predict **ICD codes** based on provided **medical embeddings**. The solution uses **deep learning (MLP)** to map input embeddings to multiple ICD codes per instance.

---

## **Dataset**

* **Input**: Pre-computed embeddings (`embeddings_1.npy`, `embeddings_2.npy`)
* **Labels**: ICD codes provided in text files (`icd_codes_1.txt`, `icd_codes_2.txt`)
* **Test Data**: `test_data.npy`
* **Label Encoding**: Multi-hot encoding (One-hot for multiple labels per sample)

The combined dataset consists of:

* **Features (X)**: Concatenated embeddings from both sources
* **Labels (y)**: Multi-hot encoded ICD codes

---

## **Preprocessing**

1. **Label Encoding**: Extract unique ICD codes and encode using multi-hot representation.
2. **Feature Scaling**: Standardized using `StandardScaler`.
3. **Dimensionality Reduction**:

   * **PCA** applied to retain **95% variance**.
4. **Train-Validation Split**: Performed with an extremely small validation size (`0.0001`) due to data constraints.

---

## **Model Architecture**

The model is a **Multi-Layer Perceptron (MLP)** with the following structure:

* **Input Layer**: Size = number of PCA components
* **Hidden Layers**:

  * Dense(1024, activation='selu') + Dropout(0.4)
  * Dense(1024, activation='selu') + Dropout(0.4)
* **Output Layer**: Dense(units = number of ICD codes, activation='sigmoid')
* **Loss Function**: Binary Crossentropy (for multi-label classification)
* **Optimizer**: Adam
* **Metrics**:

  * Accuracy
  * Custom **avg\_f2\_macro** (macro F2 score)

---

## **Custom Components**

* **Custom F2 Metric**: A Keras-compatible metric for macro-averaged F2 score.
* **Custom Learning Rate Scheduler**:

  * Reduces LR by a factor (`0.4`) after patience (`3` epochs) without improvement.
  * Minimum LR set to `1e-6`.

---

## **Training**

* **Epochs**: 100
* **Batch Size**: 1024
* **Validation Data**: Extremely small split due to constraints.
* **Callbacks**:

  * `ModelCheckpoint` to save the best model based on `avg_f2_macro`.
  * Custom LR Scheduler for adaptive learning rate.

---

## **Evaluation**

On the validation set:

* **Accuracy**: Reported using `accuracy_score`.
* **Macro F2 Score**: Reported using `fbeta_score` with β=2.

---

## **Inference & Submission**

* Load **best saved model**.
* Apply **same preprocessing (scaling + PCA)** on test data.
* Predict ICD codes using **sigmoid outputs** → thresholded at **0.5**.
* Convert predictions to **multi-label format** (`label1;label2;...`).
* Save as **submissions23.csv** with the following format:

  ```
  id,labels
  1,I10;E11
  2,J45
  ...
  ```

---

## **Dependencies**

* Python 3.x
* TensorFlow / Keras
* NumPy
* Pandas
* scikit-learn
* matplotlib, seaborn
* wandb (optional for tracking)

Install all dependencies using:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn wandb kerastuner
```

---

## **How to Run**

1. Place all required files in the same directory:

   * `embeddings_1.npy`, `embeddings_2.npy`
   * `icd_codes_1.txt`, `icd_codes_2.txt`
   * `test_data.npy`
2. Run the notebook or script:

   ```bash
   python train_icd_classifier.py
   ```
3. The final submission file will be saved as:

   ```
   submissions23.csv
   ```

---

## **Performance Metric**

The model is optimized for **macro F2 score** (β=2) to prioritize **recall** for rare ICD codes.

