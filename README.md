# Speech-based Dysarthria Classification: Improving Diagnostic Accuracy and Model Calibration

> End-to-end machine learning pipeline for early and reliable diagnosis of dysarthria and neurological disorders using speech data.

## 🧠 Motivation

Early detection of neurological disorders such as dysarthria is crucial for effective intervention, yet conventional diagnosis methods are subjective and resource-intensive. This project aims to automate and enhance the diagnostic process by leveraging deep learning on speech data, focusing on both **accuracy** and **model reliability** (calibration), which are essential for real-world clinical adoption.


## 🚀 Key Contributions

- Developed a **full machine learning pipeline** from data preprocessing to model deployment, automating the entire process using Python and TensorFlow.
- Proposed and compared multiple state-of-the-art CNN and MLP architectures, including advanced ensemble and knowledge distillation methods.
- Implemented and analyzed model calibration metrics (ECE, OE) to address the practical reliability of predictions for clinical use.
- Achieved significant performance improvements over baselines through ensemble learning and data augmentation.
- **Minhyeok Son** led the overall project design, data pipeline automation, and model calibration experiments.

## 📂 Dataset

- **Source:** AI Hub - Korean Dysarthria Speech Recognition Dataset ([Link](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=topMenu&aihubDataSe=data&dataSetSn=608))
- **Description:** 1,200+ samples of speech audio from patients with and without dysarthria.
- **Features:** Includes wav audio, JSON transcripts, and patient metadata.  
- **Preprocessing:**  
    - Segmented long audio files into manageable utterances using silence detection.
    - Extracted log-mel spectrogram features for deep learning input.
- **Note:** The dataset is based on Korean speech. Limitations regarding language diversity are discussed below.

- ![img](https://github.com/Marcus-Son/Classification_and_Calibration_of_Dysarthric_Speech/assets/137815765/1a581a17-8d5c-4ef2-b393-5aeec3fba967)

## 🔄 Project Pipeline

The project follows an end-to-end workflow, from raw data ingestion to final model evaluation and calibration. All stages were fully automated and reproducible in code, designed and implemented by Minhyeok Son.

**Pipeline Overview:**

<img width="877" alt="image" src="https://github.com/user-attachments/assets/25592b15-f8b6-4f8f-859b-703d2d18f0b8" />


1. **Data Collection & Preparation**
   - Downloaded and organized the original AI Hub dataset.
   - Automated segmentation of long audio files into short, utterance-level clips using silence detection algorithms.
   - Extracted and synchronized transcript labels.

2. **Feature Engineering**
   - Converted each audio segment into log-mel spectrogram images.
   - Standardized and augmented the dataset for robust model training.
   - (Handled via custom Python and Librosa scripts.)

3. **Modeling**
   - Designed multiple deep learning architectures:
     - Baseline CNN (custom lightweight model)
     - Ensemble of diverse CNNs (VGG, ResNet, DenseNet, SqueezeNet)
     - Baseline2: Autoencoder for feature extraction + MLP classifier
     - Knowledge Distillation (Teacher/Student framework for efficiency)
   - Implemented using TensorFlow and Keras.

4. **Training & Hyperparameter Tuning**
   - Applied stratified train/validation/test splits for reproducible experiments.
   - Used learning rate scheduling, early stopping, and model checkpointing for optimal results.

5. **Evaluation & Calibration**
   - Evaluated models on accuracy, recall, precision, F1-score, ECE (Expected Calibration Error), and OE (Overconfidence Error).
   - Compared results across all approaches and visualized key metrics.

6. **Result Analysis & Documentation**
   - Compiled results and insights into a clear, transparent report.
   - All code, logs, and visualizations are available in this repository for reproducibility.

> **Minhyeok Son** was responsible for end-to-end pipeline automation, feature engineering, and calibration analysis.

## 🛠️ Models & Methods

Multiple deep learning strategies were explored to maximize both classification accuracy and prediction reliability for dysarthria detection. All modeling, experimentation, and analysis were led by Minhyeok Son.

### 1. Baseline Model (Custom CNN)
- Lightweight CNN trained directly on 28x28 log-mel spectrogram images.
- Architecture: 3 blocks of (Conv2D → BatchNorm → AvgPooling), followed by dense layers.
- Early stopping, model checkpointing, and learning rate scheduling applied.
- Served as the main reference for all further experiments.

### 2. Baseline2 (Autoencoder + MLP)
- Features extracted using a pre-trained autoencoder.
- Classification performed using a multilayer perceptron (MLP).
- Provided insight into the performance difference between feature extraction and end-to-end approaches.

### 3. Ensemble Models
- Combined predictions from multiple CNN architectures (VGG, ResNet, DenseNet, SqueezeNet, custom CNN).
- Explored various ensemble strategies (averaging, voting).
- Significantly improved model accuracy and robustness.

### 4. Data Augmentation
- Applied image augmentations (width/height shift, zoom, etc.) on spectrograms.
- Helped mitigate overfitting and improved generalization.

### 5. Knowledge Distillation
- Leveraged a strong ensemble as the Teacher model and a compact CNN as the Student.
- Student model trained to mimic the soft-label outputs of the Teacher, using various temperature settings.
- Achieved lighter and faster models without significant loss in accuracy.

### 6. Model Calibration
- Assessed not only traditional metrics (accuracy, precision, recall, F1) but also model calibration using ECE (Expected Calibration Error) and OE (Overconfidence Error).
- Identified and addressed reliability gaps for real-world clinical deployment.

> All model design, training, and evaluation scripts were authored and maintained by Minhyeok Son, with a focus on both reproducibility and real-world applicability.

## 📈 Results

The project achieved significant improvements in both classification accuracy and model calibration, outperforming simple baselines and meeting real-world clinical requirements.

| Model                     | Accuracy | Precision | Recall | F1-score | ECE    | OE     |
|---------------------------|----------|-----------|--------|----------|--------|--------|
| Baseline CNN              | 91.03%   | 90.80%    | 91.03% | 90.45%   | 4.14%  | 4.00%  |
| Baseline2 (Autoencoder)   | 85.29%   | 85.21%    | 85.29% | 85.17%   | 4.96%  | 4.53%  |
| Ensemble (Trial 3)        | 94.41%   | 94.47%    | 94.41% | 94.14%   | 1.49%  | 1.60%  |
| Knowledge Distillation (T=5) | 91.87%   | 91.68%    | 91.87% | 91.48%   | 4.69%  | 4.36%  |

- **Ensemble models** provided the best tradeoff between accuracy and calibration.
- **Knowledge Distillation** enabled deployment of lightweight models with minimal accuracy loss, suitable for edge or mobile applications.
- Detailed metric visualizations and confusion matrices are included in the `results/` folder of this repository.

---

## ⚠️ Limitations & Challenges

- **Dataset Language:** The dataset is limited to Korean speech samples, which may limit generalizability to English or multilingual populations.
- **Clinical Application:** Real-world deployment may require larger, more diverse datasets and further clinical validation.
- **Data Imbalance:** Class imbalance and short utterance length posed challenges for robust training and evaluation.
- **Production Readiness:** The project focused on modeling and evaluation, not full deployment (e.g., REST API, UI, or edge-device integration).
- These limitations are openly discussed and addressed as potential directions for future improvement.

---

## 🧑‍💻 What I Learned & Future Work

Through this project, I (Minhyeok Son) gained extensive experience in:
- Building and automating end-to-end deep learning pipelines for real-world healthcare data.
- Applying advanced model calibration techniques to improve clinical trust in machine learning predictions.
- Experimenting with ensemble and knowledge distillation methods for both performance and efficiency.
- Collaborating in a team, managing reproducible code, and communicating findings through technical writing and visualization.

**Future directions** may include:
- Extending the methodology to English/multilingual datasets.
- Developing a web/mobile application for clinical usage.
- Integrating explainable AI (XAI) techniques for model transparency.
- Further improving model calibration and uncertainty quantification.

---

## 📚 References

- [AI Hub Korean Dysarthria Speech Dataset](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=topMenu&aihubDataSe=data&dataSetSn=608)
- Related papers and articles on dysarthria classification, CNN, and model calibration (see `references.bib`).
- [Librosa](https://librosa.org/), [TensorFlow](https://www.tensorflow.org/), [Keras](https://keras.io/)

---

## 📬 Contact

For questions or collaboration inquiries, please contact:

**Minhyeok Son**  
Email: shawn22587@gmail.com

---
