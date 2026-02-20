# toxic_comment_guard
Toxic Comment Guard: Live Gaming Chat Moderator AI using Custom RoBERTa

A state-of-the-art Real-Time Toxicity Detection System designed for online gaming communities. This project fine-tunes a RoBERTa-base transformer with a custom Multi-Sample Dropout architecture and Focal Loss to accurately classify multi-label toxicity (e.g., threats, insults, identity hate) in high-speed chat environments.


Key Features
Context-Aware Detection: Uses RoBERTa to understand slang, sarcasm, and context better than keyword filters.

Multi-Label Classification: Detects 6 levels of toxicity simultaneously: toxic, severe_toxic, obscene, threat, insult, identity_hate.

Imbalance Handling: Implements Focal Loss to penalize the model for ignoring rare, high-severity classes like threats.

Robustness: Features a Multi-Sample Dropout head (5x ensemble) to prevent overfitting and stabilize predictions.

Live Dashboard: A cyberpunk-themed Streamlit UI with real-time severity gauges, auto-ban logic, and safety scores.


 Tech Stack
Core AI: Python, PyTorch, HuggingFace Transformers

Data Processing: Pandas, NumPy, Scikit-learn

Visualization: Matplotlib, Seaborn

Deployment: Streamlit (Web App)


Dataset
The model was trained on the Jigsaw Toxic Comment Classification Challenge dataset.
(https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge)

Source: Wikipedia Talk labels.

Size: ~159,000 labeled comments.

Classes: 6 overlapping toxicity labels.


Model ArchitectureWe moved beyond standard transfer learning by engineering a custom classification head:
Base: RoBERTa-base (12-layer Transformer).
Head: MultiSampleDropout - The [CLS] token is passed through 5 separate dropout masks, and the results are averaged. This acts as a "mini-ensemble" within a single forward pass, significantly improving generalization on unseen test data.
Optimization: Trained with Focal Loss ($\gamma=2.0$) to focus learning on hard-to-classify examples.


The system achieves elite performance on unseen test data:
Micro ROC-AUC 0.9859
Hamming Loss 0.0241
Global Accuracy 89.46%

Observation: The model effectively distinguishes between "safe" gaming banter (e.g., "gg ez") and actual toxicity.
