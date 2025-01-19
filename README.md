# Fake-News-Detection-
A machine learning model to accurately classify news articles as real or fake. Mitigating the spread of misinformation on digital platforms.

Unmasking Fake News with ML.

Approach:
1) Designed a hybrid CNN-Bidirectional LSTM model for effective feature extraction and sequence learning.
2) Employed Binary Cross-Entropy as the loss function and optimized using the Adam optimizer.
3) Achieved key metrics: Accuracy, Precision, Recall, F1 Score, and ROC AUC Score of 0.97.

Data Preprocessing:
1) Loaded datasets ('train.tsv' and 'test.tsv') and performed exploratory analysis using Pandas.
2) Cleaned data by removing stopwords and applying token filtering with NLTK and Gensim.
3) Tokenized and padded data for uniform input using TensorFlow.
4) Split data into training (80%) and validation (20%) sets.

Model Architecture:
1) Used an Embedding Layer to convert words into dense vector representations.
2) Prevented overfitting using a Dropout Layer.
3) Extracted local features via Conv1D and MaxPooling Layers.
4) Captured sequence context using a Bidirectional LSTM Layer.
5) Output predictions with a Dense Layer.

Results : Achieved 97% accuracy in classifying news articles as real or fake.
