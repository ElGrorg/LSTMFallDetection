# Fall Detection via LSTM and Random Forest Classification

## Context
This project focuses on analyzing and modeling human activity data collected from wearable sensors. The dataset, sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/196/localization+data+for+person+activity), includes data from four tags placed on the left ankle, right ankle, chest, and belt of individuals. The primary goal is to classify human activities, with a specific emphasis on detecting falls, which is critical for applications in healthcare and independent living.

The dataset suffers from class imbalance, as activities like falling are rare compared to others like walking or lying. To address this, techniques such as oversampling and class weighting were employed during model training.

## Implementation
The main implementation is in `main.ipynb`, which includes the following key components:

1. **Data Preprocessing**:
   - Features were scaled using robust scaling techniques.
   - Target variables were one-hot encoded.
   - A sliding window approach was used to segment the sequential data into smaller chunks for model input.

2. **Model Design**:
   - The primary model is a Long Short-Term Memory (LSTM) Recurrent Neural Network (RNN), chosen for its ability to handle sequential data and remember long-term dependencies.
   - The LSTM model includes layers for dropout, batch normalization, and a fully connected output layer.

3. **Training and Evaluation**:
   - The model was trained using cross-entropy loss with class weights to handle the class imbalance.
   - The Adam optimizer with a learning rate scheduler was used to optimize the model.
   - Evaluation metrics include accuracy, F-score, and cross-entropy loss.

4. **Baseline Comparison**:
   - A Random Forest Classifier was implemented as a baseline, achieving an accuracy of 78%.
   - The LSTM model outperformed the baseline with an accuracy of 87% and an F-score of 0.89.

5. **Hyperparameter Tuning**:
   - Hyperparameters such as hidden size, number of layers, and dropout rate were tuned to optimize model performance.

## Results
The LSTM model demonstrated superior performance compared to the baseline Random Forest Classifier. The results highlight the effectiveness of using sequential models for activity classification tasks, especially in scenarios with imbalanced datasets.

## Future Work
Further improvements could include:
- Exploring other architectures like GRUs or Transformer-based models.
- Collecting additional data to improve model generalization.
- Implementing real-time activity detection for practical applications.