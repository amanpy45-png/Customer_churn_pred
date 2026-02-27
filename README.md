Customer Churn Prediction using Neural Networks:
This repository contains a machine learning project that predicts customer churn for a bank using a neural network (Artificial Neural Network - ANN) built with TensorFlow/Keras.

___________________________________________________________________________________________________________________________________________________________________

📂 Dataset:
The project uses the Churn_Modelling.csv dataset which contains information about bank customers. Key columns include:
CreditScore – Customer's credit score
Geography – Customer's location (Country)
Gender – Male/Female
Age – Customer's age
Tenure – Number of years as a customer
Balance – Account balance
NumOfProducts – Number of products used
HasCrCard – Whether the customer has a credit card (1 = Yes, 0 = No)
IsActiveMember – Whether the customer is active (1 = Yes, 0 = No)
EstimatedSalary – Customer's estimated salary
Exited – Target variable (1 = churned, 0 = stayed)

___________________________________________________________________________________________________________________________________________________________________

🛠️ Features Engineering:
Removed unnecessary columns: RowNumber, CustomerId, Surname.
Converted categorical variables into numerical using one-hot encoding for Geography and Gender.
Split the dataset into features (X) and target (y).

___________________________________________________________________________________________________________________________________________________________________

⚙️ Data Preprocessing:
Split data into training and test sets using train_test_split (80% train, 20% test).
Scaled the features using StandardScaler for better neural network performance.

___________________________________________________________________________________________________________________________________________________________________

🧠 Neural Network Model:
The model was built using TensorFlow Keras Sequential API:
Input layer: 11 neurons (matching feature count) with ReLU activation
Hidden layer: 11 neurons, ReLU activation
Output layer: 1 neuron, sigmoid activation (for binary classification)

___________________________________________________________________________________________________________________________________________________________________

🔧 Model Training:
Loss function: binary_crossentropy
Optimizer: Adam
Metrics: accuracy
Epochs: 100
Batch size: 32
Validation split: 20% of training data
The model was trained on the scaled training data and monitored using training and validation loss/accuracy.

___________________________________________________________________________________________________________________________________________________________________

📈 Performance:
Model evaluation on test data using accuracy_score:
accuracy_score(y_test, y_pred)
Training and validation curves can be visualized with matplotlib to check for overfitting.

___________________________________________________________________________________________________________________________________________________________________

🧾 Predicting New Customer Churn:
You can predict churn for a new customer by scaling their data and feeding it to the model:
new_customer = pd.DataFrame(
    [[650, 40, 3, 60000, 2, 1, 1, 50000, 0, 1, 1]],
    columns=columns
)
new_customer_scaled = scaler.transform(new_customer)
prob = model.predict(new_customer_scaled)[0][0]
prediction = 1 if prob > 0.5 else 0
Probability of churn: prob
Predicted class: prediction (1 = churn, 0 = stay)

___________________________________________________________________________________________________________________________________________________________________

🖼️ Visualizations:
The training process can be visualized:
Loss and validation loss
Accuracy and validation accuracy
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

___________________________________________________________________________________________________________________________________________________________________

⚡ Requirements:
Python 3.x
pandas
numpy
scikit-learn
matplotlib
tensorflow
Install dependencies via pip:  pip install pandas numpy scikit-learn matplotlib tensorflow
