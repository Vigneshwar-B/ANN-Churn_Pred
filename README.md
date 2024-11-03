# Customer Churn Prediction

![Churn Prediction](https://via.placeholder.com/800x200.png?text=Customer+Churn+Prediction)

![image](https://github.com/user-attachments/assets/d1394818-4d07-4e0a-82f2-831915ad3a8c)


### Overview

The Customer Churn Prediction project aims to predict the likelihood of customer churn based on various input features. The application utilizes a trained Artificial Neural Network (ANN) model to analyze user input and provide predictions.

## Live Demo

You can access the live Streamlit application [here](https://ann-churnpred-7.streamlit.app/).

## Table of Contents

- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Features](#features)
- [How to Run the App Locally](#how-to-run-the-app-locally)
- [Model Training](#model-training)
- [Results](#results)
- [License](#license)

## Technologies Used

- Python
- Streamlit
- TensorFlow
- Scikit-Learn
- Pandas
- NumPy

## Project Structure

```
.
├── Churn_Modelling.csv               # Dataset for training the model
├── README.md                          # Project documentation
├── app.py                             # Main Streamlit application file
├── experiments.ipynb                  # Notebook for experiments and analyses
├── hyperparametertuningann.ipynb      # Notebook for hyperparameter tuning of the ANN
├── label_encoder_gender.pkl            # Pickled Label Encoder for Gender
├── model.h5                           # Trained Keras model
├── onehot_encoder_geo.pkl              # Pickled One-Hot Encoder for Geography
├── prediction.ipynb                    # Notebook for making predictions
├── requirements.txt                    # Python package dependencies
├── salaryregression.ipynb             # Notebook related to salary regression analysis
└── scaler.pkl                         # Pickled Scaler for feature scaling
```

## Features

- User-friendly interface for inputting customer data.
- Real-time prediction of churn probability.
- Visualization of prediction results.
- Responsive design with a clean and modern UI.

## How to Run the App Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/Vigneshwar-B/ANN-Churn_Pred.git
   cd ANN-Churn_Pred
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Model Training

The model is trained using the Churn_Modelling dataset and employs various preprocessing techniques, including label encoding and one-hot encoding for categorical features. The model's architecture is built using TensorFlow and Keras.

## Results

Upon inputting customer data, the application displays the predicted churn probability along with a message indicating whether the customer is likely to churn or not.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

Built with ❤️ by Vigneshwar B.
```

Feel free to modify any sections according to your preferences or project specifics!
