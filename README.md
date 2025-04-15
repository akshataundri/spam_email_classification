# spam_mail_classification

```markdown
# 📧 Spam or Ham Classifier 📨

Welcome to the **Spam or Ham Classifier** project! 🎉 In this project, we dive into email data to predict whether a message is **spam** 🕵️‍♂️ or **ham** 🐷 (a real message). Using machine learning magic ✨, we'll automatically detect spammy emails from legit ones! 

![Spam Image](https://upload.wikimedia.org/wikipedia/commons/8/89/Spam_can.png)

---

## 💼 Project Overview

We’re using a dataset of emails 📩 to identify if the message is spam or not. With the help of machine learning models like **Logistic Regression** and some preprocessing wizardry 🧙‍♂️, we'll build a model that can predict if an email is good or junk. 

### 📊 Dataset Features

The dataset consists of two main columns:

- **Category**: Whether the message is spam (0) or ham (1).
- **Message**: The actual email content.

### 🛠️ Data Preprocessing

1. **Handle Missing Data**: We replaced any null values with empty strings, so there’s no gap in our analysis.
2. **Label Encoding**: We transformed the categorical labels:
   - **ham → 1**
   - **spam → 0**
3. **Text Vectorization**: The message content was transformed into numeric vectors using **TF-IDF** to prepare for model training.

---

## 🔮 Models Used

We used the following model to predict whether an email is spam or ham:

1. **🧮 Logistic Regression**: A simple and effective algorithm for binary classification tasks.
   - Accuracy on training data: **96.68%**
   - Accuracy on test data: **97.13%**

---

## 🚀 How to Run the Project

Excited to classify some emails? Follow these steps to get the project running on your local machine:

### Prerequisites

You’ll need a few Python libraries to run this project. Install them with pip:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost
```

### Step-by-Step Guide

1. **Clone the Repo**: Download a copy of this project onto your computer:

```bash
git clone https://github.com/Ashwadhama2004/spam-or-ham-classifier.git
cd spam-or-ham-classifier
```

2. **Get the Dataset**: The dataset `mail_data.csv` is already included in the project folder, so you can skip downloading!

3. **Run the Code**: You can run this project using a Jupyter notebook:

```bash
jupyter notebook spam_or_ham_classifier.ipynb
```

Alternatively, if you're in a rush, run the Python script directly 🏃‍♂️:

```bash
python spam_or_ham_classifier.py
```

---

## 📈 Results

Here’s how our model performed:

- **Logistic Regression**: Test Accuracy = **97.13%**

Want to play around? Try out different algorithms like **Random Forest** or **XGBoost** to see if you can beat our current accuracy!

---

## 🧪 Sample Prediction

Wanna test it? Here's an example email:

```python
input_mail = ["I've been searching for the right words to thank you for this breather. I promise I won't take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times"]

# Predict if it's spam or ham
prediction = classo.predict(input_data_features)

if (prediction[0]==1):
  print('Ham mail')
else:
  print('Spam mail')
```

Result:
```
Ham mail ✅
```

---

## 💡 Future Enhancements

You can make the classifier even better by:
- **Feature Engineering**: Add more complex features like word count, presence of links, etc.
- **Hyperparameter Tuning**: Adjust the model parameters for even better accuracy.
- **Explore Other Models**: Experiment with different machine learning algorithms like SVM, Random Forest, or XGBoost.

---

## 🛠️ Tech Stack

Here’s what we used to build this project:

- **Language**: Python 🐍
- **Libraries**: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`
- **Jupyter Notebook**: For interactive data analysis and model training.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

## 👋 Connect with Me

If you enjoyed this project or have any questions, feel free to reach out!

- GitHub: [Ashwadhama2004](https://github.com/Ashwadhama2004)

---

Thanks for checking out my **Spam or Ham Classifier** project! I hope you have fun detecting spam and learning about machine learning! 😊
```
