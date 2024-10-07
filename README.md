# Salary Prediction using Linear Regression

This project uses linear regression to predict salaries based on years of experience, test scores, and interview scores. The dataset includes various features such as experience, test scores, and interview scores.

## Requirements

- pandas
- numpy
- matplotlib
- word2number
- scikit-learn

## Dataset

The dataset used is `hiring.csv`, which contains information about candidates including their experience, test scores, interview scores, and salaries.

## Steps

1. **Data Preparation**: Load the dataset and preprocess it by handling missing values and converting text to numerical data.
2. **Feature Engineering**: Convert text-based experience data to numerical values.
3. **Model Training**: Train a linear regression model to predict salaries.
4. **Prediction**: Use the trained model to predict salaries for given inputs.
5. **Visualization**: Visualize the data using Matplotlib.

### Data Preparation

Load the dataset and handle missing values:

```python
import pandas as pd
import numpy as np
from word2number import w2n

data = pd.read_csv("hiring.csv")

# Fill missing values
data['experience'] = data['experience'].fillna('zero')
data['test_score(out of 10)'] = data['test_score(out of 10)'].fillna(data['test_score(out of 10)'].mean())

# Convert text-based experience data to numerical values
data['experience'] = data['experience'].apply(w2n.word_to_num)

print(data)
