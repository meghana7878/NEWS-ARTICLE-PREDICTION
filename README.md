# NEWS-ARTICLE-PREDICTION

**News Article Prediction** is a machine learning application that classifies news articles into categories such as **Business**, **Technology**, **Politics**, **Sports**, and **Entertainment**. It utilizes natural language processing (NLP) and a Multinomial Naïve Bayes model to predict the category of a given news article.

---

## Features

1. **Text Preprocessing:**
   Raw news data is cleaned and preprocessed using tokenization, stopword removal, and regular expressions.

2. **TF-IDF Vectorization:**
   Converts textual data into numerical form using TF-IDF to improve model performance.

3. **News Classification:**
   Implements a **Multinomial Naïve Bayes** classifier for effective multi-class text classification.

4. **Performance Evaluation:**
   Evaluates the model using accuracy score, confusion matrix, and classification report.

5. **Visualization:**

   * **WordClouds** to show the most frequent words in each category
   * **Confusion Matrix** using Seaborn
   * **Bar plots** for performance comparison

---

## Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* Scikit-learn
* Matplotlib
* Seaborn
* WordCloud

---

## Setup

To run the application locally, follow these steps:

### Clone the Repository:

```bash
git clone https://github.com/meghana7878/NEWS-ARTICLE-PREDICTION.git
cd NEWS-ARTICLE-PREDICTION
```

### Install Dependencies:

Ensure Python is installed on your system. You can download it from [python.org](https://www.python.org/).

Install the required libraries using `pip`:

```bash
pip install pandas numpy nltk matplotlib seaborn scikit-learn wordcloud
```

After installing NLTK, make sure to download the necessary resources:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## Run the Script

```bash
python main.py
```

The script will load the dataset, preprocess the text, train the model, and display the classification results and visualizations.

---

## Usage

1. Prepare or load your dataset containing news articles and their categories.
2. Run the `main.py` script.
3. View classification metrics, word clouds, and confusion matrix for model evaluation.

---

## Contributing

Contributions are welcome!
Feel free to open an issue or submit a pull request for improvements or feature additions.




