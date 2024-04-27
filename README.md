# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 3: Reddit APIs & NLP

## Subreddit Classifier: Movie vs Music

### Problem Statement

Creating a good text classification model is important to making online communities more engaging and helping users find content they are interested in. This project goal is to build a machine learning model that can automatically categorize online posts as either about "music" or "movies". We'll train this model using a dataset of labeled text from online posts to learn patterns and relationships between words and categories.

The success of the model is evaluated by looking at how accurately it can classify new posts, checking metrics like accuracy, precision, recall, and F1-score. The focus here is on sorting text from posts found in "movies" and "music" subreddits. This research is important because precise text classification can greatly improve how people discover content online and interact with it.

This project has benefits for different groups. Community users, social media platforms, and content creators will all benefit from better content discovery and happier users. Advertisers, content moderators, and researchers in natural language processing and human-computer will also benefit from the insights and methodologies developed in this project. By considering everyone's needs, the goal of this project is to create a text classification model that is practical and useful in real-world situations.

---

### Data Dictionary

Data Dictionary

| Feature     | Type    | Description                                         |
| ----------- | ------- | --------------------------------------------------- |
| author_id   | string  | Unique identifier for the Author of the Reddit post |
| post_id     | string  | Unique identifier for the Reddit post               |
| created_utc | integer | The timestamp of when the Reddit post was created   |
| title       | string  | The title of the Reddit post                        |
| self_text   | string  | The text content of the Reddit post                 |
| subreddit   | string  | The source subreddit: 'movies' or 'music'           |

---

### Executive Summary

#### Data Collection

The following are the 2 datasets were used for this project, and collected using Python Reddit API Wrapper ([PRAW](https://praw.readthedocs.io/en/stable/)), 1000 posts were scraped for each datasets for `music` and `movies` subreddits. While scraping data, only data with non missing values for features `title` and `self_text` were collected, since these are important features for the model.

**Datasets Collected Saved as csv**

- [`movies_posts.csv`](./data/movies_posts.csv): Population by Country
- [`music_posts.csv`](./data/music_posts.csv): Life Expectancy by Country

#### Data Cleaning

- While scraping data, only posts with non-missing values for features like `title` and `self_text` were retained, eliminating the need for explicitly dropping missing values during data cleaning.
- A clean post column, with `title` and `self_text`, was created by removing common or stop words, enhancing the dataset's suitability for text analysis.
- More features were engineered, including post length in characters and word count, to provide deeper insights into the dataset's structure and content.
- The text content of posts, including titles and self-text, was lemmatized for features to be used for building models.

#### Questions Explored

- Which features contribute most to the classification of Reddit posts as "Movies" or "Music"?
- What factors have the most significant impact on the accuracy of predicting Reddit post classifications?
- How do different models, such as Logistic Regression, Naive Bayes, Random Forest, and XGBoost, compare in classifying Reddit posts?

#### Conclusions and Recommendations

##### Conclusions

NLP models were successfully developed to classify Reddit posts as either "Movies" or "Music", achieving high accuracy scores ranging from 0.98 to 1.00 on both training and test sets. Our analysis revealed that certain features, such as titles and posts content, and the presence or absence of common words (movies, film, actor, song, copy, director, sound, album, starter, etc.), played an important role in distinguishing between the two categories.

XGBoost Classifier with CountVectorizer consistently demonstrated high scores (1.00) for precision, recall, and F1-score on both the training and test sets with GridSearchCV. This exceptional performance underscores its effectiveness in accurately categorizing online content. While this Shows an excellent fit to the training data, it also suggests potential overfitting as the model may have memorized the training data, leading to identical predictions on the test set.

##### Recommendations

- Given the exceptional accuracy achieved by models, prioritize their deployment in real-world applications to automate the categorization of online music/movies posts effectively.
- Despite achieving high accuracy, it's important to continuously monitor model performance over time to ensure its effectiveness as data distributions or user behaviors evolve.
- Encourage user feedback and engagement to validate model predictions and refine classification algorithms based on user preferences and behaviors. This iterative process can further enhance model accuracy and relevance.

**Steps to Run app.py and make predictions on the Front-end:**

1. Install Streamlit by running `pip install streamli`t in `terminal` or `command prompt`.
2. Run the `Streamlit app` by running `streamlit run app.py` in `terminal` or `command prompt`.

This streamlit app only run with scikit-learn==1.2.2
