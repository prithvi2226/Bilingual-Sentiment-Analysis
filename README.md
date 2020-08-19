# Bilingual Sentiment Analysis (On Two Regional Languages)

**Start here**: [`Analysis_OG.ipynb`](Analysis_OG.ipynb)

## :thought_balloon: Background
This project applies concepts and techniques from [Natural language processing](https://en.wikipedia.org/wiki/Natural_language_processing) and [Opinion mining](https://en.wikipedia.org/wiki/Sentiment_analysis).The goal here is simply to build an artificial intelligience system that differentiates [Hindi](https://en.wikipedia.org/wiki/Hindi), [Marathi](https://en.wikipedia.org/wiki/Marathi_language) code mixed with an english text on basis of their polarity. (ie positive, negative, neutral).overall.

### Sentiment vs. Software
Using natural language processing, text analysis, computational linguistics, and biometrics to systematically identify, extract, quantify, and study affective states and subjective information. Sentiment analysis is widely applied to voice of the customer materials such as reviews and survey responses, online and social media, and healthcare materials for applications that range from marketing to customer service to clinical medicine. 

## :wrench: Progress
### Mining and Collecting the data
The main goal is to get as much comments as possible for this model. We took these comments from major social media websites like [facebook](https://en.wikipedia.org/wiki/Facebook) and [Youtube](https://en.wikipedia.org/wiki/YouTube) related to social and political views from many sources which contributes in giving us data in from of polarities. We collected about 5000 comments.

### Tagging the data
Next step was to tag all the data according to their polarity(i.e. Positive, Negative, Nuetral). Tagging scheme was basically according to -- 
- Positive Comment : 3
- Negative Comment : 1
- Neutral Comment : 2

### Data Pre-Processing
As the data is all tagged, before feeding it to the model we pre-process the data.The goal of preprocessing text data is to take the data from its raw, readable form to a format that the computer can more easily work with. Most text data, and the data we will work with in this article, arrive as strings of text. Preprocessing is all the work that takes the raw input data and prepares it for insertion into a model.

While preprocessing for numerical data is dependent largely on the data, preprocessing of text data is actually a fairly straightforward process, although understanding each step and its purpose is less trivial. Our preprocessing method consists of two stages: preparation and [vectorization](https://www.geeksforgeeks.org/vectorization-in-python/). The preparation stage consists of steps that clean up the data and cut the fat. The steps are 1. removing URLs, 2. making all text lowercase, 3. removing numbers, 4. removing punctuation, 5. [tokenization](https://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html), 6. removing [stopwords](https://towardsdatascience.com/stop-words-in-nlp-5b248dadad47), and 7. [lemmatization](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html). Stopwords are words that typically add no meaning.

### Splitting data 75-25(train-test)
[train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) returns four arrays namely training data, test data, training labels and test labels. By default train_test_split, splits the data into 75% training data and 25% test data which we can think of as a good rule of thumb.

`test_size` keyword argument specifies what proportion of the original data is used for the test set. Here we have mentioned the test_size=0.3 which means 70% training data and 30% test data.

### Hyperparameter Tuning
[Hyperparameter Tuning](https://en.wikipedia.org/wiki/Hyperparameter_optimization) used on various algorithms such as [linear Regression](https://en.wikipedia.org/wiki/Linear_regression) , [XGBoost](https://en.wikipedia.org/wiki/XGBoost) used in [`Analysis_OG.ipynb`](Analysis_OG.ipynb)

### Accuracy and other Values
[Accuracy](https://mahata.github.io/machine%20learning/2014/12/31/sklearn-accuracy_score/), [precision, Recall](https://en.wikipedia.org/wiki/Precision_and_recall) and [Fscore](https://en.wikipedia.org/wiki/F1_score) for every algorithm used is given in [`Values`](Values)
Simply got overall accuracy around `70%`.

## :bulb:Work to be done
- Contextual understanding and tone
- sentiment analysis at Brandwatch?
- The caveats of sentiment analysis
- Predictions for the future of sentiment analysis

## :question: Open questions
- Is the accuracy propotional or anyway dependent on the amount of data collected?
  - the data source should closely match the intended uses? -- https://blog.infegy.com/understanding-sentiment-analysis-and-sentiment-accuracy
- Is  sentence-level cross-lingual sentiment classification enough to predict the sentiment?

## :books: Resources
### Sentiment Analysis-related publications
- https://arxiv.org/pdf/1810.03660v1.pdf
- https://www.aclweb.org/anthology/C18-1248.pdf
- https://arxiv.org/pdf/1602.07563v2.pdf
- https://arxiv.org/pdf/1810.02508v6.pdf
