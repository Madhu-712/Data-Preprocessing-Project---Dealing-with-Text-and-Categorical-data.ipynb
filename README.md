# Data-Preprocessing-Project---Dealing-with-Text-and-Categorical-data.ipynb

Data Preprocessing Project - Dealing with Text and Categorical data
In this project, I discuss various data preprocessing techniques to deal with text and categorical data. The contents of this project are categorized into various sections which are listed below:-

Table of Contents:-
Introduction
Types of data variable

• Nominal variable

• Ordinal variable

• Interval variable

• Ratio variable

Example dataset
Encoding class labels with LabelEncoder
Encoding categorical integer labels with OneHotEncoder
Encode multi-class labels to binary labels with LabelBinarizer
Encoding list of dictionaries with DictVectorizer
Converting text document to word count vectors with CountVectorizer
Converting text document to word frequency vectors with TfidfVectorizer
Transforming a counted matrix to normalized tf-idf representation with TfidfTransformer
=======================================================================================================================

1. Introduction
In the previous project, I have discussed data preprocessing techniques to handle missing numerical data. But, the real world datasets also contain text and categorical data. In this project, I will discuss various techniques to deal with text and categorical data effectively.

Machine Learning algorithms require that input data must be in numerical format. Only then the algorithms work successfully on them. So, the text data must be converted into numbers before they are fed into an algorithm.

The process of converting text data into numbers consists of two steps. First, the text data must be parsed to remove words. This process is called tokenization. Then the words need to be encoded as integers or floating point values for use as input to a machine learning algorithm. This process is called feature extraction or vectorization.

The Scikit-Learn library provides useful classes like LabelEncoder, OneHotEncoder, LabelBinarizer, DictVectorizer, CountVectorizer etc. to perform tokenization and vectorization. In this project, I will explore these classes and the process of encoding text and categorical data into numerical representation.

========================================================================================================================

2. Types of data variable
We can divide categorical data into four types. These are nominal, ordinal, interval and ratio data. These terms were developed by Stanley Smith Stevens, an American psychologist. His work was published in 1946 and these terms came into effect. These four types of data variable – nominal, ordinal, interval and ratio data are best understood with the help of examples.

Nominal variable
A categorical variable is also called a nominal variable when it has two or more categories. It is mutual exclusive, but not ordered variable. There is no ordering associated with this type of variable. Nominal scales are used for labelling variables without any quantitative value. For example, gender is a categorical variable having two categories - male and female, and there is no intrinsic ordering to the categories. Hair colour is also a categorical variable having a number of categories - black, blonde, brown, brunette, red, etc. and there is no agreed way to order these from highest to lowest. If the variable has a clear ordering, then that variable would be an ordinal variable, as described below.

Ordinal variable
In ordinal variables, there is a clear ordering of the variables. Here the order matters but not the difference between values. The order of the values is important and significant. For example, suppose we have a variable economic status with three categories - low, medium and high. In this example, we classify people into three categories. Also, we order the categories as low, medium and high. So, the categories express an order of measurement. Ordinal scales are measures of non-numeric concepts like satisfaction and happiness. It is easy to remember because it sounds like ordering into different categories. If these categories were equally spaced, then that variable would be an interval variable.

Interval variable
An interval variable is similar to an ordinal variable, except that the intervals between the values of the interval variable are equally spaced. Interval scales are numeric scales in which we know both the order and the exact differences between the values. The difference between two interval variables is measurable and constant. The example of an interval scale is Celsius temperature because the difference between each value is the same. For example, the difference between 60 and 50 degrees is a measurable 10 degrees, as is the difference between 80 and 70 degrees.

Ratio variable
A ratio variable, has all the properties of an interval variable, and also has a clear definition of 0.0. When the variable equals 0.0, there is none of that variable. Variables like height, weight, enzyme activity are ratio variables. Temperature, expressed in F or C, is not a ratio variable. A ratio variable, has all the properties of an interval variable, and also has a clear definition of 0.0. When the variable equals 0.0, there is none of that variable. Variables like height, weight, enzyme activity are ratio variables. Temperature, expressed in F or C, is not a ratio variable.

========================================================================================================================

3. Example dataset
I create an example dataset to illustrate various techniques to deal with the text and categorical data. The example dataset is about Grand Slam Tennis tournaments. The dataset contains five columns describing the Grand Slam title, host country, surface type, court speed and prize money in US Dollars Millions associated with the tournaments.

So, I will start by importing the required Python libraries.

# Import required libraries

import numpy as np

import pandas as pd
# Create an example dataset

import pandas as pd

df = pd.DataFrame([
                ['Australian Open', 'Australia', 'Hard Court','Medium', 3.2 ],
                ['French Open', 'France', 'Clay Court', 'Slow', 2.7],
                ['Wimbledon', 'UK', 'Grass Court', 'Fast', 2.91],
                ['US Open', 'USA', 'Hard Court', 'Medium', 3.8]])

df.columns = ['Grand Slam Title', 'Host Country', 'Surface Type', 'Court Speed', 'Prize Money(USD million)']
# View the first few rows of the dataset

df.head()
Grand Slam Title	Host Country	Surface Type	Court Speed	Prize Money(USD million)
0	Australian Open	Australia	Hard Court	Medium	3.20
1	French Open	France	Clay Court	Slow	2.70
2	Wimbledon	UK	Grass Court	Fast	2.91
3	US Open	USA	Hard Court	Medium	3.80
# View the summary of dataframe df

print(df.info())
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4 entries, 0 to 3
Data columns (total 5 columns):
Grand Slam Title            4 non-null object
Host Country                4 non-null object
Surface Type                4 non-null object
Court Speed                 4 non-null object
Prize Money(USD million)    4 non-null float64
dtypes: float64(1), object(4)
memory usage: 240.0+ bytes
None
# View the descriptive statistics

print(df.describe())
       Prize Money(USD million)
count                  4.000000
mean                   3.152500
std                    0.477869
min                    2.700000
25%                    2.857500
50%                    3.055000
75%                    3.350000
max                    3.800000
Interpretation

We can see that the dataframe df contains 5 columns. The columns Grand Slam Title, Host Country, Surface Type and Court Speed are of object data types while the column Prize Money is of integer data type.

=========================================================================================================================

4. Encoding class labels with LabelEncoder
The machine learning algorithms require that class labels are encoded as integers. Most estimators for classification convert class labels to integers internally. It is considered a good practice to provide class labels as integer arrays to avoid problems. Scikit-Learn provides a transformer for this task called LabelEncoder.

Suppose there are three nominal variables x1, x2, x3 given by NumPy array y

y = df[[‘x1', ‘x2’, ‘x3’]].values

The following code has been implemented in Scikit-Learn to transform y into integer values.

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(df[[‘x1’, ‘x2’, ‘x3’]].values)

print(y)

The fit_transform method is just a shortcut for calling fit and transform separately. We can use the inverse_transform method to transform the integer class labels back into their original string representation.

le.inverse_transform(y)

# Make a copy of the dataframe df

df1 = df.copy()

# I have made a copy of the dataframe df as df1.

# Now, I will work with df1.
# View the first few rows of dataframe df1

df1.head()
Grand Slam Title	Host Country	Surface Type	Court Speed	Prize Money(USD million)
0	Australian Open	Australia	Hard Court	Medium	3.20
1	French Open	France	Clay Court	Slow	2.70
2	Wimbledon	UK	Grass Court	Fast	2.91
3	US Open	USA	Hard Court	Medium	3.80
We can see that the Court Speed variable is Ordinal variable. I will now encode this Court Speed variable into integer values.

# Encode Court Speed column labels into integer values

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y1 = le.fit_transform(df1['Court Speed'].values)

print("Encoded Court Speed column labels are:\n", (y1))
Encoded Court Speed column labels are:
 [1 2 0 1]
Interpretation

We can see that the Court Speed column which contain Medium, Slow, Fast and Medium values are now encoded as 1 2 0 1.

# Print class labels

print("The class labels are:\n", (le.classes_))
The class labels are:
 ['Fast' 'Medium' 'Slow']
# Suppress future warnings

import warnings

warnings.simplefilter(action = "ignore", category = DeprecationWarning)
# Invert the encoded class labels to original class labels

print("The inverted original class labels are:\n",
      le.inverse_transform([1, 2, 0, 1]))
The inverted original class labels are:
 ['Medium' 'Slow' 'Fast' 'Medium']
Interpretation

We can view the original class labels and inverted class labels with the le.classes_ and le.inverse_transform(y) commands.

========================================================================================================================

5. Encoding categorical integer labels with OneHotEncoder
There is one problem associated with encoding class labels with LabelEncoder. Scikit-Learn’s estimator for classification treat class labels as categorical data with no order associated with it. So, we used the LabelEncoder to encode the string labels into integers. The problem arises when we apply the same approach to transform the nominal variable with LabelEncoder.

We have seen above that LabelEncoder transform NumPy array y given by

y = df[[‘x1’, ‘x2’, ‘x3’]].values

into integer array values given by

array([0, 1, 2])

So, we can map the nominal variables x1, x2, x3 to integer values 0, 1, 2 as follows.

x1 = 0

x2 = 1

x3 = 2

Although, there is no order involved with x1, x2, x3, but a learning algorithm will now assume that x1 < x2 < x3. This is wrong assumption and it will not produce desired results. We will see later, how we can solve this problem

But, first I will convert the nominal feature variable into integer values.

Now, I will encode the nominal feature variable Grand Slam Title into integer values.

# Encode Grand Slam Title column values into integer values

X1 = df1[['Grand Slam Title', 'Prize Money(USD million)']].values

title_le = LabelEncoder()

X1[:, 0] = title_le.fit_transform(X1[:, 0])

print(X1)
[[0 3.2]
 [1 2.7]
 [3 2.91]
 [2 3.8]]
Interpretation

Here the problem arises. We can see that the Grand Slam Title column values Australian Open, French Open, Wimbledon and US Open are now encoded as 0, 1, 3 and 2.

So, Australian Open is mapped to 0, French Open is mapped to 1, Wimbledon is mapped to 3 and US Open is mapped to 2. So, we can conclude that

Australian Open < French Open < US Open < Wimbledon

But, this is not true.

To fix this issue, a common solution is to use a technique called one-hot-encoding. In this technique, we create a new dummy feature for each unique value in the nominal feature column. The value of the dummy feature is equal to one when the unique value is present and zero otherwise. Similarly, for another unique value, the value of the dummy feature is equal to one when the unique value is present and zero otherwise. This is called one-hot encoding, because only one dummy feature will be equal to one (hot) , while the others will be zero (cold).

Scikit-Learn provides a OneHotEncoder transformer to convert integer categorical values into one-hot vectors. The following code accomplish this task with the NumPy array y –

y = df[[‘x1’, ‘x2’, ‘x3’]].values

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()

y = ohe.fit_transform(y).toarray()

print(y)

By default, the output is a SciPy sparse matrix, instead of a NumPy array. This way of output is very useful when we have categorical attributes with thousands of categories. If there are lot of zeros, a sparse matrix only stores the location of the non-zero elements. So, sparse matrices are a more efficient way of storing large datasets. It is supported by many Scikit-Learn functions.

To convert the dense NumPy array, we should call the toarray ( ) method. To omit the toarray() step, we could alternatively initialize the encoder as

OneHotEncoder( … , sparse = False)

to return a regular NumPy array.

Another way which is more convenient is to create those dummy features via one-hot encoding is to use the pandas.get_dummies() method. The get_dummies() method will only convert string columns and leave all other columns unchanged in a dataframe.

import pandas as pd

pd.get_dummies([[‘x1’, ‘x2’, ‘x3’]])

# Encode the converted integer values of Grand Slam Title column values into one-hot vectors

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categorical_features=[0])

ohe.fit_transform(X1).toarray()
array([[1.  , 0.  , 0.  , 0.  , 3.2 ],
       [0.  , 1.  , 0.  , 0.  , 2.7 ],
       [0.  , 0.  , 0.  , 1.  , 2.91],
       [0.  , 0.  , 1.  , 0.  , 3.8 ]])
Interpretation

We can now see that the column values in the Grand Slam Title column are now converted into one-hot-vectors.

In the first row, Australian Open is present. So, the dummy variable contains 1 for the Australian Open and 0's for the other titles since other titles are not present.

Similar explanation goes for the other Grand Slam Title column values in the second, third and fourth rows.

=======================================================================================================================

6. Encode multi-class labels to binary labels with LabelBinarizer
We can accomplish both the tasks (encoding multi-class labels to integer categories, then from integer categories to one-hot vectors or binary labels) in one shot using the Scikit-Learn’s LabelBinarizer class.

We can define a NumPy array y as follows:-

y = df[[‘x1’, ‘x2’, ‘x3’]].values

The following code transform y into binary labels using LabelBinarizer

from sklearn.preprocessing import LabelBinarizer

lb =  LabelBinarizer()

y = lb.fit_transform(df[[‘x1’, ‘x2’, ‘x3’]].values)

print(y)

This returns a dense NumPy array by default. We can get a sparse matrix by passing sparse_output = True to the LabelBinarizer constructor.

# Copy the dataframe df2

df2 = df.copy()
# View the first few rows of df2

df2.head()
Grand Slam Title	Host Country	Surface Type	Court Speed	Prize Money(USD million)
0	Australian Open	Australia	Hard Court	Medium	3.20
1	French Open	France	Clay Court	Slow	2.70
2	Wimbledon	UK	Grass Court	Fast	2.91
3	US Open	USA	Hard Court	Medium	3.80
Suppose the Prize Money column in dataframe df2 is given as a list.

Then we can use LabelBinarizer to encode the Prize Money values into one-hot-vectors.

# Slice the Prize Money column

X2 = df2.iloc[:, 4]
X2.head()
0    3.20
1    2.70
2    2.91
3    3.80
Name: Prize Money(USD million), dtype: float64
We can see that the data type of X2 is float 64. We need to convert it into integer.

# Change the data type of X2 to integer 64

X2 = X2.astype('int64')
# Check the data type of X2

X2.dtype
dtype('int64')
# Encode X2 into one-hot-vectors with LabelBinarizer

from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()

y2 = lb.fit_transform(X2)

print(y2)
[[1]
 [0]
 [0]
 [1]]
We can see that the Prize Money column is converted into one-hot-vectors.

Now, I will check its classes.

print("The class labels are:\n", lb.classes_)
The class labels are:
 [2 3]
Interpretation

The class labels of encoded values are 2 and 3. So, the values less than 3 are encoded as 0 and the values more than 3 are encoded as 1.

=========================================================================================================================

7. Encoding list of dictionaries with DictVectorizer
We have previously seen that we can use OneHotEncoder transformer to convert integer categorical values into one-hot vectors. But, when the data comes as a list of dictionaries, we can use Scikit-Learn's DictVectorizer transformer to do the same job for us.

DictVectorizer will only do a binary one-hot encoding when feature values are of type string.

Suppose there is a list of dictionaries given by y as follows:-

y = df[ {‘foo1’ :  x1}, {‘foo2’  : x2}, {‘foo3’  : x3}].

We can use DictVectorizer to do a binary one-hot encoding as follows:-

from sklearn.preprocessing import DictVectorizer

dv = DictVectorizer (sparse = False)

X_dv = dv.fit_transform(y)

print(X_dv)

With these categorical features thus encoded, we can proceed as normal with fitting a Scikit-Learn model.

To see the meaning of each column, we can inspect the feature names as follows:-

dv.get_feature_names()

tennis_df = [ 
            {'title': 'Australian Open', 'prize money': 3.20},
            {'title': 'French Open',     'prize money': 2.70},
            {'title': 'Wimbledon',       'prize money': 2.91},
            {'title': 'US Open',         'prize money': 3.80}
        ]

# We can use DictVectorizer to do a binary one-hot encoding

from sklearn.feature_extraction import DictVectorizer

dv = DictVectorizer(sparse = False)

X_dv = dv.fit_transform(tennis_df)

print(X_dv)
[[3.2  1.   0.   0.   0.  ]
 [2.7  0.   1.   0.   0.  ]
 [2.91 0.   0.   0.   1.  ]
 [3.8  0.   0.   1.   0.  ]]
# inspect the feature names

print("The feature names of tennis_df data structure are:\n" , dv.get_feature_names())
The feature names of tennis_df data structure are:
 ['prize money', 'title=Australian Open', 'title=French Open', 'title=US Open', 'title=Wimbledon']
========================================================================================================================

8. Converting text document to word count vectors with CountVectorizer
We cannot work directly with text data when using machine learning algorithms. Instead, we need to convert the text to numerical representation. Algorithms take numbers as input for further analysis. So, we need to convert text documents to vectors of numbers.

A simple yet effective model for representing text documents in machine learning is called the Bag-of-Words Model, or BoW. It focusses on the occurrence of words in a document. The Scikit-Learn’s CountVectorizer transformer is designed for representing "bag-of-words" technique. CountVectorizer takes the text data as input and count the occurrences of each word within it. The result is a sparse matrix recording the number of times each word appears.

For example, consider the following sample text data:-

corpus = [‘dog’, 
        ‘cat’
    ‘dog chases cat’]

We can use CountVectorizer to convert data as follows:-

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer ()

data = cv.fit_transform(corpus)

We can inspect the feature names and view the transformed data as follows:-

print(cv.get_feature_names())

print(data.toarray())

# View the transformed data

corpus = ['dog',
          'cat',
        'dog chases cat']


# Use CountVectorizer to convert data

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X_cv = cv.fit_transform(corpus)

print(X_cv.toarray())
[[0 0 1]
 [1 0 0]
 [1 1 1]]
# Inspect the feature names

print("The feature names of the text document corpus are:\n", cv.get_feature_names())
The feature names of the text document corpus are:
 ['cat', 'chases', 'dog']
====================================================================================================================

9. Converting text document to word frequency vectors with TfidfVectorizer
There is one problem associated with the above approach of converting text document to word count vectors with CountVectorizer. The raw word counts result in features which put too much emphasis on words that appear frequently. This cannot produce desired results in some classification algorithms.

A solution to the above problem is to calculate word frequencies. We can use Scikit-Learn’s Tfidf transformer to calculate word frequencies. It is commonly called as TF-IDF. TF-IDF stands for Term Frequency – Inverse Document Frequency. TF-IDF weights the word counts by a measure of how often they appear in the documents.

The TfidfVectorizer will tokenize documents, learn the vocabulary and inverse document frequency weightings, and allow you to encode new documents. TfidfVectorizer is equivalent to CountVectorizer followed by TfidfTransformer (described below).

The syntax for computing TF-IDF features is given below:-

from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [‘dog’,

‘cat’

‘dog chases cat’]

vec =  TfidfVectorizer()

X = vec.fit_transform(corpus)

print(X.toarray())

from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ['dog',

        'cat',

        'dog chases cat']

vec1 =  TfidfVectorizer()

X_tfv = vec1.fit_transform(corpus)

print(X_tfv.toarray())
[[0.         0.         1.        ]
 [1.         0.         0.        ]
 [0.51785612 0.68091856 0.51785612]]
We can inspect the feature names with the following command.

print(vec1.get_feature_names())

# get feature names

print(vec1.get_feature_names())
['cat', 'chases', 'dog']
======================================================================================================================

10. Transforming a counted matrix to normalized tf-idf representation with
TfidfTransformer
We have previously seen that CountVectorizer takes the text data as input and count the occurrences of each word within it. The result is a sparse matrix recording the number of times each word appears.

If we already have such a matrix, we can use it with a TfidfTransformer to calculate the inverse document frequencies (idf) and start encoding documents.

# Calculate inverse document frequencies (idf)

from sklearn.feature_extraction.text import TfidfTransformer

vec2 =  TfidfTransformer()

X_tft = vec2.fit_transform(X_cv)

print(X_tft.toarray())
[[0.         0.         1.        ]
 [1.         0.         0.        ]
 [0.51785612 0.68091856 0.51785612]]
This concludes our discussion on text and categorical data.
