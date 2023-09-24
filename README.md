# Naive Bayes Classifier

This program is an application of Machine Learning. It learns how to predict whether an email is spam or not by training. It first analyzes the patterns of 800 spam/ham emails. After analyzing the frequency of words in all the emails, it is ready to predict whether an email is spam or not. This project required us to use the Naive Bayes theorem which assumes that words are independent given the label. Laplace smoothing is applied in order to prevent a division by 0 error. In order to prevent underflow, every conditional probability term is taken the logarithm of. This works because the logarithm function is a monotonously increasing function, so the comparison between the P(spam | {w1, w2..}) and P(ham | {w1, w2..}) will not change.

## How to Run

Run the program with:

```
python naive_bayes.py
```

