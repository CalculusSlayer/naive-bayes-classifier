import numpy as np
import os

# ===================== ABOUT THE DATA ========================
# Inside the 'data' folder, the emails are separated into 'train' 
# and 'test' data. Each of these folders has nested 'spam' and 'ham'
# folders, each of which has a collection of emails as txt files.
        
# The emails used are a subset of the Enron Corpus,
# which is a set of real emails from employees at an energy
# company. The emails have a subject line and a body, both of
# which are 'tokenized' so that each unique word or bit of
# punctuation is separated by a space or newline. The starter
# code provides a function that takes a filename and returns a
# set of all of the distinct tokens in the file.
# =============================================================

class NaiveBayes():
    """
    This is a Naive Bayes spam filter, that learns word spam probabilities 
    from our pre-labeled training data and then predicts the label (ham or spam) 
    of a set of emails that it hasn't seen before.
    """
    def __init__(self):
        self.num_train_hams = 0
        self.num_train_spams = 0
        self.word_counts_spam = {}
        self.word_counts_ham = {}
        self.HAM_LABEL = 'ham'
        self.SPAM_LABEL = 'spam'

    def load_data(self, path:str='data/'):
        """
        This function loads all the train and test data and returns
        the filenames as lists.

        :param path: Expects a path such that inside are two folders:
        'train' and 'test'. Each of these two folders should have a 'spam' and
        'ham' folder inside. These four folders now all should just contain
        several txt files.
        :return: Inside a tuple, 
            1. A list with the ham training data filenames.
            2. A list with the spam training data filenames.
            3. A list with the ham test data filenames.
            4. A list with the spam test data filenames.
        """
        assert set(os.listdir(path)) == set(['test', 'train'])
        assert set(os.listdir(os.path.join(path, 'test'))) == set(['ham', 'spam'])
        assert set(os.listdir(os.path.join(path, 'train'))) == set(['ham', 'spam'])

        train_hams, train_spams, test_hams, test_spams = [], [], [], []
        for filename in os.listdir(os.path.join(path, 'train', 'ham')):
            train_hams.append(os.path.join(path, 'train', 'ham', filename))
        for filename in os.listdir(os.path.join(path, 'train', 'spam')):
            train_spams.append(os.path.join(path, 'train', 'spam', filename))
        for filename in os.listdir(os.path.join(path, 'test', 'ham')):
            test_hams.append(os.path.join(path, 'test', 'ham', filename))
        for filename in os.listdir(os.path.join(path, 'test', 'spam')):
            test_spams.append(os.path.join(path, 'test', 'spam', filename))

        return train_hams, train_spams, test_hams, test_spams

    def word_set(self, filename:str):  # fixed parameter filename type from list to str
        """ 
        This function reads in a file and returns a set of all 
        the words. It ignores the subject line.

        :param path: The filename of the email to process.
        :return: A set of all the unique word in that file.

        For example, if the email had the following content:

        Subject: Get rid of your student loans
        Hi there,
        If you work for us, we will give you money
        to repay your student loans. You will be
        debt free!
        FakePerson_22393

        This function would return to you the set:
        {'', 'work', 'give', 'money', 'rid', 'your', 'there,',
            'for', 'Get', 'to', 'Hi', 'you', 'be', 'we', 'student',
            'debt', 'loans', 'loans.', 'of', 'us,', 'will', 'repay',
            'FakePerson_22393', 'free!', 'You', 'If'}
        """
        with open(filename, 'r') as f:
            text = f.read()[9:] # Ignoring 'Subject:'
            text = text.replace('\r', '')
            text = text.replace('\n', ' ')
            words = text.split(' ')
            return set(words)

    def fit(self, train_hams:list, train_spams:list):
        """
        :param train_hams: A list of train email filenames which are ham.
        :param train_spams: A list of train email filenames which are spam.
        :return: Nothing.

        At the end of this function, the following should be true:
        1. self.num_train_hams is set to the number of ham emails given.
        2. self.num_train_spams is set to the number of spam emails given.
        3. self.word_counts_spam is a DICTIONARY where word_counts_spam[word]
        is the number of spam emails which contained this word. 
        4. self.word_counts_ham is a DICTIONARY where word_counts_ham[word]
        is the number of ham emails which contained this word. 
        """

        def get_counts(filenames:list):
            dict1 = {}
            for file in filenames:
                set1 = self.word_set(file)
                for word in set1:
                    if word in dict1:
                        dict1[word] += 1
                    else:
                        dict1[word] = 1
            return dict1

        self.num_train_hams = len(train_hams)
        self.num_train_spams = len(train_spams)
        self.word_counts_spam = get_counts(train_spams)
        self.word_counts_ham = get_counts(train_hams)

    def predict(self, filename:str):
        """
        :param filename: The filename of an email to classify.
        :return: The prediction of our Naive Bayes classifier. This
        should either return self.HAM_LABEL or self.SPAM_LABEL.
        """

        p_ham = self.num_train_hams/(self.num_train_hams + self.num_train_spams)
        p_spam = self.num_train_spams/(self.num_train_hams + self.num_train_spams)

        spam_sum = 0
        ham_sum = 0
        set2 = self.word_set(filename)
        for word in set2:
            spam_sum += np.log((self.word_counts_spam.get(word, 0)+1)/(self.num_train_spams+2))

        for word in set2:
            ham_sum += np.log((self.word_counts_ham.get(word, 0)+1)/(self.num_train_hams+2))

        spam_sum += np.log(p_spam)
        ham_sum += np.log(p_ham)

        if spam_sum > ham_sum:
            return self.SPAM_LABEL
        else:
            return self.HAM_LABEL

    def accuracy(self, hams:list, spams:list):
        """
        :param hams: A list of ham email filenames.
        :param spams: A list of spam email filenames.
        :return: The accuracy of our Naive Bayes model.
        """
        total_correct = 0
        total_datapoints = len(hams) + len(spams)
        for filename in hams:
            if self.predict(filename) == self.HAM_LABEL:
                total_correct += 1
        for filename in spams:
            if self.predict(filename) == self.SPAM_LABEL:
                total_correct += 1
        return total_correct / total_datapoints

if __name__ == '__main__':
    # Create a Naive Bayes classifier.
    nbc = NaiveBayes()

    # Load all the train/test ham/spam data.
    train_hams, train_spams, test_hams, test_spams = nbc.load_data()

    # Fit the model to the training data.
    nbc.fit(train_hams, train_spams)

    # Print out the accuracy on the train and test sets.
    print("Train Accuracy: {}".format(nbc.accuracy(train_hams, train_spams)))
    print("Test  Accuracy: {}".format(nbc.accuracy(test_hams, test_spams)))
