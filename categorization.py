from math import log
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def categorize(train, test, out):

    # Let's first initialize variables like our smoothing parameter and frequency "trackers" (explained later when used)
    k = 0.05
    word_cat_dict, cat_dict, prior_dict = dict(), dict(), dict()
    stemmer = PorterStemmer()
    predictions = []

    # First we train. Let's split each entry in our training file to separate the file from it's associated category.
    with open(train, 'r') as f:
        for line in f:
            file, category = line.split()
            # Let's open and tokenize the text (separate each token, think of these like a "piece" of the text)
            document = open(file, 'r')
            # Next, let's keep track of how many times we've seen a particular category (in specific, how many files).
            prior_dict[category] = prior_dict[category] + 1 if category in prior_dict.keys() else 1
            # Now, let's loop through all of the tokens.
            for token in word_tokenize(document.read()):
                # We'll only look at the "stem" of the word, removing morphological affixes.
                token = stemmer.stem(token)
                # Here, we'll see how many times this specific token appears in this category
                word_cat_dict[(token, category)] = word_cat_dict[(token, category)] + 1 if (token, category) in word_cat_dict else 1
                # Let's also keep track of how many tokens are in this category.
                cat_dict[category] = cat_dict[category] + 1 if category in cat_dict else 1

    # Now we test. Let's read in the testing file like we did the training file.
    with open(test, 'r') as f:
        for line in f:
            file = open(line.strip(), 'r')
            # We'll separate the tokens in the file like we did when training.
            test_dict = dict()
            # Like earlier, we only care about the "stem" of the word.
            for token in word_tokenize(file.read()):
                token = stemmer.stem(token)
                # We only care about words, not punctuation.
                if token not in list(string.punctuation):
                    test_dict[token] = test_dict[token] + 1. if token in test_dict else .1
            # This is where Naive-Bayes is used. Figure 4.2 in the textbook was referenced.
            cat_log_prob = dict()
            for category in cat_dict.keys():
                total_log_cat_prob = 0.
                cat_prior = prior_dict[category] / sum(prior_dict.values())
                for word, count in test_dict.items():
                    count_word_cat = word_cat_dict[(word, category)] + k if (word, category) in word_cat_dict else k
                    log_cat_prob = count * log(count_word_cat / (cat_dict[category] + k * len(test_dict)))
                    total_log_cat_prob += log_cat_prob
                    cat_log_prob[category] = total_log_cat_prob + log(cat_prior)
            predicted_label = max(cat_log_prob, key=cat_log_prob.get)
            predictions.append(line.strip() + ' ' + predicted_label + '\n')

    # Finally, let's write our predictions onto the output file specified earlier by the user.
    output = open(out, 'w')
    for p in predictions: output.write(p)
    output.close()
    print("Program Complete.")


train = input('Enter the name of the training file: ')
test = input('Enter the name of the testing file: ')
output = input('Enter the name you would like for the output file: ')
categorize(train, test, output)
