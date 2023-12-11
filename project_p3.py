#imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import string
import re
import nltk

EMBEDDING_FILE = "w2v.pkl"


# Function: load_w2v
# filepath: path of w2v.pkl
# Returns: A dictionary containing words as keys and pre-trained word2vec representations as numpy arrays of shape (300,)
def load_w2v(filepath):
    with open(filepath, 'rb') as fin:
        return pkl.load(fin)


# Function: load_as_list(fname)
# fname: A string indicating a filename
# Returns: Two lists: one a list of document strings, and the other a list of integers
#
# This helper function reads in the specified, specially-formatted CSV file
# and returns a list of documents (documents) and a list of binary values (label).
def load_as_list(fname):
    df = pd.read_csv(fname)
    documents = df['review'].values.tolist()
    labels = df['label'].values.tolist()
    return documents, labels


# Function: extract_user_info(user_input)
# user_input: A string of arbitrary length
# Returns: name as string
def extract_user_info(user_input):
    name = ""
    name_match = re.search(r"(^|\s)([A-Z][A-Za-z-&'\.]*(\s|$)){2,4}", user_input)
    if name_match is not None:
        name = name_match.group(0).strip()
    return name


# Function to convert a given string into a list of tokens
# Args:
#   inp_str: input string
# Returns: token list, dtype: list of strings
def get_tokens(inp_str):
    return inp_str.split()


# Function: vectorize_train, see project statement for more details
# training_documents: A list of strings
# Returns: An initialized TfidfVectorizer model, and a document-term matrix, dtype: scipy.sparse.csr.csr_matrix
def vectorize_train(training_documents):
    # Initialize the TfidfVectorizer model and document-term matrix
    vectorizer = TfidfVectorizer()
    tfidf_train = None

    vectorizer = TfidfVectorizer(tokenizer=get_tokens, lowercase=True)
    tfidf_train = vectorizer.fit_transform(training_documents)

    return vectorizer, tfidf_train


# Function: w2v(word2vec, token)
# word2vec: The pretrained Word2Vec representations as dictionary
# token: A string containing a single token
# Returns: The Word2Vec embedding for that token, as a numpy array of size (300,)
#
# This function provides access to 300-dimensional Word2Vec representations
# pretrained on Google News.  If the specified token does not exist in the
# pretrained model, it should return a zero vector; otherwise, it returns the
# corresponding word vector from the word2vec dictionary.
def w2v(word2vec, token):
    word_vector = np.zeros(300, )

    if token in word2vec.keys():
        word_vector = word2vec[token]

    return word_vector


# Function: string2vec(word2vec, user_input)
# word2vec: The pretrained Word2Vec model
# user_input: A string of arbitrary length
# Returns: A 300-dimensional averaged Word2Vec embedding for that string
#
# This function preprocesses the input string, tokenizes it using get_tokens, extracts a word embedding for
# each token in the string, and averages across those embeddings to produce a
# single, averaged embedding for the entire input.
def string2vec(word2vec, user_input):
    embedding = np.zeros(300, )

    tokens = get_tokens(user_input)
    token_embedding = np.zeros((len(tokens), 300))

    for i, token in enumerate(tokens):
        token_embedding[i] = w2v(word2vec, token)

    embedding = np.mean(token_embedding, axis=0)

    return embedding


# Function: instantiate_models()
# This function does not take any input
# Returns: Four instantiated machine learning models
#
# This function instantiates the four imported machine learning models, and
# returns them for later downstream use.  You do not need to train the models
# in this function.
def instantiate_models():
    nb = None
    logistic = None
    svm = None
    mlp = None

    nb = GaussianNB()
    logistic = LogisticRegression(random_state=100)
    svm = LinearSVC(random_state=100)
    mlp = MLPClassifier(random_state=100)

    return nb, logistic, svm, mlp


# Function: train_model_tfidf(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# tfidf_train: A document-term matrix built from the training data
# training_labels: A list of integers (all 0 or 1)
# Returns: A trained version of the input model
#
# This function trains an input machine learning model using averaged Word2Vec
# embeddings for the training documents.
def train_model_tfidf(model, tfidf_train, training_labels):
    tfidf_train_dense = tfidf_train.toarray()
    model.fit(tfidf_train_dense, training_labels)

    return model


# Function: train_model_w2v(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# word2vec: A pretrained Word2Vec model
# training_data: A list of training documents
# training_labels: A list of integers (all 0 or 1)
# Returns: A trained version of the input model
#
# This function trains an input machine learning model using averaged Word2Vec
# embeddings for the training documents.
def train_model_w2v(model, word2vec, training_documents, training_labels):
    X_train = np.array([string2vec(word2vec, doc) for doc in training_documents])
    model.fit(X_train, training_labels)

    return model


# Function: test_model_tfidf(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# vectorizer: An initialized TfidfVectorizer model
# test_data: A list of test documents
# test_labels: A list of integers (all 0 or 1)
# Returns: Precision, recall, F1, and accuracy values for the test data
#
# This function tests an input machine learning model by extracting features
# for each preprocessed test document and then predicting an output label for
# that document.  It compares the predicted and actual test labels and returns
# precision, recall, f1, and accuracy scores.
def test_model_tfidf(model, vectorizer, test_documents, test_labels):
    precision = None
    recall = None
    f1 = None
    accuracy = None

    tfidf_test = vectorizer.transform(test_documents)

    tfidf_test_dense = tfidf_test.toarray()

    predicted_labels = model.predict(tfidf_test_dense)

    precision = precision_score(test_labels, predicted_labels)
    recall = recall_score(test_labels, predicted_labels)
    f1 = f1_score(test_labels, predicted_labels)
    accuracy = accuracy_score(test_labels, predicted_labels)

    return precision, recall, f1, accuracy


# Function: test_model_w2v(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# word2vec: A pretrained Word2Vec model
# test_data: A list of test documents
# test_labels: A list of integers (all 0 or 1)
# Returns: Precision, recall, F1, and accuracy values for the test data
#
# This function tests an input machine learning model by extracting features
# for each preprocessed test document and then predicting an output label for
# that document.  It compares the predicted and actual test labels and returns
# precision, recall, f1, and accuracy scores.
def test_model_w2v(model, word2vec, test_documents, test_labels):
    precision = None
    recall = None
    f1 = None
    accuracy = None

    test_vectors = [string2vec(word2vec, doc) for doc in test_documents]

    X_test = np.array(test_vectors)

    predicted_labels = model.predict(X_test)

    precision = precision_score(test_labels, predicted_labels)
    recall = recall_score(test_labels, predicted_labels)
    f1 = f1_score(test_labels, predicted_labels)
    accuracy = accuracy_score(test_labels, predicted_labels)

    return precision, recall, f1, accuracy

# Function: count_words(user_input)
# user_input: A string of arbitrary length
# Returns: An integer value
#
# This function counts the number of words in the input string.
def count_words(user_input):
    num_words = 0

    tokens = nltk.tokenize.word_tokenize(user_input)
    filtered_tokens = [token for token in tokens if token not in string.punctuation]
    num_words = len(filtered_tokens)

    return num_words

# Function: words_per_sentence(user_input)
# user_input: A string of arbitrary length
# Returns: A floating point value
#
# This function computes the average number of words per sentence
def words_per_sentence(user_input):
    wps = 0.0

    sentences = nltk.tokenize.sent_tokenize(user_input)
    num_words = count_words(user_input)

    if len(sentences) > 0:
        wps = num_words / len(sentences)

    return wps


# Function: get_pos_tags(user_input)
# user_input: A string of arbitrary length
# Returns: A list of (token, POS) tuples
#
# This function tags each token in the user_input with a Part of Speech (POS) tag from Penn Treebank.
def get_pos_tags(user_input):
    tokens = nltk.tokenize.word_tokenize(user_input)

    pos_tags = nltk.tag.pos_tag(tokens)

    tagged_input = [(word, pos) for word, pos in pos_tags]

    return tagged_input


# Function: get_pos_categories(tagged_input)
# tagged_input: A list of (token, POS) tuples
# Returns: Seven integers, corresponding to the number of pronouns, personal
#          pronouns, articles, past tense verbs, future tense verbs,
#          prepositions, and negations in the tagged input
#
# This function counts the number of tokens corresponding to each of six POS tag
# groups, and returns those values.  The Penn Treebag tags corresponding that
# belong to each category can be found in Table 2 of the project statement.
def get_pos_categories(tagged_input):
    num_pronouns = 0
    num_prp = 0
    num_articles = 0
    num_past = 0
    num_future = 0
    num_prep = 0

    for token, pos in tagged_input:
        if pos in ['PRP', 'PRP$', 'WP', 'WP$']:
            num_pronouns += 1
        if pos in ['PRP']:
            num_prp += 1
        if pos in ['DT']:
            num_articles += 1
        if pos in ['VBD', 'VBN']:
            num_past += 1
        if pos in ['MD']:
            num_future += 1
        if pos in ['IN']:
            num_prep += 1


    return num_pronouns, num_prp, num_articles, num_past, num_future, num_prep


# Function: count_negations(user_input)
# user_input: A string of arbitrary length
# Returns: An integer value
#
# This function counts the number of negation terms in a user input string
def count_negations(user_input):
    num_negations = 0

    negation_terms = ["no", "not", "never", "n't"]

    tokens = nltk.tokenize.word_tokenize(user_input)
    filtered_tokens = [token for token in tokens if token not in string.punctuation]

    for word in filtered_tokens:
        if word.lower() in negation_terms:
            num_negations += 1

    return num_negations


# Function: summarize_analysis(num_words, wps, num_pronouns, num_prp, num_articles, num_past, num_future, num_prep, num_negations)
# num_words: An integer value
# wps: A floating point value
# num_pronouns: An integer value
# num_prp: An integer value
# num_articles: An integer value
# num_past: An integer value
# num_future: An integer value
# num_prep: An integer value
# num_negations: An integer value
# Returns: A list of three strings
#
# This function identifies the three most informative linguistic features from
# among the input feature values, and returns the psychological correlates for
# those features.  num_words and/or wps should be included if, and only if,
# their values exceed predetermined thresholds.  The remainder of the three
# most informative features should be filled by the highest-frequency features
# from among num_pronouns, num_prp, num_articles, num_past, num_future,
# num_prep, and num_negations.
def summarize_analysis(num_words, wps, num_pronouns, num_prp, num_articles, num_past, num_future, num_prep, num_negations):
    informative_correlates = []

    # Creating a reference dictionary with keys = linguistic features, and values = psychological correlates.
    # informative_correlates should hold a subset of three values from this dictionary.
    psychological_correlates = {}
    psychological_correlates["num_words"] = "Talkativeness, verbal fluency"
    psychological_correlates["wps"] = "Verbal fluency, cognitive complexity"
    psychological_correlates["num_pronouns"] = "Informal, personal"
    psychological_correlates["num_prp"] = "Personal, social"
    psychological_correlates["num_articles"] = "Use of concrete nouns, interest in objects/things"
    psychological_correlates["num_past"] = "Focused on the past"
    psychological_correlates["num_future"] = "Future and goal-oriented"
    psychological_correlates["num_prep"] = "Education, concern with precision"
    psychological_correlates["num_negations"] = "Inhibition"

    # Set thresholds
    num_words_threshold = 100
    wps_threshold = 20

    feature_values = [("num_words", num_words), ("wps", wps), ("num_pronouns", num_pronouns),
                      ("num_prp", num_prp), ("num_articles", num_articles), ("num_past", num_past),
                      ("num_future", num_future), ("num_prep", num_prep), ("num_negations", num_negations)]

    feature_values.sort(key=lambda x: x[1], reverse=True)

    informative_count = 0

    for feature_name, feature_value in feature_values:
        if feature_name == "num_words" and feature_value > num_words_threshold:
            informative_correlates.append(psychological_correlates[feature_name])
            informative_count += 1
        elif feature_name == "wps" and feature_value > wps_threshold:
            informative_correlates.append(psychological_correlates[feature_name])
            informative_count += 1
        elif informative_count < 3 and feature_name != "num_words" and feature_name != "wps":
            informative_correlates.append(psychological_correlates[feature_name])
            informative_count += 1

        if informative_count >= 3:
            break

    return informative_correlates

# Function: welcome_state()
# This function does not take any input
# Returns: A string indicating the next state
#
# This function implements the chatbot's welcome states. In this state, the chatbot greets the user.
def welcome_state():
    # Display a welcome message to the user
    user_input = print("Welcome to this chatbot!")

    return "get_user_info"


# Function: get_info_state()
# This function does not take any input
# Returns: A string indicating the next state and a string indicating the
#          user's name
#
# This function implements a state that requests the user's name and then processes
# the user's response to extract that information.
def get_info_state():
    # Request the user's name, and accept a user response of
    # arbitrary length.
    user_input = input("What is your name?\n")

    # Extract the user's name
    name = extract_user_info(user_input)

    return "sentiment_analysis", name


# Function: sentiment_analysis_state(name, model, vectorizer, word2vec)
# name: A string indicating the user's name
# model: The trained classification model used for predicting sentiment
# vectorizer: OPTIONAL; The trained vectorizer, if using TFIDF (leave empty otherwise)
# word2vec: OPTIONAL; The pretrained Word2Vec model, if using Word2Vec (leave empty otherwise)
# Returns: A string indicating the next state
#
# This function implements a state that asks the user what they want to talk about,
# and then processes their response to predict their current sentiment.
def sentiment_analysis_state(name, model, vectorizer=None, word2vec=None):
    # Check the user's sentiment
    user_input = input("Thanks {0}!  What do you want to talk about today?\n".format(name))

    # Predict the user's sentiment
    # test = vectorizer.transform([preprocessing(user_input)])  # Use if you selected a TFIDF model
    test = string2vec(word2vec, user_input)  # Use if you selected a w2v model

    label = None
    label = model.predict(test.reshape(1, -1))

    if label == 0:
        print("Hmm, it seems like you're feeling a bit down.")
    elif label == 1:
        print("It sounds like you're in a positive mood!")
    else:
        print("Hmm, that's weird.  My classifier predicted a value of: {0}".format(label))

    return "stylistic_analysis"


# Function: stylistic_analysis_state()
# This function does not take any input
# Returns: A string indicating the next state
#
# This function implements a state that asks the user what's on their mind, and
# then analyzes their response to identify informative psycholinguistic correlates.
def stylistic_analysis_state():
    user_input = input("I'd also like to do a quick stylistic analysis. What's on your mind today?\n")
    num_words = count_words(user_input)
    wps = words_per_sentence(user_input)
    pos_tags = get_pos_tags(user_input)
    num_pronouns, num_prp, num_articles, num_past, num_future, num_prep = get_pos_categories(pos_tags)
    num_negations = count_negations(user_input)

    # Generate a stylistic analysis of the user's input
    informative_correlates = summarize_analysis(num_words, wps, num_pronouns,
                                                num_prp, num_articles, num_past,
                                                num_future, num_prep, num_negations)
    print(
        "Thanks!  Based on my stylistic analysis, I've identified the following psychological correlates in your response:")
    for correlate in informative_correlates:
        print("- {0}".format(correlate))

    return "check_next_action"


# Function: check_next_state()
# This function does not take any input
# Returns: A string indicating the next state
#
# This function implements a state that checks to see what the user would like
# to do next.  The user should be able to indicate that they would like to quit
# (in which case the state should be "quit"), redo the sentiment analysis
# ("sentiment_analysis"), or redo the stylistic analysis
# ("stylistic_analysis").
def check_next_state():
    next_state = ""

    user_input = input("What would you like to do next? You can do 3 things.\n"
                       "1. You can leave the chat.\n"
                       "2. You can redo the sentiment analysis\n"
                       "3. You can redo the stylistic analysis\n")

    # user_tokens = get_tokens(user_input)

    quit_regex = re.compile(r'(leave|exit|quit|1)', re.IGNORECASE)
    sa_regex = re.compile(r'(sentiment|2)', re.IGNORECASE)
    st_regex = re.compile(r'(stylistic|3)', re.IGNORECASE)

    while(True):
        if (re.search(quit_regex, user_input) is not None):
            next_state = "quit"
            break
        elif (re.search(sa_regex, user_input) is not None):
            next_state = "sentiment_analysis"
            break
        elif (re.search(st_regex, user_input) is not None):
            next_state = "stylistic_analysis"
            break
        else:
            user_input = input("I apologize, I'm unsure of what you would like to do.\n"
                               "Please specify if you would like to quit, redo sentiment analysis, or\n"
                               "redo stylistic analysis.\n")

    return next_state


# Function: run_chatbot(model, vectorizer=None):
# model: A trained classification model
# vectorizer: OPTIONAL; The trained vectorizer, if using Naive Bayes (leave empty otherwise)
# word2vec: OPTIONAL; The pretrained Word2Vec model, if using other classification options (leave empty otherwise)
# Returns: This function does not return any values
#
# This function implements the main chatbot system---it runs different
# dialogue states depending on rules governed by the internal dialogue
# management logic, with each state handling its own input/output and internal
# processing steps.  The dialogue management logic should be implemented as
# follows:
# welcome_state() (IN STATE) -> get_info_state() (OUT STATE)
# get_info_state() (IN STATE) -> sentiment_analysis_state() (OUT STATE)
# sentiment_analysis_state() (IN STATE) -> stylistic_analysis_state() (OUT STATE - First time sentiment_analysis_state() is run)
#                                    check_next_state() (OUT STATE - Subsequent times sentiment_analysis_state() is run)
# stylistic_analysis_state() (IN STATE) -> check_next_state() (OUT STATE)
# check_next_state() (IN STATE) -> sentiment_analysis_state() (OUT STATE option 1) or
#                                  stylistic_analysis_state() (OUT STATE option 2) or
#                                  terminate chatbot
def run_chatbot(model, vectorizer=None, word2vec=None):
    welcome_state()
    next, name = get_info_state()
    sentiment_analysis_state(name, model, word2vec=word2vec)
    stylistic_analysis_state()

    while True:
        next_state = check_next_state()
        if next_state == "quit":
            return
        elif next_state == "sentiment_analysis":
            sentiment_analysis_state(name, model, word2vec=word2vec)
        elif next_state == "stylistic_analysis":
            stylistic_analysis_state()


if __name__ == "__main__":
    # Train the TfidfVectorizer and Naive Bayes model
    documents, labels = load_as_list("dataset.csv")

    # Load the Word2Vec representations so that you can make use of it later
    word2vec = load_w2v(EMBEDDING_FILE)  # Use if you selected a Word2Vec model

    # Compute TFIDF representations so that you can make use of them later
    # vectorizer, tfidf_train = vectorize_train(documents)  # Use if you selected a TFIDF model

    # Instantiate and train the machine learning models
    # Only uncomment the models you are choosing to use.

    # nb_tfidf, logistic_tfidf, svm_tfidf, mlp_tfidf = instantiate_models() # Uncomment to instantiate a TFIDF model
    nb_w2v, logistic_w2v, svm_w2v, mlp_w2v = instantiate_models()  # Uncomment to instantiate a w2v model
    # nb_tfidf = train_model_tfidf(nb_tfidf, tfidf_train, labels)
    # nb_w2v = train_model_w2v(nb_w2v, word2vec, documents, labels)
    # logistic_tfidf = train_model_tfidf(logistic_tfidf, tfidf_train, labels)
    # logistic_w2v = train_model_w2v(logistic_w2v, word2vec, documents, labels)
    # svm_tfidf = train_model_tfidf(svm_tfidf, tfidf_train, labels)
    svm_w2v = train_model_w2v(svm_w2v, word2vec, documents, labels)
    # mlp_tfidf = train_model_tfidf(mlp_tfidf, tfidf_train, labels)
    # mlp_w2v = train_model_w2v(mlp_w2v, word2vec, documents, labels)

    run_chatbot(svm_w2v, word2vec=word2vec)
