"""
Created on 21 Jun 2025
Last modification 28 Aug 2025 by AK

@author Alan Kowalczyk
"""


import os
import time
import multiprocessing
import re
import warnings
import spacy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import nltk

from nltk.corpus import sentiwordnet as swn

from joblib import Parallel, delayed
from nltk.corpus import stopwords
from sklearn.metrics import f1_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from afinn import Afinn
from flair.models import TextClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    pipeline, 
    AutoTokenizer, 
    AutoModelForSequenceClassification
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from sklearn.model_selection import StratifiedKFold
SKOPT_AVAILABLE = True


current_directory = os.path.dirname(os.path.realpath(__file__))
directory = os.path.join(current_directory, "FinancialPhraseBank")
num_cores = multiprocessing.cpu_count()  # Get the number of CPU cores on the current machine
warnings.filterwarnings("ignore")
today_date = dt.date.today()



def initialize_models():# last change 29 Jul
    """
    initialize_models - a function which is responsible for loading all sentiment analysis models
    input:
        none
    output:        
        models - a dictionary with sentiment analysis models
    """
    print("Loading sentiment analysis models...")
    
    models = {}
    
    # FinBERT model
    print("Loading FinBERT model...")
    finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels = 3)
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    models['finbert'] = pipeline("sentiment-analysis", model = finbert, tokenizer = tokenizer)

    # DistilBERT removed

    # Flair model
    print("Loading Flair model...")
    models['flair'] = TextClassifier.load('en-sentiment')

    # RoBERTa model
    print("Loading RoBERTa model...")
    roberta_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    roberta_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    models['roberta'] = pipeline("sentiment-analysis", model=roberta_model, tokenizer=roberta_tokenizer)

    # Cardiff NLP Twitter sentiment
    print("Loading Cardiff NLP model...")
    models['cardiff'] = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

    print("All models loaded successfully!")
    return models


def calculate_best_method(dfl):# last change 29 jul
    """
    calculate_best_method - function which compare methods of sentiment analysis
    input:
        dfl - a pandas DataFrame with labelled financial news headlines
    output:
        f1_scores - a dict - dictionary with method names and F1 scores
        timing_results - a dict - dictionary with method names and timing information
    """
    timing_results = {}
    
    # Run lexical-based methods
    lexical_timing = run_lexical_methods(dfl)
    timing_results.update(lexical_timing)
    
    # Run transformer-based methods  
    transformer_timing = run_transformer_methods(dfl)
    timing_results.update(transformer_timing)

    # Calculate F1 scores for all methods
    f1_scores = calculate_f1_scores(dfl)
    
    return f1_scores, timing_results


def run_lexical_methods(dfl):# last change 29 Jul
    """
    Run all lexical-based sentiment analysis methods
    input:
        dfl - dataframe - pandas DataFrame with financial news headlines
    output:
        timing_results - a dictionary - dict with timing information for lexical methods
    """
    # Clean text for lexical based methods
    dfl['Cleaned_text'] = dfl['Text'].apply(lambda x: clean_text(x))
    
    timing_results = {}

    # VADER Analysis
    print("\n1. VADER Analysis...")
    start_time = time.time()
    with multiprocessing.Pool(num_cores) as pool:
        dfl['Score_Vader'] = pool.map(vader_sentiment, dfl['Cleaned_text'])
    dfl['Label_Vader'] = [assign_sentiment_label(score, 0.05) for score in dfl['Score_Vader']]
    timing_results["VADER"] = time.time() - start_time
    print_time_taken(timing_results["VADER"], "VADER Sentiment Analysis")

    # TextBlob Analysis  
    print("\n2. TextBlob Analysis...")
    start_time = time.time()
    with multiprocessing.Pool(num_cores) as pool:
        dfl['Score_TB'] = pool.map(textblob_sentiment, dfl['Cleaned_text'])
    dfl['Label_TB'] = [assign_sentiment_label(score, 0.1) for score in dfl['Score_TB']]
    timing_results["TextBlob"] = time.time() - start_time
    print_time_taken(timing_results["TextBlob"], "TextBlob Sentiment Analysis")

    # AFINN Analysis
    print("\n3. AFINN Analysis...")
    start_time = time.time()
    with multiprocessing.Pool(num_cores) as pool:
        dfl['Score_AFINN'] = pool.map(afinn_sentiment, dfl['Cleaned_text'])
    dfl['Label_AFINN'] = [assign_sentiment_label(score, 1.0) for score in dfl['Score_AFINN']]
    timing_results["AFINN"] = time.time() - start_time
    print_time_taken(timing_results["AFINN"], "AFINN Sentiment Analysis")

    # SentiWordNet Analysis
    print("\n4. SentiWordNet Analysis...")
    start_time = time.time()
    with multiprocessing.Pool(num_cores) as pool:
        dfl['Score_SentiWordNet'] = pool.map(sentiwordnet_sentiment, dfl['Cleaned_text'])
    dfl['Label_SentiWordNet'] = [assign_sentiment_label(score, 0.1) for score in dfl['Score_SentiWordNet']]
    timing_results["SentiWordNet"] = time.time() - start_time
    print_time_taken(timing_results["SentiWordNet"], "SentiWordNet Sentiment Analysis")

    return timing_results


def run_transformer_methods(dfl):# last change 29 Jul
    """
    Run all transformer-based sentiment analysis methods
    input:
        dfl - pandas DataFrame with financial news headlines
    output:
        timing_results - dict with timing information for transformer methods
    """
    timing_results = {}

    # RoBERTa Analysis
    print("\n5. RoBERTa Analysis...")
    start_time = time.time()
    dfl['Label_RoBERTa'] = dfl['Text'].apply(roberta_sentiment)
    timing_results["RoBERTa"] = time.time() - start_time
    print_time_taken(timing_results["RoBERTa"], "RoBERTa Sentiment Analysis")

    # FinBERT Analysis
    print("\n6. FinBERT Analysis...")
    start_time = time.time()
    dfl['Label_FinBERT'] = dfl['Text'].apply(finbert_sentiment)
    timing_results["FinBERT"] = time.time() - start_time
    print_time_taken(timing_results["FinBERT"], "FinBERT Sentiment Analysis")

    # Cardiff NLP Twitter sentiment
    print("\n7. Cardiff NLP Twitter Analysis...")
    start_time = time.time()
    dfl['Label_Cardiff'] = dfl['Text'].apply(cardiffnlp_twitter_sentiment)
    timing_results["Cardiff NLP"] = time.time() - start_time
    print_time_taken(timing_results["Cardiff NLP"], "Cardiff NLP Twitter Analysis")


    
    return timing_results


def calculate_f1_scores(dfl):# last change 08 aug
    """
    Calculate F1 scores for all sentiment analysis methods
    input:
        dfl - pandas DataFrame with predictions
    output:
        f1_scores - dict with F1 scores for each method
    """
    f1_scores = {
        "VADER": f1_score(dfl['Label'], dfl['Label_Vader'], average="weighted"),
        "TextBlob": f1_score(dfl['Label'], dfl['Label_TB'], average="weighted"),
        "AFINN": f1_score(dfl['Label'], dfl['Label_AFINN'], average="weighted"),
        "RoBERTa": f1_score(dfl['Label'], dfl['Label_RoBERTa'], average="weighted"),
        "FinBERT": f1_score(dfl['Label'], dfl['Label_FinBERT'], average="weighted"),
        "Cardiff NLP": f1_score(dfl['Label'], dfl['Label_Cardiff'], average="weighted"),
    # DistilBERT removed
        "SentiWordNet": f1_score(dfl['Label'], dfl['Label_SentiWordNet'], average="weighted")
    }
    return f1_scores


def clean_text(text): # last change 26 jul
    """ 
    clean_text - function for pre-processing text for Lexical-based methods
    input:
        text - a string - text to be cleaned
    output:
        text - a string - cleaned text
    """
    # removing URLs
    text = re.sub(r"http\S+|www\S+|https\S+",
                  "",
                  text,
                  flags = re.MULTILINE)

    # removing HTML tags
    text = re.sub(r"<.*?>", "", text)

    # removing extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def format_time(seconds, detailed=False):# last change 29 jul
    """
    Format time for display
    input:
        seconds - a float - the number of seconds
        detailed - a boolean - if True, return detailed format, else short format
    output:
         - a string - formatted time string
    """
    if detailed:
        # Detailed format for individual task timing
        if seconds < 60:
            return f"{seconds:.2f} seconds"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes} minutes and {remaining_seconds:.2f} seconds"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            remaining_seconds = seconds % 60
            return f"{hours} hours, {minutes} minutes, and {remaining_seconds:.2f} seconds"
    else:
        # Short format for table display
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds:.2f}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            remaining_seconds = seconds % 60
            return f"{hours}h {minutes}m {remaining_seconds:.2f}s"


def print_time_taken(seconds, task):# last change 29 jul
    """
    Print time taken for a task
    input:
        seconds - a float - the number of seconds taken for the task
        task - a string - name of the task
    output:
        none
    """
    formatted_time = format_time(seconds, detailed=True)
    print(f"Time taken for {task}: {formatted_time}")


def assign_sentiment_label(score, threshold=0.1):# last change 29 jul
    """
    Convert numeric sentiment score to label
    input:
        score - a float - sentiment score
        threshold - a float - threshold for positive/negative classification
    output:
         - a string - sentiment label ("positive", "negative", or "neutral")
    """
    if score > threshold:
        return "positive"
    elif score < -threshold:
        return "negative"
    else:
        return "neutral"
    
    
def vader_sentiment(text):  # last change 26 jul
    """ 
    vader_sentiment - function to compute VADER sentiment score.
    input:
        text - a string - a cleaned text for which to compute the sentiment score
    output:
         - a float - the compound sentiment score from VADER
    """
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)['compound']


def textblob_sentiment(text):  # last change 26 jul
    """ 
    textblob_sentiment - function to compute TextBlob sentiment polarity.
    input:
        text - a string - a cleaned text for which to compute the sentiment polarity
    output:
        - a float - the polarity score from TextBlob
    """
    return TextBlob(text).sentiment.polarity


def sentiwordnet_sentiment(text):# last change 29 jul
    """ 
    sentiwordnet_sentiment - function to compute SentiWordNet sentiment score.
    input:
        text - a string - a cleaned text for which to compute the sentiment score
    output:
        - a float - the polarity score from SentiWordNet
    """     
    # Download required NLTK data if not already present
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    
    try:
        nltk.data.find('corpora/sentiwordnet')
    except LookupError:
        nltk.download('sentiwordnet', quiet=True)
    
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    
    total_pos_score = 0
    total_neg_score = 0
    count = 0
    
    for token in tokens:
        # Get synsets for the word
        synsets = wordnet.synsets(token)
        if synsets:
            # Take the first synset (most common meaning)
            synset = synsets[0]
            swn_synset = swn.senti_synset(synset.name())
            
            if swn_synset:
                total_pos_score += swn_synset.pos_score()
                total_neg_score += swn_synset.neg_score()
                count += 1
    
    if count == 0:
        return 0.0
    
    # Calculate overall sentiment score
    avg_pos = total_pos_score / count
    avg_neg = total_neg_score / count
    
    return avg_pos - avg_neg  # Positive if more positive, negative if more negative


def afinn_sentiment(text):# last change 29 jul
    """
    afinn_sentiment - function to compute AFINN sentiment score
    input:
        text - a string - cleaned text for sentiment analysis
    output:
         - a float - AFINN sentiment score
    """
    afinn = Afinn()
    return afinn.score(text)


def finbert_sentiment(text):# last change 08 aug
    """
    finbert_sentiment - a function to calculate FinBERT Sentiment Analysis Label
    input:
        text - a string - a raw text daata for which sentiment analisys is applied
    output:
        label - a string - label with senitment "negative", "neutral" or "positive"
    """
    result = finbert(text)
    label = result[0]['label'].lower()
    return label


def roberta_sentiment(text):# last change 08 aug
    """
    roberta_sentiment - function using RoBERTa for sentiment analysis
    input:
        text - a string - raw text for sentiment analysis
    output:
         - a string - sentiment label
    """
    result = roberta(text)
    label = result[0]['label'].lower()
    
    # Map RoBERTa labels to our format
    if 'positive' in label or label == 'pos':
        return "positive"
    elif 'negative' in label or label == 'neg':
        return "negative"
    else:
        return "neutral"


def cardiffnlp_twitter_sentiment(text):# last change 08 aug
    """
    Cardiff NLP Twitter sentiment model - a function to calculate sentiment using Cardiff NLP's Twitter model
    input:
        text - a string - raw text data for sentiment analysis
    output:
         - a string - sentiment label ("positive", "negative", or "neutral")
    """
    result = cardiff(text)
    label = result[0]['label'].lower()
    
    if 'positive' in label:
        return "positive"
    elif 'negative' in label:
        return "negative"
    else:
        return "neutral"


def create_unified_results_df(lexical_scores, lexical_timing, transformer_scores, transformer_timing, ml_results_df):#last change 08 aug
    """
    Create a unified results DataFrame combining all methods
    input:
        lexical_scores - dict - F1 scores for lexical methods
        lexical_timing - dict - timing for lexical methods
        transformer_scores - dict - F1 scores for transformer methods
        transformer_timing - dict - timing for transformer methods
        ml_results_df - DataFrame - supervised ML results
    output:
        unified_df - DataFrame - unified results for all methods
    """
    unified_results = []
    
    # Add lexical and transformer results
    all_scores = {**lexical_scores, **transformer_scores}
    all_timing = {**lexical_timing, **transformer_timing}
    
    for method_name, f1_score in all_scores.items():
        unified_results.append({
            'Method Type': 'NLP-based' if method_name in lexical_scores else 'Transformer-based',
            'Method Name': method_name,
            'F1 Score': f1_score,
            'Time [s]': all_timing.get(method_name, 0),
            'Accuracy': 'N/A',
            'Precision': 'N/A',
            'Recall': 'N/A'
        })
    
    # Add supervised ML results
    for _, row in ml_results_df.iterrows():
        unified_results.append({
            'Method Type': 'Supervised ML',
            'Method Name': row['Model name'],
            'F1 Score': row['F1 score'],
            'Time [s]': row['Time [s]'],
            'Accuracy': row['Accuracy'],
            'Precision': row['Precision'],
            'Recall': row['Recall']
        })
    
    unified_df = pd.DataFrame(unified_results)
    return unified_df


def print_unified_results_table(unified_df):# last change 08 aug
    """
    Print a comprehensive table with all methods comparison
    input:
        unified_df - DataFrame - unified results for all methods
    output:
        none
    """
    print(f"\n{'='*100}")
    print(f"{'UNIFIED RESULTS - ALL SENTIMENT ANALYSIS METHODS COMPARISON':^100}")
    print(f"{'='*100}")
    print(f"{'Method Type':<17} {'Method Name':<30} {'F1 Score':<10} {'Time':<12}")
    print(f"{'-'*100}")
    
    # Sort by F1 score (descending)
    sorted_df = unified_df.sort_values('F1 Score', ascending=False)
    
    for _, row in sorted_df.iterrows():
        method_type = row['Method Type']
        method_name = row['Method Name']
        f1_score = row['F1 Score']
        time_taken = format_time(row['Time [s]'])
        
        print(f"{method_type:<17} {method_name:<30} {f1_score:<10.4f} {time_taken:<12}")
    
    print(f"{'='*100}")


def save_results_to_latex(unified_df, filename='unified_results_table.tex'):# last change 08 aug
    """
    save_results_to_latex - save unified results in LaTeX table format
    input:
        unified_df - DataFrame - unified results for all methods
        filename - string - output LaTeX file name
    output:
        none
    """
    # Sort by F1 score (descending)
    sorted_df = unified_df.sort_values('F1 Score', ascending=False)
    
    latex_content = r"\begin{table}[ht!]" + "\n" + \
                    r"    \centering" + "\n" + \
                    r"    \begin{tabular}{|l|l|c|c|}" + "\n" + \
                    r"    \hline" + "\n" + \
                    r"    Method Type & Method Name & F1 Score & Time (s) \\ \hline" + "\n"

    for _, row in sorted_df.iterrows():
        method_type = row['Method Type'].replace('_', r'\_')
        method_name = row['Method Name'].replace('_', r'\_')
        f1_score = row['F1 Score']
        time_taken = row['Time [s]']
        
        latex_content += f"    {method_type} & {method_name} & {f1_score:.4f} & {time_taken:.2f} \\\\ \\hline\n"

    latex_content += r"    \end{tabular}" + "\n" + \
                    r"    \caption{Unified Sentiment Analysis Results: All Methods Performance Comparison}" + "\n" + \
                    r"    \label{tab:unified_sentiment_results}" + "\n" + \
                    r"\end{table}"

    # save to a new .tex file
    with open(filename, 'w') as f:
        f.write(latex_content)
        
    print(f"\nUnified LaTeX table saved to '{filename}'")


def full_clean_text(text): # last change 08 aug
    """ 
    full_clean_text - function for pre-processing text for classic ML methods
    input:
        text - a string - text to be cleaned
    output:
        - a string - fully cleaned text
    """
    # removing numbers
    text = re.sub(r"\d+", "", text)
    
    # lowercasing the text
    text = text.lower()

    # removing punctuation and special characters
    text = re.sub(r"[^\w\s]", "", text)

    # removing extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    # removing stopwords
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    filtered_text = " ".join(filtered_words)
    
    # lemmatisation
    doc = spacy_nlp(filtered_text)
    lemmatized_words = [token.lemma_ for token in doc]
    
    return " ".join(lemmatized_words)


def feature_extraction(df, vectorizer):# last change 08 aug
    """
    feature_extraction - function which implements the feature extraction
    input:
         df - a dataframe - dataframe with clean text data
         vectorizer - an object - an object for changing text data into numerical values
    output:
        dfe - a dataframe  - a dataframe with added extracted features
        matrix - matrix - the sparse matrix of features 
    """
    # fit and transform the text data to features
    matrix = vectorizer.fit_transform(df['Fully_cleaned_text'])

    # convert the matrix to a dataframe, using sparse matrix for big dataset
    df2 = pd.DataFrame.sparse.from_spmatrix(matrix, columns = vectorizer.get_feature_names_out())

    # concatenate the features with the original dataframe
    dfe = pd.concat([df, df2], axis = 1)

    return dfe, matrix


def supervised_ml(feature_sets): # last change 09 aug
    """
    supervised_ml - a function which creates a pipeline for ML models, and save results fo file, create confusion matrices
    input:
        feature_sets - a dictionary - dictionary containing matrices with features
    output:
        results_df - a dataframe - a dataframe containing results
    """
    warnings.filterwarnings("ignore")
    # dictionary of supervised ML classifiers
    classifiers = {
        "Support Vector Classifier": SVC(verbose = 0),
        "Random Forest Classifier": RandomForestClassifier(verbose = 0), 
        "MLP Classifier": MLPClassifier(verbose = 0),
        "SGD Classifier": SGDClassifier(verbose = 0),
        "Gradient Boosting Classifier": GradientBoostingClassifier(verbose = 0),    
        "XGBoost": XGBClassifier(verbosity = 0),
        "LightGBM": LGBMClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(),
        "GaussianNB": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "ExtraTrees": ExtraTreesClassifier(),
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(),
        "PassiveAggressive": PassiveAggressiveClassifier()
    }
    output_dir = f"predictions_{today_date}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = []

    # loop through each feature extraction method (BoW only)
    for feature_name, X in feature_sets.items():
        # split the data into training and testing sets (80% training, 20% testing), with random state 2 for development
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)
        
        # scale the features using StandardScaler
        scaler = StandardScaler(with_mean = False)  # use "with_mean = False" for sparse matrices
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        y_train = y_train.reset_index(drop = True)

        # Choose hyperparameter tuning strategy
        # If scikit-optimize is available, run Bayesian optimization (slower but usually better)

        print("Running Optuna (TPE + pruning) optimisation for each classifier (this may take significant time)...")
        best_classifiers = {}
        for class_name, classifier in classifiers.items():
            # Run Optuna-based Bayesian optimisation (no fallback)
            print(f"Optimising hyperparameters for {class_name}...")
            best = calculate_params_bayesian(classifier, class_name, X_train, y_train)
            best_classifiers[class_name] = best
    


        # changing sparse matrix to Compressed Sparse Column format, as multithreading is not working any other way  
        X_train = X_train.tocsc().copy()


        print(f"Starting training at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        # Train and evaluate each classifier with BoW feature set in parallel computation
        parallel_results = Parallel(n_jobs = num_cores)(
            delayed(classifier_model)(classifier, class_name, X_train, X_test, y_train, y_test)
            for class_name, classifier in best_classifiers.items()
        )


        for result, (class_name, classifier) in zip(parallel_results, best_classifiers.items()):
            result['Feature Set'] = feature_name
            results.append(result)
            
            #saving prediction to files for each classifier
            predictions_df = pd.DataFrame({
                    "True Labels": y_test,
                    "Predicted Labels": result['Predictions']
                })
            file_name_pred = f"{class_name}_{feature_name}_predictions.csv"
            file_path_pred = os.path.join(output_dir, file_name_pred)
            predictions_df.to_csv(file_path_pred, index = False)
            
            #saving confusion matrices to files
            plot_confusion_matrix(result['Confusion Matrix'], class_name, feature_name)
    results_df = pd.DataFrame(results) 
           
    return results_df


def calculate_params_bayesian(classifier, class_name, X_train, y_train, n_trials=80, timeout=None, optimize_metric='f1_weighted'):
    """
    Optuna-based hyperparameter optimisation (recommended over plain skopt for robustness).

    - Uses Optuna's TPE sampler with pruning (MedianPruner) to stop bad trials early.
    - Maximizes weighted F1 by default using stratified CV inside each trial.
    - Supports the expanded set of classifiers listed in supervised_ml.

    Parameters:
      classifier - prototype classifier instance (not modified)
      class_name - string name of the classifier (must match keys in supervised_ml)
      X_train, y_train - training data
      n_trials - number of Optuna trials to run (bigger -> better but slower)
      timeout - optional timeout in seconds for the whole study
      optimize_metric - 'f1_weighted' or 'accuracy'

    Returns: an instantiated classifier configured with best parameters (unfitted)
    """

    warnings.filterwarnings("ignore")

    # Ensure X is indexable for train/val splits
    try:
        Xc = X_train.tocsc().copy()
    except Exception:
        Xc = X_train

    # Define objective per classifier
    def objective(trial):
        try:
            # Param suggestions per classifier
            if class_name == "Support Vector Classifier":
                C = trial.suggest_loguniform('C', 1e-3, 1e2)
                kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
                max_iter = trial.suggest_int('max_iter', 100, 2000)
                model = SVC(C=C, kernel=kernel, max_iter=max_iter, random_state=42)

            elif class_name == "Random Forest Classifier":
                n_estimators = trial.suggest_int('n_estimators', 50, 800)
                max_depth = trial.suggest_int('max_depth', 3, 60)
                max_features = trial.suggest_float('max_features', 0.1, 1.0)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 50)
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                               max_features=max_features, min_samples_split=min_samples_split,
                                               min_samples_leaf=min_samples_leaf, random_state=42, n_jobs=1)

            elif class_name == "MLP Classifier":
                h1 = trial.suggest_int('hidden_layer_size1', 50, 512)
                n_hidden = trial.suggest_categorical('n_hidden_layers', [1,2,3])
                h2 = trial.suggest_int('hidden_layer_size2', 50, 512)
                alpha = trial.suggest_loguniform('alpha', 1e-7, 1e-1)
                lr = trial.suggest_loguniform('learning_rate_init', 1e-4, 1.0)
                max_iter = trial.suggest_int('max_iter', 200, 2000)
                learning_rate = trial.suggest_categorical('learning_rate', ['constant','invscaling','adaptive'])
                if n_hidden == 1:
                    hidden = (h1,)
                elif n_hidden == 2:
                    hidden = (h1, h2)
                else:
                    hidden = (h1, h2, int(h2/2))
                model = MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, learning_rate_init=lr,
                                     max_iter=max_iter, learning_rate=learning_rate, random_state=42)

            elif class_name == "Gradient Boosting Classifier":
                n_estimators = trial.suggest_int('n_estimators', 50, 800)
                learning_rate = trial.suggest_float('learning_rate', 0.01, 0.5)
                max_depth = trial.suggest_int('max_depth', 2, 20)
                subsample = trial.suggest_float('subsample', 0.3, 1.0)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 50)
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
                model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                                   max_depth=max_depth, subsample=subsample,
                                                   min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                                   random_state=42)

            elif class_name == "XGBoost":
                n_estimators = trial.suggest_int('n_estimators', 50, 800)
                learning_rate = trial.suggest_float('learning_rate', 0.01, 0.5)
                max_depth = trial.suggest_int('max_depth', 2, 20)
                subsample = trial.suggest_float('subsample', 0.5, 1.0)
                colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
                reg_alpha = trial.suggest_loguniform('reg_alpha', 1e-6, 10.0)
                reg_lambda = trial.suggest_loguniform('reg_lambda', 1e-6, 10.0)
                model = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                                      subsample=subsample, colsample_bytree=colsample_bytree,
                                      reg_alpha=reg_alpha, reg_lambda=reg_lambda, random_state=42, verbosity=0)

            elif class_name == "SGD Classifier":
                loss = trial.suggest_categorical('loss', ['hinge','log_loss','modified_huber','squared_hinge'])
                penalty = trial.suggest_categorical('penalty', ['l2','l1','elasticnet'])
                alpha = trial.suggest_loguniform('alpha', 1e-7, 1e-1)
                learning_rate = trial.suggest_categorical('learning_rate', ['constant','optimal','invscaling','adaptive'])
                eta0 = trial.suggest_float('eta0', 1e-4, 1.0)
                max_iter = trial.suggest_int('max_iter', 500, 5000)
                model = SGDClassifier(loss=loss, penalty=penalty, alpha=alpha, learning_rate=learning_rate,
                                      eta0=eta0, max_iter=max_iter, random_state=42)

            elif class_name == "LightGBM":
                n_estimators = trial.suggest_int('n_estimators', 50, 800)
                learning_rate = trial.suggest_float('learning_rate', 0.01, 0.5)
                num_leaves = trial.suggest_int('num_leaves', 8, 256)
                max_depth = trial.suggest_int('max_depth', -1, 50)
                subsample = trial.suggest_float('subsample', 0.5, 1.0)
                reg_alpha = trial.suggest_loguniform('reg_alpha', 1e-6, 10.0)
                reg_lambda = trial.suggest_loguniform('reg_lambda', 1e-6, 10.0)
                model = LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
                                       max_depth=max_depth, subsample=subsample, reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                                       random_state=42)

            elif class_name == "Logistic Regression":
                C = trial.suggest_loguniform('C', 1e-3, 1e2)
                penalty = trial.suggest_categorical('penalty', ['l2'])
                model = LogisticRegression(C=C, penalty=penalty, max_iter=1000, random_state=42)

            elif class_name == "KNN":
                n_neighbors = trial.suggest_int('n_neighbors', 1, 50)
                p = trial.suggest_categorical('p', [1,2])
                model = KNeighborsClassifier(n_neighbors=n_neighbors, p=p)

            elif class_name == "GaussianNB":
                var_smoothing = trial.suggest_loguniform('var_smoothing', 1e-12, 1e-6)
                model = GaussianNB(var_smoothing=var_smoothing)

            elif class_name == "Decision Tree":
                max_depth = trial.suggest_int('max_depth', 1, 50)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 50)
                min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
                model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split,
                                               min_samples_leaf=min_samples_leaf, random_state=42)

            elif class_name == "AdaBoost":
                n_estimators = trial.suggest_int('n_estimators', 50, 800)
                learning_rate = trial.suggest_float('learning_rate', 0.01, 2.0)
                model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)

            elif class_name == "ExtraTrees":
                n_estimators = trial.suggest_int('n_estimators', 50, 800)
                max_depth = trial.suggest_int('max_depth', 3, 60)
                model = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

            elif class_name == "LDA":
                solver = trial.suggest_categorical('solver', ['svd'])
                model = LinearDiscriminantAnalysis(solver=solver)

            elif class_name == "QDA":
                reg_param = trial.suggest_float('reg_param', 0.0, 1.0)
                model = QuadraticDiscriminantAnalysis(reg_param=reg_param)

            elif class_name == "PassiveAggressive":
                C = trial.suggest_float('C', 0.001, 10.0)
                max_iter = trial.suggest_int('max_iter', 100, 2000)
                model = PassiveAggressiveClassifier(C=C, max_iter=max_iter, random_state=42)

            else:
                raise ValueError(f"Unsupported classifier in Optuna tuner: {class_name}")

            # Stratified CV
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            for train_idx, val_idx in skf.split(Xc, y_train):
                X_tr = Xc[train_idx]
                y_tr = y_train.iloc[train_idx] if isinstance(y_train, pd.Series) else y_train[train_idx]
                X_val = Xc[val_idx]
                y_val = y_train.iloc[val_idx] if isinstance(y_train, pd.Series) else y_train[val_idx]

                # Convert to dense for models that require dense input
                try:
                    if hasattr(X_tr, 'toarray') and (class_name in ['GaussianNB', 'KNN', 'LDA', 'QDA', 'Logistic Regression']):
                        X_tr_use = X_tr.toarray()
                        X_val_use = X_val.toarray()
                    else:
                        X_tr_use = X_tr
                        X_val_use = X_val
                except Exception:
                    X_tr_use = X_tr
                    X_val_use = X_val

                model.fit(X_tr_use, y_tr)
                preds = model.predict(X_val_use)
                if optimize_metric == 'f1_weighted':
                    score = f1_score(y_val, preds, average='weighted')
                else:
                    score = accuracy_score(y_val, preds)
                scores.append(score)

            mean_score = float(np.mean(scores))
            # report to optuna as higher is better
            return mean_score

        except Exception as e:
            print(f"Optuna trial error for {class_name}: {e}")
            return 0.0

    # Create study
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)

    print(f"Starting Optuna study for {class_name} with {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best_params = study.best_params

    # Build final classifier with best_params
    if class_name == "Support Vector Classifier":
        final_clf = SVC(C=best_params.get('C'), kernel=best_params.get('kernel'), max_iter=best_params.get('max_iter'), random_state=42)
    elif class_name == "Random Forest Classifier":
        final_clf = RandomForestClassifier(n_estimators=best_params.get('n_estimators'), max_depth=best_params.get('max_depth'),
                                          max_features=best_params.get('max_features'), min_samples_split=best_params.get('min_samples_split'),
                                          min_samples_leaf=best_params.get('min_samples_leaf'), random_state=42, n_jobs=1)
    elif class_name == "MLP Classifier":
        n_hidden = best_params.get('n_hidden_layers')
        h1 = best_params.get('hidden_layer_size1')
        h2 = best_params.get('hidden_layer_size2')
        if n_hidden == 1:
            hidden = (h1,)
        elif n_hidden == 2:
            hidden = (h1, h2)
        else:
            hidden = (h1, h2, int(h2/2))
        final_clf = MLPClassifier(hidden_layer_sizes=hidden, alpha=best_params.get('alpha'), learning_rate_init=best_params.get('learning_rate_init'),
                                  max_iter=best_params.get('max_iter'), learning_rate=best_params.get('learning_rate'), random_state=42)
    elif class_name == "Gradient Boosting Classifier":
        final_clf = GradientBoostingClassifier(n_estimators=best_params.get('n_estimators'), learning_rate=best_params.get('learning_rate'),
                                               max_depth=best_params.get('max_depth'), subsample=best_params.get('subsample'),
                                               min_samples_split=best_params.get('min_samples_split'), min_samples_leaf=best_params.get('min_samples_leaf'), random_state=42)
    elif class_name == "XGBoost":
        final_clf = XGBClassifier(n_estimators=best_params.get('n_estimators'), learning_rate=best_params.get('learning_rate'),
                                  max_depth=best_params.get('max_depth'), subsample=best_params.get('subsample'), colsample_bytree=best_params.get('colsample_bytree'),
                                  reg_alpha=best_params.get('reg_alpha'), reg_lambda=best_params.get('reg_lambda'), random_state=42, verbosity=0)
    elif class_name == "SGD Classifier":
        final_clf = SGDClassifier(loss=best_params.get('loss'), penalty=best_params.get('penalty'), alpha=best_params.get('alpha'),
                                  learning_rate=best_params.get('learning_rate'), eta0=best_params.get('eta0'), max_iter=best_params.get('max_iter'), random_state=42)
    elif class_name == "LightGBM":
        final_clf = LGBMClassifier(n_estimators=best_params.get('n_estimators'), learning_rate=best_params.get('learning_rate'), num_leaves=best_params.get('num_leaves'),
                                   max_depth=best_params.get('max_depth'), subsample=best_params.get('subsample'), reg_alpha=best_params.get('reg_alpha'), reg_lambda=best_params.get('reg_lambda'), random_state=42)
    elif class_name == "Logistic Regression":
        final_clf = LogisticRegression(C=best_params.get('C'), penalty=best_params.get('penalty'), max_iter=1000, random_state=42)
    elif class_name == "KNN":
        final_clf = KNeighborsClassifier(n_neighbors=best_params.get('n_neighbors'), p=best_params.get('p'))
    elif class_name == "GaussianNB":
        final_clf = GaussianNB(var_smoothing=best_params.get('var_smoothing'))
    elif class_name == "Decision Tree":
        final_clf = DecisionTreeClassifier(max_depth=best_params.get('max_depth'), min_samples_split=best_params.get('min_samples_split'), min_samples_leaf=best_params.get('min_samples_leaf'), random_state=42)
    elif class_name == "AdaBoost":
        final_clf = AdaBoostClassifier(n_estimators=best_params.get('n_estimators'), learning_rate=best_params.get('learning_rate'), random_state=42)
    elif class_name == "ExtraTrees":
        final_clf = ExtraTreesClassifier(n_estimators=best_params.get('n_estimators'), max_depth=best_params.get('max_depth'), random_state=42)
    elif class_name == "LDA":
        final_clf = LinearDiscriminantAnalysis(solver=best_params.get('solver'))
    elif class_name == "QDA":
        final_clf = QuadraticDiscriminantAnalysis(reg_param=best_params.get('reg_param'))
    elif class_name == "PassiveAggressive":
        final_clf = PassiveAggressiveClassifier(C=best_params.get('C'), max_iter=best_params.get('max_iter'), random_state=42)
    else:
        raise ValueError(f"Unsupported classifier after Optuna tuning: {class_name}")

    print(f"Optuna best {optimize_metric} for {class_name}: {study.best_value:.4f}")
    print(f"Optuna best params for {class_name}: {best_params}")

    return final_clf


def classifier_model(classifier, class_name, X_train, X_test, y_train, y_test): # last change 08 aug
    """ 
    classifier_model - a function that trains all classic ML models and evaluates them
    input:
        classifier - an object - an actual classifier model to be trained
        class_name - a string - a classifier name
        X_train, y_train, X_test, y_test - dataframes - splited and scalled data for training and testing
    Output:
        metadata - a dictionary - predictions made by the models and evaluation of those predictions
    """
    warnings.filterwarnings("ignore")
    start_time = time.time()
    
    # train model
    classifier.fit(X_train, y_train)
    
    # make predictions
    X_test_copy = X_test.copy()
    predictions = classifier.predict(X_test_copy)
    end_time = time.time() - start_time
    
    # evaluate model
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average = "weighted")
    recall = recall_score(y_test, predictions, average = "weighted")
    f1 = f1_score(y_test, predictions, average = "weighted")
    conf_matrix = confusion_matrix(y_test, predictions)
    
    true_labels = y_test.tolist() if isinstance(y_test, pd.Series) else list(y_test)
     
    metadata = {"Model name": class_name,
                "Model": classifier,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 score": f1,
                "Confusion Matrix": conf_matrix,
                "Time [s]": end_time,
                "Predictions": predictions,
                "Label": true_labels}
    
    return metadata
   

def plot_confusion_matrix(conf_matrix, class_name, feature_name):#last change 08 aug
    """
    plot_confusion_matrix - a function which is plotting confusion matrix in graphical form, and adding them to a latex file
    input:
        conf_matrix - a ndarray - numpy multidimensional array containing confusion matrix
        class_name - a string - a classifier name
        feature_name - a string - name of the feature
    output:
        none
    """

    output_dir = f"confusion_matrices_{today_date}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  

    # Plot using Seaborn
    plt.figure(figsize = (8, 6))
    sns.heatmap(conf_matrix,
                annot = True,
                fmt = "d",
                cmap = "Blues",
                cbar = False)
    
    # Add labels and title
    plt.title(f"Confusion Matrix - {class_name} - {feature_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    
    # Save the plot
    output_file = os.path.join(output_dir, f"{class_name}_{feature_name}_confusion_matrix.png")
    plt.savefig(output_file)
    plt.close()

    tex_file = f"confusion_matrices_{today_date}.tex"
    # Add LaTeX to the end of tex file 
    with open(tex_file, 'a') as tex:
        tex.write(r"\begin{figure}[ht!]" + "\n")
        tex.write(f"    \\centering" + "\n")
        tex.write(f"    \\includegraphics[width=0.8\\textwidth]{{{output_file}}}" + "\n")
        tex.write(f"    \\caption{{Confusion Matrix - {class_name} - {feature_name}}}" + "\n")
        tex.write(r"\end{figure}" + "\n")


def create_nlp_confusion_matrices(dfl, method_scores):# last change 08 aug
    """
    Create confusion matrices for NLP and Transformer methods
    input:
        dfl - DataFrame with true and predicted labels
        method_scores - dict with method names and F1 scores  
    output:
        none
    """
    # Check what format the true labels are in
    print(f"True label column 'Label' contains: {dfl['Label'].unique()[:10]}")
    print(f"Label column type: {type(dfl['Label'].iloc[0])}")
    
    # Handle both numeric (0,1,2) and string labels
    if dfl['Label'].dtype in ['int64', 'int32', 'float64'] or isinstance(dfl['Label'].iloc[0], (int, float)):
        # Labels are numeric, convert them to strings
        label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
        y_true_original = dfl['Label'].map(label_mapping)
        print("Converted numeric labels to strings")
    else:
        # Labels are already strings
        y_true_original = dfl['Label']
        print("Using string labels as-is")
    
    # Map method names to actual column names in the dataframe
    method_column_mapping = {
        "VADER": "Label_Vader",
        "TextBlob": "Label_TB", 
        "AFINN": "Label_AFINN",
        "SentiWordNet": "Label_SentiWordNet",
        "RoBERTa": "Label_RoBERTa",
        "FinBERT": "Label_FinBERT",
        "Cardiff NLP": "Label_Cardiff"
    }
    
    for method_name in method_scores.keys():
        if method_name in method_column_mapping:
            column_name = method_column_mapping[method_name]
            print(f"\nChecking {method_name} -> {column_name}")
            
            if column_name in dfl.columns:
                print(f"  Column {column_name} found in dataframe")
                y_pred = dfl[column_name]
                print(f"  Prediction values: {y_pred.unique()[:10]}")  # Show first 10 unique values
                print(f"  Total predictions: {len(y_pred)}")
                print(f"  Non-null predictions: {y_pred.notna().sum()}")
                
                # Filter out any NaN or invalid predictions
                mask = (y_pred.notna()) & (y_true_original.notna()) & \
                       (y_pred.isin(['negative', 'neutral', 'positive'])) & \
                       (y_true_original.isin(['negative', 'neutral', 'positive']))
                
                print(f"  Valid predictions after filtering: {mask.sum()}")
                
                if mask.sum() > 0:  # Only create confusion matrix if we have valid predictions
                    y_true_filtered = y_true_original[mask]
                    y_pred_filtered = y_pred[mask]
                    
                    try:
                        conf_matrix = confusion_matrix(y_true_filtered, y_pred_filtered, 
                                                     labels=['negative', 'neutral', 'positive'])
                        plot_confusion_matrix(conf_matrix, method_name, "NLP")
                        print(f"  ✅ Created confusion matrix for {method_name}")
                    except Exception as e:
                        print(f"  ❌ Error creating confusion matrix for {method_name}: {str(e)}")
                else:
                    print(f"  ⚠️ No valid predictions found for {method_name}")
                    # Debug: show what values were actually found
                    print(f"     Unique pred values: {y_pred.unique()}")
                    print(f"     Unique true values: {y_true_original.unique()}")
            else:
                print(f"  ❌ Column {column_name} NOT found in dataframe")
                print(f"     Available columns: {[col for col in dfl.columns if 'Label' in col]}")
        else:
            print(f"❌ Method {method_name} not in mapping")


def create_performance_scatter_plot(unified_df):# last change 09 aug
    """
    Create scatter plot of F1 Score vs Time with method type color coding
    input:
        unified_df - DataFrame with unified results
    output:
        none
    """
    plt.figure(figsize=(12, 8))
    
    # Define colors for each method type
    colors = {
        'NLP-based': 'red',
        'Transformer-based': 'blue', 
        'Supervised ML': 'green'
    }
    
    # Plot each method type
    for method_type, color in colors.items():
        mask = unified_df['Method Type'] == method_type
        if mask.any():
            plt.scatter(unified_df[mask]['Time [s]'], unified_df[mask]['F1 Score'], 
                       c=color, label=method_type, alpha=0.7, s=100)
            
            # Add method name labels
            for _, row in unified_df[mask].iterrows():
                plt.annotate(row['Method Name'], 
                           (row['Time [s]'], row['F1 Score']),
                           xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, alpha=0.8)
    
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12) 
    plt.title('Sentiment Analysis Methods: F1 Score vs Processing Time', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    output_file = f"performance_scatter_plot_{today_date}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Performance scatter plot saved as '{output_file}'")



dataset_name = "ankurzing/sentiment-analysis-for-financial-news"
# licence Attribution-NonCommercial-ShareAlike 4.0 International
# credit to:
# Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). Good debt or bad debt: 
# Detecting semantic orientations in economic texts. Journal of the Association for Information 
# Science and Technology, 65(4), 782-796.


# Read Sentences_AllAgree.csv robustly: handle both header and headerless formats
dfl = pd.read_csv("Sentences_AllAgree.csv",
                  header = None,
                  names = ['Label', 'Text'],
                  encoding = 'ISO-8859-1') # dfl - data frame with labelled news headlines


# Initialize all models
models = initialize_models()

# Make models globally accessible for the sentiment functions
finbert = models['finbert']
flair = models['flair']
roberta = models['roberta']
cardiff = models['cardiff']

# Run the comparison for lexical and transformer methods
results = calculate_best_method(dfl)
all_scores, timing_results = results

# Separate lexical and transformer scores/timing for unified results
lexical_methods = ["VADER", "TextBlob", "AFINN", "SentiWordNet"]
transformer_methods = ["RoBERTa", "FinBERT", "Cardiff NLP"]
transformer_methods = ["RoBERTa", "FinBERT", "Cardiff NLP"]

lexical_scores = {k: v for k, v in all_scores.items() if k in lexical_methods}
lexical_timing = {k: v for k, v in timing_results.items() if k in lexical_methods}
transformer_scores = {k: v for k, v in all_scores.items() if k in transformer_methods}
transformer_timing = {k: v for k, v in timing_results.items() if k in transformer_methods}

print("=== RUNNING ALL SENTIMENT ANALYSIS METHODS ===")

# Clean text for both lexical methods and ML
dfl['Cleaned_text'] = dfl['Text'].apply(lambda x: clean_text(x))


# Download required NLTK data for stopwords if not present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words("english"))
spacy_nlp = spacy.load("en_core_web_sm")

# Using all cores of CPU for parallelized full cleaning 
with multiprocessing.Pool(num_cores) as pool:
    dfl['Fully_cleaned_text'] = pool.map(full_clean_text, dfl['Cleaned_text'])

# Check if cleaning resulted in empty text
empty_text_count = (dfl['Fully_cleaned_text'].str.strip() == '').sum()
if empty_text_count > 0:
    valid_text_indices = dfl['Fully_cleaned_text'].str.strip() != ''
    dfl = dfl[valid_text_indices].reset_index(drop=True)

# Convert labels as XGBoost have requirements of [0,1,2] 
# The original dataset has string labels: 'negative', 'neutral', 'positive'
y = dfl['Label'].map({'negative': 0, 'neutral': 1, 'positive': 2})

# Remove any NaN values if they exist
if y.isna().sum() > 0:
    print(f"WARNING: Found {y.isna().sum()} NaN values in labels. Removing them...")
    valid_indices = ~y.isna()
    y = y[valid_indices]
    dfl = dfl[valid_indices].reset_index(drop=True)
    print(f"After cleaning - dataset size: {len(dfl)}")

# initialize vectorizer
tfidf_vectorizer = TfidfVectorizer()

df_tfidf, tfidf_matrix = feature_extraction(dfl, tfidf_vectorizer)

# Features and labels
feature_sets = {"TF-IDF": tfidf_matrix}


results_df = supervised_ml(feature_sets)

# Save ML results (but don't print separate table)
results_df['Time [s]'] = results_df['Time [s]'].round(2)
results_df.to_csv(f"ml_results_{today_date}.csv", index=False)

# Create unified results with all methods
print("\n" + "="*60)
print("CREATING FINAL UNIFIED RESULTS")
print("="*60)

unified_df = create_unified_results_df(lexical_scores, lexical_timing, transformer_scores, transformer_timing, results_df)

# Sort by F1 Score (descending)
unified_df = unified_df.sort_values('F1 Score', ascending=False)

print("\nFinal Results Table (sorted by F1 Score):")
print_unified_results_table(unified_df)

# Save results with rounded precision
unified_df_rounded = unified_df.copy()
unified_df_rounded['F1 Score'] = unified_df_rounded['F1 Score'].round(4)
unified_df_rounded['Time [s]'] = unified_df_rounded['Time [s]'].round(4)
unified_df_rounded.to_csv(f"unified_sentiment_results_{today_date}.csv", index=False)
save_results_to_latex(unified_df)

# Create visualizations
print("\nGenerating performance scatter plot...")
create_performance_scatter_plot(unified_df)


