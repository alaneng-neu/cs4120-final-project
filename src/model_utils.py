from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os


def evaluate_model(y_true, y_pred, dataset_name=""):
    """Calculate and print evaluation metrics for binary classification."""
    print(f"{dataset_name} Metrics:")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred):.4f}")
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }


def evaluate_multiclass_model(y_true, y_pred, dataset_name="", average='weighted'):
    """
    Calculate and print evaluation metrics for multi-class classification.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    dataset_name : str
        Name of the dataset for display
    average : str
        Averaging method for multi-class metrics: 'micro', 'macro', 'weighted'
        - 'micro': Calculate metrics globally
        - 'macro': Calculate metrics for each class and take unweighted mean
        - 'weighted': Calculate metrics for each class and take weighted mean by support
    
    Returns:
    --------
    dict : Dictionary containing all metrics
    """
    print(f"{dataset_name} Metrics (average='{average}'):")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average=average, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, average=average, zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred, average=average, zero_division=0):.4f}")
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }


def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix", figsize=(14, 10)):
    """
    Plot a confusion matrix heatmap.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    class_names : list
        List of class names for axis labels
    title : str
        Title for the plot
    figsize : tuple
        Figure size (width, height)
    """
    plt.figure(figsize=figsize)
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def get_top_features_multiclass(vectorizer, classifier, n=10):
    """
    Get top features (words) for each class in a multi-class classifier.
    Works with Naive Bayes classifiers that have feature_log_prob_ attribute.
    
    Parameters:
    -----------
    vectorizer : fitted vectorizer object
        The vectorizer used to transform text
    classifier : fitted classifier object
        Must have feature_log_prob_ attribute (e.g., Naive Bayes)
    n : int
        Number of top features to return per class
    
    Returns:
    --------
    dict : Dictionary mapping class names to list of (feature, log_prob) tuples
    """
    feature_names = vectorizer.get_feature_names_out()
    log_probs = classifier.feature_log_prob_
    
    top_features = {}
    for i, class_name in enumerate(classifier.classes_):
        indices = log_probs[i].argsort()[-n:][::-1]
        top_features[class_name] = [(feature_names[j], log_probs[i][j]) for j in indices]
    
    return top_features


def get_top_features_linear(vectorizer, classifier, n=10):
    """
    Get top features (words) for each class in a linear classifier.
    Works with Logistic Regression and LinearSVC that have coef_ attribute.
    
    Parameters:
    -----------
    vectorizer : fitted vectorizer object
        The vectorizer used to transform text
    classifier : fitted classifier object
        Must have coef_ attribute (e.g., LogisticRegression, LinearSVC)
    n : int
        Number of top features to return per class
    
    Returns:
    --------
    dict : Dictionary mapping class names to list of (feature, weight) tuples
    """
    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_
    
    top_features = {}
    for i, class_name in enumerate(classifier.classes_):
        # Get indices of top n positive coefficients (most indicative of this class)
        indices = coefficients[i].argsort()[-n:][::-1]
        top_features[class_name] = [(feature_names[j], coefficients[i][j]) for j in indices]
    
    return top_features


def display_top_features(top_features, n=10, metric_name="weight"):
    """
    Display top features for each class.
    
    Parameters:
    -----------
    top_features : dict
        Output from get_top_features_multiclass() or get_top_features_linear()
    n : int
        Number of features to display per class
    metric_name : str
        Name of the metric to display (e.g., 'log prob', 'weight')
    """
    for class_name, features in top_features.items():
        print(f"Top {n} Features for '{class_name}':")
        for i, (word, value) in enumerate(features[:n], 1):
            print(f"{i:2d}. {word:25s} ({metric_name}: {value:.4f})")
        print()


def predict_phishing_type(text, vectorizer, classifier):
    """
    Predict the phishing type of a new email.
    
    Parameters:
    -----------
    text : str
        Email text to classify
    vectorizer : fitted vectorizer object
        The vectorizer used to transform text
    classifier : fitted classifier object
        The trained classifier
    
    Returns:
    --------
    tuple : (prediction, confidence, top_3_predictions)
        - prediction: the predicted class label
        - confidence: probability of the predicted class
        - top_3: list of (class_name, probability) tuples for top 3 predictions
    """
    text_vec = vectorizer.transform([text])
    prediction = classifier.predict(text_vec)[0]
    probabilities = classifier.predict_proba(text_vec)[0]
    
    # Get confidence for the predicted class
    class_idx = list(classifier.classes_).index(prediction)
    confidence = probabilities[class_idx]
    
    # Get top 3 predictions
    top_3_idx = probabilities.argsort()[-3:][::-1]
    top_3 = [(classifier.classes_[i], probabilities[i]) for i in top_3_idx]
    
    return prediction, confidence, top_3


def test_phishing_examples(examples, vectorizer, classifier):
    """
    Test the classifier on a list of example emails.
    
    Parameters:
    -----------
    examples : list
        List of email text strings to classify
    vectorizer : fitted vectorizer object
        The vectorizer used to transform text
    classifier : fitted classifier object
        The trained classifier
    """
    print("Phishing Type Predictions:")
    for i, example in enumerate(examples, 1):
        prediction, confidence, top_3 = predict_phishing_type(example, vectorizer, classifier)
        print(f"\nExample {i}:")
        print(f"Text: {example[:100]}..." if len(example) > 100 else f"Text: {example}")
        print(f"Prediction: {prediction} (Confidence: {confidence:.2%})")
        print(f"Top 3 predictions:")
        for class_name, prob in top_3:
            print(f"  - {class_name}: {prob:.2%}")


def save_model(model, vectorizer, model_name, models_dir='../models'):
    """
    Save a trained model and its vectorizer.
    
    Parameters:
    -----------
    model : fitted classifier object
        The trained model to save
    vectorizer : fitted vectorizer object
        The vectorizer to save
    model_name : str
        Base name for the model (e.g., 'phishing_nb', 'phishing_lr', 'phishing_svm')
    models_dir : str
        Directory to save models to
    
    Returns:
    --------
    tuple : (model_path, vectorizer_path)
    """
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, f'{model_name}_model.joblib')
    vectorizer_path = os.path.join(models_dir, f'{model_name}_vectorizer.joblib')
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")
    
    return model_path, vectorizer_path


def print_model_summary(model_name, df, y, X_train, X_test, train_accuracy, test_accuracy, models_dir='../models'):
    """
    Print a summary of the trained model.
    
    Parameters:
    -----------
    model_name : str
        Name of the model (e.g., 'Naive Bayes', 'Logistic Regression', 'SVM')
    df : pd.DataFrame
        The full dataset
    y : pd.Series
        Target labels
    X_train : array-like
        Training features
    X_test : array-like
        Test features
    train_accuracy : float
        Training accuracy
    test_accuracy : float
        Test accuracy
    models_dir : str
        Directory where models are saved
    """
    print(f"PHISHING TYPE CLASSIFICATION - {model_name.upper()} SUMMARY\n")
    print(f"Total samples: {len(df)}")
    print(f"Number of classes: {y.nunique()}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
