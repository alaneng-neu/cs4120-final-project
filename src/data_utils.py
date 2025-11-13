import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def load_and_preprocess_data(filepath, text_columns=['Subject', 'Message'], 
                             target_column='Spam/Ham', combine_text=True,
                             drop_na=True, positive_class='spam', 
                             file_format='csv', verbose=True):
    """
    General function to load and preprocess email/text data for spam detection.
    
    Parameters:
    -----------
    filepath : str
        Path to the data file
    text_columns : list
        List of column names containing text data to use as features
    target_column : str
        Name of the column containing labels (spam/ham)
    combine_text : bool
        Whether to combine multiple text columns into one
    drop_na : bool
        Whether to drop rows with missing values
    positive_class : str, int, or list
        Value(s) in target_column that represent spam/positive class
        Examples: 'spam', 1, ['spam', 'phishing']
    file_format : str
        Format of the file: 'csv', 'tsv', 'excel'
    verbose : bool
        Whether to print detailed information
    
    Returns:
    --------
    X : pd.Series or pd.DataFrame
        Feature data (text)
    y : pd.Series
        Target labels (binary: 1 for spam, 0 for ham)
    df : pd.DataFrame
        Full dataframe for additional analysis
    """
    # Load data based on format
    if verbose:
        print(f"Loading data from {filepath}...")
    
    if file_format == 'csv':
        df = pd.read_csv(filepath)
    elif file_format == 'tsv':
        df = pd.read_csv(filepath, sep='\t')
    elif file_format == 'excel':
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    
    if verbose:
        print(f"Dataset shape: {df.shape}")
        print(f"\nColumn names: {df.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(df.head())
    
    # Handle missing values
    if drop_na:
        initial_shape = df.shape[0]
        df = df.dropna(subset=text_columns + [target_column])
        if verbose and df.shape[0] < initial_shape:
            print(f"\nDropped {initial_shape - df.shape[0]} rows with NaN values")
            print(f"Shape after dropping NaN: {df.shape}")
    
    # Combine text columns if specified
    if combine_text and len(text_columns) > 1:
        # Fill NaN with empty string before combining
        df['combined_text'] = df[text_columns].fillna('').agg(' '.join, axis=1)
        X = df['combined_text']
        if verbose:
            print(f"\nCombined text columns: {text_columns}")
    else:
        X = df[text_columns[0]]
        if verbose:
            print(f"\nUsing single text column: {text_columns[0]}")
    
    # Convert target to binary (1 for spam/positive class, 0 for ham/negative class)
    y = convert_target_to_binary(df[target_column], positive_class)
    
    # Print class distribution
    if verbose:
        print(f"\nClass distribution:")
        print(f"Negative/Ham (0): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.2f}%)")
        print(f"Positive/Spam (1): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.2f}%)")
    
    return X, y, df


def convert_target_to_binary(target_series, positive_class='spam'):
    """
    Convert target labels to binary format (0 and 1).
    
    Parameters:
    -----------
    target_series : pd.Series
        Series containing target labels
    positive_class : str, int, bool, or list
        Value(s) that represent the positive class (spam)
        Can be:
        - String: 'spam', 'phishing', etc. (case-insensitive)
        - Integer: 1
        - Boolean: True
        - List: ['spam', 'phishing'] for multiple positive classes
    
    Returns:
    --------
    pd.Series
        Binary labels (1 for positive class, 0 for negative class)
    """
    # If already binary (0/1), check if it needs conversion
    if target_series.dtype in ['int64', 'int32'] and set(target_series.unique()).issubset({0, 1}):
        return target_series.astype(int)
    
    # If boolean, convert directly
    if target_series.dtype == 'bool':
        return target_series.astype(int)
    
    # Handle string targets (case-insensitive)
    if isinstance(positive_class, str):
        # Case-insensitive string comparison
        if target_series.dtype == 'object':
            return (target_series.str.lower() == positive_class.lower()).astype(int)
        else:
            return (target_series == positive_class).astype(int)
    
    # Handle list of positive classes
    elif isinstance(positive_class, list):
        if target_series.dtype == 'object':
            # Case-insensitive for strings
            return target_series.str.lower().isin([str(p).lower() for p in positive_class]).astype(int)
        else:
            return target_series.isin(positive_class).astype(int)
    
    # Handle numeric or boolean positive_class
    else:
        return (target_series == positive_class).astype(int)


def vectorize_text(X_train, X_test, method='tfidf', max_features=5000, 
                  verbose=True, **kwargs):
    """
    Vectorize text data using TF-IDF or Count Vectorizer.
    
    Parameters:
    -----------
    X_train : pd.Series or array-like
        Training text data
    X_test : pd.Series or array-like
        Test text data
    method : str
        'tfidf' or 'count'
    max_features : int
        Maximum number of features to extract
    verbose : bool
        Whether to print information
    **kwargs : additional arguments to pass to vectorizer
        Common options:
        - min_df: ignore terms with document frequency below threshold
        - max_df: ignore terms with document frequency above threshold
        - ngram_range: tuple (min_n, max_n) for n-gram range
        - stop_words: 'english' or list of stop words
    
    Returns:
    --------
    X_train_vec : sparse matrix
        Vectorized training data
    X_test_vec : sparse matrix
        Vectorized test data
    vectorizer : fitted vectorizer object
    """
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=max_features, **kwargs)
    elif method == 'count':
        vectorizer = CountVectorizer(max_features=max_features, **kwargs)
    else:
        raise ValueError("method must be 'tfidf' or 'count'")
    
    if verbose:
        print(f"Vectorizing text using {method.upper()}...")
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    if verbose:
        print(f"Training set shape: {X_train_vec.shape}")
        print(f"Test set shape: {X_test_vec.shape}")
        print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    
    return X_train_vec, X_test_vec, vectorizer


def get_dataset_info(filepath, target_column=None, file_format='csv'):
    """
    Quick utility to inspect dataset structure without full loading.
    Useful for figuring out column names and target values.
    
    Parameters:
    -----------
    filepath : str
        Path to the data file
    target_column : str, optional
        If provided, show unique values in this column
    file_format : str
        Format of the file: 'csv', 'tsv', 'excel'
    
    Returns:
    --------
    None (prints information)
    """
    if file_format == 'csv':
        df = pd.read_csv(filepath, nrows=100)  # Only read first 100 rows
    elif file_format == 'tsv':
        df = pd.read_csv(filepath, sep='\t', nrows=100)
    elif file_format == 'excel':
        df = pd.read_excel(filepath, nrows=100)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    
    print(f"Columns: {df.columns.tolist()}")
    print(f"Shape (first 100 rows): {df.shape}")
    print(f"Data types:")
    print(df.dtypes)
    print(f"First 3 rows:")
    print(df.head(3))
    
    if target_column and target_column in df.columns:
        print(f"Unique values in '{target_column}':")
        print(df[target_column].value_counts())
        print(f"Data type: {df[target_column].dtype}")
