import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
import string

class TextCleanerTransformer(BaseEstimator, TransformerMixin):
    '''
    Custom transformer que limpia el texto de caracteres especiales, quita stopwords y aplica lemmatizer
    '''
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.str.lower().str.strip()
        translator = str.maketrans('', '', string.punctuation+'’‘—“”–')
        X = X.map(lambda x: x.strip().lower().translate(translator))
        stopwords = nltk.corpus.stopwords.words('english')
        wordnet_lemmatizer = WordNetLemmatizer()
        X = X.map(word_tokenize).apply(lambda x: [word for word in x if word not in stopwords])
        X = X.apply(lambda x: [wordnet_lemmatizer.lemmatize(w, pos="v") for w in x])
        X = X.apply(lambda x: ' '.join(x)).to_numpy()
        return X