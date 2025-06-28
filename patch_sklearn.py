# patch_sklearn.py
import sklearn.compose._column_transformer

# Apply permanent fix for _RemainderColsList
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    setattr(sklearn.compose._column_transformer, '_RemainderColsList', list)
