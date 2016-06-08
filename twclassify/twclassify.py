# -*- coding: utf-8 -*-
from collections import Counter, defaultdict
from IPython.display import display, HTML
from itertools import product
import json
import numpy as np
import pandas as pd
import pickle
import re
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.naive_bayes import MultinomialNB
import string
from tabulate import tabulate


punc_re = '[' + re.escape(string.punctuation) + ']'

class TextVectorizer:

    def __init__(self, feature_fields, target_field, lc=True,
                 merge_field_features=False, rt_prefix=True, ignore_rt=False,
                 min_df=2, max_df=1., ngram_range=(1,1), use_idf=True, norm='l1',
                 collapse_hashtags=False, collapse_mentions=True, collapse_urls=True,
                 limit_repeats=True, retain_punc_toks=True, collapse_digits=False):
        self.feature_fields = feature_fields
        self.target_field = target_field
        self.merge_field_features = merge_field_features
        self.lc = lc
        self.rt_prefix = rt_prefix
        self.ignore_rt = ignore_rt
        self.min_df = min_df
        self.max_df = max_df
        self.ngram_range = ngram_range
        self.use_idf = use_idf
        self.norm = norm
        self.collapse_hashtags = collapse_hashtags
        self.collapse_mentions = collapse_mentions
        self.collapse_urls = collapse_urls
        self.limit_repeats = limit_repeats
        self.retain_punc_toks = retain_punc_toks
        self.collapse_digits = collapse_digits

    def tokenize(self, features):
        tokens = []
        for k, v in features.items():
            toks = []
            if isinstance(v, list):
                for vi in v:
                    toks.extend(self._tokenize(vi))
            else:
                toks = self._tokenize(v)
            if not self.merge_field_features:
                tokens.extend(k + '___' + t for t in toks)
            else:
                tokens.extend(toks)
        return ' '.join(tokens)

    def _tokenize(self, text):
        text = '' if not text else text
        text = text.lower() if self.lc else text
        if self.collapse_hashtags:
            text = re.sub('#\S+', 'THIS_IS_A_HASHTAG', text)
        else:
            text = re.sub('#(\S+)', r'HASHTAG_\1', text)
        if self.collapse_mentions:
            text = re.sub('@\S+', 'THIS_IS_A_MENTION', text)
        if self.collapse_urls:
            text = re.sub('http\S+', 'THIS_IS_A_URL', text)
        if self.limit_repeats:
            text = re.sub(r'(.)\1\1\1+', r'\1', text)
        if self.collapse_digits:
            text = re.sub(r'[0-9]+', '9', text)
        toks = []
        for tok in text.split():
            tok = re.sub(r'^(' + punc_re + '+)', r'\1 ', tok)
            tok = re.sub(r'(' + punc_re + '+)$', r' \1', tok)
            for subtok in tok.split():
                if self.retain_punc_toks or re.search('\w', subtok):
                    toks.append(subtok)
        if self.rt_prefix:
            rt_text = 'rt' if self.lc else 'RT'
            if rt_text in toks:
                toks.remove(rt_text)
                toks = ['RT_' + t for t in toks]
        return toks

    def extract_feature(self, dict_obj, field):
        """ Walk a dict to extract features.
        >>> extract_features({'a': [{'b': 1}, {'b': 2}], 'c': 3}, ['a.b', 'c'])
        {'a.b': [1, 2], 'c': 3}
        """
        parts = field.split('.')
        for pi, p in enumerate(parts):
            dict_obj = dict_obj.get(p)
            if dict_obj is None:
                break
            if isinstance(dict_obj, list):
                return [self.extract_feature(d, '.'.join(parts[pi+1:])) for d in dict_obj]
        return dict_obj

    def extract_features(self, dict_obj):
        features = {}
        for field in self.feature_fields:
            features[field] = self.extract_feature(dict_obj, field)
        return features

    def vectorize(self, data, fit=True):
        def iter_data(data):
            for d in data:
                yield (self.tokenize(self.extract_features(d)), self.extract_feature(d, self.target_field))

        labels = []
        if fit:
            self.vec = TfidfVectorizer(token_pattern='\S+', min_df=self.min_df, max_df = self.max_df,
                                       ngram_range=self.ngram_range, use_idf=self.use_idf, norm=self.norm)
            X = self.vec.fit_transform(toks for toks, label in iter_data(data) if not labels.append(label))
            self.features = np.array(self.vec.get_feature_names())
        else:
            X = self.vec.transform(toks for toks, label in iter_data(data) if not labels.append(label))
        return X, np.array(labels)

    def __repr__(self):
        attrs = vars(self)
        return ', '.join("%s: %s" % item for item in attrs.items())


class TextClassifier:
    def __init__(self, vectorizer, clf=LogisticRegression()):
        self.clf = clf
        self.vectorizer = vectorizer

    def cv(self, data, X, labels, n_folds=5, random_state=42, verbose=True):
        cv = KFold(len(labels), n_folds, random_state=random_state)
        truths = []
        preds = []
        for train, test in cv:
            self.clf.fit(X[train], labels[train])
            preds.extend(self.clf.predict(X[test]))
            truths.extend(labels[test])
        binary_truths = self.to_binary(truths)
        binary_preds = self.to_binary(preds)
        results = \
            {'accuracy': accuracy_score(truths, preds),
             'f1_pos': f1_score(binary_truths, binary_preds),
             'macro_f1': f1_score(truths, preds, average='macro', pos_label=None),
             'micro_f1': f1_score(truths, preds, average='micro', pos_label=None),
             'recall': recall_score(binary_truths, binary_preds),
             'precision': precision_score(binary_truths, binary_preds),
             'roc_auc': roc_auc_score(binary_truths, binary_preds)
             }
        if verbose:
            print(classification_report(truths, preds))
            print(self.confusion(truths, preds, self.clf.classes_))
            self.clf.fit(X, labels)
            self.top_terms()
            print('\n')
            self.top_error_terms(truths, preds, X, data)
        return results

    def fit(self, X, labels):
        self.clf.fit(X, labels)

    def predict(self, X):
        return self.clf.predict(X)

    def to_binary(self, labels):
        return np.array([1 if t.lower() == 'yes' or t.lower() == 'positive' or t.lower() == '1' else 0 for t in labels])

    def confusion(self, truths, preds, labels):
        m = confusion_matrix(truths, preds)
        m = np.vstack((labels, m))
        m = np.hstack((np.matrix([''] + list(labels)).T, m))
        return tabulate(m.tolist(), headers='firstrow')

    def top_terms(self, n=15):
        print('\n\nTOP FEATURES:')
        if len(self.clf.classes_) == 2:
            coefs = [-self.clf.coef_[0], self.clf.coef_[0]]
        else:
            coefs = self.clf.coef_
        for li, label in enumerate(self.clf.classes_):
            print('\nCLASS %s' % label)
            coef = coefs[li]
            top_coef_ind = np.argsort(coef)[::-1][:n]
            top_coef_terms = self.vectorizer.features[top_coef_ind]
            top_coef = coef[top_coef_ind]
            print('\n'.join(['%s\t%.3f' % (term, weight) for term, weight in zip(top_coef_terms, top_coef)]))

    def top_error_terms(self, truths, preds, X, data):
        print('\n\nERROR ANALYSIS:\n')
        for label in self.clf.classes_:
            print('\nincorrectly labeled %s' % label)
            iserror = np.zeros(len(truths))
            ind = [i for i, (t, p) in enumerate(zip(truths, preds)) if t != p and p == label]
            iserror[ind] = 1
            F, pval = f_classif(X, iserror)
            for fidx in np.argsort(F)[::-1][:5]:
                print('\n\t%s %d' % (self.vectorizer.features[fidx], X[ind, fidx].nnz))
                matches = []
                for midx in range(X.shape[0]):
                    if X[midx, fidx] > 0 and iserror[midx] == 1:
                        matches.append(midx)
                for m in matches[:3]:
                    print('\t\t' + str(self.vectorizer.extract_features(data[m])))

    def __repr__(self):
        return str(self.vectorizer) + '\n' + str(self.clf)


class ParameterSweeper:
    def __init__(self, n_folds=5, random_state=42):
        self.random_state = random_state
        self.n_folds = n_folds

    def sweep(self, data, vectorizer_parms, classifier_parms):
        results = []
        for vec_parms in self.iter_parms(vectorizer_parms):
            tv = TextVectorizer(**vec_parms)
            X, labels = tv.vectorize(data)
            # print(vec_parms)
            for clf_constructor, clf_parms_i in classifier_parms.items():
                #print('\t%s' % clf_constructor.__name__)
                for clf_parms in self.iter_parms(clf_parms_i):
                    # print('\t\t%s' % str(clf_parms))
                    clf = clf_constructor(**clf_parms)
                    tc = TextClassifier(tv, clf)
                    result = tc.cv(data, X, labels, n_folds=self.n_folds,
                                   random_state=self.random_state, verbose=False)
                    results.append((result, vec_parms, clf_parms, clf_constructor))
            print('best macro_f1 so far:')
            display(self.print_top_parameters('macro_f1', results, n=1, transpose=False))
        return results

    def iter_parms(self, parms):
        keys = list(parms.keys())
        option_iter = product(*list(parms.values()))
        for options in option_iter:
            opts = dict(zip(keys, options))
            yield opts

    def score_parameters(self, results):
        evaluation_metrics = list(results[0][0].keys())
        vec = DictVectorizer()
        lr = LinearRegression()
        for metric in evaluation_metrics:
            y = [r[0][metric] for r in results]
            dicts = []
            for result in results:
                d = {}
                clf_name = result[3].__name__
                for parm, val in result[1].items():
                    d['%s__vec_%s=%s' % (clf_name, parm, str(val))] = 1.0
                for parm, val in result[2].items():
                    d['%s__clf_%s=%s' % (clf_name, parm, str(val))] = 1.0
                d['clf=%s' % clf_name] = 1.0
                dicts.append(d)
            X = vec.fit_transform(dicts)
            lr.fit(X, y)
            print('\n\nRegression results for %s' % metric)
            self._print_parameter_scores(np.array(vec.get_feature_names()), lr.coef_)

    def make_classifier_from_parms(self, vec_parms, clf_parms, clf_constructor):
        tv = TextVectorizer(**vec_parms)
        clf = clf_constructor(**clf_parms)
        return TextClassifier(tv, clf)

    def get_best_classifier(self, metric, results):
        top_parms = self.top_parameters(metric, results, n=1)[0]
        return self.make_classifier_from_parms(top_parms[1], top_parms[2], top_parms[3])

    def top_parameters(self, metric, results, n=5):
        ret = []
        for result in sorted(results, key=lambda x: x[0][metric], reverse=True)[:n]:
            ret.append(result)
        return ret

    def print_top_parameters(self, metric, results, n=5, transpose=True):
        top_parms = self.top_parameters(metric, results, n)
        vec_names = set()
        clf_names = set()
        dicts = []
        for t in top_parms:
            d = dict(t[1])
            d.update(t[2])
            d['classifier'] = t[3].__name__
            d.update(t[0])
            dicts.append(d)
        pd.set_option('display.max_columns', None)
        df = pd.DataFrame(dicts)
        if transpose:
            df = df.transpose()
        return HTML(self.bold_diffs(df).to_html(escape=False))

    def bold_diffs(self, df):
        for r, row in enumerate(df.values):
            prev = None
            for c, col in enumerate(row):
                if prev and col != prev:
                    df.values[r, c] = '<b>%s</b>' % str(col)
                else:
                    df.values[r, c] = str(col)
                prev = col
        return df

    def _print_parameter_scores(self, features, coef):
        for i in np.argsort(coef)[::-1]:
            print('%.4f\t%s' % (coef[i], features[i]))
