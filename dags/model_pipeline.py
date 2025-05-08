import pandas as pd
import numpy as np
import re
import html
import emoji
import xgboost as xgb
import pickle

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

from scipy.sparse import hstack
from pythainlp import word_tokenize
from pythainlp.corpus.common import thai_stopwords
from typing import Collection, Callable


prefix_path = "/opt/airflow/dags/models/"

mlb_path = prefix_path + "mlb.pkl"
enc_path = prefix_path + "enc.pkl"
tfidf_path = prefix_path + "tfidf_vectorizer.pkl"
selector_path = prefix_path + "selector.pkl"
model_path = prefix_path + "today.pkl"


# === Custom Preprocessing Transformers ===

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb = joblib.load(mlb_path)
        self.enc = joblib.load(enc_path)
        self.list_area = ['คลองสาน',
                          'สาทร',
                          'ตลิ่งชัน',
                          'ลาดพร้าว',
                          'จอมทอง',
                          'ห้วยขวาง',
                          'บึงกุ่ม',
                          'สะพานสูง',
                          'บางเขน',
                          'ดอนเมือง',
                          'ราษฎร์บูรณะ',
                          'บางพลัด',
                          'พระนคร',
                          'บางบอน',
                          'บางรัก',
                          'บางกอกใหญ่',
                          'วังทองหลาง',
                          'ธนบุรี',
                          'ราชเทวี',
                          'สวนหลวง',
                          'หนองแขม',
                          'ยานนาวา',
                          'ทวีวัฒนา',
                          'ทุ่งครุ',
                          'ดุสิต',
                          'คันนายาว',
                          'บางซื่อ',
                          'พระโขนง',
                          'ป้อมปราบศัตรูพ่าย',
                          'ประเวศ',
                          'บางกอกน้อย',
                          'ภาษีเจริญ',
                          'หนองจอก',
                          'ลาดกระบัง',
                          'หลักสี่',
                          'บางกะปิ',
                          'สัมพันธวงศ์',
                          'มีนบุรี',
                          'คลองสามวา',
                          'บางแค',
                          'จตุจักร',
                          'คลองเตย',
                          'บางคอแหลม',
                          'สายไหม',
                          'บางนา',
                          'ดินแดง',
                          'พญาไท',
                          'ปทุมวัน',
                          'บางขุนเทียน',
                          'วัฒนา']

    def split_types(self, types):
        if types is None:
            return []
        types = types.strip()
        return types.split(',')

    def match_area(self, list_type):
        for e in list_type:
            for j in self.list_area:
                if j in e:
                    return j
        return 'อื่นๆ'

    def fit(self, X, y=None):
        df = X.copy()

        df['type'] = df['type'].apply(self.split_types)
        df = df[df['organization'].apply(lambda x: any(re.match('เขต', s) for s in x.split(',')))]
        df['organization'] = df['organization'].apply(lambda x: self.match_area(x.split(',')))
        self.mlb.fit(df['type'])
        self.enc.fit(df[['organization']])
        return self

    def transform(self, X):
        df = X.copy()

        df['type'] = df['type'].apply(self.split_types)
        df['organization'] = df['organization'].apply(lambda x: self.match_area(x))

        type_encoded = self.mlb.transform(df['type'])
        type_encoded_df = pd.DataFrame(type_encoded, columns=self.mlb.classes_, index=df.index)

        org_encoded = self.enc.transform(df[['organization']])
        org_encoded_df = pd.DataFrame(org_encoded.toarray(), columns=self.enc.get_feature_names_out(), index=df.index)

        # df['last_activity'] = pd.to_datetime(df['last_activity'].apply(lambda x: x.split('+')[0].split('.')[0]))
        # df['timestamp'] = pd.to_datetime(df['timestamp'].apply(lambda x: x.split('+')[0].split('.')[0]))
        df["time_taken"] = ((df['last_activity'] - df['timestamp']) / np.timedelta64(1, 'h')).apply(int)

        df = pd.concat([df, type_encoded_df, org_encoded_df], axis=1)
        df.drop(['type', 'organization', 'timestamp', 'last_activity'], axis=1, inplace=True)
        return df


# === Text Cleaner ===

def process_transformers(text: str):
    def fix_html(text):
        re1 = re.compile(r"  +")
        text = (
            text.replace("#39;", "'")
            .replace("amp;", "&")
            .replace("#146;", "'")
            .replace("nbsp;", " ")
            .replace("#36;", "$")
            .replace("\\n", "\n")
            .replace("quot;", "'")
            .replace("<br />", "\n")
            .replace('\\"', '"')
            .replace(" @.@ ", ".")
            .replace(" @-@ ", "-")
            .replace(" @,@ ", ",")
            .replace("\\", " \\ ")
        )
        return re1.sub(" ", html.unescape(text))

    def replace_url(text):
        URL_PATTERN = r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
        return re.sub(URL_PATTERN, "<url>", text)

    def rm_brackets(text):
        new_line = re.sub(r"\(\)", "", text)
        new_line = re.sub(r"\{\}", "", new_line)
        new_line = re.sub(r"\[\]", "", new_line)
        new_line = re.sub(r"\([^a-zA-Z0-9ก-๙]+\)", "", new_line)
        new_line = re.sub(r"\{[^a-zA-Z0-9ก-๙]+\}", "", new_line)
        new_line = re.sub(r"\[[^a-zA-Z0-9ก-๙]+\]", "", new_line)
        new_line = re.sub(r"(?<=\()[^a-zA-Z0-9ก-๙]+(?=[a-zA-Z0-9ก-๙])", "", new_line)
        new_line = re.sub(r"(?<=\{)[^a-zA-Z0-9ก-๙]+(?=[a-zA-Z0-9ก-๙])", "", new_line)
        new_line = re.sub(r"(?<=\[)[^a-zA-Z0-9ก-๙]+(?=[a-zA-Z0-9ก-๙])", "", new_line)
        new_line = re.sub(r"(?<=[a-zA-Z0-9ก-๙])[^a-zA-Z0-9ก-๙]+(?=\))", "", new_line)
        new_line = re.sub(r"(?<=[a-zA-Z0-9ก-๙])[^a-zA-Z0-9ก-๙]+(?=\})", "", new_line)
        new_line = re.sub(r"(?<=[a-zA-Z0-9ก-๙])[^a-zA-Z0-9ก-๙]+(?=\])", "", new_line)
        return new_line

    def rm_useless_spaces(text):
        return re.sub(" {2,}", " ", text)

    def replace_rep_after(text):
        def _replace_rep(m):
            c, cc = m.groups()
            return f"{c}"
        re_rep = re.compile(r"(\S)(\1{3,})")
        return re_rep.sub(_replace_rep, text)

    def replace_stop_words(text):
        words = word_tokenize(text, keep_whitespace=False)
        words = [word for word in words if word not in thai_stopwords()]
        return ' '.join(words)

    def ungroup_emoji(toks):
        res = []
        for tok in toks:
            if emoji.emoji_count(tok) == len(tok):
                res.extend(list(tok))
            else:
                res.append(tok)
        return res

    def replace_wrep_post(toks):
        previous_word = None
        rep_count = 0
        res = []
        for current_word in toks + ["</s>"]:
            if current_word == previous_word:
                rep_count += 1
            elif (current_word != previous_word) and (rep_count > 0):
                res += [previous_word]
                rep_count = 0
            else:
                res.append(previous_word)
            previous_word = current_word
        return res[1:]

    def remove_space(toks):
        return [t.strip() for t in toks if t.strip()]

    # Run all
    text = text.lower()
    for f in [fix_html, replace_url, rm_brackets, rm_useless_spaces, replace_rep_after, replace_stop_words]:
        text = f(text)
    toks = word_tokenize(text)
    for f in [ungroup_emoji, replace_wrep_post, remove_space]:
        toks = f(toks)
    return "".join(toks)


# === Model Pipeline ===

# Load pre-trained model, vectorizer, and selector
# with open("tfidf.pkl", "rb") as f:
#     tfidf = pickle.load(f)

# with open("selector.pkl", "rb") as f:
#     selector = pickle.load(f)

# with open("model.pkl", "rb") as f:
#     model = pickle.load(f)

tfidf_load = joblib.load(tfidf_path)
selector_load = joblib.load(selector_path)
model_load = joblib.load(model_path)

def final_pipeline_fn(df):
    preprocessor = CustomPreprocessor()
    df = preprocessor.transform(df)
    df['comment'] = df['comment'].apply(process_transformers)
    X_text = tfidf_load.transform(df['comment'])

    X_other = df.drop(columns=['comment', 'time_taken'])
    X_final = hstack([X_text, X_other])

    X_final = selector_load.transform(X_final).astype('float32')
    y_pred = model_load.predict(X_final)
    return np.expm1(y_pred).astype(int)

