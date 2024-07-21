import spacy
import os
import json
import pandas as pd
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
nlp = spacy.load("en_core_web_sm")
CITIES_LIST = ['charlotte', 'edinburgh', 'lasvegas', 'london', 'phoenix', 'pittsburgh', 'singapore']
print(CITIES_LIST)


def mkdir(idir):
    if not os.path.isdir(idir):
        os.makedirs(idir)


def get_noun_phrases(doc, output=None, keep=None):
    if keep is None:
        return list([np.text.lower() for np in doc.noun_chunks])
    if output is None:
        output = {}
    kws = []
    for nc in doc.noun_chunks:
        ws = []
        for word in nc:
            if (word.pos_ in keep) and (len(word) > 2):
                ws.append(word.text.lower())
        if len(ws) > 0:
            n = ' '.join(ws)
            output[n] = output.get(n, 0) + 1
            kws.append(n)
    return output, kws


def increase_count(idict, key, freq):
    if key not in idict:
        idict[key] = 0
    idict[key] += freq


def add_to_dict(idict, key, value, freq=1):
    if key not in idict:
        idict[key] = {}
    if value not in idict[key]:
        idict[key][value] = 0
    idict[key][value] += freq


def get_unique_values(idict, count_only=False):
    if count_only:
        return {k: len(set(v)) for k, v in idict.items()}
    else:
        return {k: list(set(v)) for k, v in idict.items()}


def save_np_info(np2count, np2reviews, np2rest, np2users, ofile, count_only=False):
    output = {"np2count": np2count, "np2reviews": np2reviews,
            'np2rests': np2rest, 'np2users': np2users}
    json.dump(output, open(ofile, 'w'))
    print("Saved to", ofile)


def extract_raw_keywords_for_reviews(data, ofile, keep=['ADJ', 'NOUN', 'PROPN', 'VERB'],
                                     overwrite=False, review2keyword_ofile=None):
    if os.path.isfile(ofile) and not overwrite:
        print("Existing output file. Stop! (set overwrite=True to overwrite)")
        return
    np2count = {}   # frequency
    np2review2count = {}  # reviews
    np2rest2count = {}  #
    np2user2count = {}
    review2keywords = {}
    for rid, uid, restid, text in tqdm(zip(data['review_id'],
        data['user_id'], data['rest_id'], data['text']), total=len(data)):
        doc = nlp(text)
        tmp, keywords = get_noun_phrases(doc, keep=keep)  # np for this review
        for np, freq in tmp.items():
            increase_count(np2count, np, freq)
            add_to_dict(np2review2count, np, rid, freq)
            add_to_dict(np2rest2count, np, restid, freq)
            add_to_dict(np2user2count, np, uid, freq)
        review2keywords[rid] = keywords
    save_np_info(np2count, np2review2count, np2rest2count, np2user2count, ofile)
    if review2keyword_ofile is not None:
        df = pd.DataFrame({"Review_ID": list(review2keywords.keys()), "Keywords": list(review2keywords.values())})
        df.to_csv(review2keyword_ofile)


def load_split(sfile='./data/reviews/splits.json', city='singapore', setname='test'):
    return json.load(open(sfile))[city][setname]


def filter_keywords(ifile, ofile, min_freq=3):
    data = json.load(open(ifile))
    np2count = data['np2count']
    valid_kws = [a for a, b in np2count.items() if b >= min_freq]
    new_dict = {}
    for k, v in data.items():
        tmp = {}
        for k2 in valid_kws:
            tmp[k2] = v[k2]
        new_dict[k] = tmp
    json.dump(new_dict, open(ofile, 'w'))
    print("Saved to", ofile)


def group_keywords_for_users(ifile, ofile):
    dt = json.load(open(ifile))
    np2users = dt['np2users']
    u2kw = {}  # {user: {keyword: freq}}
    for kw, u2c in np2users.items():
        for u, c in u2c.items():
            if u not in u2kw:
                u2kw[u] = {}
            u2kw[u][kw] = c
    json.dump(u2kw, open(ofile, 'w'))
    print("Saved to", ofile)


def group_keywords_for_rests(ifile, ofile):
    dt = json.load(open(ifile))
    np2rests = dt['np2rests']
    u2kw = {}  # {rest: {keyword: freq}}
    for kw, u2c in np2rests.items():
        for u, c in u2c.items():
            if u not in u2kw:
                u2kw[u] = {}
            u2kw[u][kw] = c
    json.dump(u2kw, open(ofile, 'w'))
    print("Saved to", ofile)


def compute_tfirf(ifile, ofile, irf, default_irf=0.01, sorting=True):
    dt = json.load(open(ifile))
    u2kw2score = {}
    for u, kw2f in dt.items():
        kw2score = {}
        for kw, f in kw2f.items():
            kw2score[kw] = f * irf.get(kw, default_irf)
        u2kw2score[u] = kw2score
    # sort
    if sorting:
        tmp = {}
        for k, v in u2kw2score.items():
            vs = sorted(v.items(), key=lambda x: x[1], reverse=True)
            tmp[k] = vs
        u2kw2score = tmp
    json.dump(u2kw2score, open(ofile, 'w'))
    print("Saved to", ofile)


def get_irf(city, irf_dict, irf_dir):
    if city not in irf_dict:
        irf = json.load(open(os.path.join(irf_dir, city)))
        irf_dict[city] = irf
    return irf_dict[city]


def compute_irf(num, N=1000):
    return np.log(N / num)


def compute_irf_for_dir(idir, odir, N=1000):
    for fname in os.listdir(idir):
        # print(fname)
        if fname.startswith(".") or not fname.endswith(".json"):
            continue
        # print(fname)
        ifile = os.path.join(idir, fname)
        # print(ifile)
        ofile = os.path.join(odir, fname)
        dt = json.load(open(ifile))['np2rests']
        np2irf = {}
        for n, r in dt.items():
            np2irf[n] = compute_irf(len(r), N=N)
        json.dump(np2irf, open(ofile, 'w'))
        print("Saved to", ofile)


def compute_iUf_for_dir(idir, odir, N=1000):
    for fname in os.listdir(idir):
        # print(fname)
        if fname.startswith(".") or not fname.endswith(".json"):
            continue
        # print(fname)
        ifile = os.path.join(idir, fname)
        # print(ifile)
        ofile = os.path.join(odir, fname)
        dt = json.load(open(ifile))['np2users']
        np2iuf = {}
        for n, r in dt.items():
            np2iuf[n] = compute_irf(len(r), N=N)
        json.dump(np2iuf, open(ofile, 'w'))
        print("Saved to", ofile)


def compute_tfiuf(ifile, ofile, irf, default_irf=0.01, sorting=True):
    dt = json.load(open(ifile))
    u2kw2score = {}
    for u, kw2f in dt.items():
        kw2score = {}
        for kw, f in kw2f.items():
            kw2score[kw] = f * irf.get(kw, default_irf)
        u2kw2score[u] = kw2score
    # sort
    if sorting:
        tmp = {}
        for k, v in u2kw2score.items():
            vs = sorted(v.items(), key=lambda x: x[1], reverse=True)
            tmp[k] = vs
        u2kw2score = tmp
    json.dump(u2kw2score, open(ofile, 'w'))
    print("Saved to", ofile)


CITIES = ['edinburgh']
sets = ['train', 'test']
for city in CITIES:
    dt = pd.read_csv('./data/reviews/{}.csv'.format(city))
    for setname in sets:
        print("Processing for", city, setname)
        uids = load_split(city=city, setname=setname)
        dt_set = dt[dt['user_id'].isin(uids)]
        odir = './data/preprocessed/by_city-users_min_3_reviews/keywords_spacy/' + setname
        mkdir(odir)
        extract_raw_keywords_for_reviews(dt_set, ofile=os.path.join(odir, city + '-keywords.json'), keep=['ADJ', 'NOUN', 'PROPN', 'VERB'],
                                        overwrite=True, review2keyword_ofile=os.path.join(odir,city+"-review2keywords.csv"))


CITIES = ['edinburgh']
min_freq = 3
for city in CITIES:
    print("Processing for", city)
    ifile = './data/preprocessed/by_city-users_min_3_reviews/keywords_spacy/train/{}-keywords.json'.format(city)
    ofile = './data/preprocessed/by_city-users_min_3_reviews/keywords_spacy-min_{}/train/{}-keywords.json'.format(min_freq, city)
    mkdir('./data/preprocessed/by_city-users_min_3_reviews/keywords_spacy-min_{}/train/'.format(min_freq))
    filter_keywords(ifile, ofile, min_freq=min_freq)


# names = ['train']
# for setname in names:
#     # idir = './data/preprocessed/by_city-users_min_3_reviews/keywords_spacy/' + setname
#     idir = './data/preprocessed/by_city-users_min_3_reviews/keywords_spacy-min_3/' + setname
#     odir = './data/preprocessed/by_city-users_min_3_reviews/user_to_keywords/raw_freq/' + setname
#     mkdir(odir)
#     for fname in os.listdir(idir):
#         if fname.startswith('.') or not fname.endswith(".json"):
#             continue
#         print("Processing for", fname)
#         ifile = os.path.join(idir, fname)
#         ofile = os.path.join(odir, fname)
#         group_keywords_for_users(ifile, ofile)
#     print("------------")


# # idir = './data/preprocessed/by_city-users_min_3_reviews/keywords_spacy/train'
# idir = './data/preprocessed/by_city-users_min_3_reviews/keywords_spacy-min_3/train'
# odir = './data/preprocessed/by_city-users_min_3_reviews/keywords_IRF'
# mkdir(odir)
# compute_irf_for_dir(idir, odir, N=1000)


# irf_dir = './data/preprocessed/by_city-users_min_3_reviews/keywords_IRF'
# idir_root = './data/preprocessed/by_city-users_min_3_reviews/user_to_keywords/raw_freq'
# odir_root = './data/preprocessed/by_city-users_min_3_reviews/user_to_keywords/tf_irf'

# city2irf = {}

# for setname in os.listdir(idir_root):
#     if setname.startswith("."):
#         continue
#     idir = os.path.join(idir_root, setname)
#     odir = os.path.join(odir_root, setname)
#     mkdir(odir)
#     for fname in os.listdir(idir):
#         if fname.startswith("."):
#             continue
#         ifile = os.path.join(idir, fname)
#         ofile = os.path.join(odir, fname)
#         print("Processing for", ifile)
#         compute_tfirf(ifile, ofile, irf=get_irf(fname, city2irf, irf_dir))


names = ['train']
for setname in names:
    # idir = './data/preprocessed/by_city-users_min_3_reviews/keywords_spacy/' + setname
    idir = './data/preprocessed/by_city-users_min_3_reviews/keywords_spacy-min_3/' + setname
    odir = './data/preprocessed/by_city-users_min_3_reviews/user_to_keywords/raw_freq/' + setname
    mkdir(odir)
    for fname in os.listdir(idir):
        if fname.startswith('.') or not fname.endswith(".json"):
            continue
        print("Processing for", fname)
        ifile = os.path.join(idir, fname)
        ofile = os.path.join(odir, fname)
        group_keywords_for_rests(ifile, ofile)
    print("------------")


idir = './data/preprocessed/by_city-users_min_3_reviews/keywords_spacy-min_3/train'
odir = './data/preprocessed/by_city-users_min_3_reviews/keywords_IUF'
mkdir(odir)
compute_iUf_for_dir(idir, odir, N=1000)

iuf_dir = './data/preprocessed/by_city-users_min_3_reviews/keywords_IUF'
idir_root = './data/preprocessed/by_city-users_min_3_reviews/user_to_keywords/raw_freq'
odir_root = './data/preprocessed/by_city-users_min_3_reviews/user_to_keywords/tf_iuf'

city2iuf = {}

for setname in os.listdir(idir_root):
    if setname.startswith("."):
        continue
    idir = os.path.join(idir_root, setname)
    odir = os.path.join(odir_root, setname)
    mkdir(odir)
    for fname in os.listdir(idir):
        if fname.startswith("."):
            continue
        ifile = os.path.join(idir, fname)
        ofile = os.path.join(odir, fname)
        print("Processing for", ifile)
        compute_tfiuf(ifile, ofile, irf=get_irf(fname, city2iuf, iuf_dir))


model = SentenceTransformer('all-MiniLM-L6-v2')
# model = SentenceTransformer('whaleloops/phrase-bert')

typeFile = ['train', 'test']
CITIES = ['edinburgh']
for city in CITIES:
    for tp in typeFile:
        if tp == 'train':
            f = open(f'./data/preprocessed/by_city-users_min_3_reviews/keywords_spacy-min_3/{tp}/{city}-keywords.json')
        else:
            f = open(f'./data/preprocessed/by_city-users_min_3_reviews/keywords_spacy/{tp}/{city}-keywords.json')
        data = json.load(f)
        keys = [ii for ii in data]
        kwEB = []
        kwL = []
        kws = [kw for kw in data[keys[0]]]
        print("Encoding")
        inputs = model.encode(kws)
        kwEB_pad = np.asarray(inputs)
        np.save(f'./data/embedding/{city}_kwSenEB_pad_{tp}.npy', kwEB_pad)

# Download train: train is filtered file at ./data/preprocessed/by_city-users_min_3_reviews/keywords_spacy-min_3/train) then rename as: {city}-keyword_train.json
# Download test: train is filtered file at ./data/preprocessed/by_city-users_min_3_reviews/keywords_spacy/test) then rename as: {city}-keyword_test.json
# move those train/test keywords file to ./data/keywords directory

# Download irf and tf_irf;  rename as {city}-keyword-IRF.json {city}-keyword-TFIRF.json ; move to  ./data/score/{city}-keyword-TFIRF.json
# Download iuf and tf_iuf;  rename as {city}-keyword-IUF.json {city}-keyword-TFIUF.json ; move to  ./data/score/{city}-keyword-TFIUF.json
