import os
from collections import namedtuple
from operator import itemgetter
from pathlib import Path
import pickle

from flask import Flask
from flask import render_template, request
import gensim.downloader as gensim_api
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

import tar_framework.ranking as ranking
from tar_framework.assessing import Assessor
import parameter_utilities

app = Flask(__name__)
corpus_name = "reuters21578"
file_params = parameter_utilities.make_file_params(corpus_name, corpus_name, "alum", "alum")
del file_params["data_name"]
del file_params["topic_id"]
del file_params["topic_set"]
assessor = Assessor(**file_params)
complete_dids = assessor.get_complete_document_ids()
complete_pseudo_dids = assessor.get_complete_document_ids_with_pseudo_document()
complete_pseudo_texts = assessor.get_complete_document_texts_with_pseudo_document()

ranker = ranking.Ranker(model_type="famdg", committed_beta=1.0, number_of_mapping_nodes=50, scheduler_address="localhost:8786")

@app.route('/')
def hello():
	return render_template('index.html')

@app.route("/search", methods=["POST"])
async def getSeedDocs():
    if request.form["vectorizer"] == "tf-idf":
        vectorizer_type = ranking.VectorizerType.tf_idf
    elif request.form["vectorizer"] == "word2vec":
        vectorizer_type = ranking.VectorizerType.word2vec
    elif request.form["vectorizer"] == "glove":
        vectorizer_type = ranking.VectorizerType.glove

    ranker.model = None
    ranker.set_document_ids_to_features(dids=complete_pseudo_dids, texts=complete_pseudo_texts, corpus_texts=complete_pseudo_texts, vectorizer_type=vectorizer_type, corpus_name=corpus_name)
    ranker.set_features_by_name("complete_dids", complete_dids)
    await ranker.cache_corpus_in_model(complete_dids)
    
    doc_ids, scores = ranking.bm25_okapi_rank(assessor.get_complete_document_ids(), assessor.get_complete_texts(), request.form["seedKeywords"])
    documents = [assessor.document_id_to_text[doc_id] for doc_id in doc_ids[:10]]
    return {"documents": documents, "document_ids": doc_ids[:10], "scores": scores[:10]}

@app.route("/reset", methods=["POST"])
def resetModel():
    ranker.model = None
    return {}

@app.route("/score", methods=["POST"])
async def updateModel():
    evaluated_document_ids = request.json["documentIds"]
    relevance_scores = request.json["relevanceScores"]
    features = ranker.get_feature_by_document_ids(evaluated_document_ids)
    await ranker.train(features, relevance_scores, evaluated_document_ids)
    assessor.update_assessment(evaluated_document_ids)
    
    unassessed_document_ids = assessor.get_unassessed_document_ids()
    scores = await ranker.predict_with_doc_id(unassessed_document_ids)
    zipped = sorted(scores, key=itemgetter(0), reverse=True)
    
    if len(zipped) > 0:
        _, ranked_document_ids, artmap_categories = zip(*zipped)
    else:
        ranked_document_ids = []
    documents = [assessor.document_id_to_text[doc_id] for doc_id in ranked_document_ids[:10]]
    return {"documents": documents, "document_ids": ranked_document_ids[:10], "doc_categories": artmap_categories[:10]}


@app.route("/explain", methods=["POST"])
async def explainDocument():
    explanation = f"No explanation for document: {request.json['documentId']}"
    if request.json["vectorizer"] == "tf-idf":
        global features
        if features is None:
            features = get_tf_idf_features()
        category_and_feature_info = get_category_and_feature_info(ranker.model.training_fuzzy_artmap, int(request.json["categoryId"]))
        rule = convert_category_and_feature_info_to_rule(category_and_feature_info[0])
        explanation = f"Rule for category ({request.json['categoryId']}) associated with document {request.json['documentId']}:\n{rule}"
    elif request.json["vectorizer"] == "word2vec":
        global word2vec_model
        if word2vec_model is None:
            word2vec_model = get_word2vec_features()
        descriptors = build_word2vec_category_descriptors(ranker.model.training_fuzzy_artmap, int(request.json["categoryId"]))
        explanation = f"Category ({request.json['categoryId']}) associated with document {request.json['documentId']} is described by:\n{', '.join(descriptors)}"
    else:
        global glove_model
        if glove_model is None:
            load_glove_model()
        descriptors = build_glove_category_descriptors(ranker.model.training_fuzzy_artmap, int(request.json["categoryId"]))
        explanation = f"Category ({request.json['categoryId']}) associated with document {request.json['documentId']} is described by:\n{', '.join(descriptors)}"
    return {"explanation_type":"string", "explanation": explanation}

features = None
word2vec_model = None
base_word2vec_model = None

glove_model = None
glove_map = None

CategoryFeatures = namedtuple("CategoryFeatures", ["relevant", "features"])
FeatureRange = namedtuple("FeatureRange", ["min", "max"])

def load_glove_model():
    if not container_running:
        glove_file_location = (Path.cwd() / "data" / "glove.6B.300d.txt").resolve()
    else:
        glove_file_location = (Path.cwd().parent / "data" / "glove.6B.300d.txt").resolve()
    print("Loading Glove Model")
    global glove_model
    global glove_map
    glove_map = {}
    glove_words = []
    glove_vectors = []
    with open(glove_file_location,'r') as glove_file:
        for line in glove_file:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_words.append(word)
            glove_vectors.append(embedding)
    glove_model = torch.tensor(np.array(glove_vectors))
    print(f"{len(glove_words):,} words loaded")
    print("Scaling GloVe features")
    scalar = MinMaxScaler(feature_range=(0,1), copy=False)
    scalar.fit_transform(glove_model)
    for index, word in enumerate(glove_words):
        glove_map[index] = word
    print("Scaling complete")

def build_glove_category_descriptors(model, category_index):
    ones = torch.ones([300])
    global glove_model
    global glove_map
    point_a = model.weight_a[category_index,0:300]
    point_b = ones - model.weight_a[category_index,300:]        
    category_center = (point_a + point_b) / 2
    cos_sim = torch.nn.functional.cosine_similarity(category_center, glove_model, dim=1)

    # Get the top 10 cosine similarity values along with their corresponding indices
    _, top_indices = torch.topk(cos_sim, k=10, dim=0, largest=True, sorted=True)
    
    category_words = []
    for word_index in top_indices:
        category_words.append(glove_map[word_index.item()])
    
    return category_words

def get_word2vec_features():
    global base_word2vec_model
    base_word2vec_model = gensim_api.load('word2vec-google-news-300')
    scalar = MinMaxScaler(feature_range=(0,1), copy=False)
    scaled_model = torch.tensor(base_word2vec_model.vectors)
    scalar.fit_transform(scaled_model)
    torch.clip(scaled_model, min=0.0, max=1.0, out=scaled_model)
    return scaled_model

def build_word2vec_category_descriptors(model, category_index):
    ones = torch.ones([300])
    global word2vec_model
    global base_word2vec_model

    point_a = model.weight_a[category_index,0:300]
    point_b = ones - model.weight_a[category_index,300:]        
    category_center = (point_a + point_b) / 2
    cos_sim = torch.nn.functional.cosine_similarity(category_center, word2vec_model, dim=1)

    # Get the top 10 cosine similarity values along with their corresponding indices
    _, top_indices = torch.topk(cos_sim, k=10, dim=0, largest=True, sorted=True)
    
    category_words = []
    for word_index in top_indices:
        category_words.append(base_word2vec_model.index_to_key[word_index])
    
    return category_words

def get_tf_idf_features():
    if not container_running:
        pickle_path = (Path.cwd() / "data" / "pickels" / "vec_tf_idf_ff7732097eca0366d681c66cdb6e8d66_reuters21578.pkl").resolve()
    else:
        pickle_path = (Path.cwd().parent / "data" / "pickels" / "vec_tf_idf_ff7732097eca0366d681c66cdb6e8d66_reuters21578.pkl").resolve()
    with open(pickle_path, "rb") as pickled_vectorizer_file:
        tfidf_vectorizer = pickle.load(pickled_vectorizer_file)

    return tfidf_vectorizer.get_feature_names_out()

def get_category_and_feature_info(model, category_index):
    feature_weights_by_category = {}
    category_features = CategoryFeatures(bool(model.weight_ab[category_index][0]), [])
    feature_min = 1.0
    feature_max = 0.0
    for index, feature in enumerate(features):
        presence = model.weight_a[category_index][index].item()
        if presence == 0.0:
            continue
        category_features.features.append((feature, presence))
        feature_min = min(feature_min, presence)
        feature_max = max(feature_max, presence)
    feature_weights_by_category[category_index] = (category_features, FeatureRange(feature_min, feature_max))
    return feature_weights_by_category


def convert_category_and_feature_info_to_rule(category_and_feature_info):
    if len(category_and_feature_info[0].features) == 0:
        return
    predicates = []
    bin_size = (category_and_feature_info[1].max - category_and_feature_info[1].min) / 3
    for feature, weight in category_and_feature_info[0].features:
        quantitized_weight = ""
        formatted_feature = f"and '{feature}'"
        predicate_end = ""
        if (category_and_feature_info[1].max - bin_size) < weight and weight <= category_and_feature_info[1].max:
            quantitized_weight = "highly"
        elif (category_and_feature_info[1].min + bin_size) < weight and weight <= (category_and_feature_info[1].max - bin_size):
            quantitized_weight = "somewhat"
        else:
            quantitized_weight = "rarely"
        predicates.append(f"{formatted_feature} is {quantitized_weight} prevalent in document{predicate_end}")
    if len(predicates) > 0:
        predicates[0] = predicates[0][4:]
    combined_predicates = "\n".join(predicates)
    relevance = "Relevant" if category_and_feature_info[0].relevant else "Not Relevant"
    rule_if = "\nIF"
    rule = f"Document is {relevance}{rule_if} {combined_predicates}"
    return rule


container_running = False
if __name__ == '__main__':
    container_running = True
    app.run(host='0.0.0.0', port=8000, debug=True)