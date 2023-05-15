import os
from operator import itemgetter

from flask import Flask
from flask import render_template, request

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
    return {"explanation_type":"string", "explanation": explanation}


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8000, debug=True)
