import os
from pathlib import Path
import json

from reuters_small_parser import ReutersParser


def get_corpus_files(full_path):
    for root, _, file in os.walk(full_path):
        for file in file:
            if ".sgm" in file:
                # print(os.path.join(root,file))
                yield os.path.join(root,file)

def get_text_corpus(full_path):
    parser = ReutersParser()
    documents = {}
    for file in get_corpus_files(full_path):
        with open(file, 'rb') as corpus_file:
            documents.update(parser.parse(corpus_file))
    
    return {document_id: document for document_id, document in documents.items() if len(document["body"]) > 0}

def get_categories(corpus_location):
    with open(os.path.join(corpus_location, "all-topics-strings.lc.txt")) as category_file:
        return [line.strip() for line in category_file.readlines()]


# topics - query file
def make_topics(target_location, corpus_location):
    topics_path = os.path.join(target_location, "topics")
    if not Path(topics_path).exists():
        os.mkdir(topics_path)
    for category in get_categories(corpus_location):
        topic = {"id": category, "query": "", "title": category}
        with open(os.path.join(topics_path, category), "w") as topic_file:
            topic_file.write(json.dumps(topic))
        
        # {"id": "grain", "query": "", "title": "grain" }

# qrels
def make_qrels(target_location, topic, text_corpus):
    qrels_path = os.path.join(target_location, "qrels")
    if not Path(qrels_path).exists():
        os.mkdir(qrels_path)
    qrels = {}
    for document_id, document in text_corpus.items():
        relevant = topic in document["topics"]
        qrel = f"{topic}     0  {document_id:5}     {int(relevant)}\n"
        if document_id not in qrels:
            qrels[document_id] = qrel
        elif relevant:
            qrels[document_id] = qrel
    
    with open(os.path.join(qrels_path, topic), "w") as qrel_file:
        qrel_file.writelines(qrels.values())

    #alt.atheism \s+ 0 \s+ doc_id \s+ [0|1] (rel/not rel)


# docids
def make_doc_ids(target_location, doc_ids):
    docids_path = os.path.join(target_location, "docids")
    if not Path(docids_path).exists():
        os.mkdir(docids_path)
    
    with open(os.path.join(docids_path, "reuters21578.docids.txt"), "w") as docids_file:
        docids_file.writelines(doc_ids)
    #docid\n

# doctext
def make_doctext(target_location, docs):
    doctext_path = os.path.join(target_location, "doctexts")
    if not Path(doctext_path).exists():
        os.mkdir(doctext_path)
    
    with open(os.path.join(doctext_path, "reuters21578.collection.json"), "w") as doctext_file:
        for doc_id, doc in docs.items():
            doctext_file.write(json.dumps({"id": str(doc_id), "title": doc["title"], "content": doc["body"]}) + "\n")
    #{"id": "doc_id", "title": "", "content": "body"}\n

if __name__ == "__main__":
    corpora_root_path = os.path.abspath("../corpora")
    reuters_small_corpus = "reuters21578"
    full_reuters_small_path = os.path.join(corpora_root_path, reuters_small_corpus)

    target_location = "./data/reuters21578"
    
    text_corpus = get_text_corpus(full_reuters_small_path)
    make_topics(target_location, full_reuters_small_path)

    categories = get_categories(full_reuters_small_path)

    for category in categories:
        make_qrels(target_location, category, text_corpus)

    doc_ids = list([str(doc_id)+"\n" for doc_id in text_corpus.keys()])
    make_doc_ids(target_location, doc_ids)

    make_doctext(target_location, text_corpus)