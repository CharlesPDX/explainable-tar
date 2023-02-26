import os 

corpora_doctext_mapping = {
    '20newsgroups': '20newsgroups.collection.json',
    'reuters21578': 'reuters21578.collection.json',
    'athome1': 'athome1.collection.json',
    'athome4': 'athome4.collection.json',
    'legal': 'collection.json',
    'reuters-rcv1': 'reuters-rcv1.collection.json',
    'down-reuters': 'reuters-rcv1.collection.json',
    'down-tr': 'athome1.collection.json',
    'down-tr4': 'athome4.collection.json',
}

corpora_docid_mapping = {
    '20newsgroups': '20newsgroups.docids.txt',
    'reuters21578': 'reuters21578.docids.txt',
    'athome1': 'athome1.docids.txt',
    'athome4': 'athome4.docids.txt',
    'legal': 'docids.txt',
    'reuters-rcv1': 'rcv1v2-ids.dat',
    'down-reuters': 'down-reuters.docids.txt',
    'down-tr': 'down-tr.docids.txt',
    'down-tr4': 'down-tr4.docids.txt',
}

def make_file_params(collection_name: str, corpus_name: str, topic_id: str, topic_set: str) -> dict:
    PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    params = {'data_name': corpus_name,
    'topic_id': topic_id,
    'topic_set': topic_set,
    'query_file': os.path.join(PARENT_DIR, 'data', collection_name, 'topics', topic_id),
    'qrel_file': os.path.join(PARENT_DIR, 'data', collection_name, 'qrels', topic_id)}

    if corpus_name in corpora_doctext_mapping:
        params['doc_text_file'] = os.path.join(PARENT_DIR, 'data', collection_name, 'doctexts', corpora_doctext_mapping[corpus_name])
    else:
        params['doc_text_file'] = os.path.join(PARENT_DIR, 'data', collection_name, 'doctexts', topic_id)

    if corpus_name in corpora_docid_mapping:
        params['doc_id_file'] = os.path.join(PARENT_DIR, 'data', collection_name, 'docids', corpora_docid_mapping[corpus_name])
    else:
        params['doc_id_file'] = os.path.join(PARENT_DIR, 'data', collection_name, 'docids', topic_id)


    return params
