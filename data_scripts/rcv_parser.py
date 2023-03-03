# based on https://github.com/ductri/reuters_loader/blob/master/main.py

import json
import logging
import os
from pathlib import Path
import gzip
import shutil
from urllib import request
from collections import defaultdict
import xml.etree.ElementTree as ET



def might_download_file(url):
    file_name = url.split('/')[-1]
    file = ROOT/file_name
    if not file.exists():
        logging.info('File %s does not exist. Downloading ...\n', file_name)
        file_data = request.urlopen(url)
        data_to_write = file_data.read()

        with file.open('wb') as f:
            f.write(data_to_write)
    else:
        logging.info(f"File {file_name} already exists")


def might_extract_gz(path):
    path = Path(path)
    file_output_name = '.'.join(path.name.split('.')[:-1])
    file_name = path.name
    if not (path.parent/file_output_name).exists():
        logging.info('Extracting %s ...', file_name)

        with gzip.open(str(path), 'rb') as f_in:
            with open(str(path.parent/file_output_name), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        logging.info(f"File {file_name} already exists")


def get_doc_ids_v2():
    file = ROOT/'rcv1v2-ids.dat'
    with file.open('rt', encoding='ascii') as i_f:
        doc_ids = i_f.readlines()
    doc_ids = [item[:-1] for item in doc_ids]
    logging.info('There are %s docs in RCV1-v2', len(doc_ids))
    return doc_ids


def get_doc_topics_mapping():
    file = ROOT / 'rcv1-v2.topics.qrels'
    with file.open('rt', encoding='ascii') as i_f:
        lines = i_f.readlines()
    lines = [item[:-1] for item in lines]
    doc_topics = defaultdict(list)
    for item in lines:
        topic, doc_id, _ = item.split()
        doc_topics[doc_id].append(topic)
    logging.info(f'Mapping dictionary contains {len(doc_topics)} docs')
    return doc_topics


def get_topic_desc():
    file = ROOT / 'rcv1'/'codes'/'topic_codes.txt'
    with file.open('rt', encoding='ascii') as i_f:
        lines = i_f.readlines()
    lines = [item[:-1] for item in lines if len(item)>1][2:]
    topic_desc = [item.split('\t') for item in lines]
    topic_desc = {item[0]:item[1] for item in topic_desc}

    logging.info(f'There are totally {len(topic_desc)} topics')
    return topic_desc


def fetch_docs(doc_ids):
    all_path_docs = list(ROOT.glob('rcv1/199*/*'))    

    docid2path = {p.name[:-10]:p for p in all_path_docs}
    for idx, doc_id in enumerate(doc_ids):
        # text = docid2path[doc_id].open('rt', encoding='iso-8859-1').read()
        tree = ET.parse(str(docid2path[doc_id]))
        root = tree.getroot()
        text = "".join(root.find("text").itertext())
        title = "".join(root.find("title").itertext())
        if idx % 100000 == 0:
            logging.info('Fetched %s/%s docs', idx, len(doc_ids))

        yield doc_id, text, title


def write_docids():
    docid_path = target_base.joinpath("docids")
    if not docid_path.exists():
        os.mkdir(docid_path)
    file = ROOT/'rcv1v2-ids.dat'
    dest_file = docid_path/'rcv1v2-ids.dat'
    if dest_file.exists():
        logging.info(f"docid file {dest_file} already exists")
        return
    shutil.copy(file, docid_path)


def write_doctext():
    doctext_path = target_base.joinpath("doctexts")
    if not doctext_path.exists():
        os.mkdir(doctext_path)
    doc_ids = get_doc_ids_v2()
    file = doctext_path / "reuters-rcv1.collection.json"
    if file.exists():
        logging.info(f"doctext file {file} already exists")
        return
    
    with file.open("w") as doctext_file:
        for doc_id, doc, title in fetch_docs(doc_ids):
            doctext_file.write(json.dumps({"id": str(doc_id), "title": title, "content": doc}) + "\n")


def write_qrels():
    qrels_path = target_base.joinpath("qrels")
    if not qrels_path.exists():
        os.mkdir(qrels_path)
    doc_topic_mappings = get_doc_topics_mapping()
    topics_and_descriptions = get_topic_desc()
    for topic, _ in topics_and_descriptions.items():
        file = qrels_path / topic
        if file.exists():
            logging.info(f"qrel file {file} already exists")
            continue
        with file.open("w") as qrel_file:
            for document_id, topics in doc_topic_mappings.items():
                relevant = topic in topics
                qrel_file.write(f"{topic}     0  {document_id:5}     {int(relevant)}\n")


def write_topics():
    topics_path = target_base.joinpath("topics")
    if not topics_path.exists():
        os.mkdir(topics_path)

    topics_and_descriptions = get_topic_desc()
    for topic, description in topics_and_descriptions.items():
        file = topics_path / topic
        if file.exists():
            logging.info(f"topic file {file} already exists")
            continue
        trec_topic = {"id": topic, "query": description, "title": description}
        with file.open("w") as topic_file:
            topic_file.write(json.dumps(trec_topic))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    root_dir = os.getcwd() + "/data/"
    ROOT = Path(root_dir)
    target_base = Path(ROOT.joinpath("reuters-rcv1"))

    logging.info('Downloading rcv1v2-ids.dat.gz')
    might_download_file('http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a07-rcv1-doc-ids/rcv1v2-ids.dat.gz')

    logging.info('Downloading rcv1-v2.topics.qrels.gz')
    might_download_file('http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a08-topic-qrels/rcv1-v2.topics.qrels.gz')
    might_extract_gz(ROOT / 'rcv1-v2.topics.qrels.gz')
    write_qrels()

    might_extract_gz(ROOT / 'rcv1v2-ids.dat.gz')
    write_docids()
    
    write_doctext()
    write_topics()
    