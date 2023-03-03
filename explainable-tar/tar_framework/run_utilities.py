# coding=utf-8

import os
import pandas as pd

import logging
logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)
file_logging_handler = logging.FileHandler('auto_tar.log')
file_logging_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_logging_handler.setFormatter(file_logging_format)
LOGGER.addHandler(file_logging_handler)


PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RET_DIR = os.path.join(PARENT_DIR, 'ret')
DATA_DIR = os.path.join(PARENT_DIR, 'data')

INF = 1e+10

REL = 1
NONREL = 0



def check_path(mdir):
    if not os.path.exists(mdir):
        os.makedirs(mdir)


def name_tar_run_file(data_name, model_name, topic_set, exp_id, topic_id):
    mdir = os.path.join(RET_DIR, data_name, 'tar_run', model_name, topic_set, str(exp_id))
    check_path(mdir)
    fname = topic_id + '.run'
    return os.path.join(mdir, fname)


def write_tar_run_file(f, topic_id, check_func, shown_dids):
    """

    "Three types of interactions are supported:
    # INTERACTION = AF, relevance feedback is used by the system to compute the ranking of subsequent documents
    # INTERACTION = NF, relevance feedback is not being used by the system
    # INTERACTION = NS, the document is not shown to the user (these documents can be excluded from the output"

    See CLEF-TAR 2017 (https://sites.google.com/site/clefehealth2017/task-2).

    @param f:
    @param topic_id:
    @param check_func:
    @param shown_dids:
    @return:
    """
    for i, did in enumerate(shown_dids):

        if check_func(did) is True:
            screen = 'AF'
        else:
            screen = 'NF'
        f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(topic_id, screen, did, i + 1, -i, 'mrun'))
    return


def name_interaction_file(data_name, model_name, topic_set, exp_id, topic_id):
    mdir = os.path.join(RET_DIR, data_name, 'interaction', model_name, topic_set, str(exp_id))
    check_path(mdir)
    fname = topic_id + '.csv'
    return os.path.join(mdir, fname)


def read_interaction_file(data_name, model_name, topic_set, exp_id, topic_id):
    mdir = os.path.join(RET_DIR, data_name, 'interaction', model_name, exp_id, topic_set, str(exp_id))
    filename = topic_id + '.csv'
    df = pd.read_csv(os.path.join(mdir, filename))
    if len(df) == 0:
        print('Empty df', os.path.join(mdir, filename))

    return df
