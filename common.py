import hashlib
import os
from pathlib import Path

import pandas as pd
from loguru import logger

# For testing only - to see environ variable is availabe in environ
env_vars = list(os.environ.keys())
logger.info("*" * 20)
logger.info("Environment Variables")
logger.info(env_vars)
logger.info("*" * 20)

# Detect if the program is running in AWS or local
LOCALTEST = False  # Default is aws
found = False
for var in env_vars:
    if "SIA_ENV" in var:
        found = True
if not found:
    LOCALTEST = True
logger.info(f"Environment variables found? = {found}")

base_path = Path("/Users/sahil_sharma/Desktop/Joey 2.0/streamlit")
#if LOCALTEST:
#    base_path = Path(f"/Users/sahil_sharma/Desktop/Joey 2.0/streamlit")#/Users/{os.environ['USER']}/Projects/de-faqiller/v2assests")
#else:
#    base_path = Path("/Users/sahil_sharma/Desktop/Joey 2.0/streamlit")#/opt/ml/")

################################################################################
# Haystack DB files
# model path
MODEL_FOLDER = base_path# / "model"
MODEL_FILE = MODEL_FOLDER / "faiss_document_store_Joey.db"
FAISS_PREFIX = "faiss_index_sap"

# data path
DATA_FOLDER = base_path / "input/data/training/data"
DATA_FILE = DATA_FOLDER / "SAPqa.xlsx"

# General Parameters
MIN_QUERY_LENGTH = 2
LEV_DISTANCE_THRESHOLD = 0.9  # compare query with kb question

################################################################################
# Information Retrieval / Retrieve and Re-Rank Algorithm parameters

# Original models
# EMBEDDING_MODEL = 'distilroberta-base-msmarco-v2' # Original
# EMBEDDING_MODEL = 'msmarco-distilbert-dot-v5' # Seems to be better than orig mod
# EMBEDDING_MODEL = "multi-qa-mpnet-base-dot-v1"
# EMBEDDING_LENGTH = 768
# EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
# EMBEDDING_LENGTH = 384

# CROSSENCODING_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"  # Original

# EMBEDDING_NUM_RESULTS = 10
RANKER_NUM_RESULTS = 5

# Min Information Retrieval score to include in OpenAI qury
# IR_ANSWER_THRESHOLD = 0.3
# IR_ANSWER_THRESHOLD = 0.0001

################################################################################
# GPT Parameters - for openai api call
GPT_MODEL_PARAMS = {
    "api_type": "azure",
    "api_base": "https://siaec-data-gpt.openai.azure.com/",
    "api_version": "2023-03-15-preview",
    # "model_name": "DaVinci-003",
    "deployment_name": "gpt3516k"#"gpt-35-turbo",
}

MIN_GPT_QUERY_LENGTH = 2

# GPT Similarity Threshold score
GPT_ANSWER_THRESHOLD = 1.0

# GPT Prefix in result. Include any required symbols or characters
# GPT_ANSWER_PREFIX = "[GPT3 BETA] "
GPT_ANSWER_PREFIX = (
    "**Disclaimer:**\nPlease note that we are still in a testing phase "
    + "for this GenAI app. You may want to validate the answer provided against "
    + "the cited sources.\n\n"
)


################################################################################
def get_hash(df_: pd.DataFrame, col: str = "hval"):
    """Get the hash value for the hval column in the dataframe

    :param df_: Dataframe with a column named hval. This column should contain the text.
    :type df_: pd.DataFrame
    :return: Dataframe
    :rtype: pd.DataFrame
    """

    for i, row in df_.iterrows():
        df_.loc[i, col] = hashlib.md5(row[col]).hexdigest()

    return df_
