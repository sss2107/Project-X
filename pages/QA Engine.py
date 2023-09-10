import streamlit as st
from urllib.error import URLError
import openai
import os
from collections import deque
from loguru import logger
import sys
import time
import altair as alt
from urllib.error import URLError
# sys.path.append("..")

import re
import sys
import traceback
import json
import time
from typing import Callable
from loguru import logger
from functools import wraps
import pandas as pd

# Langchain related
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chat_models import AzureChatOpenAI
from common import (
    GPT_ANSWER_THRESHOLD,
    MIN_GPT_QUERY_LENGTH,
    MIN_QUERY_LENGTH,
    RANKER_NUM_RESULTS,
    GPT_MODEL_PARAMS,
    MODEL_FOLDER,
    GPT_ANSWER_PREFIX,
)

#There are other ways to mask the OpenAI key however, due to shortage of time, I'll embed the key in the code. Please don't use this anywhere else.
#from config import API_KEY
os.environ['OPENAI_API_KEY'] = 'e1ad721cd0fc4e5a89a8c67d1ce6e75d'
# os.environ['GPT_API_KEY'] = 'e1ad721cd0fc4e5a89a8c67d1ce6e75d'

logger.remove()
logger.add(sys.stderr, level="DEBUG")

################################# Langchain#####################################
# If the GPT api key is not set, then fallback to just the semantic search
# if os.getenv("GPT_API_KEY") is None:
#     logger.info("No GPT3 key, disabling this feature")
#     GPT_STATUS = False
# else:
#     logger.info("GPT3 API Key found")
#     os.environ["OPENAI_API_KEY"] = os.getenv("GPT_API_KEY")
#     GPT_STATUS = True
GPT_STATUS = True

os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(page_title="Ask anything from Sahils resume", page_icon="😇")
html_temp = """
<div style="background-color:brown;padding:10px">
<h2 style="color:white;text-align:center;">Ask anything from Sahils resume</h2>
</div>
"""
st.markdown(html_temp,unsafe_allow_html=True)
st.sidebar.header("Ask anything from Sahils resume")

class Chatbot:
    def __init__(self) -> None:
        self.embeddings = HuggingFaceEmbeddings()
        logger.info(f"Embedding model = {self.embeddings.model_name}")

        logger.info("Reading in vectorstore DB from training script")
        self.db = FAISS.load_local("faiss_index_sap", self.embeddings)

        # Change model to GPT4 on our Azure Openai resource
        model = AzureChatOpenAI(
            openai_api_base=GPT_MODEL_PARAMS["api_base"],
            openai_api_version=GPT_MODEL_PARAMS["api_version"],
            deployment_name=GPT_MODEL_PARAMS["deployment_name"],
            openai_api_key="e1ad721cd0fc4e5a89a8c67d1ce6e75d",#os.getenv("GPT_API_KEY"),
            openai_api_type=GPT_MODEL_PARAMS["api_type"],
        )
        self.chain = load_qa_with_sources_chain(
            model,
            chain_type="stuff",
        )

    def get_response(
        self,
        query: str,
        # hr_grade: str,
        database_name: str = "joey",
        query_gpt: bool = GPT_STATUS,
    ) -> dict:
        """Executes the query and returns the results in the expected format to return
        as a response to the user.

        :param query: User query
        :type query: str
        :param hr_grade: User's hr grade, to be passed in the request
        :type hr_grade: str
        :param database_name: Currently not used; meant to enable querying from multiple vecdbs,
        defaults to "joey"
        :type database_name: str, optional
        :param query_gpt: Flag to denote if GPT should be called, defaults to GPT_STATUS
        :type query_gpt: bool, optional
        :return: A dict of dicts containing the results. Top level key is the answer index,
        inner keys are ['Question','Answer','Image','SimScore']
        :rtype: dict
        """

        # 28Mar23: Prepare the query
        orig_query = query  # make a copy to display in the final output
        query = pre_process(query)
        logger.info(f"Modified query: {query}")

        # Semantic similarity results. Outputs are [(doc,score)]
        #! distance is returned, not similarity scores
        raw_results = self.db.similarity_search_with_score(query, k=RANKER_NUM_RESULTS)

        #! Process the results
        #! To create the final answer, the best score from the semantic search is
        #! extracted; if it's below a threshold, then GPT is called to generate the answer
        #! otherwise, the semantic answers are returned. To prepare for this possibility, the
        #! semantic answers are also formatted.
        docs = []  # To pass to the chain as the context
        results = {}  # Formatted results as a dict for potential display
        ans_idx = 0  # Initial index for the formatted results
        min_distance = 999  # To store the min distance found among the results
        all_semantic_questions = (
            ""  # To keep the questions only. Concatenated as a single string
        )
        top_answer = ""  # To keep the top answer only
        for i, raw_result in enumerate(raw_results):
            logger.debug(raw_result)

            # Get the question,answer and score from the db entries
            doc, score = raw_result
            (q, a) = doc.page_content.split("##")

            if i == 0:
                top_answer = q.strip() + "\n" + a.strip() + "\n"
            else:
                # Keep only the questions
                all_semantic_questions += "- " + q.strip() + "\n"

            docs.append(doc)
            results["Q" + str(ans_idx)] = {
                "QUESTION": q.strip(),
                "ANSWER": a.strip(),
                "REFER": doc.metadata["REFER"],
                "IMAGE": doc.metadata["IMAGE"],
#                "URL": doc.metadata["URL"],
                "SimScore": score,
            }
            min_distance = min([min_distance, score])

            ans_idx += 1
        logger.info(f"Min distance from Semantic results: {min_distance}")


        #! next blocks are only executed if gpt should be called which are based
        #! on the min_distance from the semantic search and the query_gpt flag
        #! The output can be 2 types: a) gpt is able to answer the query, b) gpt
        #! does not know how to answer the query. In the former case, the gpt
        #! response is formatted to extract the answer and the sources (where the
        #! text comes from the vecdb). In the latter case, a default answer is given
        #! whereby the top answer from the semantic search is returned along with
        #! a list of other possible questions to try (comes from the questions in
        #! the semantic search)
        gpt_answer = GPT_ANSWER_PREFIX
        # Call GPT only if the best answer from the semantic search is below threshold:
        # i.e. there is a potential answer to the semantic answers
        if min_distance <= GPT_ANSWER_THRESHOLD and query_gpt:
            # Run the chain for the query
            gpt_result = self.chain(
                {
                    "input_documents": docs,
                    "question": query,
                },
                return_only_outputs=True,
            )
            logger.debug(gpt_result)

            gpt_answer_text = gpt_result["output_text"]

            # gpt_answer_flag is used to indicate if the answer is a valid answer
            # if false, a default answer is returned to indicate that the answer
            # cannot be found.
            if re.search(r"I don't know", gpt_answer_text):
                gpt_answer_flag = False
            else:
                if re.search("(?=Source|Sources|SOURCE|SOURCES)", gpt_answer_text):
                    gpt_answer_flag = True
                    parts = re.split(
                        "(?=Source|Sources|SOURCE|SOURCES)", gpt_answer_text
                    )

                    gpt_answer += parts[0].strip() + "\n"
                    logger.info(f"source_part: {parts[1]}")
                    sources = [x for x in re.findall(r"S\d+", parts[1].strip())]
                    sources = list(set(sources))
                    logger.debug(f"SOURCES: {sources}")

                    gpt_answer += f"\n{len(sources)} SOURCES:\n"

                    for idx, source in enumerate(sources):
                        for doc in docs:
                            if source == doc.metadata["source"]:
                                (q, a) = doc.page_content.split("##")
                                gpt_answer += "\n"
                                gpt_answer += f"[{idx+1}] " + q.strip() + "\n"
                                gpt_answer += a.strip() + "\n"
                                gpt_answer = re.sub(r"\(S\d+\)", "", gpt_answer)
                else:
                    gpt_answer_flag = False
        else:
            gpt_answer_flag = False

        # If an answer is not found, modify the output by returning the top answer
        # (in case it happens to be correct) and a list of possible questions coming
        # from the semantic search
        if not gpt_answer_flag and query_gpt:
            gpt_answer += (
                "We're trying to fetch the best match for you.\n"
            )
            gpt_answer += "This is the closest answer from the KnowledgeBase:\n\n"
            gpt_answer += top_answer
            gpt_answer += (
                "\nAlternatively, you may want to try one of these questions instead:\n"
            )
            gpt_answer += all_semantic_questions
            gpt_answer += "\nOtherwise, please reach out to sahil_sharma."

        if query_gpt:
            logger.info(f"[FINAL ANSWER]\n{gpt_answer}")

            results = {}
            results["Q0"] = {
                "QUESTION": orig_query.strip(),
                "ANSWER": gpt_answer,
                "REFER": "EMPTY",
                "IMAGE": "EMPTY",
                "URL": "",
                "SimScore": 1,
            }

        return results#hr_extra_result.hr_response(
            # pd.DataFrame.from_dict(results, orient="index")
        # )


############################ HELPER FUNCTIONS ##################################
# def send_error_response(error_message: str, container_code: str, error_code: str):
#     return error_message,container_code,error_code


def measure(func: Callable) -> Callable:
    """This is a decorator function that measures the execution time of the function
    it decorates.

    :param func: A function to measure
    :type func: Callable
    :return: A wrapped function
    :rtype: Callable
    """

#    @wraps(func)
# def _time_it(*args, **kwargs):
#     start = int(round(time() * 1000000))
#     try:
#         return func(*args, **kwargs)
#     finally:
#         end_ = int(round(time() * 1000000)) - start
#         logger.info(f"Total execution/response time: {end_ if end_ > 0 else 0} us ")

#     return _time_it


def count_query_length(text: str) -> int:
    """Count the number of tokens in the query

    :param text: Query
    :type text: str
    :return: Number of tokens
    :rtype: int
    """

    parts = " ".join(re.compile(r"\W+", re.UNICODE).split(text))
    zz = re.sub("[^a-zA-Z]+", " ", parts).strip()
    text_length = len(zz.split(" "))
    return text_length


def pre_process(query: str) -> str:
    """Modify the query before running the search

    :param query: The user query
    :type query: str
    :return: Modified query
    :rtype: str
    """
    if re.search(r"(travel\s)?subload", query):
        query = re.sub(
            r"(travel\s)?subload", "leisure travel subject-to-load-basis", query
        )

    return query


################################# SETUP ########################################
logger.info("Start of predictor")
logger.info("Creating the chatbot database using previously saved document store")

logger.info("Creating the chatbot object")
################################# FLASK ########################################



#@app.route("/ping", methods=["GET"])
def ping():
    status = 200
    return "Ping Successful." #flask.Response(
#        response="Ping Successful.", status=status, mimetype="application/json"
#    )

#@app.route("/invocations", methods=["POST"])
#@measure
def transformation():
    try:
        logger.info("*" * 20)
        logger.info("Start /invocations")

        # Return error response if query is too short
        query_length = count_query_length(query)
        logger.info(f"Query: {query}\nQuery length: {query_length}")
        if query_length < MIN_QUERY_LENGTH:
            logger.info("User query too short. Returning error")

            return print("Query is too short, please input at least 2 words for the query.")#send_error_response(
#                "Query is too short, please input at least 2 words for the query.",
#                4016,
#                416,
#            )

        # The openai_query flag is to indicate to use GPT answer where possible
        # use_openai = False
        # use_openai = bool(data["openai_query"])
        # if "openai_query" in data and str(data["openai_query"]).lower() in [
        #     "true",
        #     "1",
        #     "y",
        #     "yes",
        # ]:
        use_openai = True
        logger.info(f"Use Generative Answer = {use_openai}")

        if query_length < MIN_GPT_QUERY_LENGTH:
            """
            If the query is too short, the Info Retrieval search could match
            easily to many docs in the KB because the intent of the query
            is not clear. This in turn will create a Context with potentially
            irrelevant info. When sent to GPT, a low quality answer could
            result.
            The parameter MIN_GPT_QUERY_LENGTH is set in common.py is currently
            from heuristics. Can be adjusted higher if shorter queries still
            gives poor GPT results.
            """
            logger.info("Query is too short for GPT")
            use_openai = False  # overrides the request flag

        result = kris_chat.get_response(
            query,
            database_name="Joey",
            query_gpt=use_openai,
        )

        logger.debug(json.dumps(result, indent=4))

        return result["Q0"]["ANSWER"]#flask.Response(
#                response=json.dumps(result), status=200, mimetype="application/json"
#            )

        # else:
        #     return send_error_response("Only JSON format is supported.", 4000, 400)
    except Exception:
        traceback.print_exc()
        logger.exception("Error is transformation() function")
        return "Sorry not sure if I fully understand- please elaborate your question."


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "w2k_hash" not in st.session_state:
    st.session_state.w2k_hash = ""
if "w2k" not in st.session_state:
    st.session_state.w2k = ""
if "message" not in st.session_state:
    st.session_state.message = ""


if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.history.append(
        {"role": "assistant", "content": "I'm an HR AI assistant. How can i help you today?"}
    )

if st.session_state.authenticated is True and st.session_state.usr_input in "GenAIapp" and st.session_state.department in "AI":
    # auth=st.session_state.authenticated
    st.write("""When writing prompt, be as detailed as possible. You can check the examples on the left for reference.""")

    with st.form("gpt_form"):
        query = st.text_area("Please input your query here:","What are Sahils achievements?; What is Sahils role as AI research at NUS?; Is Sahil married?")
        submit_button = st.form_submit_button("Send", type='primary')
        if submit_button and query:
            st.session_state['message']+=query
            start_time = time.time()
            st.session_state.history.append({"role": "user", "content": query})
            try:
                kris_chat = Chatbot()
                res=transformation()
                st.success('{}'.format(res))
                end_time = time.time()
                elapsed_time = end_time - start_time
                # Add assistant response to the conversation history and limit to 5 messages
                st.session_state.history.append(
                    {"role": "assistant", "content": res}
                )
                if len(st.session_state.history) > 10:
                    st.session_state.history.pop(0)
                    st.session_state.history.pop(0)
                # token count
#                prompt_token_count = int(response["usage"]["prompt_tokens"])
#                answer_token_count = int(response["usage"]["completion_tokens"])
                # how many seconds used for the whole query
                query_time = round(elapsed_time, 2)

            except Exception as e:
                logger.error(e)
                st.warning(
                    "We apologize for the inconvenience. The Microsoft service is currently beyond capacity. We have sent a message to our Data team to investigate and we hope to have it resolved soon. Please check back later for updates. Thank you for your patience."
                )
    acco = st.expander("Conversation history", expanded=True)
    for message in st.session_state.history:
        acco.write(f'{message["role"].capitalize()}: {message["content"]}')


    if st.button("About"):
#        st.text("Lets Learn")
        st.text("Please reach out to sahil21@u.nus.edu for feedback.")
else: st.warning("Sorry, you don't have access to this app.")


#query='How many paid annual leave I have?'
#kris_chat = Chatbot()
#res=transformation()

# used to modify the output based on user grade. Currently this is a passthrough
# hr_extra_result = HrResult()
