import utils

import os
import openai
openai.api_key = utils.get_openai_api_key()

from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_files=["./eBook-How-to-Build-a-Career-in-AI.pdf"]
).load_data()

from llama_index import Document
document = Document(text="\n\n".join([doc.text for doc in documents]))

from llama_index.llms import OpenAI
from trulens_eval import Tru
tru = Tru()
tru.reset_database()

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
from utils import build_sentence_window_index

sentence_index = build_sentence_window_index(
    document,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="sentence_index"
)
from utils import get_sentence_window_query_engine

sentence_window_engine = get_sentence_window_query_engine(sentence_index)
window_response = sentence_window_engine.query(
    "how do I get started on a personal project in AI?"
)
print(str(window_response))
tru.reset_database()

tru_recorder_sentence_window = get_prebuilt_trulens_recorder(
    sentence_window_engine,
    app_id = "Sentence Window Query Engine"
)
for question in eval_questions:
    with tru_recorder_sentence_window as recording:
        response = sentence_window_engine.query(question)
        print(question)
        print(str(response))
tru.get_leaderboard(app_ids=[])
# launches on http://localhost:8501/
tru.run_dashboard()

# Auto-merging retrieval

from utils import build_automerging_index

automerging_index = build_automerging_index(
    documents,
    llm,
    embed_model="local:BAAI/bge-small-en-v1.5",
    save_dir="merging_index"
)
from utils import get_automerging_query_engine

automerging_query_engine = get_automerging_query_engine(
    automerging_index,
)
auto_merging_response = automerging_query_engine.query(
    "How do I build a portfolio of AI projects?"
)
print(str(auto_merging_response))
tru.reset_database()

tru_recorder_automerging = get_prebuilt_trulens_recorder(automerging_query_engine,
                                                         app_id="Automerging Query Engine")

for question in eval_questions:
    with tru_recorder_automerging as recording:
        response = automerging_query_engine.query(question)
        print(question)
        print(response)

tru.get_leaderboard(app_ids=[])
# launches on http://localhost:8501/
tru.run_dashboard()