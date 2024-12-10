from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3")  #"llama3.1"

from langchain_community.utilities import SQLDatabase
uri= f"Your URI database URL"
db = SQLDatabase.from_uri(database_uri=uri)

examples = [
    {   "input": "List all Rosas.", 
        "query": "SELECT * FROM Rosas;"},
    {
        "input": "How many Cantidad are in CamaRosas 1",
        "query": "SELECT * FROM CamaRosas WHERE ID_CamaRosas = 1 ;",
    },
    {
        "input": "List all Camas in the Bloque A Ubicacion.",
        "query": "SELECT * FROM Cama WHERE Ubicacion LIKE 'Bloque A%';",
    },
    {
        "input": "Find the total of Rosas.",
        "query": "select count(*) from Rosas;",
    },
    {
        "input": "What is the ID of the TallosPodados where the Cantidad is the highest?",
        "query": "SELECT TOP (1) ID_TallosPodados from TallosPodados order by Cantidad desc;",
    },
]



from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    embeddings,
    FAISS,
    k=2,
    input_keys=["input"],
    )

# example_selector.vectorstore.search("How many Rosas are there?", search_type = "mmr")



system_prefix = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQL SERVER query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 7 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.

You have access to the following tools for interacting with the database:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of {tool_names}
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.
If you see you are repeating yourself, just provide final answer and exit.

Here are some examples of user inputs and their corresponding SQL queries:"""


from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

dynamic_few_shot_prompt = FewShotPromptTemplate(
    example_selector = example_selector,
    example_prompt=PromptTemplate.from_template(
        "User input: {input}\nSQL query: {query}"
    ),
    input_variables=["input"],
    prefix=system_prefix,
    suffix=""
)


from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate

full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=dynamic_few_shot_prompt),
        ("human", "{input}"),
        ("system", "{agent_scratchpad}"),
    ]
)

from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool, InfoSQLDatabaseTool, ListSQLDatabaseTool, QuerySQLCheckerTool

tools = [QuerySQLDataBaseTool(db = db), 
        #  InfoSQLDatabaseTool(db = db), 
        #  ListSQLDatabaseTool(db = db), 
         QuerySQLCheckerTool(db = db, llm = llm)]

# print(ListSQLDatabaseTool(db = db).description)

prompt_val = full_prompt.invoke(
    {
        "input": "count Roses",
        "tool_names" : [tool.name for tool in tools],
        "tools" : [tool.name + " - " + tool.description.strip() for tool in tools],
        "agent_scratchpad": [],
    }
)

# print(prompt_val.to_string())


from langchain.agents import AgentExecutor, create_react_agent
agent = create_react_agent(llm, tools, full_prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, verbose=True)

last_k_messages = 4




from langchain_community.chat_message_histories import SQLChatMessageHistory

def get_session_history(session_id):
    chat_message_history = SQLChatMessageHistory(
    session_id=session_id, connection = uri
    )

    messages = chat_message_history.get_messages()
    chat_message_history.clear()
    
    for message in messages[-last_k_messages:]:
        chat_message_history.add_message(message)
    
    print("chat_message_history ", chat_message_history)
    return chat_message_history


from langchain_core.runnables.history import RunnableWithMessageHistory

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    
)


import gradio as gr
import uuid


with gr.Blocks() as demo:
    
    state = gr.State("")
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])


    def respond(message, chatbot_history, session_id):
        if not chatbot_history:
            session_id = uuid.uuid4().hex

        print("Session ID: ", session_id)

        response = agent_with_chat_history.invoke(
                                        {"input": message},
                                        {"configurable": {"session_id": session_id}},
                                        )

        chatbot_history.append((message, response['output']))
        return "", chatbot_history, session_id

    msg.submit(respond, [msg, chatbot, state], [msg, chatbot, state])

demo.launch()