
from langchain_community.llms.ollama import Ollama


llm = Ollama(model="llama3.1:8b-instruct-q4_0")



from langchain.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentType
uri= f"Your URI database URL"
remote_db = SQLDatabase.from_uri(database_uri=uri)


query = "how many Cama are ?"



agent_executor = create_sql_agent(llm, db = remote_db, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose = True)


agent_executor.invoke(query)









# from langchain_community.document_loaders import PyPDFLoader

# loader = PyPDFLoader("202003291621085101sanjeev_rdbms_unit-I_sql_bba_ms_4_sem.pdf")
# pages = loader.load_and_split()

# from langchain_text_splitters import CharacterTextSplitter

# text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=7500, chunk_overlap=100
# )
# doc_splits = text_splitter.split_documents(pages)

# import tiktoken

# # encoding = tiktoken.get_encoding("cl100k_base")
# # encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
# # for d in doc_splits:
# #     print("The document is %s tokens" % len(encoding.encode(d.page_content)))

# from langchain_chroma import Chroma
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnableLambda, RunnablePassthrough
# from langchain_community.llms.ollama import Ollama

# from langchain_ollama import OllamaEmbeddings

# embeddings = OllamaEmbeddings(model="nomic-embed-text")


# # Add to vectorDB
# vectorstore = Chroma.from_documents(
#     documents=doc_splits,
#     collection_name="rag-chroma",
#     embedding=embeddings,
# )
# retriever = vectorstore.as_retriever()

# from langchain_core.prompts import ChatPromptTemplate


# # Prompt
# template = """Answer the question based only on the following context but only answer the question, nothing more:
# {context}

# Question: {question}
# """
# prompt = ChatPromptTemplate.from_template(template)

# # LLM API
# llm = Ollama(model="llama3", temperature=0.1)



# # Chain
# chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# # Question
# print(chain.invoke("como puedo selecionar todos los datos de una tabla, como seria la query?"))




