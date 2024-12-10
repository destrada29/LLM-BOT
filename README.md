# LLM-BOT

## Project Overview

This repository contains Python scripts leveraging **LangChain**, **Gradio**, and SQL utilities to build intelligent agents and interactive applications. The main features include prompt templates, LLM-powered SQL querying, and a chatbot interface. Below is a breakdown of the scripts:

---

### **Files Description**

#### 1. **`promptTemplates.py`**
This script demonstrates how to create and format prompt templates for language models. It includes:
- Examples of structured prompts for question-answering tasks.
- Utilization of `FewShotPromptTemplate` and `PromptTemplate` to format questions and answers effectively.

#### Key Features:
- **Few-shot examples**: Designed to guide the LLM in understanding complex queries.
- **Dynamic Prompt Formatting**: Customizes prompts based on input variables.

---

#### 2. **`llama3.1.py`**
This script integrates the **LangChain Ollama** model and tools to enable SQL database interactions, semantic similarity-based prompt selection, and chatbot enhancements.

#### Key Features:
- **SQL Database Agent**: Automatically generates SQL queries based on user input and fetches results.
- **Semantic Example Selector**: Dynamically selects examples based on input similarity for enhanced LLM understanding.
- **Gradio Chatbot**: An interactive UI for user interaction powered by Gradio and LangChain tools.

#### Includes:
- **FAISS Vector Store**: For semantic similarity-based example selection.
- **Custom SQL Agent**: Supports user-defined queries and restricts unsafe operations like `INSERT` or `DELETE`.

---

#### 3. **`bot.py`**
This script builds a lightweight SQL agent that utilizes the **Ollama LLM** for querying a remote database. 

#### Key Features:
- **Zero-Shot SQL Agent**: Provides direct answers to database-related questions without requiring additional context.
- **Easy-to-Integrate**: Focused functionality for querying and retrieving database information.

---

### **Requirements**
To run the scripts, the following dependencies must be installed:
- Python >= 3.8
- [LangChain](https://www.langchain.com/)
- [Gradio](https://gradio.app/)
- SQL Database library (`pymssql`, `psycopg2`, etc., depending on your DB)
- FAISS for vector-based retrieval
- Any compatible LLM, e.g., Ollama's Llama3

Install required libraries using:
```bash
pip install langchain gradio pymssql faiss-cpu
```

- And If you like to install all de dependencies that langchain has, there is how:
```bash
pip install -r requirements.txt
```



## Usage

### Run the Chatbot
1. Navigate to the directory containing the scripts.
2. Execute the chatbot script:
   ```bash
   python llama3.1.py
   ```
3. Interact with the Gradio interface to ask database-related questions.

### Customizing Prompts
- Modify examples in `promptTemplates.py` to tailor the LLM responses based on specific use cases.

### Database Configuration
- Replace `Your URI database URL` with the appropriate connection string in `llama3.1.py` and `bot.py`.

---

## Contributing
Feel free to fork this repository and submit pull requests for improvements or new features. Ensure proper documentation for any added functionality.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

