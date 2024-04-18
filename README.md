# Prototype : Multi-PDF-Chat-Extract-Documents-Sources-with-Responses 
chat with multiple Pdfs | get sources | all open source ( LLMs , Embeddings) | Runs on CPU

This project utilizes the following tools:

- [mistral-7b-instruct-v0.1.Q8_0.gguf] Quantized LLm .
- [HuggingFaceEmbeddings] for embeddings 'sentence-transformers/all-MiniLM-L6-v2 ' model .
- [faiss] vector store db .
- [langchain] to build a RAG system .
- [chainlit] UI.
- 
# Project Setup Guide

1. **Install git for  Linux (Debian/Ubuntu)**
   - Commands:
     ```
     sudo apt update
     sudo apt install git
     ```
2. **Clone the repository**
   - Command:
     ```
     git clone <this-repository_url>
     ```
3. **Install Python 3.10 Virtual Environment**
   - Description: This step installs the Python 3.10 virtual environment package, necessary for creating isolated Python environments.
   - Command:
     ```
     !apt install python3.10-venv
     ```
4. **Create Virtual Environment**
   - Description: This step creates a virtual environment named `.venv` in the current directory and activates it.
   - Commands:
     ```
     !python -m venv .venv
     !source .venv/bin/activate
     ```
5. **Install Requirements**
   - Description: This step installs Python dependencies listed in the `requirements.txt` file.
   - Command:
     ```
     !pip install -r requirements.txt
     ```
 6. **Create a 'data'  folder , add your Pdfs into , create 'vs-db' folder and Run the  script to prepare the vectore store db**
    - Commands:
     ```
     `mkdir data`
     `mkdir vs-db`
     `python Generate-vs-db.py`
     ```
7. **Download Pretrained Model**
   - Description: This step downloads a pretrained language model required for the project's implementation.
   - Command:
     ```
     !wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q8_0.gguf
     ```
8. **Get your authtoken key , Run the Model and Setup Tunnel**
   - Description: This step runs the Python script `chains-llm.py` using the `chainlit` tool, and sets up an ngrok tunnel to expose a local server.
   - Commands:
     ```
     !chainlit run chains-llm.py &>/content/logs.txt &
     !ngrok config add-authtoken '<put-your-authtoken-key-here>'
     from pyngrok import ngrok
     ngrok_tunnel = ngrok.connect(8000)
     print('Public URL:', ngrok_tunnel.public_url)
     ```
9. **Visit Site**
   - Description: After completing the previous steps, visit the provided Public URL in a web browser. Wait until the chatbot says Hi!and starts asking questions about PDFs in the data folder.
   
10. **Stop Model and Tunnel**
   - Description: This step stops the execution of the `chainlit` tool and terminates the ngrok tunnel.
   - Commands:
     ```
     !killall chainlit
     ngrok.kill()
     ```

## Note
Please ensure you have a stable internet connection and necessary permissions to execute the commands listed above.
