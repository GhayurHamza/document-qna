# 📄 Document Qna App
install requirements

Pull docker image of qdrant if image not already present
```docker pull qdrant/qdrant```

Run the contaner for qdrant
```docker run -p 6333:6333 -v .:/qdrant/storage qdrant/qdrant```

Download ollama for your system (mac, windows): https://ollama.com/download

For linux run command:
```curl -fsSL https://ollama.com/install.sh | sh```

After downloading run command:
```ollama pull llama3.2:3b```

Now your app is ready to use.

In the root directory, run command to start the application on your browser:
```streamlit run doc_qna_app/st_app```
