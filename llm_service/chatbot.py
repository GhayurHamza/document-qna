from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_ollama import ChatOllama
from qdrant_client import QdrantClient
from langchain import PromptTemplate
from langchain.chains import RetrievalQA


class ChatbotManager:
    def __init__(
        self,
        model_name = "BAAI/bge-small-en",
        device = "cpu",
        encode_kwargs = {"normalize_embeddings": True},
        llm_model = "llama3.2:3b",
        llm_temperature = 0.7,
        qdrant_url = "http://localhost:6333",
        collection_name = "vector_db",
    ):

        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name

        # Embeddings
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs,
        )

        # Local LLM
        self.llm = ChatOllama(
            model=self.llm_model,
            temperature=self.llm_temperature,
        )

        # Prompt template
        self.prompt_template = """Use the following pieces of information to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Context: {context}
        Question: {question}
        
        Only return the helpful answer. Answer must be detailed and well explained.
        Helpful answer:
        """

        # Qdrant client
        self.client = QdrantClient(
            url=self.qdrant_url, prefer_grpc=False
        )

        # Qdrant vector store
        self.db = Qdrant(
            client=self.client,
            embeddings=self.embeddings,
            collection_name=self.collection_name
        )

        # Prompt initialization
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=['context', 'question']
        )

        # Retriever initialization
        self.retriever = self.db.as_retriever(search_kwargs={"k": 1})

        # Define chain type kwargs
        self.chain_type_kwargs = {"prompt": self.prompt}

        # Initialize the RetrievalQA chain with return_source_documents=False
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=False,
            chain_type_kwargs=self.chain_type_kwargs,
            verbose=False
        )

    def get_response(self, query: str) -> str:
        """
        Processes the user's query and returns the chatbot's response.

        Args:
            query (str): The user's input question.

        Returns:
            str: The chatbot's response.
        """
        try:
            response = self.qa.run(query)
            return response
        except Exception as e:
            return "⚠️ Sorry, I couldn't process your request at the moment."