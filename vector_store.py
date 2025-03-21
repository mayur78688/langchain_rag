import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from uuid import uuid4
class Vector_Store:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.index = faiss.IndexFlatL2(len(self.embeddings.embed_query("hello world")))

    def embed_store_vectors(self,documents):
        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=self.index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vector_store.add_documents(documents=documents, ids=uuids)
    def get_vector_res(self,text):
        results = self.vector_store.similarity_search(
            text,
            k=2,
        )
        return results