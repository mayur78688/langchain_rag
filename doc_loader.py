from langchain_community.document_loaders import PyPDFLoader
class Load_data:
    def __init__(self,path:str):
        self.loader = PyPDFLoader(path)
    def small_pdf_data(self):
        data = self.loader.load()
        return data
    def large_pdf_data(self):
        for doc in self.loader.lazy_load():
            yield doc
        
            