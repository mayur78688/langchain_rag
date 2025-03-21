
 
from doc_loader import Load_data
from text_splites import Splitter
from vector_store import Vector_Store

#load the data from the doc
#create object of load_data and pass the path 
load_data = Load_data(r"file.pdf")
data = load_data.small_pdf_data()

#split the data into ckunks 
vector = Vector_Store()
splitter = Splitter(chunk_size=200,chunk_overlap=5)

chunks = splitter.rec_text_splitter(data[0].page_content)

vector.embed_store_vectors(data)

output = vector.get_vector_res("education details")

print(output)


