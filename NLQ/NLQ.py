from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
import os
import pickle as pk

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'

class DataBase:
    def __init__(
            self,
            data_loc:str,
            cache_dir:str = '/',
        ) -> None:

        self.data_loc = data_loc
        self.files = None
        self.embeddings = HuggingFaceEmbeddings()
        self.cache_dir = cache_dir
        self.vec_db = None
        self.llm_chain = None
    
    def __load_data(self):
        loader = CSVLoader(file_path = self.data_loc)
        self.files = loader.load()
        bin_file = open(f'{self.cache_dir}file', 'ab')
        pk.dump(self.files, bin_file)
    
    def __vector_database(self):
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(self.files)
        self.vec_db = FAISS.from_documents(docs, self.embeddings)
    
    def __llm(self):
        flan_t5 = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.1, "max_length":100})
        template ="""Content: {content}//
        Question: {question}
        Answer: """
        
        prompt = PromptTemplate(template=template, input_variables=["question", "content"])

        self.llm_chain = LLMChain(
            prompt=prompt,
            llm=flan_t5
        )

    def query(self, keyword, query):
        docs = self.vec_db.similarity_search(keyword)
        content = docs[0].page_content
        question = question
        return self.llm_chain.run(question, content)
    
obj = DataBase(
    data_loc='data.csv'
)

pk.dump(obj, open("llm, 'ab"))

obj.query(
    "King of Jungle",
    "who is the king of the jungle"
)