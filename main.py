from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer

load_dotenv()

# Step 1: Read Text from PDF File
text = ""
pdf_reader = PdfReader("incorrect_facts.pdf")
for page in pdf_reader.pages:
    text += page.extract_text() + "\n"

# Step 2: Parent-Child Chunking
parent_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=50)
child_splitter = CharacterTextSplitter(separator=" ", chunk_size=250, chunk_overlap=50)

parent_chunks = parent_splitter.split_text(text)
parent_child_map = {}
child_chunks = []

for parent in parent_chunks:
    children = child_splitter.split_text(parent)
    parent_child_map[parent] = children
    child_chunks.extend(children)

    # Print check - Limit number of parents to display for readability
# max_parents_to_show = 5

# for i, (parent, children) in enumerate(parent_child_map.items()):
#     if i >= max_parents_to_show:
#         break  # Avoid printing too many chunks
    
#     print(f"\n🔷 Parent Chunk {i+1}:\n{parent}\n")
#     print("   └── Child Chunks:")
    
#     for j, child in enumerate(children):
#         print(f"      🟢 Child {j+1}: {child}")
    
#     print("=" * 80)  # Separator for clarity

# Step 3: Initialize Vector Database
pc = Pinecone()
index_name = "rag-pc"
index = pc.Index(index_name)

#creating index and then commenting once the index is created
# pc.create_index(
#     name=index_name,
#     dimension=384, # Replace with your model dimensions
#     metric="cosine", # Replace with your model metric
#     spec=ServerlessSpec(
#         cloud="aws",
#         region="us-east-1"
#     ) 
# )

# Step 4: Select Embedding Model
embedding_model = SentenceTransformer("BAAI/bge-small-en")

# Step 5: Store Chunks in Pinecone
# for i, parent in enumerate(parent_chunks):
#     parent_embedding = embedding_model.encode(parent, normalize_embeddings=True)
#     index.upsert([(f"p-{i+1}", parent_embedding.tolist(), {"chunk": parent, "type": "parent"})])
    
#     for j, child in enumerate(parent_child_map[parent]):
#         child_embedding = embedding_model.encode(child, normalize_embeddings=True)
#         index.upsert([(f"c-{i+1}-{j+1}", child_embedding.tolist(), {"chunk": child, "parent": f"p-{i+1}", "type": "child"})])

# Step 6: User Query
llm = ChatGroq(temperature=0, model="llama3-70b-8192")
query = "How do Birds Migrate"
question_embedding = embedding_model.encode(query, normalize_embeddings=True)

# Step 7: Retrieve Top-K Results
result = index.query(vector=question_embedding.tolist(), top_k=3, include_metadata=True)
retrieved_chunks = []

for match in result.matches:
    retrieved_chunks.append(match.metadata["chunk"])
    
    # If retrieved chunk is a child, fetch its parent
    if "parent" in match.metadata:
        parent_id = match.metadata["parent"]
        parent_result = index.query(id=parent_id, top_k=1, include_metadata=True)
        retrieved_chunks.append(parent_result.matches[0].metadata["chunk"])

augmented_text = "\n\n".join(set(retrieved_chunks))

#Printing retrieved chunks to varify
# print("\nRetrieved Chunks:")

# for i, match in enumerate(result.matches):
#     print(f"\nChunk {i+1}:")
#     print(f"   ID: {match.id}")
#     print(f"   Score: {match.score}")
#     print(f"   Text: {match.metadata['chunk'][:500]}...")  # Truncate for readability

#     # If it's a child, fetch its parent chunk
#     if "parent" in match.metadata:
#         parent_id = match.metadata["parent"]
#         print(f"   Parent ID: {parent_id}")

#         # Fetch parent chunk
#         parent_result = index.query(id=parent_id, top_k=1, include_metadata=True)

#         if parent_result.matches:
#             parent_chunk = parent_result.matches[0].metadata["chunk"]
#             print(f"   Parent Text: {parent_chunk[:500]}...")  # Truncate for readability
#         else:
#             print("   Parent chunk not found!")

# print("\nAll retrieved chunks displayed.\n")

# Step 8: Create Chatbot
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are a helpful assistant. Use the context provided to answer the question accurately.
    Only use this context, not your knowledge.
    \n\nContext:{context}\nQuestion:{question}
    """
)

chain = prompt | llm 
response = chain.invoke({"context": augmented_text, "question": query})

print(response.content)
