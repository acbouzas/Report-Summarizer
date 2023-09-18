import openai
import os
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import CharacterTextSplitter

openai.api_key ='ENTER YOUR OPENAI API KEY HERE'

def get_pdf_text(equity):
    pdf = os.path.join('reportes-moodys', f"{equity}.pdf")
    with open(pdf, 'rb') as pdf_file:
        text = ''
        
        # Create a PDF reader object
        pdf_reader = PdfReader(pdf_file)
        text+= pdf_reader.pages[0].extract_text()
        
        #Get the total number of pages in the PDF
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return text

def get_chunks(text):
    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 100,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_text(text)
    return texts


def semantic_search(query_embedding, embeddings, chunks, top_k=5, fl_verbose=False):
    
    # Calculate cosine similarity scores using scikit-learn
    cosine_scores = cosine_similarity([query_embedding], embeddings)[0]

    # Create a list of (text, similarity_score) pairs
    results = [(chunks[i], cosine_scores[i]) for i in range(len(chunks))]

    # Sort the results by similarity score in descending order
    results.sort(key=lambda x: x[1], reverse=True)

    results = results[:top_k]
    
    if fl_verbose:
        # Print the sorted results
        for text, score in results:
            print("-------------------------------")
            print(f"Similarity Score: {score}")
            print(f"Text: {text}")
            print("")
    
    return "".join([x[0] for x in results])

def get_response(content):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": content},
        ],
        max_tokens=500
    )
    answer = response['choices'][0]['message']['content']
    return answer



# Beggining of the APP


st.header(f"Report Summarizer")

equities = ['','AMC', 'GT', 'EC', 'GEO', 'IEP', 'JWN', 'LUMN', "MACY'S", 'NAVI', 'OPI', 'PEMEX', 'QVC']
equity = st.selectbox('Choose your desired Equity', options= equities, index=0, key='eq')

# MODEL ECODDING
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
query = '''provide the company debt status and key indicators'''

content = f"""Given this Credit report about {equity}, 
1) Give me a brief summarization about the current state of the company debt and how likely it is to default.
Mention relevant ratios such as EBIT/interest expense of the latest periods. Use less than 100 words
2)If you find the 'key indicators' table, give me that table in Markdown format titled 'Key Indicators Table'. Take this table as an output format example: 

|                         | 12/31/2019 | 12/31/2020 | 12/31/2021 | 09/30/2022 | LTM12/31/2022 | Proj.12/31/2023 | Proj.12/31/2024 |
|-------------------------|-----------|-----------|------------|------------|---------------|-----------------|-----------------|
| Revenue (USD Million)   | $5,471.0  | $1,242.4  | $2,527.9   | $4,092.2   | $4,050.0      | $4,670.0        | $5,020.0        |
| EBITA Margin            | 11.7%     | -92.8%    | -13.4%     | 4.3%       | 3.5%          | 5.0%            | 6.4%            |
| Debt / EBITDA           | 6.6x      | -28.1x    | 34.1x      | 11.5x      | 12.0x         | 9.6x            | 8.4x            |
| EBITA / Interest        | 0.9x      | -1.3x     | -0.4x      | 0.2x       | 0.2x          | 0.3x            | 0.4x            |
| RCF / Net Debt          | 7.5%      | -10.1%    | -5.5%      | -0.4%      | -0.5%         | 4.4%            | 6.2%            |

Do not include any extra output after finishing this task.
"""


if st.session_state['eq'] != '' :     
    
    #get pdf text
    text = get_pdf_text(equity)

    #get pdf text
    chunks = get_chunks(text)

    # embed query
    query_embedding = model.encode(query)

    # generate embeddings on text chunks
    embeddings = model.encode(chunks)

    # Retrieve Context
    context_text = semantic_search(query_embedding, embeddings, chunks, top_k=5, fl_verbose=False)

    all_text = content + context_text
    # get response from the query
    ans = get_response(all_text)
  
    st.markdown(ans)
    print (ans) 
else: 
    pass        