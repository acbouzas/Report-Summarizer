# Report-Summarizer

## Introduction

The Report Summarizer is designed to extract information and provide summaries of PDF documents. This time I'm using Sentence Transformers and the OpenAI API, instead of Langchain directly. That gives us room to work with the code and get a more customized output fro the given PDF.       

## Example

1. Choose a desired asset from the dropdown menu.

2. The application provides a summary based on the content of the financial report, and a table if it's available.

3.You can change the prompts to get a customized report (optional) 

![image](https://github.com/acbouzas/Report-Summarizer/blob/main/images/Screenshot%20from%202023-09-18%2011-17-43.png)
![image](https://github.com/acbouzas/Report-Summarizer/blob/main/images/Screenshot%20from%202023-09-18%2011-18-35.png)
![image](https://github.com/acbouzas/Report-Summarizer/blob/main/images/Screenshot%20from%202023-09-18%2011-12-38.png)

## Requirements

Before running the code, you need to install the following Python libraries:

- openai
- streamlit
- langchain
- PyPDF2
- sentence_transformers
- scikit-learn



