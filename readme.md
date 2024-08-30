# Spotify Bot Review

Creating a webapp that will go through millions of review and find insight using RAG and LLM.

## Table of Contents
1. [Installation](#installation)
2. [Data Processing and Cleaning](#data-processing-and-cleaning)
3. [ANN Creation](#ann-creation)
4. [Streamlit Web App](#streamlit-web-app)
5. [Test Results](#test-results)
6. [Future Improvements](#future-improvements)

## Installation
This project is using python 3.11.8. requirement.txt is attached in this project git. 
Please use the following command to install all required library. 

```bash
pip install -r requirements.txt
```

Note : the data folder and text video can be found on the link in the attached email.

## Data Processing and Cleaning
The first process in this whole pipeline is to clean the review data. 
This process is done using the 01. Data Processing.ipynb notebook
Inside there are multiple logic to clean the review data including: 
- Lower Casing Word
- Remove Special Character
- Remove Extra Spaces
- Remove Numeric Only Review
- Remove Review with Low Word Count
- Remove Review with Low Character Count

## ANN Creation
After cleaning the review data we need to prepare it so that it can be used in RAG system. 
We need to convert the cleaned text into embeddings. In this process sentence-transformers library will be used. 
The model paraphrase-MiniLM-L6-v2 is used in this proccess because it is fast and reliable for most task including this one.

## Streamlit Web App
To host everything, we will use streamlit app, there are multiple feature inside this app. Here are some notable feature:
1. RAG System using FAISS: In this app before we ask LLM, we need to provide relevant information about the review so LLM can answer it correctly. In order to do that we use RAG system. Embeddings that are generated from the previous step is feed into FAISS. FAISS is used because it is open-source, easy to use and pretty fast. With this system we will pull top 200 review that are related to the user's question. 
THis information will provide enought information to the LLM to answer the user questions.
2. Dynamic FAISS Index : Following from the previous feature. We can adjust the FAISS index depending in our use case. For example, if we only want to ask about information from review that are coming from certain date, we can filter out the review so that only review on that date will be used as extra information for the LLM. We can do similar thing with the review ratings and app version.
3. Fully Conversational : user can ask question to chatbot and user can ask more question about what the LLM just said. This is because we fully implement converstaional LLM where chat history on the same session will be saved and forwarded to the LLM when new question is asked. 

## Test Result
Test Result can be seen in the test folder in this repository.

## Future Improvement
### Data Processing and Cleaning
In this process, there are multiple step that can be implemented to improve the data quality: 
1. Language Detection and Translation : There are few review that are coming from different language other than english. There are two options, we can remove it entirely or we can detect it and convert it to english. Combination of langid and google_translate library would be sufficient for this step
2. NER / Sentiment Analysis : Detecting what the review about (UI, Feature, etc) and How the reviewer sentyment about the app (Positive/Negative/Netral) can provide an extra information for the LLM to work with. 

### ANN Creation
1. Better Encoding Model : Right now we use  paraphrase-MiniLM-L6-v2 but with further research there might be a more suitable model that can extract better embeddings.
2. Combining Embeddings : Using two encoding models and combining the embeddings might provide a usefull step in our usecase. For example we can add one additional model that works well with multi language, that way we don't need to worry about translating the review because the model done it for us. 

### Streamlit Web App
1. Better Vector Database : As of now, we use FAISS as vector database, and it works well, but one lacking feature is the abilty to do metadata filtering before doing ANN search. We manage to do it in a janky way by utilizing the dataframe index to filter the embedings but it would be better to have it built in. Right now FAISS is also hosted inside the app, as the review grows, this will be a problem, we need a more scalabel solutions. 
Some thing like AWS ES should be able to handle this task better. 
2. LLM Selection : In this project gpt-4o-mini is used because it is fast and cheap. But depending on the requirement we may need to get a better model like gpt-4o or move towards open source option like Llamma. 
3. Using Agent : As of now we are not using any LLM library like Langchain or LlamaIndex, the main reason is mostly because time constrain. But using this library we can create an LLM agent that can more intellegenty gather information about the user question. some of the benefits of using LLM agents and these library are 
- We can remove the user side filtering and rely on the LLM agent to do it automatically. For example if user want to know what is the latest trend based on the review, insted of manually selecting the start and end date of the review, LLM agent can intelegently do it based on the user question. Although we need to make sure that the LLM understand the filtering function clearly. 
- Langchain and LLammaIndex also allow use to integrate the RAG system better because in these library there are built in feature that handle most of the heavy works, instead of 
- A lot of prebuilt function that will make development of complex function easier. 
4. Quality Scoring : Adding some kind of score to the LLM response so that we can use it to make the LLM better. Unable to implement this feature because of time constrains.