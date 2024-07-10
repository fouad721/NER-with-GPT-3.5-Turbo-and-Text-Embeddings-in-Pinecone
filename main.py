import openai
import torch
from transformers import BertTokenizer, BertModel
from pinecone import Pinecone, ServerlessSpec
import hashlib


def generate_embeddings(text):
    """
    Generate embeddings for the given text using BERT.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)

    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    print("Embedding shape:", embeddings.shape)
    return embeddings

def store_embeddings_in_vector_db(embeddings, text, pinecone_api_key, pinecone_index_name):
    """
    Store the embeddings in a vector database using Pinecone, with an ASCII ID.
    """
    pc = Pinecone(api_key=pinecone_api_key)

    # Generate a valid ASCII ID (using MD5 hash of the text for uniqueness)
    text_id = hashlib.md5(text.encode('utf-8')).hexdigest()

    # Check if the Pinecone index exists, if not, create one with the correct dimension
    if pinecone_index_name not in pc.list_indexes().names():
        pc.create_index(
            name=pinecone_index_name,
            dimension=768,
            metric='euclidean',
            spec=ServerlessSpec(cloud='aws', region='us-west1-gcp')
        )

    # Connect to the Pinecone index
    vector_db = pc.Index(pinecone_index_name)

    # Ensure the embeddings are a flat list of floats
    embeddings_list = embeddings[0].tolist()

    # Store the embeddings
    vector_db.upsert(vectors=[(text_id, embeddings_list)])

# Update the gpt3_ner function call accordingly
def gpt3_ner(text, language, api_key, pinecone_api_key, pinecone_index_name):
    """
    Perform Named Entity Recognition using GPT-3.5 Turbo.
    """
    # Generate embeddings
    embeddings = generate_embeddings(text)

    # Store embeddings in vector database
    store_embeddings_in_vector_db(embeddings, text, pinecone_api_key, pinecone_index_name)

    # GPT-3 API call
    openai.api_key = api_key
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"Extract and categorize named entities in this text: {text}. Language: {language}.",
        max_tokens=100,
        temperature=0.3
    )

    return response.choices[0].text.strip()

# Example usage
text_en = "In 2022, the Nobel Prize in Physics was awarded for groundbreaking research in quantum mechanics."
text_es = "En 2022, el Premio Nobel de Física fue otorgado por investigaciones innovadoras en mecánica cuántica."
text_ru = "В 2022 году Нобелевская премия по физике была присуждена за прорывные исследования в области квантовой механики."

api_key = ''  # Replace with your actual OpenAI API key
pinecone_api_key = '' # Replace with your actual Pinecone API key
pinecone_index_name = ""  # Replace with your Pinecone index name

entities_en = gpt3_ner(text_en, "English", api_key, pinecone_api_key, pinecone_index_name)
print("English NER:", entities_en)

# Process Spanish sentence
entities_es = gpt3_ner(text_es, "Spanish", api_key, pinecone_api_key, pinecone_index_name)
print("Spanish NER:", entities_es)

# Process Russian sentence
entities_ru = gpt3_ner(text_ru, "Russian", api_key, pinecone_api_key, pinecone_index_name)
print("Russian NER:", entities_ru)