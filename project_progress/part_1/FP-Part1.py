import json
import math
import re
from collections import defaultdict, Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


#NOT INCLUDED IN THE PROJECT, REMEMNANT OF WRONG AND TEMPLATELESS APPROACH FOR PART 1 THAT HAS BEEN REIMAGINED INSIDE THE TEMPLATE

# Download required NLTK resources once
nltk.download('stopwords', quiet=True)

# Initialize text processing tools
STOP_WORDS = set(stopwords.words('english'))
STEMMER = SnowballStemmer('english')


def clean_text(text):
    if not text:
        return []
    
    # Lowercasing & Tokenization / Removing Punctuation
    tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    
    # Removing Stop Words & Stemming
    cleaned_tokens = [
        STEMMER.stem(token) 
        for token in tokens 
        if token not in STOP_WORDS
    ]
    
    return cleaned_tokens

#the load_data function is not really relevant for this one becouse of the template 
def load_data(file_path):
    """
    Loads JSON data, preserves all metadata fields, and creates a merged search text.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    docs = {}
    for item in data:
        
        pid = str(item.get('pid', item.get('_id', '')))
        if not pid:
            continue

        # Turn the dataset into something usable
        title = item.get('title', '') or ''
        desc = item.get('description', '') or ''
        brand = item.get('brand', '') or ''
        category = item.get('category', '') or ''
        sub_category = item.get('sub_category', '') or ''
        product_details = item.get('product_details', '') or ''
        seller = item.get('seller', '') or ''

        searchable_text = f"{title} {brand} {category} {sub_category} {desc} {product_details} {seller}"

        # Store full document details (Requirement 2 & 4)
        docs[pid] = {
            'pid': pid,
            'title': title,
            'description': desc,
            'brand': brand,
            'category': category,
            'sub_category': sub_category,
            'product_details': product_details,
            'seller': seller,
            'out_of_stock': item.get('out_of_stock', False),
            'selling_price': item.get('selling_price', None),
            'discount': item.get('discount', None),
            'actual_price': item.get('actual_price', None),
            'average_rating': item.get('average_rating', None),
            'url': item.get('url', ''),
            'searchable_text': searchable_text
        }
        
    return docs


def build_inverted_index(docs):
  
    index = defaultdict(dict)  # { term: { pid: frequency } }
    doc_lengths = {}
    
    for pid, doc in docs.items():
        tokens = clean_text(doc['searchable_text'])
        doc_lengths[pid] = len(tokens)
        counts = Counter(tokens)
        
        for term, freq in counts.items():
            index[term][pid] = freq
            
    return index, doc_lengths


def search_tfidf(query, index, docs, doc_lengths):
 
    query_tokens = clean_text(query)
    if not query_tokens:
        return []
    
    N = len(docs)
    scores = defaultdict(float)
    
    for term in query_tokens:
        if term in index:
            df = len(index[term])
            idf = math.log((N + 1) / (df + 1)) + 1
            for pid, tf in index[term].items():
                scores[pid] += tf * idf

    for pid in scores:
        if doc_lengths[pid] > 0:
            scores[pid] /= doc_lengths[pid]

    ranked_results = []
    for pid, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        doc_data = docs[pid].copy()
        doc_data['score'] = score
        ranked_results.append(doc_data)
        
    return ranked_results


if __name__ == "__main__":
    file_path = "fashion_products_dataset.json"
    
    # Load Data
    docs = load_data(file_path)
    
    # Build Inverted Index
    index, doc_lengths = build_inverted_index(docs)
    
    # Test Queries
    test_query = "women full sleeve sweatshirt cotton"
    results = search_tfidf(test_query, index, docs, doc_lengths)
    
    print(f"Top 3 Results for Query: '{test_query}'\n" + "="*50)
    for doc in results[:3]:
        print(f"PID: {doc['pid']}")
        print(f"Title: {doc['title']}")
        print(f"Brand: {doc['brand']} | Price: {doc['selling_price']}")
        print(f"Score: {doc['score']:.4f}")
        print("-" * 50)