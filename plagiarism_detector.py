import re
import time
import requests
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer
from datetime import datetime

# Download required NLTK models
nltk.download('punkt')
nltk.download('stopwords')

# 🔐 Your API keys
SERP_API_KEY = '1261693f87db8fb9430c3b5074df0eb3a0dca34521eabc3c3ceb7e021a1be3f5'
GOOGLE_API_KEY = 'AIzaSyDVDzBhQrdy47pHIdBBI0-1Ui6OewxlASk'  # ⬅️ Replace this
GOOGLE_CSE_ID = 'c5720123c5efb4f61'  # ⬅️ Replace this

LOG_FILE = 'plagiarism_log.txt'

def log_event(message):
    """Appends a log message with timestamp."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[{datetime.now()}] {message}\n")

def check_plagiarism_online(input_text):
    """
    First tries SerpAPI. If it fails or reaches quota, switches to Google CSE.
    Logs all steps and errors.
    """
    try:
        log_event("Attempting SerpAPI")
        result = check_with_serpapi(input_text)
        result['used_api'] = 'serpapi'
        return result
    except Exception as e:
        log_event(f"SerpAPI failed: {e}")
        try:
            log_event("Falling back to Google CSE")
            result = check_with_google_cse(input_text)
            result['used_api'] = 'google_cse'
            return result
        except Exception as ge:
            log_event(f"Google CSE also failed: {ge}")
            return {
                'copied': False,
                'sources': [],
                'percentage_copied': 0.0,
                'used_api': 'none',
                'error': 'Both APIs failed'
            }

def check_with_serpapi(input_text):
    SEARCH_URL = 'https://serpapi.com/search'
    tokenizer = PunktSentenceTokenizer()
    sentences = tokenizer.tokenize(input_text)

    chunk_size = 10
    copied = False
    matches = []
    seen_sources = set()
    seen_snippets = set()
    total_matched_length = 0

    for i in range(0, len(sentences), chunk_size):
        chunk = ' '.join(sentences[i:i + chunk_size])
        if not chunk.strip():
            continue

        params = {
            'api_key': SERP_API_KEY,
            'engine': 'google',
            'q': chunk,
            'num': 5,
        }

        log_event(f"[SerpAPI] Querying chunk {i // chunk_size + 1}")
        response = requests.get(SEARCH_URL, params=params)

        if response.status_code != 200:
            raise Exception(f"SerpAPI HTTP error: {response.status_code}")
        results = response.json()

        if 'error' in results:
            raise Exception(f"SerpAPI error: {results['error']}")

        if 'organic_results' in results:
            for item in results['organic_results']:
                snippet = item.get('snippet', '').strip()
                source = item.get('link', '').strip()

                if snippet and snippet not in seen_snippets and source not in seen_sources:
                    copied = True
                    matches.append({
                        'matching_text': snippet,
                        'source': source
                    })
                    total_matched_length += len(snippet)
                    seen_snippets.add(snippet)
                    seen_sources.add(source)

        time.sleep(1.5)

    percentage_copied = (total_matched_length / len(input_text)) * 100 if len(input_text) > 0 else 0
    return {
        'copied': copied,
        'sources': matches,
        'percentage_copied': min(round(percentage_copied, 2), 100)
    }

def check_with_google_cse(input_text):
    SEARCH_URL = 'https://www.googleapis.com/customsearch/v1'
    tokenizer = PunktSentenceTokenizer()
    sentences = tokenizer.tokenize(input_text)

    chunk_size = 10
    copied = False
    matches = []
    seen_sources = set()
    seen_snippets = set()
    total_matched_length = 0

    for i in range(0, len(sentences), chunk_size):
        chunk = ' '.join(sentences[i:i + chunk_size])
        if not chunk.strip():
            continue

        params = {
            'key': GOOGLE_API_KEY,
            'cx': GOOGLE_CSE_ID,
            'q': chunk,
            'num': 5,
        }

        log_event(f"[Google CSE] Querying chunk {i // chunk_size + 1}")
        response = requests.get(SEARCH_URL, params=params)

        if response.status_code != 200:
            raise Exception(f"Google CSE HTTP error: {response.status_code}")
        results = response.json()

        if 'error' in results:
            raise Exception(f"Google CSE error: {results['error'].get('message', 'Unknown error')}")

        if 'items' in results:
            for item in results['items']:
                snippet = item.get('snippet', '').strip()
                source = item.get('link', '').strip()

                if snippet and snippet not in seen_snippets and source not in seen_sources:
                    copied = True
                    matches.append({
                        'matching_text': snippet,
                        'source': source
                    })
                    total_matched_length += len(snippet)
                    seen_snippets.add(snippet)
                    seen_sources.add(source)

        time.sleep(1.5)

    percentage_copied = (total_matched_length / len(input_text)) * 100 if len(input_text) > 0 else 0
    return {
        'copied': copied,
        'sources': matches,
        'percentage_copied': min(round(percentage_copied, 2), 100)
    }

def clean_text(text):
    """
    Cleans excessive whitespace from text.
    """
    return re.sub(r'\s+', ' ', text).strip()
