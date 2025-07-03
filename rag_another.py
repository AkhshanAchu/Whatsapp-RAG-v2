
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Union
import json
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
import emoji
warnings.filterwarnings('ignore')
from sentence_transformers import SentenceTransformer
import torch
from concurrent.futures import ThreadPoolExecutor
import pickle
import os
from functools import lru_cache

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

@dataclass
class ChatResult:
    """Simple result structure"""
    index: int
    user: str
    date_time: str
    message: str
    message_type: str
    score: float
    method: str

class TanglishChatRAG:
    """Optimized RAG system for Tamil-English (Tanglish) chat data"""
    
    def __init__(self, csv_path: str = None, df: pd.DataFrame = None, cache_dir: str = "./cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        if df is not None:
            self.df = df.copy()
        elif csv_path:
            self.df = pd.read_csv(csv_path)
        else:
            raise ValueError("Provide either csv_path or df")
        
        # Initialize model with better device handling
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emo_model = SentenceTransformer("paraphrase-MiniLM-L3-v2", device=self.device)
        
        # Pre-compute emotion vectors for faster lookup
        self.emotions = [
            "Joy", "Happiness", "Sadness", "Anger", "Love", "Fear", 
            "Surprise", "Excitement", "Frustration", "Gratitude", "Despair", "Normal"
        ]
        self.emotion_embeddings = self.emo_model.encode(self.emotions)
        
        self.setup_data()
        self.build_indexes()
    
    @lru_cache(maxsize=1000)
    def convert_emojis(self, text):
        """Cached emoji conversion"""
        return emoji.demojize(text)
    
    def get_emotion_similarity_batch(self, messages: List[str]):
        """Batch process emotion similarity for better performance"""
        message_embeddings = self.emo_model.encode(messages)
        similarities = cosine_similarity(message_embeddings, self.emotion_embeddings)
        most_similar_indices = np.argmax(similarities, axis=1)
        
        return [
            (similarities[i], self.emotions[most_similar_indices[i]], message_embeddings[i])
            for i in range(len(messages))
        ]
    
    def setup_data(self):
        """Setup and clean the data with optimizations"""
        print("Setting up data...")
        
        # Basic cleaning - vectorized operations
        self.df['date_time'] = pd.to_datetime(self.df['date_time'])
        self.df = self.df.dropna(subset=['message'])
        self.df['message'] = self.df['message'].astype(str)
        self.df.reset_index(drop=True, inplace=True)
        self.df['msg_index'] = self.df.index
        
        # Vectorized text cleaning
        self.df['clean_text'] = self.df['message'].apply(self.clean_text_optimized)
        
        print(f"Loaded {len(self.df)} messages")
        print(f"Users: {self.df['user'].nunique()}")
        print(f"Date range: {self.df['date_time'].min()} to {self.df['date_time'].max()}")
        
        # Batch process emotion similarities
        print("Computing emotion similarities...")
        clean_texts = self.df['clean_text'].tolist()
        
        # Process in batches for memory efficiency
        batch_size = 1000
        all_similarities = []
        all_emotions = []
        all_embeddings = []
        
        for i in range(0, len(clean_texts), batch_size):
            batch = clean_texts[i:i+batch_size]
            batch_results = self.get_emotion_similarity_batch(batch)
            
            for sim, emotion, embedding in batch_results:
                all_similarities.append(sim)
                all_emotions.append(emotion)
                all_embeddings.append(embedding)
        
        self.df['emotion_similarities'] = all_similarities
        self.df['dominant_emotion'] = all_emotions
        self.df['message_embedding'] = all_embeddings
        
        print("Data setup complete!")
    
    def clean_text_optimized(self, text):
        """Optimized text cleaning with better regex"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        # Combined regex operations
        text = re.sub(r'http\S+|www\S+|https\S+|\s+|[^\w\s]', lambda m: '' if m.group().startswith(('http', 'www')) else ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return self.convert_emojis(text)
    
    def build_indexes(self):
        """Build optimized search indexes"""
        print("Building optimized indexes...")
        
        # Check for cached indexes
        cache_file = os.path.join(self.cache_dir, "search_indexes.pkl")
        if os.path.exists(cache_file):
            print("Loading cached indexes...")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.tfidf = cache_data['tfidf']
                self.tfidf_matrix = cache_data['tfidf_matrix']
                self.word_freq = cache_data['word_freq']
                self.word_to_messages = cache_data['word_to_messages']
                self.user_stats = cache_data['user_stats']
                print("Cached indexes loaded!")
                return
        
        # 1. Optimized TF-IDF
        self.tfidf = TfidfVectorizer(
            max_features=10000,  # Reduced for speed
            ngram_range=(1, 3),
            min_df=2,  # Increased to reduce noise
            max_df=0.9
        )
        
        texts = self.df['clean_text'].fillna('').tolist()
        self.tfidf_matrix = self.tfidf.fit_transform(texts)
        
        # 2. Optimized word frequency index
        self.word_freq = Counter()
        self.word_to_messages = defaultdict(set)  # Use set for faster lookups
        
        for idx, text in enumerate(texts):
            words = set(text.split())  # Use set to avoid duplicates
            for word in words:
                if len(word) > 2:
                    self.word_freq[word] += 1
                    self.word_to_messages[word].add(idx)
        
        # Convert sets to lists for serialization
        self.word_to_messages = {k: list(v) for k, v in self.word_to_messages.items()}
        
        # 3. Optimized user stats
        self.user_stats = {}
        for user in self.df['user'].unique():
            user_data = self.df[self.df['user'] == user]
            self.user_stats[user] = {
                'message_count': len(user_data),
                'avg_length': user_data['message'].str.len().mean(),
                'common_words': Counter(' '.join(user_data['clean_text']).split()).most_common(10)
            }
        
        # Cache the indexes
        cache_data = {
            'tfidf': self.tfidf,
            'tfidf_matrix': self.tfidf_matrix,
            'word_freq': self.word_freq,
            'word_to_messages': self.word_to_messages,
            'user_stats': self.user_stats
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print("Optimized indexes built and cached!")
    
    def semantic_search_optimized(self, query: str, top_k: int = 15) -> List[ChatResult]:
        """Ultra-fast semantic search using precomputed embeddings"""
        if not query.strip():
            return []
        
        # Encode query once
        query_embedding = self.emo_model.encode([query])[0]
        
        # Vectorized similarity computation
        embeddings_matrix = np.vstack(self.df['message_embedding'].values)
        similarities = cosine_similarity([query_embedding], embeddings_matrix)[0]
        
        # Get top-k indices efficiently
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Threshold for relevance
                row = self.df.iloc[idx]
                results.append(ChatResult(
                    index=int(idx),
                    user=row['user'],
                    date_time=str(row['date_time']),
                    message=row['message'],
                    message_type=row['message_type'],
                    score=float(similarities[idx]),
                    method='semantic_optimized'
                ))
        
        return results
    
    def keyword_search_optimized(self, keywords: Union[str, List[str]], top_k: int = 15) -> List[ChatResult]:
        """Optimized keyword search"""
        if isinstance(keywords, str):
            keywords = keywords.lower().split()
        else:
            keywords = [k.lower() for k in keywords]
        
        # Use sets for faster operations
        message_scores = defaultdict(float)
        
        for keyword in keywords:
            keyword = keyword.strip()
            if len(keyword) < 2:
                continue
            
            # Exact matches
            if keyword in self.word_to_messages:
                for msg_idx in self.word_to_messages[keyword]:
                    message_scores[msg_idx] += 2.0
            
            # Optimized partial matches using set operations
            matching_words = [word for word in self.word_to_messages.keys() 
                            if keyword in word or word in keyword]
            
            for word in matching_words:
                if word != keyword:  # Avoid double counting
                    similarity = len(set(keyword) & set(word)) / max(len(set(keyword) | set(word)), 1)
                    if similarity > 0.5:
                        for msg_idx in self.word_to_messages[word]:
                            message_scores[msg_idx] += similarity * 0.8
        
        # Get top results efficiently
        if not message_scores:
            return []
        
        top_items = sorted(message_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for msg_idx, score in top_items:
            row = self.df.iloc[msg_idx]
            results.append(ChatResult(
                index=int(msg_idx),
                user=row['user'],
                date_time=str(row['date_time']),
                message=row['message'],
                message_type=row['message_type'],
                score=float(score),
                method='keyword_optimized'
            ))
        
        return results
    
    def tfidf_search_optimized(self, query: str, top_k: int = 15) -> List[ChatResult]:
        """Fast TF-IDF search"""
        if not query.strip():
            return []
        
        query_vec = self.tfidf.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        # Get top-k indices efficiently
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.01:  # Minimum relevance threshold
                row = self.df.iloc[idx]
                results.append(ChatResult(
                    index=int(idx),
                    user=row['user'],
                    date_time=str(row['date_time']),
                    message=row['message'],
                    message_type=row['message_type'],
                    score=float(similarities[idx]),
                    method='tfidf_optimized'
                ))
        
        return results
    
    def best_search_optimized(self, query: str, top_k: int = 15, **kwargs) -> List[ChatResult]:
        """Optimized combined search with parallel processing"""
        if not query.strip():
            return []
        
        # Use ThreadPoolExecutor for parallel search
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            
            # Submit search tasks
            futures.append(executor.submit(self.semantic_search_optimized, query, top_k))
            futures.append(executor.submit(self.keyword_search_optimized, query.split(), top_k))
            futures.append(executor.submit(self.tfidf_search_optimized, query, top_k))
            
            # Collect results
            all_results = []
            for future in futures:
                try:
                    results = future.result(timeout=10)  # 10 second timeout
                    all_results.extend(results)
                except Exception as e:
                    print(f"Search method failed: {e}")
                    continue
        
        if not all_results:
            return []
        
        # Optimized result combination using dict for O(1) lookup
        message_groups = {}
        for result in all_results:
            if result.index not in message_groups:
                message_groups[result.index] = []
            message_groups[result.index].append(result)
        
        # Calculate combined scores efficiently
        final_results = []
        for msg_idx, results_list in message_groups.items():
            # Weight different methods
            method_weights = {
                'semantic_optimized': 1.0,
                'keyword_optimized': 0.8,
                'tfidf_optimized': 0.6
            }
            
            # Calculate weighted average score
            total_score = 0
            total_weight = 0
            methods = set()
            
            for result in results_list:
                weight = method_weights.get(result.method, 0.5)
                total_score += result.score * weight
                total_weight += weight
                methods.add(result.method)
            
            if total_weight > 0:
                avg_score = total_score / total_weight
                # Boost for multiple methods
                method_boost = len(methods) * 0.1
                final_score = avg_score + method_boost
                
                # Use the best result as template
                best_result = max(results_list, key=lambda x: x.score)
                best_result.score = final_score
                best_result.method = 'combined_optimized'
                final_results.append(best_result)
        
        # Sort and return top results
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:top_k]
    
    def get_user_messages(self, username: str, limit: int = 10) -> List[ChatResult]:
        """Optimized user message retrieval"""
        # Use vectorized string operations
        mask = self.df['user'].str.contains(username, case=False, na=False)
        user_data = self.df[mask].nlargest(limit, 'date_time')
        
        return [
            ChatResult(
                index=int(row['msg_index']),
                user=row['user'],
                date_time=str(row['date_time']),
                message=row['message'],
                message_type=row['message_type'],
                score=1.0,
                method='user_filter'
            )
            for _, row in user_data.iterrows()
        ]
    
    def get_recent_messages(self, hours: int = 24, limit: int = 10) -> List[ChatResult]:
        """Optimized recent message retrieval"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        mask = self.df['date_time'] >= cutoff_time
        recent_data = self.df[mask].nlargest(limit, 'date_time')
        
        return [
            ChatResult(
                index=int(row['msg_index']),
                user=row['user'],
                date_time=str(row['date_time']),
                message=row['message'],
                message_type=row['message_type'],
                score=1.0,
                method='recent'
            )
            for _, row in recent_data.iterrows()
        ]
    
    def print_results(self, results: List[ChatResult]):
        """Print search results efficiently"""
        if not results:
            print("No results found!")
            return
        
        print(f"\n=== Found {len(results)} Results ===")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. [{result.method.upper()}] Score: {result.score:.3f}")
            print(f"   User: {result.user}")
            print(f"   Time: {result.date_time}")
            print(f"   Type: {result.message_type}")
            print(f"   Message: {result.message}")
            print(f"   {'='*60}")
    
    def get_stats(self):
        """Get dataset statistics efficiently"""
        stats = {
            'total_messages': len(self.df),
            'unique_users': self.df['user'].nunique(),
            'date_range': f"{self.df['date_time'].min()} to {self.df['date_time'].max()}",
            'message_types': self.df['message_type'].value_counts().to_dict(),
            'avg_message_length': self.df['message'].str.len().mean(),
            'top_users': self.df['user'].value_counts().head(5).to_dict(),
            'most_common_words': [word for word, count in self.word_freq.most_common(20)]
        }
        return stats
    
    def clear_cache(self):
        """Clear cached indexes"""
        cache_file = os.path.join(self.cache_dir, "search_indexes.pkl")
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print("Cache cleared!")

# Example usage and performance test
def performance_test():
    """Test performance improvements"""
    import time
    
    # Create sample data
    sample_data = pd.DataFrame({
        'user': ['User1', 'User2', 'User3'] * 1000,
        'date_time': pd.date_range('2024-01-01', periods=3000, freq='H'),
        'message': ['Hello world', 'How are you', 'Good morning'] * 1000,
        'message_type': ['text'] * 3000
    })
    
    print("Testing optimized RAG performance...")
    start_time = time.time()
    
    rag = TanglishChatRAG(df=sample_data)
    setup_time = time.time() - start_time
    
    print(f"Setup time: {setup_time:.2f} seconds")
    
    # Test search performance
    start_time = time.time()
    results = rag.best_search_optimized("hello world", top_k=10)
    search_time = time.time() - start_time
    print(results)
    print(f"Search time: {search_time:.3f} seconds")
    print(f"Found {len(results)} results")
    
    return rag

if __name__ == "__main__":
    performance_test()