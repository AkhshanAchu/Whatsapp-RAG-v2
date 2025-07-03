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
import numpy as np
import torch

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
    """RAG system for Tamil-English (Tanglish) chat data"""
    
    def __init__(self, csv_path: str = None, df: pd.DataFrame = None):
        if df is not None:
            self.df = df.copy()
        elif csv_path:
            self.df = pd.read_csv(csv_path)
        else:
            raise ValueError("Provide either csv_path or df")
        
        self.emo_model = SentenceTransformer("paraphrase-MiniLM-L3-v2").cuda()
        if not hasattr(self, "emo_model"):
            raise RuntimeError("emo_model has not been initialized. Did you run model_builder before setup?")
        self.setup_data()
        self.build_indexes()

    
    def convert_emojis(self,text):
        return emoji.demojize(text)
    
    def model_builder(self):
        return SentenceTransformer("paraphrase-MiniLM-L3-v2").cuda()
    def get_emotion_similarity(self,message):
        emotions = [
            "Joy",
            "Happiness",
            "Sadness",
            "Anger",
            "Love",
            "Fear",
            "Surprise",
            "Excitement",
            "Frustration",
            "Gratitude",
            "Despair",
            "Normal",
        ]
        emotion_embeddings = self.emo_model.encode(emotions)
        message_embedding = self.emo_model.encode([message])
        similarity = self.emo_model.similarity(message_embedding, emotion_embeddings)[0]
        most_similar_emotion_index = np.argmax(similarity)
        return similarity,emotions[most_similar_emotion_index],message_embedding 
    
    def setup_data(self):
        """Setup and clean the data"""
        # Basic cleaning
        self.df['date_time'] = pd.to_datetime(self.df['date_time'])
        self.df = self.df.dropna(subset=['message'])
        self.df['message'] = self.df['message'].astype(str)
        self.df.reset_index(drop=True, inplace=True)
        # Add index column
        self.df['msg_index'] = self.df.index
        
        # Clean text
        self.df['clean_text'] = self.df['message'].apply(self.clean_text)
        
        print(f"Loaded {len(self.df)} messages")
        print(f"Users: {self.df['user'].nunique()}")
        print(f"Date range: {self.df['date_time'].min()} to {self.df['date_time'].max()}")
        self.df[['semantic_sim', 'emotion_sim', 'final_score']] = self.df['clean_text'].apply(self.get_emotion_similarity).apply(pd.Series)
    
    def clean_text(self, text):
        """Basic text cleaning"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove special characters but keep Tamil-English words
        text = re.sub(r'[^\w\s]', ' ', text)
        
        return self.convert_emojis(text)

    def find_top_n_similar_tensors(self, df, column_name, target_tensor, n):
        target_tensor = torch.tensor(target_tensor).float()

        similarities = df[column_name].apply(
            lambda x: torch.nn.functional.cosine_similarity(
                torch.tensor(x).float(), target_tensor, dim=0
            ).item()
        )

        top_n_indices = similarities.nlargest(n).index
        return top_n_indices, similarities.loc[top_n_indices]



    def batch_text_similarity_search(self, query, top_k: int = 15, column_name: str = 'final_score') -> Dict[str, List[ChatResult]]:
        """
        Compute emotion-aware similarity between a single query and all messages.
        Uses cosine similarity between the query embedding and precomputed 'final_score' vectors.
        """
        if not query.strip():
            return []

        # Encode the query using the emotion model (expects a list)
        query_embedding = self.emo_model.encode([query])[0]  # shape: (384,)

        # Convert to tensor
        query_tensor = torch.tensor(query_embedding).float()

        # Compute similarity with all message embeddings in the specified column
        similarities = self.df[column_name].apply(
            lambda x: torch.nn.functional.cosine_similarity(torch.tensor(x[0]).float(), query_tensor, dim=0).item()
        )

        # Get top-k similar messages
        top_indices = similarities.nlargest(top_k).index

        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            results.append(ChatResult(
                index=int(idx),
                user=row['user'],
                date_time=str(row['date_time']),
                message=row['message'],
                message_type=row['message_type'],
                score=float(similarities[idx]),
                method='text_emotion_similarity'
            ))

        return results

    def build_indexes(self):
        """Build search indexes"""
        print("Building indexes...")
        
        # 1. TF-IDF for semantic search
        self.tfidf = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        texts = self.df['clean_text'].fillna('').tolist()
        self.tfidf_matrix = self.tfidf.fit_transform(texts)
        
        # 2. Word frequency index for keywords
        self.word_freq = Counter()
        self.word_to_messages = defaultdict(list)
        
        for idx, text in enumerate(texts):
            words = text.split()
            for word in words:
                if len(word) > 2:  # Skip very short words
                    self.word_freq[word] += 1
                    self.word_to_messages[word].append(idx)
        
        # 3. User stats
        self.user_stats = {}
        for user in self.df['user'].unique():
            user_data = self.df[self.df['user'] == user]
            self.user_stats[user] = {
                'message_count': len(user_data),
                'avg_length': user_data['message'].str.len().mean(),
                'common_words': Counter(' '.join(user_data['clean_text']).split()).most_common(10)
            }
        
        print("Indexes built successfully!")
    
    def semantic_search_optimized(self, query: str, top_k: int = 15) -> List[ChatResult]:
        """Optimized search using semantic similarity and emotion similarity"""
        if not query.strip():
            return []
        query_emotion1, query_emotion2, query_emotion3 = self.get_emotion_similarity(query)
        
        semantic_similarities = self.df['semantic_sim'].apply(
            lambda x: torch.nn.functional.cosine_similarity(
                x.unsqueeze(0) if x is not None else torch.zeros_like(query_emotion1).unsqueeze(0), 
                query_emotion1.unsqueeze(0)
            ).item() if x is not None else 0
        )

        final_scores = semantic_similarities
        # Get top N indices efficiently
        top_indices = final_scores.nlargest(top_k).index
        
        # Build results
        results = []
        for idx in top_indices:
            if final_scores[idx] > 0:  # Only include relevant results
                row = self.df.iloc[idx]
                result = ChatResult(
                    index=int(idx),
                    user=row['user'],
                    date_time=str(row['date_time']),
                    message=row['message'],
                    message_type=row['message_type'],
                    score=float(final_scores[idx]),
                    method='semantic_emotion'
                )
                results.append(result)
        
        return results
    
    def keyword_search(self, keywords: Union[str, List[str]], top_k: int = 15) -> List[ChatResult]:
        """Search using keyword matching"""
        if isinstance(keywords, str):
            keywords = keywords.lower().split()
        else:
            keywords = [k.lower() for k in keywords]
        
        # Find messages containing keywords
        message_scores = defaultdict(float)
        
        for keyword in keywords:
            keyword = keyword.strip()
            if len(keyword) < 2:
                continue
            
            # Exact matches
            if keyword in self.word_to_messages:
                for msg_idx in self.word_to_messages[keyword]:
                    message_scores[msg_idx] += 2.0
            
            # Partial matches
            for word in self.word_to_messages:
                if keyword in word or word in keyword:
                    similarity = len(set(keyword) & set(word)) / max(len(set(keyword) | set(word)), 1)
                    if similarity > 0.5:
                        for msg_idx in self.word_to_messages[word]:
                            message_scores[msg_idx] += similarity
        
        # Sort by score
        sorted_messages = sorted(message_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for msg_idx, score in sorted_messages[:top_k]:
            row = self.df.iloc[msg_idx]
            result = ChatResult(
                index=int(msg_idx),
                user=row['user'],
                date_time=str(row['date_time']),
                message=row['message'],
                message_type=row['message_type'],
                score=float(score),
                method='keyword'
            )
            results.append(result)
        
        return results
    
    def stats_search(self, filters: Dict, top_k: int = 15) -> List[ChatResult]:
        """Search based on statistics and filters"""
        filtered_df = self.df.copy()
        
        # Apply filters
        if 'user' in filters:
            filtered_df = filtered_df[filtered_df['user'].str.contains(filters['user'], case=False, na=False)]
        
        if 'min_length' in filters:
            filtered_df = filtered_df[filtered_df['message'].str.len() >= filters['min_length']]
        
        if 'max_length' in filters:
            filtered_df = filtered_df[filtered_df['message'].str.len() <= filters['max_length']]
        
        if 'message_type' in filters:
            filtered_df = filtered_df[filtered_df['message_type'] == filters['message_type']]
        
        if 'date_from' in filters:
            filtered_df = filtered_df[filtered_df['date_time'] >= filters['date_from']]
        
        if 'date_to' in filters:
            filtered_df = filtered_df[filtered_df['date_time'] <= filters['date_to']]
        
        if 'contains' in filters:
            search_term = filters['contains'].lower()
            filtered_df = filtered_df[filtered_df['clean_text'].str.contains(search_term, case=False, na=False)]
        
        # Score by recency and length
        if len(filtered_df) == 0:
            return []
        
        # Simple scoring: newer messages and longer messages get higher scores
        max_date = filtered_df['date_time'].max()
        filtered_df['days_old'] = (max_date - filtered_df['date_time']).dt.days
        filtered_df['recency_score'] = 1 / (1 + filtered_df['days_old'] / 30)  # Decay over 30 days
        filtered_df['length_score'] = filtered_df['message'].str.len() / 100  # Normalize length
        filtered_df['final_score'] = filtered_df['recency_score'] + filtered_df['length_score']
        
        # Get top results
        top_results = filtered_df.nlargest(top_k, 'final_score')
        
        results = []
        for _, row in top_results.iterrows():
            result = ChatResult(
                index=int(row['msg_index']),
                user=row['user'],
                date_time=str(row['date_time']),
                message=row['message'],
                message_type=row['message_type'],
                score=float(row['final_score']),
                method='stats'
            )
            results.append(result)
        
        return results
    
    def best_search(self, query: str, top_k: int = 15, **kwargs) -> List[ChatResult]:
        """Combined search using multiple methods"""
        all_results = []
        
        # 1. Semantic search
        if query.strip():
            semantic_results = self.semantic_search_optimized(query, top_k)
            all_results.extend(semantic_results)

        if query.strip():
            batch_sim = self.batch_text_similarity_search(query, top_k)
            all_results.extend(batch_sim)
        
        # 2. Keyword search from query
        if query.strip():
            query_words = query.split()
            keyword_results = self.keyword_search(query_words, top_k)
            all_results.extend(keyword_results)
        
        # 3. Stats search if filters provided
        if 'filters' in kwargs and kwargs['filters']:
            # Add query to contains filter if not already there
            filters = kwargs['filters'].copy()
            if query.strip() and 'contains' not in filters:
                filters['contains'] = query
            stats_results = self.stats_search(filters, top_k)
            all_results.extend(stats_results)
        
        # Combine results and re-rank
        if not all_results:
            return []
        
        # Group by message index
        message_groups = defaultdict(list)
        for result in all_results:
            message_groups[result.index].append(result)
        
        # Calculate combined scores
        final_results = []
        for msg_idx, results_list in message_groups.items():
            # Use maximum score from different methods
            best_result = max(results_list, key=lambda x: x.score)
            
            # Boost if found by multiple methods
            method_count = len(set(r.method for r in results_list))
            boost = (method_count - 1) * 0.5
            best_result.score += boost
            best_result.method = 'combined'
            
            final_results.append(best_result)
        
        # Sort by final score and return top_k
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:top_k+top_k]
    
    def get_user_messages(self, username: str, limit: int = 10) -> List[ChatResult]:
        """Get recent messages from a specific user"""
        user_data = self.df[self.df['user'].str.contains(username, case=False, na=False)]
        user_data = user_data.sort_values('date_time', ascending=False).head(limit)
        
        results = []
        for _, row in user_data.iterrows():
            result = ChatResult(
                index=int(row['msg_index']),
                user=row['user'],
                date_time=str(row['date_time']),
                message=row['message'],
                message_type=row['message_type'],
                score=1.0,
                method='user_filter'
            )
            results.append(result)
        
        return results
    
    def get_recent_messages(self, hours: int = 24, limit: int = 10) -> List[ChatResult]:
        """Get recent messages within specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_data = self.df[self.df['date_time'] >= cutoff_time]
        recent_data = recent_data.sort_values('date_time', ascending=False).head(limit)
        
        results = []
        for _, row in recent_data.iterrows():
            result = ChatResult(
                index=int(row['msg_index']),
                user=row['user'],
                date_time=str(row['date_time']),
                message=row['message'],
                message_type=row['message_type'],
                score=1.0,
                method='recent'
            )
            results.append(result)
        
        return results
    
    def print_results(self, results: List[ChatResult]):
        """Print search results nicely"""
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
        """Get dataset statistics"""
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
