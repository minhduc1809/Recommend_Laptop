import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import sys
import io
import warnings
import os
csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'laptop_prices_cleaned.csv'))

# Cấu hình encoding và warning
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except:
    pass
warnings.filterwarnings('ignore')

# Download NLTK data với error handling
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt')
        except:
            print("Warning: Could not download punkt tokenizer")
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('stopwords')
        except:
            print("Warning: Could not download stopwords")

download_nltk_data()

class LaptopRecommendationChatbot:
    def __init__(self, csv_path=None):
        """
        Khởi tạo chatbot với dữ liệu laptop
        """
        self.df = None
        self.tfidf_vectorizer = None
        self.feature_matrix = None
        self.scaler = StandardScaler()
        self.price_model = None
        self.stemmer = PorterStemmer()
        
        # Khởi tạo stop words với fallback
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
                             'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
                             'to', 'was', 'will', 'with'}
        
        # Load data if provided
        self.load_data(csv_path)
        self._preprocess_data()
        self._build_content_based_model()
        self._build_price_prediction_model()
    def load_data(self, csv_path):
        """Load dữ liệu từ file CSV"""
        try:
            self.df = pd.read_csv(csv_path)
            print(f"Đã load {len(self.df)} laptop từ {csv_path}")
        except Exception as e:
            print(f"Lỗi khi load data: {e}")
            self._create_sample_data()
    
    def _preprocess_data(self):
        """Tiền xử lý dữ liệu"""
        if self.df is None or self.df.empty:
            print("Không có dữ liệu để xử lý!")
            return
            
        # Xử lý missing values
        self.df = self.df.fillna('Unknown')
        
        # Tạo cột mô tả tổng hợp cho content-based filtering
        self.df['description'] = (
            self.df['brand'].astype(str) + ' ' +
            self.df['model'].astype(str) + ' ' +
            self.df['cpu'].astype(str) + ' ' +
            self.df['ram'].astype(str) + ' ' +
            self.df['os'].astype(str) + ' ' +
            self.df['special_features'].astype(str) + ' ' +
            self.df['graphics'].astype(str) + ' ' +
            self.df['performance_tier'].astype(str)
        )
        
        # Chuyển đổi các cột số
        numeric_columns = ['rating', 'price']
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Xử lý RAM và Storage với regex cải tiến
        self.df['ram_gb'] = self.df['ram'].str.extract(r'(\d+)').fillna(8).astype(float)
        self.df['storage_gb'] = self.df['harddisk'].str.extract(r'(\d+)').fillna(500).astype(float)
        
        print("Đã hoàn thành tiền xử lý dữ liệu")
    
    def _preprocess_text(self, text):
        """Tiền xử lý văn bản cho NLP với fallback"""
        try:
            text = str(text).lower()
            
            # Simple tokenization nếu NLTK không hoạt động
            try:
                tokens = word_tokenize(text)
            except:
                # Fallback tokenization
                tokens = re.findall(r'\b\w+\b', text)
            
            # Loại bỏ stopwords và stemming
            processed_tokens = []
            for token in tokens:
                if token.isalnum() and token not in self.stop_words:
                    try:
                        stemmed = self.stemmer.stem(token)
                        processed_tokens.append(stemmed)
                    except:
                        processed_tokens.append(token)
            
            return ' '.join(processed_tokens)
            
        except Exception as e:
            print(f"Lỗi trong _preprocess_text: {e}")
            return str(text).lower()
    
    def _build_content_based_model(self):
        """Xây dựng mô hình Content-Based Filtering sử dụng TF-IDF"""
        try:
            # Tiền xử lý văn bản
            processed_descriptions = [self._preprocess_text(desc) for desc in self.df['description']]
            
            # TF-IDF Vectorization
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                stop_words='english'
            )
            
            self.feature_matrix = self.tfidf_vectorizer.fit_transform(processed_descriptions)
            print("Đã xây dựng mô hình Content-Based với TF-IDF")
        except Exception as e:
            print(f"Lỗi khi xây dựng mô hình TF-IDF: {e}")
    
    def _build_price_prediction_model(self):
        """Xây dựng mô hình dự đoán giá sử dụng Random Forest"""
        try:
            # Chuẩn bị features cho price prediction
            X_numeric = pd.DataFrame()
            
            # Encode categorical features
            categorical_cols = ['brand', 'cpu', 'os', 'graphics', 'performance_tier']
            
            for col in categorical_cols:
                if col in self.df.columns:
                    le = LabelEncoder()
                    X_numeric[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
            
            # Add numeric features
            numeric_cols = ['rating', 'ram_gb', 'storage_gb']
            for col in numeric_cols:
                if col in self.df.columns:
                    X_numeric[col] = self.df[col].fillna(self.df[col].median())
            
            if len(X_numeric.columns) > 0 and 'price' in self.df.columns:
                X = X_numeric
                y = self.df['price'].fillna(self.df['price'].median())
                
                # Train model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                self.price_model = RandomForestRegressor(n_estimators=100, random_state=42)
                self.price_model.fit(X_train, y_train)
                
                # Evaluate
                score = self.price_model.score(X_test, y_test)
                print(f"Mô hình dự đoán giá - R² Score: {score:.3f}")
        except Exception as e:
            print(f"Lỗi khi xây dựng mô hình giá: {e}")
    
    def _extract_requirements(self, user_input):
        """Trích xuất yêu cầu từ input của user sử dụng NLP"""
        user_input = user_input.lower()
        requirements = {}
        
        # Budget extraction
        budget_patterns = [
            r'under (\d+)', r'below (\d+)', r'less than (\d+)',
            r'around (\d+)', r'about (\d+)', r'(\d+) dollar', r'\$(\d+)',
            r'dưới (\d+)', r'khoảng (\d+)', r'tầm (\d+)'
        ]
        
        for pattern in budget_patterns:
            match = re.search(pattern, user_input)
            if match:
                requirements['max_budget'] = int(match.group(1))
                break
        
        # Usage purpose
        if any(word in user_input for word in ['gaming', 'game', 'gamer', 'chơi game']):
            requirements['purpose'] = 'gaming'
        elif any(word in user_input for word in ['business', 'work', 'office', 'làm việc', 'văn phòng']):
            requirements['purpose'] = 'business'
        elif any(word in user_input for word in ['student', 'study', 'school', 'học sinh', 'sinh viên']):
            requirements['purpose'] = 'student'
        elif any(word in user_input for word in ['design', 'creative', 'video', 'photo', 'thiết kế']):
            requirements['purpose'] = 'creative'
        
        # Brand preference
        brands = ['apple', 'dell', 'hp', 'asus', 'lenovo', 'acer', 'msi']
        for brand in brands:
            if brand in user_input:
                requirements['brand'] = brand.title()
                break
        
        # Screen size
        if any(word in user_input for word in ['small', 'compact', 'portable', 'nhỏ', 'gọn']):
            requirements['screen_preference'] = 'small'
        elif any(word in user_input for word in ['large', 'big', '17', 'lớn', 'to']):
            requirements['screen_preference'] = 'large'
        
        # Performance
        if any(word in user_input for word in ['high performance', 'powerful', 'fast', 'mạnh', 'nhanh']):
            requirements['performance'] = 'high'
        elif any(word in user_input for word in ['budget', 'cheap', 'affordable', 'rẻ', 'giá rẻ']):
            requirements['performance'] = 'budget'
        
        return requirements
    
    def _filter_by_requirements(self, requirements):
        """Lọc laptop theo yêu cầu của user"""
        filtered_df = self.df.copy()
        
        # Budget filter
        if 'max_budget' in requirements:
            filtered_df = filtered_df[filtered_df['price'] <= requirements['max_budget']]
        
        # Brand filter
        if 'brand' in requirements:
            filtered_df = filtered_df[filtered_df['brand'].str.contains(requirements['brand'], case=False, na=False)]
        
        # Purpose-based filtering
        if 'purpose' in requirements:
            purpose = requirements['purpose']
            if purpose == 'gaming':
                filtered_df = filtered_df[
                    (filtered_df['graphics'] == 'Dedicated') |
                    (filtered_df['special_features'].str.contains('gaming', case=False, na=False))
                ]
            elif purpose == 'business':
                filtered_df = filtered_df[
                    (filtered_df['special_features'].str.contains('business|fingerprint', case=False, na=False)) |
                    (filtered_df['brand'].isin(['Dell', 'Lenovo', 'HP']))
                ]
            elif purpose == 'student':
                filtered_df = filtered_df[filtered_df['price'] <= 800]
            elif purpose == 'creative':
                filtered_df = filtered_df[
                    (filtered_df['ram_gb'] >= 16) &
                    (filtered_df['performance_tier'].isin(['High-End', 'Mid-Range']))
                ]
        
        # Screen size preference
        if 'screen_preference' in requirements:
            try:
                screen_sizes = filtered_df['screen_size'].str.extract(r'(\d+\.?\d*)').astype(float)[0]
                if requirements['screen_preference'] == 'small':
                    filtered_df = filtered_df[screen_sizes <= 14]
                elif requirements['screen_preference'] == 'large':
                    filtered_df = filtered_df[screen_sizes >= 15.6]
            except:
                pass
        
        # Performance filter
        if 'performance' in requirements:
            if requirements['performance'] == 'high':
                filtered_df = filtered_df[filtered_df['performance_tier'] == 'High-End']
            elif requirements['performance'] == 'budget':
                filtered_df = filtered_df[filtered_df['performance_tier'].isin(['Budget', 'Entry-Level'])]
        
        return filtered_df
    
    def get_content_based_recommendations(self, query, top_k=5):
        """Gợi ý sử dụng Content-Based Filtering"""
        try:
            if self.tfidf_vectorizer is None or self.feature_matrix is None:
                return []
                
            # Xử lý query
            processed_query = self._preprocess_text(query)
            query_vector = self.tfidf_vectorizer.transform([processed_query])
            
            # Tính similarity
            similarities = cosine_similarity(query_vector, self.feature_matrix).flatten()
            
            # Lấy top recommendations
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            recommendations = []
            for idx in top_indices:
                if similarities[idx] > 0:  # Chỉ lấy những cái có similarity > 0
                    laptop = self.df.iloc[idx]
                    recommendations.append({
                        'laptop': laptop,
                        'similarity_score': similarities[idx],
                        'reason': f"Độ tương đồng: {similarities[idx]:.3f}"
                    })
            
            return recommendations
        except Exception as e:
            print(f"Lỗi trong get_content_based_recommendations: {e}")
            return []
    
    def predict_price_range(self, requirements):
        """Dự đoán khoảng giá dựa trên yêu cầu"""
        try:
            avg_price = self.df['price'].mean()
            
            if 'purpose' in requirements:
                purpose = requirements['purpose']
                if purpose == 'gaming':
                    return f"Khoảng giá dự kiến: ${avg_price * 1.5:,.0f} - ${avg_price * 2.5:,.0f}"
                elif purpose == 'business':
                    return f"Khoảng giá dự kiến: ${avg_price * 0.8:,.0f} - ${avg_price * 1.5:,.0f}"
                elif purpose == 'student':
                    return f"Khoảng giá dự kiến: ${avg_price * 0.5:,.0f} - ${avg_price:,.0f}"
            
            return f"Khoảng giá dự kiến: ${avg_price * 0.7:,.0f} - ${avg_price * 1.3:,.0f}"
        except:
            return "Không thể dự đoán giá"
    
    def chat(self, user_input):
        """Main chat function - xử lý input và trả về gợi ý"""
        print(f"\n👤 User: {user_input}")
        
        # Xử lý các câu hỏi chung
        user_input_lower = user_input.lower()
        
        if any(greeting in user_input_lower for greeting in ['hello', 'hi', 'chào', 'xin chào']):
            return """🤖 Chatbot: Xin chào! Tôi là chatbot gợi ý laptop. 
Tôi có thể giúp bạn tìm laptop phù hợp dựa trên:
- Ngân sách của bạn
- Mục đích sử dụng (gaming, học tập, làm việc, sáng tạo)
- Thương hiệu yêu thích
- Kích thước màn hình
- Yêu cầu về hiệu năng

Hãy cho tôi biết bạn đang tìm laptop như thế nào nhé!"""
        
        if 'help' in user_input_lower or 'giúp' in user_input_lower:
            return """🤖 Chatbot: Tôi có thể giúp bạn:
1. Tìm laptop theo ngân sách: "Tôi cần laptop dưới $1000"
2. Tìm laptop theo mục đích: "Laptop để gaming" hoặc "Laptop cho học sinh"
3. So sánh laptop: "So sánh MacBook với Dell XPS"
4. Dự đoán giá: "Giá laptop gaming khoảng bao nhiêu?"

Hãy mô tả yêu cầu cụ thể của bạn!"""
        
        try:
            # Trích xuất yêu cầu
            requirements = self._extract_requirements(user_input)
            
            # Lọc laptop theo yêu cầu
            filtered_laptops = self._filter_by_requirements(requirements)
            
            if len(filtered_laptops) == 0:
                return "🤖 Chatbot: Xin lỗi, tôi không tìm thấy laptop nào phù hợp với yêu cầu của bạn. Bạn có thể thử mở rộng tiêu chí không?"
            
            # Lấy gợi ý content-based
            content_recommendations = self.get_content_based_recommendations(user_input, top_k=3)
            
            # Tạo response
            response = "🤖 Chatbot: Dựa trên yêu cầu của bạn, tôi gợi ý những laptop sau:\n\n"
            
            # Hiển thị dự đoán giá nếu có
            if requirements:
                price_prediction = self.predict_price_range(requirements)
                response += f"💰 {price_prediction}\n\n"
            
            # Hiển thị top recommendations
            if content_recommendations:
                for i, rec in enumerate(content_recommendations[:3], 1):
                    laptop = rec['laptop']
                    response += f"📱 **{i}. {laptop['brand']} {laptop['model']}**\n"
                    response += f"   • CPU: {laptop['cpu']}\n"
                    response += f"   • RAM: {laptop['ram']}\n"
                    response += f"   • Storage: {laptop['harddisk']}\n"
                    response += f"   • Screen: {laptop['screen_size']}\n"
                    response += f"   • Price: ${laptop['price']:,.0f}\n"
                    response += f"   • Rating: {laptop['rating']}/5.0\n"
                    response += f"   • {rec['reason']}\n\n"
            else:
                # Nếu không có content-based recommendations, hiển thị filtered results
                for i, (idx, laptop) in enumerate(filtered_laptops.head(3).iterrows(), 1):
                    response += f"📱 **{i}. {laptop['brand']} {laptop['model']}**\n"
                    response += f"   • CPU: {laptop['cpu']}\n"
                    response += f"   • RAM: {laptop['ram']}\n"
                    response += f"   • Storage: {laptop['harddisk']}\n"
                    response += f"   • Price: ${laptop['price']:,.0f}\n"
                    response += f"   • Rating: {laptop['rating']}/5.0\n\n"
            
            # Thêm thông tin hữu ích
            if 'max_budget' in requirements:
                budget_options = filtered_laptops[filtered_laptops['price'] <= requirements['max_budget']]
                response += f"📊 Tôi tìm thấy {len(budget_options)} laptop trong ngân sách ${requirements['max_budget']:,} của bạn.\n"
            
            response += "\n❓ Bạn có muốn tôi giải thích thêm về laptop nào không? Hoặc bạn có tiêu chí khác cần xem xét?"
            
            return response
            
        except Exception as e:
            print(f"Lỗi trong chat function: {e}")
            return "🤖 Chatbot: Xin lỗi, có lỗi xảy ra. Bạn có thể thử lại với câu hỏi khác không?"
    
    def get_laptop_details(self, laptop_name):
        """Lấy thông tin chi tiết về laptop"""
        try:
            laptop_matches = self.df[
                self.df['model'].str.contains(laptop_name, case=False, na=False) |
                self.df['brand'].str.contains(laptop_name, case=False, na=False)
            ]
            
            if len(laptop_matches) == 0:
                return "Không tìm thấy laptop này trong database."
            
            laptop = laptop_matches.iloc[0]
            details = f"""
📱 **Chi tiết {laptop['brand']} {laptop['model']}**

🔧 **Thông số kỹ thuật:**
• CPU: {laptop['cpu']}
• RAM: {laptop['ram']}
• Storage: {laptop['harddisk']}
• Graphics: {laptop['graphics']}
• Screen: {laptop['screen_size']}
• OS: {laptop['os']}
• Color: {laptop['color']}

💰 **Giá cả & Đánh giá:**
• Price: ${laptop['price']:,.0f}
• Rating: {laptop['rating']}/5.0
• Performance Tier: {laptop['performance_tier']}

⭐ **Tính năng đặc biệt:**
{laptop['special_features']}

🎯 **Phù hợp cho:** {self._get_suitable_usage(laptop)}
            """
            return details
        except Exception as e:
            return f"Lỗi khi lấy thông tin laptop: {e}"
    
    def _get_suitable_usage(self, laptop):
        """Xác định laptop phù hợp cho mục đích gì"""
        usages = []
        
        try:
            if laptop['graphics'] == 'Dedicated' or 'gaming' in str(laptop['special_features']).lower():
                usages.append("Gaming")
            
            if laptop['performance_tier'] in ['High-End', 'Mid-Range'] and laptop['ram_gb'] >= 16:
                usages.append("Creative Work")
            
            if 'business' in str(laptop['special_features']).lower() or laptop['brand'] in ['Dell', 'Lenovo', 'HP']:
                usages.append("Business")
            
            if laptop['price'] <= 600:
                usages.append("Students")
            
            if not usages:
                usages.append("General Use")
            
            return ", ".join(usages)
        except:
            return "General Use"

# Demo function
def run_chatbot_demo():
    """Chạy demo chatbot"""
    print("=== LAPTOP RECOMMENDATION CHATBOT ===")
    print("Khởi tạo chatbot...")
    
    try:
        # Khởi tạo chatbot (sử dụng sample data)
        chatbot = LaptopRecommendationChatbot()
        
        print("✅ Chatbot đã sẵn sàng!")
        print("Gõ 'quit' để thoát\n")
        
        # Sample conversations
        sample_queries = [
            "Hello! Tôi cần laptop để gaming dưới $1500",
            "Laptop nào tốt cho học sinh với ngân sách $800?",
            "Tôi muốn MacBook để làm việc",
            "help"
        ]
        
        print("=== DEMO CONVERSATIONS ===")
        for query in sample_queries:
            response = chatbot.chat(query)
            print(response)
            print("-" * 80)
            
        print("\n=== INTERACTIVE MODE ===")
        while True:
            try:
                user_input = input("\n👤 You: ")
                if user_input.lower() in ['quit', 'exit', 'bye', 'thoát']:
                    print("🤖 Chatbot: Cảm ơn bạn đã sử dụng! Chúc bạn tìm được laptop ưng ý!")
                    break
                
                response = chatbot.chat(user_input)
                print(response)
            except KeyboardInterrupt:
                print("\n🤖 Chatbot: Cảm ơn bạn đã sử dụng!")
                break
            except Exception as e:
                print(f"Lỗi: {e}")
                
    except Exception as e:
        print(f"Lỗi khởi tạo chatbot: {e}")

if __name__ == "__main__":
    run_chatbot_demo()
