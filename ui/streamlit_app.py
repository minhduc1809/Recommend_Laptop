import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from textblob import TextBlob
try:
    from laptop_recommender import LaptopRecommendationChatbot
except ImportError:
    st.error("Cannot import LaptopRecommendationChatbot. Please check the path and module.")
    st.stop()

# Cấu hình trang
st.set_page_config(
    page_title="Laptop Recommendation Chatbot",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 5px solid #9c27b0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

def analyze_sentiment(text):
    """Simple sentiment analysis using TextBlob"""
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            return "Tích cực 😊"
        elif polarity < -0.1:
            return "Tiêu cực 😞"
        else:
            return "Trung tính 😐"
    except:
        return "Trung tính 😐"

@st.cache_resource
def load_chatbot():
    """Load chatbot với caching"""
    try:
        return LaptopRecommendationChatbot()
    except Exception as e:
        st.error(f"Error loading chatbot: {str(e)}")
        return None

@st.cache_data
def load_data():
    """Load và cache dữ liệu"""
    chatbot = load_chatbot()
    if chatbot is not None:
        return chatbot.df
    else:
        # Return dummy data if chatbot fails to load
        return pd.DataFrame({
            'name': ['Sample Laptop'],
            'brand': ['Sample'],
            'price': [1000000],
            'performance_tier': ['Mid'],
            'rating': [4.0],
            'cpu': ['Intel i5'],
            'ram': [8],
            'storage': [256],
            'screen': [15.6],
            'weight': [2.0],
            'battery': [8],
            'description': ['Sample laptop for testing']
        })

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'laptop_data' not in st.session_state:
        st.session_state.laptop_data = load_data()

def main():
    # Initialize session state
    initialize_session_state()
    
    st.markdown('<h1 class="main-header">💻 Laptop Recommendation Chatbot</h1>', unsafe_allow_html=True)
    
    # Load chatbot and data
    chatbot = load_chatbot()
    df = st.session_state.laptop_data
    
    if df is None or df.empty:
        st.error("Cannot load data. Please check your data source.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 Quick Filters")
        
        # Budget filter
        try:
            min_price = int(df['price'].min())
            max_price = int(df['price'].max())
            budget_range = st.slider(
                "Budget Range (VND)",
                min_value=min_price,
                max_value=max_price,
                value=(min_price, max_price),
                format="%d"
            )
        except:
            budget_range = (500000, 50000000)
        
        # Brand filter
        try:
            brands = ['All'] + sorted(df['brand'].unique().tolist())
            selected_brand = st.selectbox("Brand", brands)
        except:
            selected_brand = 'All'
        
        # Performance filter
        try:
            performance_tiers = ['All'] + sorted(df['performance_tier'].unique().tolist())
            selected_performance = st.selectbox("Performance Tier", performance_tiers)
        except:
            selected_performance = 'All'
        
        # Purpose
        st.markdown("### 🎮 Purpose")
        purpose_options = {
            "Any": "any",
            "Gaming": "gaming", 
            "Business": "business",
            "Student": "student",
            "Creative Work": "creative"
        }
        selected_purpose = st.radio("Select Purpose", list(purpose_options.keys()))
        
        st.markdown("---")
        st.markdown("### 📊 Dataset Info")
        st.metric("Total Laptops", len(df))
        try:
            st.metric("Price Range", f"{df['price'].min()/1000000:.1f}M - {df['price'].max()/1000000:.1f}M VND")
            st.metric("Avg Rating", f"{df['rating'].mean():.1f}/5.0")
        except:
            st.metric("Price Range", "N/A")
            st.metric("Avg Rating", "N/A")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat interface
        st.subheader("💬 Trò chuyện với AI")
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Hãy mô tả nhu cầu sử dụng laptop của bạn..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Process and respond
            with st.chat_message("assistant"):
                with st.spinner("Đang phân tích và tìm kiếm..."):
                    try:
                        # Analyze sentiment
                        sentiment = analyze_sentiment(prompt)
                        
                        # Extract price information from prompt
                        price_match = re.findall(r'(\d+)\s*(?:triệu|tr|million)', prompt.lower())
                        max_price = None
                        if price_match:
                            max_price = int(price_match[0]) * 1000000
                        else:
                            max_price = budget_range[1]
                        
                        # Filter data based on sidebar
                        filtered_data = df.copy()
                        
                        if selected_brand != 'All':
                            filtered_data = filtered_data[filtered_data['brand'] == selected_brand]
                        
                        filtered_data = filtered_data[
                            (filtered_data['price'] >= budget_range[0]) &
                            (filtered_data['price'] <= budget_range[1])
                        ]
                        
                        if len(filtered_data) > 0:
                            # Simple recommendation based on price and rating
                            top_laptops = filtered_data.nlargest(3, ['rating', 'price'])
                            
                            response = f"**Cảm xúc của bạn:** {sentiment}\n\n"
                            response += f"**Dựa trên yêu cầu của bạn, tôi gợi ý {len(top_laptops)} laptop sau:**\n\n"
                            
                            for idx, (_, laptop) in enumerate(top_laptops.iterrows()):
                                response += f"**{idx+1}. {laptop['name']}** ({laptop['brand']})\n"
                                response += f"- 💰 Giá: {laptop['price']/1000000:.1f} triệu VNĐ\n"
                                
                                if 'cpu' in laptop:
                                    response += f"- 💻 CPU: {laptop['cpu']} | RAM: {laptop['ram']}GB | Storage: {laptop['storage']}GB\n"
                                if 'screen' in laptop:
                                    response += f"- 📺 Màn hình: {laptop['screen']}\" | ⚖️ Trọng lượng: {laptop['weight']}kg\n"
                                if 'rating' in laptop:
                                    response += f"- ⭐ Rating: {laptop['rating']}/5"
                                if 'battery' in laptop:
                                    response += f" | 🔋 Pin: {laptop['battery']}h\n"
                                if 'description' in laptop:
                                    response += f"- 📝 Mô tả: {laptop['description']}\n\n"
                                else:
                                    response += "\n\n"
                            
                            response += "Bạn có cần thêm thông tin chi tiết về laptop nào không? 😊"
                        else:
                            response = "Xin lỗi, không tìm thấy laptop phù hợp với bộ lọc hiện tại. Hãy thử điều chỉnh bộ lọc bên trái! 😅"
                    
                    except Exception as e:
                        response = f"Xin lỗi, có lỗi xảy ra: {str(e)}. Vui lòng thử lại! 😅"
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

    with col2:
        # Visualization
        st.subheader("📈 Phân tích thị trường")
        
        try:
            # Price distribution by brand
            if 'brand' in df.columns and 'price' in df.columns:
                fig_price = px.box(
                    df, 
                    x='brand', 
                    y='price',
                    title="Phân bố giá theo thương hiệu",
                    labels={'price': 'Giá (VNĐ)', 'brand': 'Thương hiệu'}
                )
                fig_price.update_layout(height=300)
                st.plotly_chart(fig_price, use_container_width=True)
            
            # Rating vs price scatter plot
            if 'rating' in df.columns and 'price' in df.columns:
                fig_scatter = px.scatter(
                    df,
                    x='price', 
                    y='rating',
                    size='ram' if 'ram' in df.columns else None,
                    color='brand' if 'brand' in df.columns else None,
                    title="Rating vs Giá",
                    labels={'price': 'Giá (VNĐ)', 'rating': 'Rating'},
                    hover_data=['name'] if 'name' in df.columns else None
                )
                fig_scatter.update_layout(height=300)
                st.plotly_chart(fig_scatter, use_container_width=True)
        except Exception as e:
            st.write("Không thể hiển thị biểu đồ:", str(e))

    # Quick actions
    st.subheader("🚀 Gợi ý nhanh")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🎮 Gaming"):
            query = "gaming laptop mạnh chơi game"
            st.session_state.messages.append({"role": "user", "content": query})
            st.rerun()

    with col2:
        if st.button("💼 Văn phòng"):
            query = "laptop văn phòng nhẹ pin trâu"
            st.session_state.messages.append({"role": "user", "content": query})
            st.rerun()

    with col3:
        if st.button("🎨 Thiết kế"):
            query = "laptop thiết kế đồ họa màn hình đẹp"
            st.session_state.messages.append({"role": "user", "content": query})
            st.rerun()

    with col4:
        if st.button("🎓 Sinh viên"):
            query = "laptop sinh viên giá rẻ học tập"
            st.session_state.messages.append({"role": "user", "content": query})
            st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>🤖 <strong>AI Laptop Advisor</strong> - Sử dụng Machine Learning để gợi ý laptop phù hợp</p>
            <p>Công nghệ: TF-IDF, Random Forest, Content-based Filtering, Sentiment Analysis</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Reset chat button
    if st.sidebar.button("🗑️ Xóa lịch sử chat"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()
