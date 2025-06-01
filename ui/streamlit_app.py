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

@st.cache_resource
def load_chatbot():
    """Load chatbot với caching"""
    return LaptopRecommendationChatbot()

@st.cache_data
def load_data():
    """Load và cache dữ liệu"""
    chatbot = load_chatbot()
    return chatbot.df

def analyze_sentiment(text):
    """Simple sentiment analysis"""
    positive_words = ['tốt', 'good', 'great', 'excellent', 'amazing', 'love', 'like', 'best']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'xấu', 'tệ']
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        return "Tích cực 😊"
    elif neg_count > pos_count:
        return "Tiêu cực 😞"
    else:
        return "Trung tính 😐"

def main():
    st.markdown('<h1 class="main-header">💻 Laptop Recommendation Chatbot</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Load chatbot and data
    chatbot = load_chatbot()
    df = load_data()
    
    # Store data in session state for filtering
    if "laptop_data" not in st.session_state:
        st.session_state.laptop_data = df
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 Quick Filters")
        
        # Convert price to millions for Vietnam currency display
        df_price_millions = df['price'] / 1000000 if df['price'].max() > 10000 else df['price']
        
        # Budget filter
        budget_range = st.slider(
            "Budget Range (triệu VNĐ)" if df['price'].max() > 10000 else "Budget Range ($)",
            min_value=float(df_price_millions.min()),
            max_value=float(df_price_millions.max()),
            value=(float(df_price_millions.min()), float(df_price_millions.max()))
        )
        
        # Brand filter
        brands = ['All'] + sorted(df['brand'].unique().tolist())
        selected_brand = st.selectbox("Brand", brands)
        
        # Performance filter
        if 'performance_tier' in df.columns:
            performance_tiers = ['All'] + sorted(df['performance_tier'].unique().tolist())
            selected_performance = st.selectbox("Performance Tier", performance_tiers)
        else:
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
        price_display = f"{df_price_millions.min():.1f} - {df_price_millions.max():.1f} dolar" if df['price'].max() > 10000 else f"${df['price'].min():.0f} - ${df['price'].max():.0f}"
        st.metric("Price Range", price_display)
        st.metric("Avg Rating", f"{df['rating'].mean():.1f}/5.0")
    
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
                        # Get chatbot response
                        response = chatbot.chat(prompt)
                        
                        # Display response
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                    except Exception as e:
                        error_message = f"Xin lỗi, có lỗi xảy ra: {str(e)}"
                        st.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})

    with col2:
        # Visualization
        st.subheader("📈 Phân tích thị trường")
        
        try:
            # Price distribution by brand
            fig_price = px.box(
                df, 
                x='brand', 
                y='price',
                title="Phân bố giá theo thương hiệu",
                labels={'price': 'Giá (VNĐ)' if df['price'].max() > 10000 else 'Giá ($)', 'brand': 'Thương hiệu'}
            )
            fig_price.update_layout(height=300)
            st.plotly_chart(fig_price, use_container_width=True)
            
            # Rating vs price scatter plot
            fig_scatter = px.scatter(
                df,
                x='price', 
                y='rating',
                color='brand',
                title="Rating vs Giá",
                labels={'price': 'Giá (VNĐ)' if df['price'].max() > 10000 else 'Giá ($)', 'rating': 'Rating'},
                hover_data=['brand'] if 'model' not in df.columns else ['brand', 'model']
            )
            fig_scatter.update_layout(height=300)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        except Exception as e:
            st.error(f"Lỗi hiển thị biểu đồ: {str(e)}")

    # Quick actions
    st.subheader("🚀 Gợi ý nhanh")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🎮 Gaming"):
            query = "Tôi cần laptop gaming mạnh để chơi game"
            st.session_state.messages.append({"role": "user", "content": query})
            st.rerun()

    with col2:
        if st.button("💼 Văn phòng"):
            query = "Tôi cần laptop văn phòng nhẹ pin trâu"
            st.session_state.messages.append({"role": "user", "content": query})
            st.rerun()

    with col3:
        if st.button("🎨 Thiết kế"):
            query = "Tôi cần laptop thiết kế đồ họa màn hình đẹp"
            st.session_state.messages.append({"role": "user", "content": query})
            st.rerun()

    with col4:
        if st.button("🎓 Sinh viên"):
            query = "Tôi cần laptop sinh viên giá rẻ cho học tập"
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
