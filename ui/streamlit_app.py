import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
sys.path.append(r'C:\CNS\Hackathon_AI\recommend_laptop\src')
from laptop_recommender import LaptopRecommendationChatbot

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

def main():
    st.markdown('<h1 class="main-header">💻 Laptop Recommendation Chatbot</h1>', unsafe_allow_html=True)
    
    # Load chatbot
    chatbot = load_chatbot()
    df = load_data()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 Quick Filters")
        
        # Budget filter
        budget_range = st.slider(
            "Budget Range ($)",
            min_value=int(df['price'].min()),
            max_value=int(df['price'].max()),
            value=(int(df['price'].min()), int(df['price'].max()))
        )
        
        # Brand filter
        brands = ['All'] + sorted(df['brand'].unique().tolist())
        selected_brand = st.selectbox("Brand", brands)
        
        # Performance filter
        performance_tiers = ['All'] + sorted(df['performance_tier'].unique().tolist())
        selected_performance = st.selectbox("Performance Tier", performance_tiers)
        
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
        st.metric("Price Range", f"${df['price'].min():.0f} - ${df['price'].max():.0f}")
        st.metric("Avg Rating", f"{df['rating'].mean():.1f}/5.0")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    with col1:
        # Chat interface
        st.subheader("💬 Trò chuyện với AI")
        
        # Hiển thị lịch sử chat
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Input chat
        if prompt := st.chat_input("Hãy mô tả nhu cầu sử dụng laptop của bạn..."):
            # Thêm tin nhắn người dùng
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Xử lý và phản hồi
            with st.chat_message("assistant"):
                with st.spinner("Đang phân tích và tìm kiếm..."):
                    # Phân tích sentiment
                    sentiment = analyze_sentiment(prompt)
                    
                    # Trích xuất thông tin giá từ prompt
                    price_match = re.findall(r'(\d+)\s*(?:triệu|tr|million)', prompt.lower())
                    max_price = None
                    if price_match:
                        max_price = int(price_match[0]) * 1000000
                    else:
                        max_price = price_range[1] * 1000000
                    
                    # Lọc dữ liệu theo sidebar
                    filtered_data = st.session_state.laptop_data[
                        (st.session_state.laptop_data['brand'].isin(brands)) &
                        (st.session_state.laptop_data['price'] >= price_range[0] * 1000000) &
                        (st.session_state.laptop_data['price'] <= price_range[1] * 1000000)
                    ]
                    
                    if len(filtered_data) > 0:
                        # Tạo recommender mới với dữ liệu đã lọc
                        temp_recommender = LaptopRecommendationSystem(filtered_data)
                        recommendations, scores = temp_recommender.hybrid_recommend(prompt, max_price, top_k=3)
                        
                        response = f"**Cảm xúc của bạn:** {sentiment}\n\n"
                        response += f"**Dựa trên yêu cầu của bạn, tôi gợi ý {len(recommendations)} laptop sau:**\n\n"
                        
                        for idx, (_, laptop) in enumerate(recommendations.iterrows()):
                            response += f"**{idx+1}. {laptop['name']}** ({laptop['brand']})\n"
                            response += f"- 💰 Giá: {laptop['price']/1000000:.1f} triệu VNĐ\n"
                            response += f"- 💻 CPU: {laptop['cpu']} | RAM: {laptop['ram']}GB | Storage: {laptop['storage']}GB\n"
                            response += f"- 📺 Màn hình: {laptop['screen']}\" | ⚖️ Trọng lượng: {laptop['weight']}kg\n"
                            response += f"- ⭐ Rating: {laptop['rating']}/5 | 🔋 Pin: {laptop['battery']}h\n"
                            response += f"- 📝 Mô tả: {laptop['description']}\n\n"
                        
                        response += "Bạn có cần thêm thông tin chi tiết về laptop nào không? 😊"
                    else:
                        response = "Xin lỗi, không tìm thấy laptop phù hợp với bộ lọc hiện tại. Hãy thử điều chỉnh bộ lọc bên trái! 😅"
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

    with col2:
        # Visualization
        st.subheader("📈 Phân tích thị trường")
        
        # Biểu đồ giá theo thương hiệu
        fig_price = px.box(
            st.session_state.laptop_data, 
            x='brand', 
            y='price',
            title="Phân bố giá theo thương hiệu",
            labels={'price': 'Giá (VNĐ)', 'brand': 'Thương hiệu'}
        )
        fig_price.update_layout(height=300)
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Biểu đồ rating vs price
        fig_scatter = px.scatter(
            st.session_state.laptop_data,
            x='price', 
            y='rating',
            size='ram',
            color='brand',
            title="Rating vs Giá",
            labels={'price': 'Giá (VNĐ)', 'rating': 'Rating'},
            hover_data=['name']
        )
        fig_scatter.update_layout(height=300)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Quick actions
    st.subheader("🚀 Gợi ý nhanh")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🎮 Gaming"):
            query = "gaming laptop mạnh chơi game"
            st.session_state.messages.append({"role": "user", "content": query})
            st.experimental_rerun()

    with col2:
        if st.button("💼 Văn phòng"):
            query = "laptop văn phòng nhẹ pin trâu"
            st.session_state.messages.append({"role": "user", "content": query})
            st.experimental_rerun()

    with col3:
        if st.button("🎨 Thiết kế"):
            query = "laptop thiết kế đồ họa màn hình đẹp"
            st.session_state.messages.append({"role": "user", "content": query})
            st.experimental_rerun()

    with col4:
        if st.button("🎓 Sinh viên"):
            query = "laptop sinh viên giá rẻ học tập"
            st.session_state.messages.append({"role": "user", "content": query})
            st.experimental_rerun()

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
        st.experimental_rerun()