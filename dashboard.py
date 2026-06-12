# dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import json

# Page configuration
st.set_page_config(
    page_title="Airline Sentiment Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for sentiment tags
st.markdown("""
<style>
.sentiment-positive {
    background-color: #28a745;
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    display: inline-block;
    font-weight: bold;
}
.sentiment-neutral {
    background-color: #ffc107;
    color: #333;
    padding: 4px 12px;
    border-radius: 20px;
    display: inline-block;
    font-weight: bold;
}
.sentiment-negative {
    background-color: #dc3545;
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    display: inline-block;
    font-weight: bold;
}
.metric-card {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #007bff;
}
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    """Load historical predictions data"""
    try:
        df = pd.read_csv('outputs/predictions_full.csv')
        
        # Convert date column if exists
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Map sentiment to numeric score
        sentiment_score_map = {
            'positive': 1,
            'neutral': 0,
            'negative': -1
        }
        
        if 'predicted_sentiment' in df.columns:
            df['score'] = df['predicted_sentiment'].map(sentiment_score_map)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Calculate net sentiment score
def calculate_net_sentiment(df):
    """Calculate net sentiment score from score column"""
    if 'score' in df.columns and len(df) > 0:
        return df['score'].mean()
    return 0

# Get most frequent negative reason
def get_most_frequent_negative_reason(df):
    """Get the most common negative reason"""
    if 'negativereason' in df.columns and len(df) > 0:
        negative_df = df[df['predicted_sentiment'] == 'negative']
        if len(negative_df) > 0:
            most_common = negative_df['negativereason'].mode()
            if len(most_common) > 0 and pd.notna(most_common[0]):
                return most_common[0]
    return "No negative reasons recorded"

# Sidebar filters
st.sidebar.title("✈️ Airline Sentiment Dashboard")
st.sidebar.markdown("---")

# Load data
df = load_data()

if df.empty:
    st.error("No data found. Please ensure 'outputs/predictions_full.csv' exists.")
    st.stop()

# Sidebar filters
st.sidebar.subheader("Filters")

# Airline filter
airlines = ['All'] + sorted(df['airline'].unique().tolist()) if 'airline' in df.columns else ['All']
selected_airline = st.sidebar.selectbox("Select Airline", airlines)

# Date range filter
if 'date' in df.columns and len(df) > 0:
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)
        filtered_df = df[mask].copy()
    else:
        filtered_df = df.copy()
else:
    filtered_df = df.copy()
    st.sidebar.warning("No date column found in data")

# Apply airline filter
if selected_airline != 'All' and 'airline' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['airline'] == selected_airline]

# Main dashboard
st.title("📊 US Airline Sentiment Analysis Dashboard")
st.markdown(f"**Champion Model:** SVM Pipeline | **Accuracy:** 79% | **Macro F1:** 0.7232")
st.markdown("---")

# KPI Header
col1, col2, col3 = st.columns(3)

with col1:
    total_tweets = len(filtered_df)
    st.metric(
        label="📝 Total Tweets Analyzed",
        value=f"{total_tweets:,}",
        delta=None
    )

with col2:
    net_sentiment = calculate_net_sentiment(filtered_df)
    sentiment_color = "normal" if net_sentiment >= 0 else "inverse"
    st.metric(
        label="💯 Average Net Sentiment Score",
        value=f"{net_sentiment:.3f}",
        delta="Positive" if net_sentiment > 0 else "Negative" if net_sentiment < 0 else "Neutral",
        delta_color=sentiment_color
    )

with col3:
    top_reason = get_most_frequent_negative_reason(filtered_df)
    st.metric(
        label="⚠️ Most Frequent Negative Reason",
        value=top_reason if top_reason != "No negative reasons recorded" else "N/A",
        delta=None
    )

st.markdown("---")

# Visualizations
col1, col2 = st.columns(2)

# 1. Net Brand Sentiment Score
with col1:
    st.subheader("🎯 Net Brand Sentiment Score")
    
    if 'airline' in filtered_df.columns and 'score' in filtered_df.columns:
        airline_sentiment = filtered_df.groupby('airline')['score'].mean().sort_values().reset_index()
        airline_sentiment.columns = ['airline', 'net_score']
        
        # Color mapping
        colors = ['#dc3545' if x < 0 else '#28a745' for x in airline_sentiment['net_score']]
        
        fig = go.Figure(data=[
            go.Bar(
                x=airline_sentiment['net_score'],
                y=airline_sentiment['airline'],
                orientation='h',
                marker_color=colors,
                text=airline_sentiment['net_score'].round(3),
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            height=400,
            xaxis_title="Net Sentiment Score",
            yaxis_title="Airline",
            showlegend=False,
            xaxis=dict(range=[-1, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Required columns missing for Net Brand Sentiment Score")

# 2. Brand Perception Map (Stacked percentage bar chart)
with col2:
    st.subheader("📊 Brand Perception Map")
    
    if 'airline' in filtered_df.columns and 'predicted_sentiment' in filtered_df.columns:
        # Calculate percentages
        sentiment_counts = filtered_df.groupby(['airline', 'predicted_sentiment']).size().reset_index(name='count')
        airline_totals = sentiment_counts.groupby('airline')['count'].transform('sum')
        sentiment_counts['percentage'] = (sentiment_counts['count'] / airline_totals) * 100
        
        fig = px.bar(
            sentiment_counts,
            x='airline',
            y='percentage',
            color='predicted_sentiment',
            color_discrete_map={
                'positive': '#28a745',
                'neutral': '#ffc107',
                'negative': '#dc3545'
            },
            title="Sentiment Distribution by Airline",
            labels={'percentage': 'Percentage (%)', 'airline': 'Airline'},
            barmode='stack'
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Required columns missing for Brand Perception Map")

# 3. Temporal Sentiment Trend
st.subheader("📈 Temporal Sentiment Trend")

if 'date' in filtered_df.columns and 'predicted_sentiment' in filtered_df.columns:
    # Daily sentiment counts
    daily_sentiment = filtered_df.groupby([filtered_df['date'].dt.date, 'predicted_sentiment']).size().reset_index(name='count')
    daily_sentiment.columns = ['date', 'sentiment', 'count']
    
    fig = px.line(
        daily_sentiment,
        x='date',
        y='count',
        color='sentiment',
        color_discrete_map={
            'positive': '#28a745',
            'neutral': '#ffc107',
            'negative': '#dc3545'
        },
        title="Daily Sentiment Trends Over Time",
        labels={'count': 'Number of Tweets', 'date': 'Date', 'sentiment': 'Sentiment'}
    )
    
    fig.update_layout(height=450, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Date or sentiment columns missing for Temporal Trend")

# 4. Negative Reason Breakdown
st.subheader("🔍 Negative Reason Breakdown")

if 'negativereason' in filtered_df.columns and 'predicted_sentiment' in filtered_df.columns:
    negative_df = filtered_df[filtered_df['predicted_sentiment'] == 'negative']
    
    if len(negative_df) > 0 and negative_df['negativereason'].notna().any():
        reason_counts = negative_df['negativereason'].value_counts().head(10).reset_index()
        reason_counts.columns = ['reason', 'count']
        
        fig = px.bar(
            reason_counts,
            x='count',
            y='reason',
            orientation='h',
            color='count',
            color_continuous_scale='Reds',
            title="Top Reasons for Negative Sentiment",
            labels={'count': 'Number of Tweets', 'reason': 'Negative Reason'}
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No negative sentiment tweets with reasons found in the selected data")
else:
    st.warning("Negative reason column not found in data")

# Live Prediction Interface
st.markdown("---")
st.subheader("🤖 Live Prediction Interface")
st.markdown("**Test the Champion Model**")

col1, col2 = st.columns([3, 1])

with col1:
    user_input = st.text_area(
        "Enter a tweet to analyze sentiment:",
        height=100,
        placeholder="e.g., I love flying with United Airlines! The service was amazing."
    )

with col2:
    st.markdown("")
    st.markdown("")
    analyze_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)

# API endpoint
API_URL = "http://localhost:8000/predict-sentiment/"

if analyze_button and user_input:
    with st.spinner("Analyzing sentiment..."):
        try:
            # Send request to FastAPI backend
            response = requests.post(
                API_URL,
                json={"text": user_input},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                sentiment = result.get('sentiment', 'unknown')
                confidence = result.get('confidence', 0)
                
                # Display result
                st.markdown("### Analysis Result:")
                
                sentiment_class = f"sentiment-{sentiment}"
                sentiment_display = sentiment.upper()
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f'<div class="{sentiment_class}" style="text-align: center; font-size: 24px;">{sentiment_display}</div>', 
                               unsafe_allow_html=True)
                    st.markdown(f"**Confidence Score:** {confidence:.2%}")
                    st.markdown(f"**Input Text:** \"{user_input}\"")
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to FastAPI backend. Please ensure the server is running at http://localhost:8000")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            
elif analyze_button and not user_input:
    st.warning("Please enter some text to analyze")

# Sample Tweets Section
st.markdown("---")
st.subheader("📝 Sample Tweets from Dataset")

if len(filtered_df) > 0:
    # Get sample tweets (assuming there's a 'text' or 'tweet' column)
    text_column = None
    for col in ['text', 'tweet', 'cleaned_text', 'content']:
        if col in filtered_df.columns:
            text_column = col
            break
    
    if text_column:
        sample_tweets = filtered_df[[text_column, 'predicted_sentiment', 'airline']].sample(min(5, len(filtered_df)))
        
        for idx, row in sample_tweets.iterrows():
            sentiment = row['predicted_sentiment']
            sentiment_class = f"sentiment-{sentiment}"
            
            with st.expander(f"Tweet from {row['airline']}"):
                st.markdown(f"**Text:** {row[text_column][:200]}...")
                st.markdown(f'**Sentiment:** <span class="{sentiment_class}">{sentiment.upper()}</span>', 
                           unsafe_allow_html=True)
    else:
        st.info("No text column found to display sample tweets")
else:
    st.info("No data available to display sample tweets")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: gray;">
        <p>Dashboard powered by Streamlit | Champion Model: SVM Pipeline (Accuracy: 79%, Macro F1: 0.7232)</p>
        <p>Data Source: US Airline Sentiment Analysis | API Endpoint: http://localhost:8000</p>
    </div>
    """,
    unsafe_allow_html=True
)
