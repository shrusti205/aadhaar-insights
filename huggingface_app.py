import streamlit as st
import pandas as pd
import time
from huggingface_utils import AadhaarAnalyzer, analyze_aadhaar_feedback, generate_insights

# Set page config
st.set_page_config(
    page_title="Aadhaar Feedback Analysis",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .stTextArea textarea {
        min-height: 150px;
    }
    .feedback-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .positive {
        border-left: 5px solid #4CAF50;
    }
    .negative {
        border-left: 5px solid #F44336;
    }
    .neutral {
        border-left: 5px solid #FFC107;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🤖 Aadhaar Feedback Analysis with Hugging Face")
    st.markdown("""
    This tool uses Hugging Face's Transformers to analyze and categorize Aadhaar-related feedback.
    You can either enter text directly or upload a CSV file containing feedback data.
    """)

    # Initialize analyzer
    analyzer = AadhaarAnalyzer()

    # Sidebar for model options
    st.sidebar.header("Analysis Options")
    analysis_type = st.sidebar.radio(
        "Select Analysis Type",
        ["Single Text Analysis", "Batch Analysis (CSV)"]
    )

    if analysis_type == "Single Text Analysis":
        st.subheader("🔍 Analyze Single Feedback")
        
        # Text input
        feedback = st.text_area("Enter your feedback about Aadhaar services:", 
                              placeholder="I had a great experience with the Aadhaar update process...")
        
        if st.button("Analyze Sentiment"):
            if feedback.strip():
                with st.spinner("Analyzing..."):
                    # Show progress
                    progress_bar = st.progress(0)
                    for percent_complete in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(percent_complete + 1)
                    
                    # Analyze sentiment
                    sentiment_result = analyzer.analyze_sentiment(feedback)
                    
                    # Categorize feedback
                    categories = [
                        "Aadhaar Enrollment", "Aadhaar Update", "Biometric Issues",
                        "OTP Problems", "Document Verification", "Other"
                    ]
                    category_result = analyzer.categorize_feedback(feedback, categories)
                    
                    # Display results
                    st.subheader("📊 Analysis Results")
                    
                    # Sentiment card
                    sentiment = sentiment_result['sentiment']
                    score = sentiment_result['score']
                    
                    # Determine sentiment color
                    sentiment_class = "neutral"
                    if sentiment == "POSITIVE":
                        sentiment_class = "positive"
                    elif sentiment == "NEGATIVE":
                        sentiment_class = "negative"
                    
                    st.markdown(f"""
                    <div class='feedback-card {sentiment_class}'>
                        <h4>Sentiment Analysis</h4>
                        <p><strong>Sentiment:</strong> {sentiment}</p>
                        <p><strong>Confidence:</strong> {score:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Category results
                    st.markdown("### 🏷️ Detected Categories")
                    for cat, score in zip(category_result['categories'], category_result['scores']):
                        st.progress(score, text=f"{cat} ({score:.1%})")
                    
                    # Raw feedback
                    st.markdown("### 📝 Your Feedback")
                    st.info(f'"{feedback}"')
            else:
                st.warning("Please enter some feedback to analyze.")
    
    else:  # Batch Analysis
        st.subheader("📊 Batch Analyze Feedback Data")
        
        # File upload
        uploaded_file = st.file_uploader("Upload a CSV file with feedback data", type=["csv"])
        
        if uploaded_file is not None:
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
                
                # Check if the required columns exist
                if 'feedback' not in df.columns:
                    st.error("The uploaded CSV must contain a 'feedback' column.")
                    return
                
                st.subheader("📋 Preview of Uploaded Data")
                st.dataframe(df.head())
                
                if st.button("Start Analysis"):
                    with st.spinner("Analyzing feedback... This may take a few minutes..."):
                        # Create a progress bar
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Analyze the feedback
                        status_text.text("Analyzing sentiment...")
                        df = analyze_aadhaar_feedback(df)
                        progress_bar.progress(50)
                        
                        # Generate insights
                        status_text.text("Generating insights...")
                        insights = generate_insights(df)
                        progress_bar.progress(100)
                        
                        # Display results
                        st.success("Analysis complete!")
                        
                        # Show sentiment distribution
                        st.subheader("📈 Sentiment Distribution")
                        sentiment_data = pd.DataFrame(
                            list(insights['sentiment_distribution'].items()),
                            columns=['Sentiment', 'Percentage']
                        )
                        st.bar_chart(sentiment_data.set_index('Sentiment'))
                        
                        # Show top feedback
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("👍 Top Positive Feedback")
                            for i, feedback in enumerate(insights.get('top_positive_feedback', []), 1):
                                st.markdown(f"{i}. {feedback[:150]}...")
                        
                        with col2:
                            st.subheader("👎 Top Negative Feedback")
                            for i, feedback in enumerate(insights.get('top_negative_feedback', []), 1):
                                st.markdown(f"{i}. {feedback[:150]}...")
                        
                        # Download button for analyzed data
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Analyzed Data",
                            data=csv,
                            file_name="analyzed_feedback.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
