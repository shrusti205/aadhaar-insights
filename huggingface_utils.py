from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd

class AadhaarAnalyzer:
    def __init__(self):
        # Initialize sentiment analysis pipeline
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        
        # Initialize a zero-shot classifier for categorization
        self.classifier = pipeline("zero-shot-classification",
                                 model="facebook/bart-large-mnli")
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of the given text"""
        try:
            result = self.sentiment_analyzer(text[:512])  # Limit to 512 tokens
            return {
                'sentiment': result[0]['label'],
                'score': float(result[0]['score'])
            }
        except Exception as e:
            return {'error': str(e)}
    
    def categorize_feedback(self, text, categories):
        """Categorize feedback into predefined categories"""
        try:
            result = self.classifier(
                text[:512],  # Limit to 512 tokens
                candidate_labels=categories,
                multi_label=True
            )
            return {
                'categories': result['labels'][:3],  # Top 3 categories
                'scores': [float(score) for score in result['scores'][:3]]
            }
        except Exception as e:
            return {'error': str(e)}

def analyze_aadhaar_feedback(df, text_column='feedback'):
    """
    Analyze a DataFrame containing Aadhaar-related feedback
    
    Args:
        df (pd.DataFrame): DataFrame containing feedback text
        text_column (str): Name of the column containing text to analyze
        
    Returns:
        pd.DataFrame: Original DataFrame with added analysis columns
    """
    analyzer = AadhaarAnalyzer()
    
    # Add sentiment analysis
    sentiment_results = []
    for text in df[text_column]:
        result = analyzer.analyze_sentiment(text)
        sentiment_results.append(result)
    
    sentiment_df = pd.DataFrame(sentiment_results)
    df = pd.concat([df, sentiment_df], axis=1)
    
    return df

def generate_insights(df, text_column='feedback', sentiment_column='sentiment'):
    """
    Generate insights from analyzed feedback data
    
    Args:
        df (pd.DataFrame): DataFrame containing analyzed feedback
        text_column (str): Name of the column containing feedback text
        sentiment_column (str): Name of the column containing sentiment analysis
        
    Returns:
        dict: Dictionary containing various insights
    """
    insights = {}
    
    # Basic sentiment distribution
    sentiment_dist = df[sentiment_column].value_counts(normalize=True) * 100
    insights['sentiment_distribution'] = sentiment_dist.to_dict()
    
    # Most common positive and negative feedback
    if 'score' in df.columns:
        top_positive = df.nlargest(3, 'score')[text_column].tolist()
        top_negative = df.nsmallest(3, 'score')[text_column].tolist()
        insights['top_positive_feedback'] = top_positive
        insights['top_negative_feedback'] = top_negative
    
    return insights
