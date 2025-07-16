#importing the necessary libraries
#streamlit is an open-source python library
import streamlit as st
import pandas  as pd
import numpy as np
import nltk
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#stopwords are words in a sentence that does not carry much meaning.... 
#   i.e the sentence can do without it like "the"
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
nltk.download('punkt_tab')

#according to streamlit documentation, st.spinner displays a loading spinner while executing a block of code

with st.spinner("Downloading the NLTK resources..."):
    try:
        #standard tokenizer
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        #VADER for sentiment analysis
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')

   

    try:
         #to remove stopwords
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

#shows that the NLTK resources has been downloaded successfully
st.success("NLTK resources has been downloaded!")

#initialize VADER sentiment intensity analayzer
Analyzer = SentimentIntensityAnalyzer()

#load the stopwords

STOP_WORDS = set(stopwords.words('english'))

#columns definition

Open_Ended_Columns = ['describe_me', 'areas_i_could_improve_on', 'any_other_feedback']



#helper functions
#cacheing helps prevent the data from recomputing everytime the program is run 
#   instead, it calls the same input everytime

@st.cache_data
def preprocess_text(text):

    '''
    Preprocesses a single text string:
    - Converts to lowercase.
    - Removes punctuation and special characters.
    - Tokenizes the text.
    - Removes English stopwords
    '''

    if isinstance(text,str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        words = word_tokenize(text, language='english')

        #removing stopwords from the texts

        filtered_words = [word for word in words if word not in STOP_WORDS and word.strip() != '']
        return ' '.join(filtered_words)
    
    else:
        #returns an empty string for non-string inputs
        return ""
    
#preprocessing the data
@st.cache_data
#practiced with the twitter dataset which only analysed the text column
#    but since my feedback has multiple columns i figured it would be better
# to use a function to preprocess the data instead of working with a single column everytime
def preprocess_multiple_columns(dataframe, columns_to_preprocess):
    #created a copy of the dataframe so as not to directly affect the original dataset
    df_copy = dataframe.copy()
    for col in columns_to_preprocess:
        if col in dataframe.columns:
            df_copy[col] = df_copy[col].apply(preprocess_text)
        else:
            st.warning(f"Column '{col}' not found in the dataframe")

    return df_copy


# performing sentiment analysis with VADER: Valence Aware Dictionary Sentiment Reasoner
@st.cache_data
def get_vader_sentiment(text):
    if isinstance(text,str):
        return Analyzer.polarity_scores(text)
    
    else:
        return {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}
    

#classifying the sentiment based on the VADER compound score

@st.cache_data
def classify_sentiment(compound_score):
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'


@st.cache_data # Cache the entire data loading and processing pipeline
def load_and_process_data(file_path):
    """
    Loads the dataset, preprocesses text, and performs VADER sentiment analysis
    on the specified open-ended columns.
    Returns the DataFrame with new sentiment-related columns.
    """
    df = pd.read_csv(file_path)
    
    # Apreprocess text columns by calling the preprocess function
    df_processed_text = preprocess_multiple_columns(df.copy(), Open_Ended_Columns)

    # perform sentiment analysis and extract scores by calling the get_vader_sentiment function
    for col in Open_Ended_Columns  :
        # Get VADER scores for the preprocessed text
        df_processed_text[f'{col}_vader_scores'] = df_processed_text[col].apply(get_vader_sentiment)
        
        # Extract individual sentiment scores (neg, neu, pos, compound)
        df_processed_text[f'{col}_vader_neg'] = df_processed_text[f'{col}_vader_scores'].apply(lambda x: x['neg'])
        df_processed_text[f'{col}_vader_neu'] = df_processed_text[f'{col}_vader_scores'].apply(lambda x: x['neu'])
        df_processed_text[f'{col}_vader_pos'] = df_processed_text[f'{col}_vader_scores'].apply(lambda x: x['pos'])
        df_processed_text[f'{col}_vader_compound'] = df_processed_text[f'{col}_vader_scores'].apply(lambda x: x['compound'])
        
        # Classify overall sentiment based on compound score
        df_processed_text[f'{col}_vader_sentiment'] = df_processed_text[f'{col}_vader_compound'].apply(classify_sentiment)
        
    return df_processed_text # Returns the processed dataFrame



#creating the wordcloud....yayyy!!! finally did it :)

# according to the st documentation: 
#   st.cache_resource operates as a decorator to cache functions that return global resources
#       in this case, uses it to create a matplotlib figure object

@st.cache_resource 
def generate_wordcloud_plot(text_data, title, background_color="white", width=800, height=400):
    """
    Generates a word cloud plot from text data.
    Returns a Matplotlib figure object. Returns None if text_data is empty.
    """
    # Check if text_data is meaningful i.e it is not empty or just a whitespace
    if not text_data or text_data.isspace():
        return None 

    wordcloud = WordCloud(
        background_color=background_color,
        width=width,
        height=height,
        collocations=False,
        max_words=100 
    ).generate(text_data)
    
    # a Matplotlib figure and axes for the word cloud
    #i should try it out with different values
    fig, ax = plt.subplots(figsize=(10, 5)) 
    ax.imshow(wordcloud, interpolation="bilinear") 
    ax.axis("off")
    ax.set_title(title, fontsize=16, color='black') 
    
    return fig



# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Feedback Sentiment Analysis")

st.title("Spring'25 Feedback Sentiment Analysis Dashboard")

st.write(
    "This dashboard visualizes sentiment analysis of my friends' feedbacks at the end of the spring'25 semester. "
    "Decided to analyze the feedback so as to improve my sentiment analysis skills with the use of VADER while visualizing with wordclouds and barchart"
)

# Load and process data 
processed_df = load_and_process_data("Personal Feedback.csv")


# --- Interactive Widgets to select feedback column ---
selected_column = st.selectbox(
    "**Select a Feedback Category:**",
    options=Open_Ended_Columns,
    help="Choose which feedback column you want to analyze."
)

st.markdown("---") # Visual separator


# --- Displays Sentiment Analysis Results for the Selected Column ---
st.subheader(f"Sentiment Analysis for: **'{selected_column.replace('_', ' ').title()}'**")

# Section for Overall Sentiment Distribution (Bar Chart)
st.write("#### Overall Sentiment Distribution")
# Get value counts and reindex to ensure consistent order (Positive, Neutral, Negative)
sentiment_counts = processed_df[f'{selected_column}_vader_sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative'], fill_value=0)
st.bar_chart(sentiment_counts)


# Section for Word Clouds by Sentiment 
st.write("#### Word Clouds by Sentiment")

# Create tabs for each sentiment category
tab_positive, tab_negative, tab_neutral = st.tabs(["Positive Feedback", "Negative Feedback", "Neutral Feedback"])

with tab_positive:
    st.write(f"Words frequently appearing as **positive** feedback for '{selected_column.replace('_', ' ').title()}':")
    # Filter DataFrame for positive sentiment in the selected column
    positive_df = processed_df[processed_df[f'{selected_column}_vader_sentiment'] == 'Positive']
    
    if not positive_df.empty:
        # Join all positive feedback text for word cloud generation
        positive_text = ' '.join(positive_df[selected_column].dropna().tolist())
        # Generate and display the word cloud plot
        fig_pos = generate_wordcloud_plot(positive_text, f'Positive Feedback: "{selected_column.replace("_", " ").title()}"', background_color="lightgreen")
        if fig_pos: # Check if a figure was actually returned (i.e., not None)
            st.pyplot(fig_pos)
            plt.close(fig_pos)
        else:
            st.info("No meaningful words to display for positive feedback in this category.")
    else:
        st.info("No positive feedback found for this category.")

with tab_negative:
    st.write(f"Words frequently appearing as **negative** feedback for '{selected_column.replace('_', ' ').title()}':")
    # Filter DataFrame for negative sentiment
    negative_df = processed_df[processed_df[f'{selected_column}_vader_sentiment'] == 'Negative']
    
    if not negative_df.empty:
        negative_text = ' '.join(negative_df[selected_column].dropna().tolist())
        fig_neg = generate_wordcloud_plot(negative_text, f'Negative Feedback: "{selected_column.replace("_", " ").title()}"', background_color="lightcoral")
        if fig_neg:
            st.pyplot(fig_neg)
            plt.close(fig_neg)
        else:
            st.info("No meaningful words to display for negative feedback in this category.")
    else:
        st.info("No negative feedback found for this category.")

with tab_neutral:
    st.write(f"Words frequently appearing in **neutral** feedback for '{selected_column.replace('_', ' ').title()}':")
    # Filter DataFrame for neutral sentiment
    neutral_df = processed_df[processed_df[f'{selected_column}_vader_sentiment'] == 'Neutral']
    
    if not neutral_df.empty:
        neutral_text = ' '.join(neutral_df[selected_column].dropna().tolist())
        fig_neu = generate_wordcloud_plot(neutral_text, f'Neutral Feedback: "{selected_column.replace("_", " ").title()}"', background_color="lightgray")
        if fig_neu:
            st.pyplot(fig_neu)
            plt.close(fig_neu)
        else:
            st.info("No meaningful words to display for neutral feedback in this category.")
    else:
        st.info("No neutral feedback found for this category.")

# Display of raw processed data
st.markdown("---")
with st.expander("View The Raw Processed Data Samples"):
    # Display relevant columns: original text columns + their sentiment classification
    st.dataframe(processed_df[Open_Ended_Columns + [f'{col}_vader_sentiment' for col in Open_Ended_Columns]].head(10))



                    
