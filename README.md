# 📈 Streamlit Feedback Sentiment Analysis Dashboard

## Project Overview

This project presents an interactive **Streamlit dashboard** designed for in-depth sentiment analysis of feedback of my friends view of me at the end of the Spring 2025 semester. The project is mainly for me to learn sentiment analysis. The application is built entirely in Python, leveraging powerful NLP libraries and Streamlit's intuitive framework for web deployment.

## ✨ Features

The dashboard offers the following key functionalities:

* **Interactive Web Interface:** A user-friendly, responsive dashboard built with Streamlit.
* **Multi-Category Feedback Analysis:** Analyze sentiment across distinct open-ended feedback columns (e.g., `describe_me`, `areas_i_could_improve_on`, `any_other_feedback`).
* **VADER Sentiment Analysis:** Utilizes NLTK's Valence Aware Dictionary and sEntiment Reasoner (VADER) for nuanced, lexicon-based sentiment scoring, robust for general feedback.
* **Comprehensive Text Preprocessing:** Includes automated steps for text cleaning:
    * Conversion to lowercase.
    * Removal of punctuation and special characters.
    * Word tokenization.
    * Elimination of common English stopwords to focus on meaningful terms.
* **Sentiment-Specific Word Clouds:** Generates visually distinct word clouds for Positive, Negative, and Neutral feedback. These clouds highlight the most frequently occurring words within each sentiment category, allowing for rapid identification of dominant themes and keywords.
* **Sentiment Distribution Bar Charts:** Provides a clear visual summary of the sentiment breakdown (percentage of Positive, Negative, Neutral feedback) for the selected category.
* **Cached Performance:** Utilizes Streamlit's caching mechanisms (`@st.cache_data`, `@st.cache_resource`) to ensure efficient data processing and fast dashboard interactions.

## 🛠️ Technologies Used

* **Python 3.13.1** 
* **Streamlit:** For creating the interactive web application.
* **Pandas:** For robust data manipulation and analysis.
* **NLTK (Natural Language Toolkit):** Core library for VADER sentiment analysis, word tokenization, and managing stopwords.
* **WordCloud:** For generating visually appealing word cloud graphics.
* **Matplotlib:** For rendering the word clouds and sentiment distribution bar charts.
* **Git & GitHub:** For version control and project hosting.

## ⚙️ How to Run Locally

Follow these steps to set up and run the Streamlit dashboard on your local machine.

### Prerequisites

Ensure you have the following installed:

* **Python 3+
* **pip** (Python package installer, usually comes with Python)
* **Git** (for cloning the repository)

### Installation Steps

1.  **Clone the repository:**
    Open your terminal or command prompt and clone the project:
    ```bash
    git clone [https://github.com/GraceJulius/StreamlitSentimentAnalysis.git](https://github.com/GraceJulius/StreamlitSentimentAnalysis.git)
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd StreamlitSentimentAnalysis
    ```
3.  **Create a virtual environment:**
    This isolates your project's dependencies from your system's Python packages.
    ```bash
    python -m venv venv
    ```
4.  **Activate the virtual environment:**
    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS / Linux:**
        ```bash
        source venv/bin/activate
        ```
5.  **Install the required Python packages:**
    The project dependencies are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Ensure your data file `Personal Feedback.csv` is located in the same directory** as `app.py`.
2.  **Run the Streamlit application:**
    With your virtual environment activated, execute the following command:
    ```bash
    streamlit run app.py
    ```
3.  This command will automatically open the interactive dashboard in your default web browser (typically at `http://localhost:8501`).

## 📊 Data

The dashboard operates on my feedback data supplied via the `Personal Feedback.csv` file. This CSV file is expected to contain open-ended text columns for analysis. The current `app.py` is configured to process the following columns:

* `describe_me`
* `areas_i_could_improve_on`
* `any_other_feedback`


## 📸 Screenshots / Demo
<p align="center">
  <img width="1830" height="854" alt="image" src="https://github.com/user-attachments/assets/ce613971-1986-4773-8da4-e27615909ca2" />
  <br>
  <em> Main dashboard view showing sentiment selection.</em>
</p>

<p align="center">
  <img width="1743" height="815" alt="image" src="https://github.com/user-attachments/assets/4393cbb3-360d-40da-8a48-c38ad3893e94" />
  <br>
  <em> Word Cloud for Positive Feedback.</em>
</p>

<p align="center">
 <img width="1785" height="141" alt="image" src="https://github.com/user-attachments/assets/ada1fef0-f899-4f3a-8f49-613faf5a6981" />
  <br>
  <em> Word Cloud for Negative Feedback.</em>
</p>


## 📧 Contact

Feel free to connect or reach out if you have any questions, feedback, or collaboration opportunities!

* **GitHub:** [GraceJulius](https://github.com/GraceJulius)
* **Email:** [your.email@example.com](mailto:juliusgrace65@gmail.com) 
* **LinkedIn:** [[LinkedIn Profile](https://www.linkedin.com/in/grace-julius/)] 


---
