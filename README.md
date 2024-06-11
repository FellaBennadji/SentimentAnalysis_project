# Sentiment Analysis Model for Restaurant Reviews

This project implements a sentiment analysis model for evaluating restaurant reviews. The model identifies aspects such as food, drinks, and service, and extracts related sentiment expressions from the reviews.

## Installation
### Prerequisites

    Python 3.x
    Gensim
    SpaCy
    SciPy

### Setup

1. Clone the repository:
```
git clone https://github.com/yourusername/restaurant-sentiment-analysis.git
cd restaurant-sentiment-analysis
```

2. Install required libraries:

```
pip install gensim spacy scipy
```

3. Download the SpaCy French model:

```
python -m spacy download fr_core_news_md
```

4. Download the word embeddings file frWac_non_lem_no_postag_no_phrase_200_skip_cut100.bin and place it in the project directory.

### Usage

Run the analysis on a text file with reviews:

```
python sentiment_analysis.py reviews.txt
```

Results are saved in resultats.json.


### Files

    sentiment_analysis.py: Main script.
    frWac_non_lem_no_postag_no_phrase_200_skip_cut100.bin: Word embeddings (not included).
    reviews.txt: Input file.
    resultats.json: Output file.

### Functionality

    Loads word embeddings and SpaCy model.
    Defines aspect keywords for food, drinks, and service.
    Calculates similarity between words and aspect keywords.
    Identifies aspect terms and associated sentiments in reviews.
    Saves results in JSON format.
