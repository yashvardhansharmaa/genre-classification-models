# Comprehensive Music Machine Learning Project

## Overview
Utilizing NLTK's VADER for lyrical sentiment analysis, SpotiPy for extracting Spotify audio features to train logistic regression and random forest classifiers, and Librosa for processing raw audio data with convolutional neural networks, this project comprehensively analyzed and classified music genres like indie folk and rock through natural language processing on lyrics, modeling of acoustic properties like danceability and tempo, and deep learning on the intricate audio textures.

## Part 1: Lyric Sentiment Analysis

### Problem
The objective is to analyze the sentiment of lyrics from indie folk and rock songs to determine how lyrical content reflects the emotional and thematic elements of these genres.

### Approach
- **Data Collection**: Lyrics for selected songs were retrieved from the Genius API.
- **Sentiment Analysis**: Utilized the NLTK library, particularly the VADER tool, to analyze and score the sentiment of each song's lyrics.
- **Visualization**: Plotted sentiment scores to compare and contrast between indie folk and rock genres.

### Results
The analysis provided a deeper understanding of the emotional undertones in song lyrics, revealing distinct differences between the two studied genres. This segment highlighted the reflective and often melancholic sentiment in indie folk as opposed to the vibrant energy found in rock lyrics.

## Part 2: Spotify Audio Features Analysis

### Problem
To classify music into genres based on audio features provided by Spotify, reflecting physical and acoustic properties like danceability, energy, and tempo.

### Approach
- **Feature Extraction**: Used the SpotiPy library to fetch audio features for a curated list of songs.
- **Machine Learning Models**: Applied logistic regression and random forest classifiers to predict genres based on audio features.
- **Evaluation**: Assessed the accuracy and effectiveness of the models in genre classification.

### Results
The models successfully classified songs into genres with considerable accuracy, underscoring the significance of audio features in music genre classification.

## Part 3: Direct Audio Data Processing

### Problem
The final part of the project addresses the classification of music genres directly from audio files, exploring the raw audio data for deeper insights.

### Approach
- **Feature Extraction**: Utilized the Librosa library to extract sophisticated audio features from raw music files.
- **Deep Learning**: Implemented convolutional neural networks to classify music samples based on their audio profiles.
- **Visualization and Evaluation**: Analyzed the performance of the neural network and visualized the classification results.

### Results
This approach provided a granular analysis of audio data, achieving high accuracy in genre classification and offering insights into the complex textures of music.

## Installation and Usage
Ensure you have Python 3.x installed along with required libraries such as NLTK, TensorFlow, Librosa, SpotiPy, Pandas, NumPy, Matplotlib, and Scikit-learn. To run the different components of the project:

```bash
python lyrics_analysis.py
python spotify_features_analysis.py
python audio_data_processing.py
