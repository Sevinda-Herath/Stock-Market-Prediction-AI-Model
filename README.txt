# LSTM Stock Price Prediction with Sentiment Analysis

This project implements Long Short-Term Memory (LSTM) neural networks for stock price prediction, enhanced with sentiment analysis to improve forecasting accuracy. The models are trained on historical price data combined with sentiment scores derived from financial news and social media.

## Project Overview

This repository contains a machine learning pipeline that:

1. Processes historical stock price data for major tech companies
2. Integrates sentiment analysis from financial news sources
3. Trains LSTM models to predict future stock prices
4. Evaluates model performance with comprehensive metrics
5. Visualizes predictions and training statistics

## Data Sources

- **Stock Prices**: Daily historical data for 10 major tech companies
- **Sentiment Data**: Sentiment scores (positive, negative, neutral) derived from financial news and social media

## Stocks Analyzed

The project covers 10 global tech companies:

| Symbol | Company | Exchange |
|--------|---------|----------|
| AMZN | Amazon | NASDAQ |
| AAPL | Apple | NASDAQ |
| GOOGL | Alphabet (Google) | NASDAQ |
| 005930.KS | Samsung Electronics | Korea Exchange |
| 2317.TW | Foxconn (Hon Hai Precision) | Taiwan Stock Exchange |
| MSFT | Microsoft | NASDAQ |
| JD | JD.com | NASDAQ |
| BABA | Alibaba | NYSE |
| T | AT&T | NYSE |
| META | Meta Platforms (Facebook) | NASDAQ |

## Model Architecture

The LSTM model architecture consists of:

- Input layer with 60-day time windows of price and sentiment data
- First LSTM layer with 64 units and sequence return
- Dropout layer (10%)
- Second LSTM layer with 32 units
- Dropout layer (10%)
- Dense output layer (1 unit)

## Performance Metrics

The table below shows the performance metrics for each stock's LSTM model:

| Symbol | Train MAE | Test MAE | Train MSE | Test MSE | Train R² | Test R² | Next Day Prediction |
|--------|-----------|----------|-----------|----------|----------|---------|---------------------|
| AMZN | 0.8977 | 2.9957 | 1.2734 | 15.8557 | 0.9985 | 0.9865 | 221.8782 |
| AAPL | 0.4286 | 2.6541 | 0.3578 | 12.9553 | 0.9988 | 0.9909 | 210.6605 |
| GOOGL | 0.4509 | 2.2886 | 0.5806 | 9.3957 | 0.9989 | 0.9876 | 177.9490 |
| 005930.KS | 351.5538 | 996.4562 | 287329.2619 | 1673499.6517 | 0.9981 | 0.9806 | 62037.1200 |
| 2317.TW | 0.9901 | 1.7062 | 1.7034 | 7.6566 | 0.9965 | 0.9947 | 163.1873 |
| MSFT | 0.6750 | 5.3860 | 1.5232 | 50.0006 | 0.9987 | 0.9924 | 500.5797 |
| JD | 0.9054 | 0.7351 | 2.0185 | 1.0504 | 0.9939 | 0.9686 | 31.1280 |
| BABA | 2.6676 | 1.8792 | 15.0308 | 7.7138 | 0.9955 | 0.9784 | 106.5826 |
| T | 0.1653 | 0.2342 | 0.0480 | 0.1073 | 0.9975 | 0.9918 | 27.3832 |
| META | 3.5761 | 8.9833 | 35.4067 | 167.8872 | 0.9952 | 0.9941 | 722.6456 |

Note: Samsung Electronics (005930.KS) shows larger MAE/MSE values due to its higher absolute price in Korean Won.

## How It Works

1. **Data Preparation**:
   - Historical stock prices are loaded and preprocessed
   - Sentiment data is collected, processed, and merged with price data
   - Data is scaled using MinMaxScaler to normalize values

2. **Sequence Creation**:
   - Time-series data is transformed into sequences of 60 days
   - Each sequence has a corresponding target (next day's closing price)

3. **Model Training**:
   - LSTM model is trained on 80% of the data
   - Early stopping prevents overfitting
   - Best model weights are saved

4. **Evaluation**:
   - Model is evaluated on 20% test data
   - Performance metrics are calculated and saved
   - Predictions are visualized against actual prices

5. **Forecasting**:
   - The model predicts the next day's price using the most recent data

## Results

The models achieved impressive R² scores (generally above 0.98 on test data), indicating strong predictive performance. Visualizations of predictions against actual values are saved in the `test_set_predictions` directory.

## Conclusions

- LSTM networks combined with sentiment analysis can effectively predict stock price movements
- Models show excellent performance on test data, with high R² scores across all stocks
- The integration of sentiment data provides additional signal beyond pure price information
- Taiwanese stock 2317.TW (Foxconn) shows the best test R² score (0.9947)
- JD.com has the lowest test MAE (0.7351), while META has the highest (8.9833)

## Technologies Used

- Python 3.12
- TensorFlow/Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Future Work

- Incorporate additional features (volume, technical indicators)
- Experiment with different LSTM architectures (bidirectional, attention)
- Add ensemble methods to improve prediction stability
- Implement real-time prediction pipeline

---

Last Updated: 2025-08-06  
Author: [Sevinda-Herath](https://github.com/Sevinda-Herath)
