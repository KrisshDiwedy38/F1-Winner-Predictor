# 🏎️ F1 Race Winner Predictor

A Streamlit web application that predicts the winner of Formula 1 races using an XGBoost machine learning model. This app allows users to select a race from historical data, view participating drivers, and see the top 5 predicted winners with their win probabilities.

## ✨ Features

- **Interactive Race Selection**: Browse and select from a list of historical F1 races
- **Dynamic Predictions**: Get real-time winner predictions using a pre-trained XGBoost model
- **Detailed Results**: View the top 5 predicted winners, their teams, and their win probability scores
- **Podium View**: See a clear "🥇🥈🥉" podium prediction for the top three drivers
- **Maintenance Mode**: Includes a built-in maintenance page that can be toggled with an environment variable for seamless updates during deployment
- **Train On-Demand**: Option to train the model directly from the web interface if a trained model is not found

## 🛠️ Technology Stack

- **Backend**: Python
- **Machine Learning**: XGBoost, Scikit-learn, Pandas, NumPy
- **Web Framework**: Streamlit
- **Data Source**: Historical F1 data (Supabase, CSVs, etc.)

## 📂 Project Structure

The project is organized to separate the application interface from the backend logic.

```
F1WINNERPREDICTOR/
├── app/
│   ├── app.py          # Main entry point that routes to main or maintenance
│   ├── main.py         # The core Streamlit application logic
│   └── maintenance.py  # The maintenance page UI
├── backend/
│   ├── data/           # Scripts for data cleaning and preparation
│   └── ml/             # Scripts for model training, prediction, and loading
├── .gitignore
├── requirements.txt    # Python package dependencies
└── README.md           # This file
```

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### Prerequisites

- Python 3.8 or higher
- pip and venv

### Installation

1. **Clone the Repository**
   ```bash
   git clone <your-repository-url>
   cd F1WINNERPREDICTOR
   ```

2. **Create and Activate a Virtual Environment**
   
   It's highly recommended to use a virtual environment to manage dependencies.
   
   **Windows:**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
   
   **macOS / Linux:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   
   Install all the required Python packages using the requirements.txt file.
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Running the Application

### 1. Train the Model (if needed)

If you don't have a pre-trained model file (`models/xgboost_model.json`), you need to train one first.

```bash
python backend/ml/train.py
```

### 2. Run the Streamlit App

Launch the application using the main entry point `app/app.py`.

```bash
streamlit run app/app.py
```

The application should now be open and running in your web browser at `http://localhost:8501`!

## 🔧 Configuration

### Environment Variables

- `MAINTENANCE_MODE`: Set to `true` to enable maintenance mode and display the maintenance page instead of the main application

## 📊 How It Works

1. **Data Processing**: Historical F1 race data is cleaned and prepared for machine learning
2. **Feature Engineering**: Various features are extracted from race and driver statistics
3. **Model Training**: An XGBoost model is trained on historical race outcomes
4. **Prediction**: The trained model predicts win probabilities for each driver in a selected race
5. **Visualization**: Results are displayed with an intuitive interface showing top predictions
