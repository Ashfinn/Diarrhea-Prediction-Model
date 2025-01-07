# Environmental Determinants of Diarrheal Disease: A Multi-Regional Analysis in Bangladesh

## Overview
This research project analyzes the relationship between environmental factors and diarrheal disease incidence across four major divisions in Bangladesh. Using machine learning models and time series analysis, the study explores how environmental variables influence disease patterns and develops predictive models for public health applications.

## Key Features
- Multi-regional data analysis covering Rajshahi, Khulna, Dhaka, and Chattogram divisions
- Interactive visualization of disease patterns and environmental correlations
- Time series decomposition and seasonal trend analysis
- Predictive modeling using various machine learning algorithms
- Interactive web dashboard for data exploration and model predictions

## Technologies Used
- **Data Analysis**: Python, Pandas, NumPy
- **Machine Learning**: Scikit-learn, Random Forest, Gradient Boosting
- **Visualization**: Plotly, Seaborn, Matplotlib
- **Web Dashboard**: Streamlit
- **Time Series Analysis**: Statsmodels

## Data Sources
The study utilizes data from four divisions in Bangladesh, including:
- Daily diarrheal disease cases
- Environmental parameters:
  - Maximum temperature
  - Minimum temperature
  - Humidity
  - Precipitation

Data source: [Data-Lab-CU/diarrhea](https://github.com/Data-Lab-CU/diarrhea/tree/main/data)

## Analysis Components
1. **Data Preprocessing**
   - Outlier detection and handling
   - Missing value treatment
   - Feature engineering

2. **Exploratory Data Analysis**
   - Regional comparison of disease patterns
   - Seasonal trend analysis
   - Environmental factor correlations

3. **Predictive Modeling**
   - Implementation of multiple ML algorithms
   - Model performance comparison
   - Feature importance analysis

4. **Interactive Dashboard**
   - Real-time data visualization
   - Model prediction interface
   - Time series analysis tools

## Installation and Usage

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Running the Dashboard
```bash
streamlit run app.py
```

### Project Structure
```
diarrhea-env-analysis/
│
├── app.py                 # Streamlit dashboard
├── requirements.txt       # Project dependencies
├── notebooks/            
│   └── analysis.ipynb    # Research analysis notebook
├── datasets/                 # Data directory
│   └── data.csv        # Data documentation
└── README.md             # Project documentation
```

## Results and Insights
- Identified lack of correlations between environmental factors and disease incidence
- Developed predictive models with R² scores ranging from 0.2 to 0.5
- Discovered few seasonal patterns in disease occurrence
- Quantified the relative importance of different environmental factors

## Future Work
- Integration of additional environmental parameters
- Extension of analysis to other regions
- Implementation of advanced time series models
- Development of early warning systems

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For any queries regarding this research, please open an issue in this repository.

## Acknowledgments
- Research collaborators and advisors