import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

st.set_page_config(page_title="Diarrhea Analysis Dashboard", layout="wide")

st.title("Bangladesh Regional Diarrhea Analysis & Prediction Dashboard")
st.markdown("""
This dashboard presents a comprehensive analysis of diarrhea cases across four major divisions 
in Bangladesh, examining the relationship between environmental factors and disease occurrence.
""")

# Data Loading Function
@st.cache_data
def load_data():
    # Loading the datasets
    rajshahi = pd.read_csv('datasets/Rajshahi.csv')
    khulna = pd.read_csv('datasets/Khulna.csv')
    dhaka = pd.read_csv('datasets/Dhaka.csv')
    chattogram = pd.read_csv('datasets/Chattogram.csv')
    
    # Converting dates
    for df in [rajshahi, khulna, dhaka, chattogram]:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Adding division labels
    rajshahi['Division'] = 'Rajshahi'
    khulna['Division'] = 'Khulna'
    dhaka['Division'] = 'Dhaka'
    chattogram['Division'] = 'Chattogram'
    
    # Combining data
    combined_df = pd.concat([rajshahi, khulna, dhaka, chattogram])
    return combined_df, rajshahi, khulna, dhaka, chattogram

# Load data
combined_df, rajshahi, khulna, dhaka, chattogram = load_data()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", 
    ["Overview", "Time Series Analysis", "Correlation Analysis", "Prediction Model"])

if page == "Overview":
    st.header("Regional Overview")
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(combined_df))
    with col2:
        st.metric("Date Range", f"{combined_df['Date'].min().year} - {combined_df['Date'].max().year}")
    with col3:
        st.metric("Average Cases", int(combined_df['Diarrhea'].mean()))
    with col4:
        st.metric("Max Cases", int(combined_df['Diarrhea'].max()))
    
    # Time series plot
    st.subheader("Diarrhea Cases Across Divisions")
    fig = px.line(combined_df, x='Date', y='Diarrhea', color='Division',
                  title='Diarrhea Cases Over Time by Division')
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribution plot
    st.subheader("Distribution of Cases by Division")
    fig = px.box(combined_df, x='Division', y='Diarrhea', 
                 title='Distribution of Diarrhea Cases by Division')
    st.plotly_chart(fig, use_container_width=True)

elif page == "Time Series Analysis":
    st.header("Time Series Analysis")
    
    # Division selector
    division = st.selectbox("Select Division", 
                          ['Rajshahi', 'Khulna', 'Dhaka', 'Chattogram'])
    
    # Get corresponding dataframe
    df_dict = {'Rajshahi': rajshahi, 'Khulna': khulna, 
               'Dhaka': dhaka, 'Chattogram': chattogram}
    df = df_dict[division]
    
    # Time series decomposition
    st.subheader("Seasonal Decomposition")
    decomposition = seasonal_decompose(df['Diarrhea'], period=30)
    
    # Plot components
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10))
    decomposition.observed.plot(ax=ax1)
    ax1.set_title('Observed')
    decomposition.trend.plot(ax=ax2)
    ax2.set_title('Trend')
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonal')
    decomposition.resid.plot(ax=ax4)
    ax4.set_title('Residual')
    plt.tight_layout()
    st.pyplot(fig)

elif page == "Correlation Analysis":
    st.header("Correlation Analysis")
    
    # Division selector
    division = st.selectbox("Select Division", 
                          ['Rajshahi', 'Khulna', 'Dhaka', 'Chattogram'])
    
    # Get corresponding dataframe
    df_dict = {'Rajshahi': rajshahi, 'Khulna': khulna, 
               'Dhaka': dhaka, 'Chattogram': chattogram}
    df = df_dict[division]
    
    # Correlation matrix
    correlation = df[['Diarrhea', 'Minimum Temperature', 'Maximum Temperature', 
                     'Humidity', 'Preceptation']].corr()
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax)
    plt.title(f'Correlation Matrix for {division}')
    st.pyplot(fig)
    
    # Scatter plots
    st.subheader("Feature Relationships")
    feature = st.selectbox("Select Feature", 
                          ['Minimum Temperature', 'Maximum Temperature', 
                           'Humidity', 'Preceptation'])
    
    fig = px.scatter(df, x=feature, y='Diarrhea', 
                     trendline="ols", 
                     title=f'{feature} vs Diarrhea Cases')
    st.plotly_chart(fig, use_container_width=True)

elif page == "Prediction Model":
    st.header("Prediction Model")
    
    # Model selection
    model_type = st.selectbox("Select Model", 
                             ["Random Forest", "Gradient Boosting"])
    
    # Feature selection
    st.subheader("Select Features for Prediction")
    use_min_temp = st.checkbox("Minimum Temperature", value=True)
    use_max_temp = st.checkbox("Maximum Temperature", value=True)
    use_humidity = st.checkbox("Humidity", value=True)
    use_precip = st.checkbox("Precipitation", value=True)
    
    # Prepare features
    features = []
    if use_min_temp:
        features.append('Minimum Temperature')
    if use_max_temp:
        features.append('Maximum Temperature')
    if use_humidity:
        features.append('Humidity')
    if use_precip:
        features.append('Preceptation')
    
    if len(features) > 0:
        X = combined_df[features]
        y = combined_df['Diarrhea']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                           test_size=0.2, 
                                                           random_state=42)
        
        # Train model
        if model_type == "Random Forest":
            model = RandomForestRegressor(random_state=42)
        else:
            model = GradientBoostingRegressor(random_state=42)
            
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("R² Score", round(r2_score(y_test, y_pred), 3))
        with col2:
            st.metric("RMSE", round(np.sqrt(mean_squared_error(y_test, y_pred)), 3))
        
        # Plot actual vs predicted
        fig = px.scatter(x=y_test, y=y_pred, 
                        labels={'x': 'Actual Cases', 'y': 'Predicted Cases'},
                        title='Actual vs Predicted Diarrhea Cases')
        fig.add_trace(px.line(x=[y_test.min(), y_test.max()], 
                             y=[y_test.min(), y_test.max()]).data[0])
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        if len(features) > 1:
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(importance_df, x='Importance', y='Feature',
                        title='Feature Importance',
                        orientation='h')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please select at least one feature for prediction.")