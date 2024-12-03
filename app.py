# Imports
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

# Sidebar Navigation
page = st.sidebar.selectbox("Select a Page", ["Home", "Overview", "Exploratory Data Analysis", "Model Training and Evaluation"])

# Load Dataset
data = pd.read_csv("data/train_cleaned.csv")

# Home Page
if page == "Home":
    st.title("Welcome to the Airline Satisfaction Predictions App! ✈️")
    st.write("Explore insights derived from airline customer feedback. Dive into the data, discover trends, and visualize key metrics.")
    st.image("images/airport.jpg")

# Overview of the Data
if page == "Overview":
    st.header("Overview of the Data 🔍")
    st.write("This dataset contains information from an airline passenger satisfaction survey.")

    tab1, tab2, tab3 = st.tabs(["Data Dictionary", "Data Types", "Sample Data"])
    with tab1:
        st.write("### Data Dictionary:")
        st.write("- `Gender`: Gender of the passengers - female, male.")
        st.write("- `Customer Type`: The customer type - loyal customer, disloyal customer.")
        st.write("- `Age`: The actual age of the passengers.")
        st.write("- `Type of Travel`: Purpose of the flight of the passengers - personal travel, business travel).")
        st.write("- `Class`: Travel class in the plane of the passengers - business, eco, eco plus.")
        st.write("- `Flight distance`: The flight distance of this journey.")
        st.write("- `Inflight wifi service`: Satisfaction level of the inflight wifi service - 0 : not applicable, 1 - 5.")
        st.write("- `Departure/Arrival time convenient`: Satisfaction level of departure/arrival time convenient.")
        st.write("- `Ease of Online booking`: Satisfaction level of online booking.")
        st.write("- `Gate location`: Satisfaction level of gate location.")
        st.write("- `Food and drink`: Satisfaction level of food and drink.")
        st.write("- `Online boarding`: Satisfaction level of online boarding.")
        st.write("- `Seat comfort`: Satisfaction level of Seat comfort.")
        st.write("- `Inflight entertainment`: Satisfaction level of inflight entertainment.")
        st.write("- `On-board service`: Satisfaction level of On-board service.")
        st.write("- `Leg room service`: Satisfaction level of Leg room service.")
        st.write("- `Baggage handling`: Satisfaction level of baggage handling.")
        st.write("- `Check-in service`: Satisfaction level of check-in service.")
        st.write("- `Inflight service`: Satisfaction level of inflight service.")
        st.write("- `Cleanliness`: Satisfaction level of cleanliness.")
        st.write("- `Departure Delay in Minutes`: Minutes delayed when departure.")
        st.write("- `Arrival Delay in Minutes`: Minutes delayed when arrival.")
        st.write("- `Satisfaction`: Airline satisfaction level - satisfaction, neutral or dissatisfaction.")
    with tab2:
        st.write("### Data Types:")
        st.write(data.dtypes)
    with tab3:
        st.write("### Sample Data:")
        st.write(data.head())

# EDA Page
if page == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis (EDA) 📊")

    # Count Plot
    st.subheader("Distribution of Customer Type")
    plt.figure(figsize = (10, 5))
    sns.countplot(data = data, x = "Customer Type", hue = "Type of Travel")
    plt.title("Distribution of Customer Type by Type of Travel")
    plt.ylabel("Count")
    st.pyplot(plt)
    container = st.container(border = True)
    container.write("The above countplot displays the distribution of types of customers by their type of travel. There are more loyal customers over disloyal customers and business travel seems to be more popular over personal travel in both categories. 0 represents personal travel and 1 represents business travel.")

    # Histogram
    st.subheader("Distribution of Age")
    plt.figure(figsize = (10, 5))
    sns.histplot(data = data, x = "Age", bins = 25)
    plt.title("Distribution of Customer Age")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    st.pyplot(plt)
    container = st.container(border = True)
    container.write("The above histogram displays the distribution of ages from the dataset, the data is normally distributed.")

    # Scatter Plot
    st.subheader("Departure Delay vs Arrival Delay")
    plt.figure(figsize = (10, 5))
    sns.scatterplot(x = "Departure Delay in Minutes", y = "Arrival Delay in Minutes", data = data, hue = "satisfaction")
    plt.title("Relationship between Departure Delay vs Arrival Delay")
    st.pyplot(plt)
    container = st.container(border = True)
    container.write("The above scatterplot shows a positive relationship between departure delay and arrival delay grouped by satisfaction. As departure delay time increases, arrival delay time also increases and most are neutral or dissatisfied due to lack of unpunctuality. 0 represents neutral or dissatisfied and 1 represents satisfied.")

    # Box Plot
    st.subheader("Flight Distances by Seat Comfort Rating")
    plt.figure(figsize = (10, 5))
    sns.boxplot(x = "Seat comfort", y = "Flight Distance", data = data)
    plt.title("Distributions of Flight Distances by Seat Comfort Rating")
    plt.xticks(rotation = 45)
    st.pyplot(plt)
    container = st.container(border = True)
    container.write("The above side-by-side boxplots display the distributions of flight distances by seat comfort rating. The data shows that as flight distance increases, average seat comfort rating also increases.")

elif page == "Model Training and Evaluation":
    st.title("Model Training and Evaluation 🛠️")

    # Sidebar for Model Selection
    st.sidebar.subheader("Choose a Machine Learning Model")
    model_option = st.sidebar.selectbox("Select a Model", ["K-Nearest Neighbors", "Logistic Regression", "Random Forest"])

    # Prepare the Data
    X = data.drop(columns = "satisfaction")
    y = data["satisfaction"]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, stratify = y)

    # Scale the Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the Selected Model
    if model_option == "K-Nearest Neighbors":
        k = st.sidebar.slider("Select the Number of Neighbors (k)", min_value = 1, max_value = 51, value = 2)
        model = KNeighborsClassifier(n_neighbors = k)
    elif model_option == "Logistic Regression":
        model = LogisticRegression()
    else:
        model = RandomForestClassifier()

    # Train the Model on the Scaled Data
    model.fit(X_train_scaled, y_train)

    # Display Training and Test Accuracy
    st.write(f"**Model Selected: {model_option}**")
    st.write(f"Training Accuracy: {model.score(X_train_scaled, y_train):.2f}")
    st.write(f"Test Accuracy: {model.score(X_test_scaled, y_test):.2f}")

    # Display Confusion Matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, ax = ax, cmap = "Blues")
    st.pyplot(fig)