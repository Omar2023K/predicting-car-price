import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import joblib 
from numerize.numerize import numerize
import streamlit as st 

st.write("please run بقى")

df = pd.read_csv("cleaned_data.csv")






# Welcome Page
def welcome_page():
    
    st.title("Welcome to Car Price Predictor! 🚗")
    nv = st.sidebar.title("Navigation")
    st.sidebar.markdown("""
    - [Home](#Home)
    - [Analysis](#Analysis)
    - [Prediction](#Prediction)
    """)
    
    st.markdown(
        """
        <h2 style="color: #2F4F4F; text-align: center;">Your Car Pricing Assistant</h2>
        <p style="text-align: center; font-size: 18px;">We help you find the best price for your car based on key features.</p>
        <br>
        """,
        unsafe_allow_html=True
    )
    
    st.image("images/car.jpg", use_container_width=True)

    st.markdown("""
    🚗 **Find the best price** for your car based on market data.<br>
    📊 **Explore car data insights** and trends to understand the market.<br>
    💰 **Predict the price** of your car based on features like make, model, mileage, fuel type, and more.
    """, unsafe_allow_html=True)

    st.markdown("### How it works?")
    
    st.markdown("""
    <ul style="list-style-type:none; font-size: 18px;">
        <li>🔍 **Step 1:** Choose the model and features of your car</li>
        <li>📊 **Step 2:** Analyze car prices using historical data trends</li>
        <li>💡 **Step 3:** Get a predicted price for your car's sale</li>
    </ul>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    



def Analysis():

    col1, col2 = st.columns(2)
    with col1:
        car_total = int(df['name'].count())
        st.metric('Number of cars', numerize(car_total))

    with col2:
        car_avg = int(df['selling_price'].mean())
        st.metric('Average_pricing', numerize(car_avg))    

    st.title("Vehicle Market Analysis 📊")
    st.image("images/car_2.jpg", use_container_width=True)

    numeric_columns = df[['selling_price', 'km_driven', 'mileage', 'engine', 'max_power', 'seats', 'year', 'Torque']]
    correlation_matrix = numeric_columns.corr()
    fig10 = px.imshow(correlation_matrix,
                      title="Correlation between each Features",
                      labels={'x': 'Features', 'y': 'Features'},
                      color_continuous_scale='Blues')
    st.plotly_chart(fig10, use_container_width=True, key="correlation_matrix")  # Unique key
    fig10.update_layout(
        width=1000,  # Set the width of the figure
        height=800,  # Set the height of the figure
        title_font_size=20  # Adjust the title font size if necessary
    )
    st.write("NOTICE THAT: The closer the correlation is to 1, the stronger the relationship")

    c1, c2 = st.columns(2)
    with c1:

        df_seller = df['seller_type'].value_counts().reset_index()
        df_seller.columns = ['seller_type', 'count']
        fig1 = px.bar(data_frame=df_seller, x='count', y='seller_type', title='Number of Cars Sold by Seller Type')
        st.plotly_chart(fig1, use_container_width=True, key="seller_type_chart")  # Unique key

        df_tra = df['transmission'].value_counts().reset_index()
        df_tra.columns = ['transmission_type', 'count']
        fig2 = px.bar(data_frame=df_tra, x='transmission_type', y='count', title='Number of Cars Sold each Seller Transmission Type')
        st.plotly_chart(fig2, use_container_width=True, key="transmission_chart")  # Unique key

        df_fue = df['fuel'].value_counts().reset_index()
        df_fue.columns = ['Fuel_type', 'count']
        fig3 = px.bar(data_frame=df_fue, x='Fuel_type', y='count', title='Number of Cars Sold each Fuel Type')
        st.plotly_chart(fig3, use_container_width=True, key="fuel_type_chart")  # Unique key

        df_own = df['owner'].value_counts().reset_index()
        df_own.columns = ['owner_type', 'count']
        fig4 = px.bar(data_frame=df_own, x='owner_type', y='count', title='Number of Cars Sold each owner Status')
        st.plotly_chart(fig4, use_container_width=True, key="owner_type_chart")  # Unique key

        fig11 = px.bar(data_frame=df, x='fuel', y='selling_price', title='Selling Price Distribution by Fuel Type')
        st.plotly_chart(fig11, use_container_width=True, key="fuel_price_chart")  # Unique key

        fig13 = px.bar(data_frame=df, x='seller_type', y='selling_price', title='Selling Price Distribution by Seller Type')
        st.plotly_chart(fig13, use_container_width=True, key="seller_price_chart")  # Unique key

        fig15 = px.scatter(data_frame=df, x='engine', y='selling_price', title='Selling Price vs Engine Capacity')
        st.plotly_chart(fig15, use_container_width=True, key="engine_price_chart")  # Unique key

        fig16 = px.scatter(data_frame=df, x='Torque', y='selling_price', title='Selling Price vs Torque')
        st.plotly_chart(fig16, use_container_width=True, key="torque_price_chart")  # Unique key

    with c2:

        fig5 = px.histogram(df, x='selling_price', title='Distribution of Selling Price')
        st.plotly_chart(fig5, use_container_width=True, key="selling_price_distribution")  # Unique key

        fig7 = px.histogram(df, x='year', title='Number of Cars by Model Year')
        st.plotly_chart(fig7, use_container_width=True, key="cars_by_year")  # Unique key

        fig8 = px.histogram(df, 
                            x='seller_type', 
                            color='fuel', 
                            title='Count of Seller Type by Fuel Type',
                            barmode='stack')
        st.plotly_chart(fig8, use_container_width=True, key="seller_fuel_distribution")  # Unique key

        transmission_percentages = df['transmission'].value_counts(normalize=True) * 100
        fig9 = px.pie(names=transmission_percentages.index, 
                      values=transmission_percentages.values,
                      title='Percentage of Manual vs Automatic Transmission',
                      color=transmission_percentages.index,
                      color_discrete_map={'Manual': 'blue', 'Automatic': 'gray'})
        st.plotly_chart(fig9, use_container_width=True, key="transmission_percentage")  # Unique key

        df_avg_year = df.groupby('year')['selling_price'].mean().reset_index()
        fig10 = px.line(data_frame=df_avg_year, x='year', y='selling_price', title='Average Selling Price vs Year of Manufacture')
        st.plotly_chart(fig10, use_container_width=True, key="average_price_by_year")  # Unique key

        fig12 = px.box(data_frame=df, x='transmission', y='selling_price', title='Selling Price Distribution by Transmission Type')
        st.plotly_chart(fig12, use_container_width=True, key="transmission_price_distribution")  # Unique key

        df_avg_own = df.groupby('owner')['selling_price'].mean().reset_index()
        fig14 = px.bar(data_frame=df_avg_own, x='owner', y='selling_price', title='Average Selling Price for each Owner')
        st.plotly_chart(fig14, use_container_width=True, key="average_price_by_owner")  # Unique key

        fig15 = px.scatter(data_frame=df, x='max_power', y='selling_price', title='Selling Price vs max_power')
        st.plotly_chart(fig15, use_container_width=True, key="max_power_price_chart")  # Unique key






def prediction():

    st.write("### Welcome to the Car Price Prediction Tool! 🚗💰")
    st.image('images/predict_image.png')
    st.write("Please input the following features of your car to predict the best price:")

    year_model = options=sorted(df['year'].unique())
    year_filter = st.selectbox('Select Model Year', options=year_model)

    mileage = st.number_input(
    'Select Maximum Mileage (kmpl)', 
    min_value=int(df['mileage'].min()), 
    max_value=int(df['mileage'].max())+ 10, 
    value=int(df['mileage'].max()),  
    step=5)

    max_power  = st.slider(
    'Select max power (bhp)', 
    min_value=int(df['max_power'].min()), 
    max_value=int(df['max_power'].max())+ 10, 
    value=int(df['max_power'].max()),  
    step=10)

    km_driven  = st.slider(
    'Select km_driven (km)', 
    min_value=int(df['km_driven'].min()), 
    max_value=int(df['km_driven'].max())+ 10, 
    value=int(df['km_driven'].max()),  
    step=100)

    engine  = st.slider(
    'Select engine power (cc)', 
    min_value=int(df['engine'].min()), 
    max_value=int(df['engine'].max())+ 10, 
    value=int(df['engine'].max()),  
    step=100)

    seats  = st.number_input(
    'Select Number of seats', 
    min_value=int(df['seats'].min()), 
    max_value=int(df['seats'].max())+ 10, 
    value=int(df['seats'].max()),  
    step=5)

    Torque  = st.number_input(
    'Select Power of Torque', 
    min_value=int(df['Torque'].min()), 
    max_value=int(df['Torque'].max())+ 10, 
    value=int(df['Torque'].max()),  
    step=100)

    fuel = st.selectbox("Fuel Type",['Diesel','Petrol','LPG','CNG'])
    owner = st.selectbox("Owner",['First Owner','Second Owner','Third Owner','Fourth & Above Owner','Test Drive Car'])

    seller_group = df['seller_type'].unique()
    seller_type = st.radio('Seller',options=seller_group)
    transmission_type = df['transmission'].unique()
    transmission = st.radio('Transmission_Type',options=transmission_type)

    bt = st.button('Predict price')

    if bt == True:
        scaler = joblib.load('model/scaler.pkl')
        target = joblib.load('model/target_scaler.pkl')
        model = joblib.load('model/model.pkl')

        fuel_mapping = {'Diesel': 1, 'Petrol': 3, 'LPG': 2, 'CNG': 0}
        seller_mapping = {'Individual': 1, 'Dealer': 0, 'Trustmark Dealer': 2}
        transmission_mapping = {'Manual': 1, 'Automatic': 0}
        owner_mapping = {
            'First Owner': 0, 
            'Second Owner': 2, 
            'Third Owner': 4, 
            'Fourth & Above Owner': 1, 
            'Test Drive Car': 3}
        fuel_encoded = fuel_mapping[fuel]
        seller_type_encoded = seller_mapping[seller_type]
        transmission_encoded = transmission_mapping[transmission]
        owner_encoded = owner_mapping[owner]

        data_input = np.array([[year_filter,mileage,max_power,km_driven,engine,seats,Torque,fuel_encoded,seller_type_encoded,transmission_encoded,owner_encoded]])
        scaled_data = scaler.transform(data_input)
        mod_pre = model.predict(scaled_data)
        real_result = target.inverse_transform(mod_pre.reshape(-1,1))
        st.success(f"### The predicted price for your car is : $ {real_result[0][0]:,.2f}")

       # st.write(real_result)
            

def main():
    nv = st.sidebar.selectbox("Navigation", ["Home", "Analysis", "Prediction"])

    if nv == "Home":
        welcome_page()
    elif nv == "Analysis":
        Analysis()
    elif nv == "Prediction":
        prediction()

if __name__ == "__main__":
    main()


               
 






        






