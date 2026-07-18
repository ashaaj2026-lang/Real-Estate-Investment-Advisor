import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df1 = pd.read_csv(r"E:\GUVI_PROJ\Project3\df_cleaned.csv", index_col=0)


st.title("🏠 Real Estate Investment Advisor")
st.subheader("Predicting Property Profitability & Future Value")

page = st.sidebar.radio(
    "Navigate",
    ["Intro", "EDA: Visualizations", "Prediction"]
) 
        
if page == "Intro":
    st.write("PROJECT 3")
    st.write("This project predicts future property value after 5 years & Good Investment Recommendation")

    st.write("Dataset Shape:", df1.shape)

    if st.checkbox("View Dataset"):
        st.dataframe(df1)



elif page == "EDA: Visualizations":
     option = st.radio("select the option: ",["avg_price_sqft", "BHK Distribution", "Expensive_loca", "Facing VS Pricepersqft", 
                                              "Prop_owner_type", "Availability", "Amenities_score", "Public_Transport"],  horizontal=True)
         

     if option == "avg_price_sqft":
         avg_price = df1.groupby('Property_Type')['Price_per_SqFt'].mean()
         st.write(avg_price)
         st.bar_chart(avg_price)

     elif option == "BHK Distribution":
         top10_city = df1["City"].value_counts().head(10).index
         temp = df1[df1["City"].isin(top10_city)]
         st.write(temp)
         fig, ax = plt.subplots(figsize=(12,6))
         sns.countplot(data=temp,x="City",hue="BHK",ax=ax)
         plt.xticks(rotation=45)
         plt.title("BHK Distribution Across Top 10 Cities")
         st.pyplot(fig)
         
     elif option == "Expensive_loca":
         expensive_loca = (df1[['Price', 'Locality']].sort_values(by = 'Price', ascending=False).head(5))
         st.write(expensive_loca)
         st.bar_chart(expensive_loca)
     
     elif option == "Facing VS Pricepersqft":
         dir_price= df1.groupby('Facing')['Price_per_SqFt'].mean()
         st.write(dir_price)
         st.scatter_chart(dir_price)
         
     elif option == "Prop_owner_type":
         no_prop_owner_type = df1['Owner_Type'].value_counts()
         st.write(no_prop_owner_type)
         fig, ax = plt.subplots(figsize=(6,6))
         ax.pie(no_prop_owner_type.values,labels=no_prop_owner_type.index,autopct="%1.1f%%",startangle=90)
         ax.set_title("Number of Properties by Owner Type")
         st.pyplot(fig) 

     elif option == "Availability":
         no_availability_type = df1['Availability_Status'].value_counts()
         st.write(no_availability_type)
         st.bar_chart(no_availability_type)

     elif option == "Amenities_score":
         pps_amenities = df1.groupby("Amenities_score")["Price_per_SqFt"].mean()
         st.write(pps_amenities)
         fig, ax = plt.subplots(figsize=(8,5))
         sns.scatterplot(x=pps_amenities.index,y=pps_amenities.values,hue=pps_amenities.index,palette="Set1",ax=ax)
         ax.set_xlabel("Amenities Score")
         ax.set_ylabel("Price per SqFt")
         ax.set_title("Price per SqFt by Amenities Score")
         st.pyplot(fig)

     elif option == "Public_Transport":
         pps_public_trans=  df1.groupby('Public_Transport_Accessibility')['Price_per_SqFt'].mean()
         st.write(pps_public_trans)
         fig, ax = plt.subplots(figsize=(8,5))
         sns.barplot(x=pps_public_trans.index,y=pps_public_trans.values,hue=pps_public_trans.index,palette="Set2",ax=ax)
         ax.set_xlabel("Transport_Availability")
         ax.set_ylabel("Price per SqFt")
         ax.set_title("Price per SqFt by Public_Transport")
         st.pyplot(fig)



    #  columns = [
    #                 "State", "City", "Locality", "Property_Type", "BHK",
    #                 "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt",
    #                 "Year_Built", "Furnished_Status", "Floor_No",
    #                 "Total_Floors", "Age_of_Property", "Nearby_Schools",
    #                 "Nearby_Hospitals", "Public_Transport_Accessibility",
    #                 "Parking_Space", "Security", "Amenities", "Facing",
    #                 "Owner_Type", "Availability_Status", "Price",
    #                 "Amenities_score", "Pool", "Clubhouse", "Garden",
    #                 "Gym", "Playground", "Future_price_5yrs",
    #                 "Good_Investment"
    #            ] 
     
elif page == "Prediction":
    # CATEGORICAL INPUTS
    state_list = sorted(df1["State"].dropna().unique())
    state = st.selectbox("Select State", state_list)

    city_list = sorted(df1["City"].dropna().unique())
    city = st.selectbox("Select City", city_list)

    locality_list = sorted(df1["Locality"].dropna().unique())
    locality = st.selectbox("Select Locality", locality_list)

    property_type_list = sorted(df1["Property_Type"].dropna().unique())
    property_type = st.selectbox("Select Property Type", property_type_list)

    furnished_list = sorted(df1["Furnished_Status"].dropna().unique())
    furnished = st.selectbox("Furnished Status", furnished_list)

    facing_list = sorted(df1["Facing"].dropna().unique())
    facing = st.selectbox("Facing", facing_list)

    owner_list = sorted(df1["Owner_Type"].dropna().unique())
    owner = st.selectbox("Owner Type", owner_list)

    availability_list = sorted(df1["Availability_Status"].dropna().unique())
    availability = st.selectbox("Availability Status", availability_list)

# -------------------------------
# NUMERICAL INPUTS
# -------------------------------
    bhk = st.number_input("Enter BHK", min_value=1, max_value=10, value=2)

    size = st.number_input("Size in SqFt", min_value=100, value=1000) 

    price_per_sqft = st.number_input("Price per SqFt",min_value=1000,value=5000)

    year_built = st.number_input("Year Built", min_value=1950, max_value=2026, value=2015)

    floor_no = st.number_input("Floor Number", min_value=0, value=1)

    total_floors = st.number_input("Total Floors", min_value=1, value=5)

    age = st.number_input("Age of Property", min_value=0, value=5)

    nearby_schools = st.number_input("Nearby Schools", min_value=0, value=2)

    nearby_hospitals = st.number_input("Nearby Hospitals", min_value=0, value=2)

    public_transport = st.selectbox("Public Transport Accessibility",sorted(df1["Public_Transport_Accessibility"].dropna().unique()))


# -------------------------------
# CATEGORICAL / BINARY INPUTS
# -------------------------------

    parking = st.selectbox("Parking Space", [0, 1])

    security = st.selectbox("Security", [0, 1])

    pool = st.selectbox("Pool", [0, 1])

    clubhouse = st.selectbox("Clubhouse", [0, 1])

    garden = st.selectbox("Garden", [0, 1])

    gym = st.selectbox("Gym", [0, 1])

    playground = st.selectbox("Playground", [0, 1])

    amenities_score = st.number_input("Amenities Score",min_value=0,max_value=5,value=3)

    input_data = pd.DataFrame({
        "State": [state],
        "City": [city],
        "Locality": [locality],
        "Property_Type": [property_type],
        "BHK": [bhk],
        "Size_in_SqFt": [size],
        "Price_per_SqFt": [price_per_sqft],
        "Year_Built": [year_built],
        "Furnished_Status": [furnished],
        "Floor_No": [floor_no],
        "Total_Floors": [total_floors],
        "Age_of_Property": [age],
        "Nearby_Schools": [nearby_schools],
        "Nearby_Hospitals": [nearby_hospitals],
        "Public_Transport_Accessibility": [public_transport],
        "Parking_Space": [parking],
        "Security": [security],
        "Facing": [facing],
        "Owner_Type": [owner],
        "Availability_Status": [availability],
        "Amenities_score": [amenities_score],
        "Pool": [pool],
        "Clubhouse": [clubhouse],
        "Garden": [garden],
        "Gym": [gym],
        "Playground": [playground]
    }) 


    # Encode categorical columns

    import pickle

    le_state = pickle.load(open("state_encoder.pkl", "rb"))
    le_city = pickle.load(open("city_encoder.pkl", "rb"))
    le_locality = pickle.load(open("locality_encoder.pkl", "rb"))
    le_property = pickle.load(open("property_encoder.pkl", "rb"))
    le_facing = pickle.load(open("facing_encoder.pkl", "rb"))
    le_owner = pickle.load(open("owner_encoder.pkl", "rb"))


    input_data["State"] = le_state.transform(input_data["State"])

    input_data["City"] = le_city.transform(input_data["City"])

    input_data["Locality"] = le_locality.transform(input_data["Locality"])

    input_data["Property_Type"] = le_property.transform(input_data["Property_Type"])

    input_data["Facing"] = le_facing.transform(input_data["Facing"])

    input_data["Owner_Type"] = le_owner.transform(input_data["Owner_Type"]) 


    # Map categorical values

    input_data["Availability_Status"] = input_data["Availability_Status"].map({"Under_Construction": 0,"Ready_to_Move": 1})

    input_data["Furnished_Status"] = input_data["Furnished_Status"].map({"Unfurnished": 0,"Semi-furnished": 1,"Furnished": 2})

    input_data["Public_Transport_Accessibility"] = input_data["Public_Transport_Accessibility"].map({"Low": 0,"Medium": 1,"High": 2}) 

    input_data_xgb = input_data.copy() 
    price = st.number_input("Current Price in Rupees", min_value=100000, value=5000000)

    input_data_xgb.insert(20, "Price", price)

    @st.cache_resource
    def load_model():
        with open("rf_model.pkl", "rb") as file:
          model = pickle.load(file)
        return model 
    rf_model = load_model()

    @st.cache_resource
    def load_xgb_model():
        with open("xgb_model.pkl", "rb") as file:
          model = pickle.load(file)
        return model
    xgb_model = load_xgb_model()
        

    if st.button("Predict Future Price"):
        prediction = rf_model.predict(input_data)
        st.success(f"Predicted Future Price: ₹{prediction[0]:,.2f}")


    investment_prediction = xgb_model.predict(input_data_xgb)
    
    if investment_prediction[0] == 1:
        st.success("Good Investment: Yes")
    else:
        st.warning("Good Investment: No")

# prediction = model.predict(input_data)



# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestRegressor
# import pickle

# categorical_columns = ["City"]
# numerical_columns = ["Area", "Bedrooms", "Property_Age"]

# preprocessor = ColumnTransformer(
#     transformers=[
#         (
#             "categorical",
#             OneHotEncoder(handle_unknown="ignore"),
#             categorical_columns
#         ),
#         (
#             "numerical",
#             StandardScaler(),
#             numerical_columns
#         )
#     ]
# )

# pipeline = Pipeline([
#     ("preprocessor", preprocessor),
#     ("model", RandomForestRegressor(random_state=42))
# ])

# X = df[
#     ["City", "Area", "Bedrooms", "Property_Age"]
# ]

# y = df["Price"]

# pipeline.fit(X, y)

# with open("house_price_model.pkl", "wb") as file:
#     pickle.dump(pipeline, file)



# input_data = pd.DataFrame({
#         "City": [city],
#         "Area": [area],
#         "Bedrooms": [bedrooms],
#         "Property_Age": [age]
#     })
        


    #  elif option == "property_type":
    #     query = " select  property_type, avg(price) as avg_price from listings GROUP BY property_type"
    #     vs2 = pd.read_sql(query, conn) 
    #     st.line_chart(vs2, x= "property_type", y= "avg_price")
    #  elif option == "furnished_price":
    #     query = """
    #     select p.furnishing_status, (ROUND(AVG(l.price),2)) as avg_price
    #     from listings l join property p ON l.listing_id = p.listing_id  
    #     GROUP BY 1
    #     """
    #     vs3 = pd.read_sql(query, conn) 
    #     st.bar_chart(vs3, x= "furnishing_status", y= "avg_price")
    #  elif option == "available":
    #     query = """
    #     SELECT l.listing_id,l.price
    #     FROM listings l LEFT JOIN sales s 
    #     ON l.listing_id = s.listing_id 
    #     WHERE s.date_sold IS NULL
    #     """
    #     vs4 = pd.read_sql(query, conn) 
    #     st.scatter_chart(vs4, x= "listing_id", y= "price")








# city  = input()
# input(city) = "?"    
# update listings set city = "mexico", price = 1200000, where listing_id =  l200045
#                  input("city") = "?", ;

# if page == "intro":
#   st.title("BRICK VIEW ") 
#   st.write(" ") 
# elif page == "filter": 
#   query ="SELECT city, (ROUND(AVG(price),2)) FROM listings GROUP BY city" 
#   df1 = pd.read_sql(query, conn) 
#   st.write(df1)
# elif page == "filter":
#   query = "SELECT property_type, AVG(price/sqft) AS avg_price_per_sqft FROM listings GROUP BY property_type" 
#   df2 = pd.read_sql(query, conn) 
#   st.write(df2)


# if page == "filters":
#     option = st.radio("select the option: "
#                       ["listings", "city", "property type", "sold", "available"],
#                       horizontal=True)    
#     if option == "listings":
#         query = " select * from listings "
#         opt1 = pd.read_sql(query, conn) 
#         st.write(opt1)
#     elif option == "city":
#         query = " select city, avg(price) as avg_price from listings GROUP BY city"
#         opt2 = pd.read_sql(query, conn) 
#         st.write(opt2)
#     elif option == "property type":
#         query = " select  property_type, avg(price) as avg_price from listings GROUP BY property_type"
#         opt3 = pd.read_sql(query, conn) 
#         st.write(opt3)
#     elif option == "sold":
#         query = " select  listing_id, sale_price as price from sales"
#         opt4 = pd.read_sql(query, conn) 
#         st.write(opt4) 
#     elif option == "available":
#         query = " SELECT l.listing_id, l.price FROM listings l LEFT JOIN sales s ON l.listing_id = s.listing_id WHERE s.date_sold IS NULL"
#         opt5 = pd.read_sql(query, conn) 
#         st.write(opt5)



# str.set_page_config(page_title= "Real Estate Investment Advisor", layout = "centered")
# str.title("Predicting Property Profitability & Future Value")
# str.markdown("Enter Parameters to check the Price after 5 yrs")
# str.sidebar.header("Metrics")

# # input values
# state = str.sidebar.number_input("STATE", min_value=1, max_value=20, value=1)
# city = str.sidebar.number_input("CITY", min_value=1, max_value=41, value=1)
# locality = str.sidebar.number_input("LOCALITY", min_value=1, max_value=500, value=1)
# property_type = str.sidebar.selectbox("Property Type", ["Apartment= 0,  Independent House= 1,    Villa=2"])




# with open("model.pkl", "rb") as file:
#     loaded_model = pickle.load(file)

# person_income = str.sidebar.number_input("Applicant Anual Salary INR", min_value=0, value=65000, step= 1000)
# person_emp_length = str.sidebar.slider("Applicant Work Experiance", min_value=2, max_value=40, value =5) 
# loan_amt = str.sidebar.number_input("Requested Loan Amount", min_value=1000, value=15000, step= 100) 
# home_ownership = str.sidebar.selectbox("Ownership Type", ["RENT", "MORTGAGE", "OWN"]) 
# credit_score = str.sidebar.number_input("Cibil Score", min_value=400, max_value=800, value=700)