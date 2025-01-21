import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(
    page_title="Prediksi Asuransi",
    page_icon="💰",
    layout="centered"
)

# Load the saved model
try:
    model = pickle.load(open('model_uas.pkl', 'rb'))
    scaler = StandardScaler()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    
def predict_premium(age, sex, bmi, children, smoker):
    # Create input array and reshape
    features = np.array([age, sex, bmi, children, smoker]).reshape(1, -1)
    
    # Make prediction
    try:
        prediction = model.predict(features)
        return prediction[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def main():
    # Title and student info
    st.title("Insurance Premium Prediction")
    st.markdown("### Student Information")
    st.write("NIM: 2021230002")
    st.write("Name: Rama Haddaf Syachriza")
    
    st.markdown("---")
    
    # Input form
    st.subheader("Enter Customer Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=100, value=25)
        sex = st.selectbox("Sex", options=["Female", "Male"])
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
        
    with col2:
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
        smoker = st.selectbox("Smoker", options=["No", "Yes"])
        
    # Convert categorical inputs
    sex = 1 if sex == "Male" else 0
    smoker = 1 if smoker == "Yes" else 0
    
    # Submit button
    if st.button("Calculate Premium"):
        # Make prediction
        premium = predict_premium(age, sex, bmi, children, smoker)
        
        if premium is not None:
            # Display result
            st.markdown("---")
            st.subheader("Predicted Insurance Premium")
            st.markdown(f"### ${premium:,.2f} per month")
            
            # Additional information
            st.markdown("---")
            st.markdown("### Input Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Age:", age)
                st.write("Sex:", "Male" if sex == 1 else "Female")
                st.write("BMI:", bmi)
            with col2:
                st.write("Children:", children)
                st.write("Smoker:", "Yes" if smoker == 1 else "No")

if __name__ == "__main__":
    main()
