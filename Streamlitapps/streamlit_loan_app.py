import streamlit as st
import numpy as np
import joblib

#path of locally stored model file
model_name = 'D:\Mlops1_experiment_Packages\Streamlitapps\model.pkl'

try:
    _model = joblib.load(model_name)
except Exception as e:
    st.error(f"Error loding model file - {e}")
    st.stop()
    #print(e)


#prediction function to predict
def predict(Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,
            LoanAmount,Loan_Amount_Term,Credit_History,Property_Area):
    gender_encode = lambda x : 0 if x == 'Female' else 1
    married_encode = lambda x : 0 if x == 'No' else 1
    dependents_encode = lambda x : 0 if x == '0' else 1 if x == '1' else 2 if x == '2' else 3
    education_encode = lambda x : 0 if x == 'Graduate' else 1
    self_employed_encode = lambda x : 0 if x == 'No' else 1
    property_area_encode = lambda x : 0 if x == 'Rural' else 1 if x == 'Semiurban' else 2


    gender = gender_encode(Gender)
    married = married_encode(Married)
    dependents = dependents_encode(Dependents)
    education = education_encode(Education)
    selfemployed = self_employed_encode(Self_Employed)
    credithistory = Credit_History
    propertyarea = property_area_encode(Property_Area)

    applicantincome = ApplicantIncome + CoapplicantIncome
    applicantincome_log = np.log(applicantincome)
    loanamount_log = np.log(LoanAmount)
    loanamountterm = Loan_Amount_Term

    #combine all features into a single array
    input_data = np.array([[gender,married,dependents,education,selfemployed,applicantincome_log,
                            loanamount_log,loanamountterm,credithistory,propertyarea]])
    
    #make predictions
    prediction = _model.predict(input_data)[0]
    return prediction


#print(predict("Male","Yes","0","Graduate","No",10000,20000,100000,240,1.0,"Rural"))


#Streamlit app
def main():
    st.title("Welcome to loan application")
    st.header("Please fill your details to proceed for loan application")

    Gender = st.selectbox("Gender" , ["Male","Female"])
    Married = st.selectbox("Married" , ["Yes","No"])
    Dependents = st.selectbox("Dependents" , ["0","1","2","3+"]) 
    Education = st.selectbox("Education" , ["Graduate","Not Graduate"])
    Self_Employed= st.selectbox("Self Employed" , ["Yes","No"])
    ApplicantIncome = st.number_input("Applicant Income")
    CoapplicantIncome = st.number_input("Coapplicant Income")
    LoanAmount = st.number_input("Loan amount")
    Loan_Amount_Term = st.number_input("Loan amount term")
    Credit_History = st.number_input("Credit history")
    Property_Area = st.selectbox("Property Area" , ["Rural","Urban","Semi Urban"])

    if st.button("Predict"):
        result = predict(Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,
                         LoanAmount,Loan_Amount_Term,Credit_History,Property_Area)
        if result == 1:
            st.success("Application Status:- Approved")
        else:
            st.error("Application Status:- Not approved")

    

if __name__=="__main__":
    main()
