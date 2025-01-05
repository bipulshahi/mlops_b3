import streamlit as st

st.title("Streamlit Demo")

st.header("Header for the app")

st.text("This is example text")

st.success("Success")
st.warning("Warning")
st.info("Information")
st.error("Error")

if st.checkbox("Select/Deselect"):
    st.text("User selected the checkbox")
else:
    st.text("User has not selected the checkbox")

state = st.radio("What is your favorite color?" , ("Red","Green","Blue"))
if state == 'Green':
    st.success("Well that is my favorite too")


occupation = st.selectbox("What do you do?" , ["Student","Vlogger","Engineer"])
st.text(f"Selected option is {occupation}")

user_name = st.text_input("Enter your name")
user_age = st.number_input("Enter your age")

if st.button("Example button"):
    #st.success("You clicked it")
    st.write(f"Name : {user_name}, Age : {user_age}")