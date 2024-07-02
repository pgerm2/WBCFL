
#***********THIS IS ANOTHER VERSION OF THE MAIN CODE****************************

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier


backg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background: rgba(0,0,0,0)

}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
</style>
"""
#Load the model
classifier = joblib.load('classifier.joblib')

st.markdown(backg_img, unsafe_allow_html=True)


def main(title = "We Build Champions for LIFE!".upper()):
    st.markdown("<h1 style='text-align: center; font-size: 50px; color: blue; '>{}</h1>".format(title), 
    unsafe_allow_html=True)
    
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.subheader("SOCOM")
        st.image("images/SOCOM.png",width=110)

    with col2:
            st.subheader("Green Berets")
            st.image("images/SF.png",width=110)

    with col3:
            st.subheader("Rangers")
            st.image("images/Rangers.png",width=110)

    with col4:
            st.subheader("Navy Seals")
            st.image("images/Navy_SEALs.png",width=110)

    with col5:
            st.subheader("Pilot")
            st.image("images/Wings.png",width=110)

    with col6:
            st.subheader("OCS")
            st.image("images/OCS.png",width=110)
    info = ''
    
    
def result():
    st.subheader("PLEAES ANSWER THE FOLLOWING QUESTIONS:", divider='red')


    with st.form('form', clear_on_submit= True, border= True):
        with st.expander("1. Please Answer Question 1 :fire:"):
            pred = st.text_input ('Are you a college grad?')
            pred1 = int(pred)

        with st.expander("2. Please Answer Question 2 :fire:"):
            answer = st.text_input('What is your 5 mile run time?')
           

        with st.expander("3. Please Answer Question 3 :fire:"):
            answer = st.text_input('What is your 1500m swim time?')
          

        with st.expander("4. Please Answer Question 4 :fire:"):
            answer = st.text_input('Placeholder')

        if st.form_submit_button("Submit"):
            # subbut = st.form_submit_button("Submit")
            predansw = np.array(pred)
            answer = predansw.reshape(1 ,1)
            prediction = classifier.predict(([[30,87,44,55]])) #prints a 1D array with with one predicted value.
        
            # if st.session_state.clicked[1]:# 1 = True
            #     uploaded_file = st.file_uploader("Choose a file", type='csv')
            #     if uploaded_file is not None:
            #         df = pd.read_csv(uploaded_file, low_memory = True)
            #         st.header('Uploaded data sample')
            #         st.write(df.head())
            #         classifier = joblib.load('classifier.joblib')   

      

            if pred1 < 10:
                info = 'U.S. Army Special Forces'
                st.success('We Recommend: {}'.format(info)) 

            # elif(prediction[0] == 1):
            #                 info = 'Navy'

            # elif(prediction[0] == 2):
            #                 info = 'OCS'

            else:
                info = 'Pilot'
                st.success('Prediction: {}'.format(info)) 
          

            
        

        
main()  

result()








        # dataset = pd.read_csv('SOF_RFC_Recommend.csv')
        # X = dataset.iloc[:, :-1].values
        # y = dataset.iloc[:, -1].values

        # # Splitting the dataset into the Training set and Test set
        # from sklearn.model_selection import train_test_split
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
        
        # # Feature Scaling
        # from sklearn.preprocessing import StandardScaler
        # sc = StandardScaler()
        # X_train = sc.fit_transform(X_train)
        # X_test = sc.transform(X_test)

        # # Training the Random Forest model on the Training set
        # from sklearn.ensemble import RandomForestClassifier
        # classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        # classifier.fit(X_train, y_train)

        # y_pred = classifier.predict(X_test)
        # print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


      
        # y_pred = classifier.predict_proba(X_test)
        # y_pred = pd.DataFrame(y_pred, columns = ['Salary', 'Purchased'])
        # st.header('Predicted values')
        # st.write(y_pred)

        # pred = y_pred.to_csv(index=False).encode('utf-8')
        # st.download_button('Download prediction',
        #                 pred,
        #                 'prediction.csv',
        #                 'text/csv',
        #                 key='download-csv')

# def clicked_button(button):
#     st.session_state.clicked[button] = True

# st.button("Show Graph", on_click = clicked, args = [1])

# if st.session_state.clicked[1]:# 1 = True
#     chart_data = pd.read_csv('Social_Network_Ads_Short.csv')

#     st.line_chart(data=chart_data, x='Age', y='EstimatedSalary')

# if __name__ == "__main__":
#     main()
  
