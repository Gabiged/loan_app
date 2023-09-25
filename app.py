import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

model = joblib.load("default_model_prediction.joblib")


def encode_categorical_columns(user_inputs, selected_values):
    encoded_columns = {}
    categorical_columns = {
        'term': [' 36 months', ' 60 months'],
        'emp_length': ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years',
                       '8 years', '9 years', '10+ years'],
        'home_ownership': ['MORTGAGE', 'RENT', 'OWN', 'ANY', 'OTHER'],
        'purpose': ['car', 'credit_card', 'debt_consolidation', 'educational', 'home_improvement', 'house',
                    'major_purchase', 'medical', 'moving', 'other', 'renewable_energy', 'small_business', 'vacation',
                    'wedding'],
        'addr_state': ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN',
                       'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ',
                       'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA',
                       'WI', 'WV', 'WY'],
        'application_type': ['Individual', 'Joint App']
    }
    for column, possible_values in categorical_columns.items():
        for value in possible_values:
            if value == selected_values[column]:
                encoded_columns[
                    f'{column}__{value.replace(" ", "_").replace("+", "plus").replace("<", "less_than")}'] = int(1)
            else:
                encoded_columns[
                    f'{column}__{value.replace(" ", "_").replace("+", "plus").replace("<", "less_than")}'] = int(0)

    user_inputs_encoded = pd.concat([user_inputs, pd.Series(encoded_columns)], axis=1)
    return user_inputs_encoded


def replace_outliers_with_iqr(data, column_name, multiplier=1.5):
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    data[column_name] = data[column_name].apply(
        lambda x: lower_bound
        if x < lower_bound
        else (upper_bound if x > upper_bound else x)
    )
    return data


def convert_to_datetime(date_str):
    if date_str.strip():
        try:
            return pd.to_datetime(date_str, format="%b-%Y")
        except ValueError:
            return None
    else:
        return None


def main():
    bg = """<div style='background-color:grey; padding:10px'>
              <h1 style='color:white'>Streamlit Loan Acceptance Prediction App</h1>
       </div>"""

    st.markdown(bg, unsafe_allow_html=True)

    left, right = st.columns((2, 2))
    loan_amnt = right.number_input('Loan Amount (in $)/loan_amnt', min_value=0.0)
    term = right.selectbox('Term', [' 36 months', ' 60 months'], index=0)
    emp_length = left.selectbox("Employment length in years/emp_length", ('< 1 year', '1 year', '2 years', '3 years',
                                                                          '4 years', '5 years', '6 years', '7 years',
                                                                          '8 years', '9 years', '10+ years'))
    home_ownership = left.selectbox("Home ownership status/home_ownership", ['MORTGAGE', 'RENT', 'OWN', 'ANY', 'OTHER'])
    annual_inc = right.number_input('Annual Income/annual_inc', min_value=0.0)
    purpose = left.selectbox('Purpose for the loan/purpose', ['car', 'credit_card', 'debt_consolidation', 'educational',
                                                              'home_improvement', 'house', 'major_purchase', 'medical',
                                                              'moving', 'other', 'renewable_energy', 'small_business',
                                                              'vacation', 'wedding'])
    addr_state = left.selectbox("State/addr_state", ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA',
                                                     'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME',
                                                     'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM',
                                                     'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
                                                     'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY'])
    dti = right.number_input('Debt-to-Income Ratio (DTI)', min_value=0.0)
    earliest_cr_line = left.text_input("Enter a Date (MMM-YYYY) for the First Credit line/earliest_cr_line")
    fico_range_low = right.number_input("The lower boundary of FICO range/fico_range_low", min_value=0.0,
                                        max_value=1000.0)
    fico_range_high = right.number_input("The upper boundary of FICO range/fico_range_high", min_value=0.0,
                                         max_value=1000.0)
    open_acc = right.number_input('Open credit account count/open_acc', min_value=0.0, step=1.0, max_value=500.0)
    revol_util = right.number_input("Revolving credit utilization(%)/revol_util:", min_value=0.0, step=5.0)
    total_acc = left.number_input("Enter your total number of credit accounts/total_acc:", min_value=0.0, value=0.0,
                                  step=1.0)
    application_type = left.selectbox("Individual or a Joint Application/application_type", ['Individual', 'Joint App'])
    tot_cur_bal = left.number_input("Total Current Balance of All Accounts/tot_cur_bal", min_value=0.0, value=0.0,
                                    step=1.0)
    acc_open_past_24mths = left.number_input("Total count of credit accounts/acc_open_past_24mths", min_value=0.0,
                                             value=0.0, step=1.0)
    mo_sin_old_rev_tl_op = left.number_input("Months since oldest revolving account opened/mo_sin_old_rev_tl_op",
                                             min_value=0.0, value=0.0, step=1.0)
    mo_sin_rcnt_rev_tl_op = left.number_input("Months since most recent revolving account opened/mo_sin_rcnt_rev_tl_op",
                                              min_value=0.0, value=0.0, step=1.0)
    mort_acc = right.number_input("Number of mortgage accounts/mort_acc", min_value=0.0, value=0.0, step=1.0)
    mths_since_recent_inq = left.number_input("Months since most recent inquiry/mths_since_recent_inq")
    num_actv_bc_tl = right.number_input("Number of currently active bankcard accounts/num_actv_bc_tl", min_value=0.0,
                                        value=0.0, step=1.0)
    pub_rec_bankruptcies = right.number_input("Number of public record bankruptcies/pub_rec_bankruptcies",
                                              min_value=0.0, value=0.0, step=1.0)
    user_inputs = pd.DataFrame({
        'loan_amnt': [loan_amnt],
        'term': [term],
        'emp_length': [emp_length],
        'home_ownership':  [home_ownership],
        'annual_inc': [annual_inc],
        'purpose':  [purpose],
        'addr_state': [addr_state],
        'dti': [dti],
        'earliest_cr_line': [earliest_cr_line],
        'fico_range_low': [fico_range_low],
        ''"fico_range_high": [fico_range_high],
        'open_acc': [open_acc],
        'revol_util': [revol_util],
        'total_acc': [total_acc],
        'application_type': [application_type],
        'tot_cur_bal': [tot_cur_bal],
        'acc_open_past_24mths': [acc_open_past_24mths],
        'mo_sin_old_rev_tl_op': [mo_sin_old_rev_tl_op],
        'mo_sin_rcnt_rev_tl_op': [mo_sin_rcnt_rev_tl_op],
        'mort_acc': [mort_acc],
        'mths_since_recent_inq': [mths_since_recent_inq],
        'num_actv_bc_tl': [num_actv_bc_tl],
        'pub_rec_bankruptcies': [pub_rec_bankruptcies]
    })
    user_inputs['earliest_cr_line'] = user_inputs['earliest_cr_line'].apply(convert_to_datetime)
    if user_inputs['earliest_cr_line'].notnull().all():
        date_today = pd.Timestamp.now()
        user_inputs['credit_history'] = (date_today - user_inputs['earliest_cr_line']).dt.days // 365
        user_inputs['credit_history'].fillna(0, inplace=True)
        user_inputs['credit_history'] = user_inputs['credit_history'].astype(int)
        user_inputs.drop('earliest_cr_line', axis=1, inplace=True)
    else:
        st.warning("Invalid date format. Please enter dates in MMM-YYYY format.")

    user_inputs["fico_mean"] = user_inputs[["fico_range_low", "fico_range_high"]].mean(axis=1)
    user_inputs.drop(["fico_range_low", "fico_range_high"], axis=1, inplace=True)
    user_inputs["term"] = user_inputs["term"].str.extract("(\d+)")
    columns_to_replace_outliers = ['loan_amnt', 'annual_inc', 'dti', 'open_acc', 'revol_util', 'total_acc',
                                   'tot_cur_bal', 'acc_open_past_24mths', 'mo_sin_old_rev_tl_op',
                                   'mo_sin_rcnt_rev_tl_op', 'mort_acc', 'mths_since_recent_inq', 'num_actv_bc_tl',
                                   'pub_rec_bankruptcies', 'fico_mean']

    for column in columns_to_replace_outliers:
        user_inputs = replace_outliers_with_iqr(user_inputs, column)

    selected_values = {
        'term': term,
        'emp_length': [emp_length],
        'home_ownership': home_ownership,
        'purpose': purpose,
        'addr_state': addr_state,
        'application_type': application_type
    }

    user_inputs_encoded = encode_categorical_columns(user_inputs, selected_values)
    scaler = StandardScaler()
    user_inputs_encoded[columns_to_replace_outliers] = scaler.fit_transform(user_inputs_encoded[columns_to_replace_outliers])

    button = st.button('Predict')
    if button:

        predictions = model.predict(user_inputs_encoded)
        st.write("Predicted Loan Acceptance:", predictions)


if __name__ == "__main__":
    main()
