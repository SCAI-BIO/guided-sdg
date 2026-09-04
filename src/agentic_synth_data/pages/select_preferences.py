import streamlit as st

from agentic_synth_data.preference_setup import SynthesisRequest
from agentic_synth_data.utils import display_synthesis_request_st, output_default_parameters

# TODO: implement show after selection (more natural feeling)
st.title("Agentic synthetic data workflow")
st.header("Select preferences")
st.write("---")
st.write("Select preferences -> missing value handling -> another thing")
st.write("---")

request = SynthesisRequest()

st.subheader(
        "Would you like to generate the same amount of samples as the dataset or a fixed " \
        "number of samples?"
    )
choice_1 = st.selectbox(
    "sample size", ["fixed value", "same as original"], label_visibility="hidden", index=None
    )

if choice_1 == "fixed value":
    request.sample_size.mode = "fixed_value"
    st.write("#### How many synthetic samples would you like to generate?")
    value_1 = st.number_input("", min_value=1)
    request.sample_size.fixed_value = value_1
else:
    request.sample_size.mode = "same_as_original"

st.write("---")
st.subheader("Are you interested in differential privacy?")
response_2 = st.selectbox(
    "privacy specification", ["yes", "no"], label_visibility="hidden", index=None
    )

if response_2 == "yes":
    request.privacy.mode = "differential_privacy"

st.write("---")
# Fill data generator preferences
st.subheader("Is there a specific data generator that you want to use on your data?")
st.write("Select 'auto' if you would like a suggestion.")

choice_4 = st.selectbox("data generator",
    [
        "auto",
        "CTGAN_r4r",
        "CTGAN_sdr",
        "PATEGAN",
        "BayesianNet",
        "PrivBayes",
        "CTGAN_sdv",
        "TVAE",
        "GaussianCopula",
        "TabSyn", 
        "STaSY",
    ],
    label_visibility="hidden",
    index=None
)

if choice_4 == "auto":
    st.write("\t=> You will receive a recommendation based on the dataset's characteristics.")
    request.generator.name = choice_4

else:
    if choice_4 != "None":
        st.write(f"\t=> Your data will be generated using {choice_4}.")
request.generator.name = choice_4
request.generator.parameters = output_default_parameters(choice_4) # default parameters

# Missing value handling preferences
st.write("---")
st.subheader("How would you like missing values to be handled?")
st.write("Select 'auto' if you would like a suggestion.")

choice_5 = st.selectbox("", ["auto", "impute", "remove"], label_visibility="hidden", index=None)

if choice_5 == "auto":
    request.missing_values.mode = choice_5
    st.write("\t=> You will receive a recommendation based on the dataset's characteristics.")

elif choice_5 == "remove":
    request.missing_values.mode = choice_5

elif choice_5 == "impute":
    request.missing_values.mode = choice_5
    st.write(
        "Currently, only the SimpleImputer imputation method is currently supported."
    )
    request.missing_values.impute.method = "SimpleImputer"
    st.write("\t=> Your missing data will be imputed by SimpleImputer.")

    # Specify categorical handling.
    st.write(
        "How would you like to handle missing categorical values? Select 'auto' if you " \
        "would like a suggestion."
    )

    choice_6 = st.selectbox(
        "", ["auto", "most frequent", "constant",], label_visibility="hidden", index=None
    )

    if choice_6 == "auto":
        st.write("\t=> You will receive a recommendation based on the dataset's characteristics.")

    if choice_6 == "most frequent":
        choice = "most_frequent"

    else:
        st.write(f"\t=> Your missing categorical data will be imputed using {choice_6}.")

    request.missing_values.impute.categorical_strategy = choice_6

    # Specify numerical handling.
    st.write(
        "How would you like to handle missing numerical values? Select 'auto' if you would" \
       " like a suggestion."
    )

    choice_7 = st.selectbox(
        "", ["auto", "mean", "median", "most frequent"], label_visibility="hidden", index=None
        )
    if choice_7 == "most frequent":
        choice_7 = "most_frequent"

    if choice_7 == "auto":
        st.write("\t=> You will receive a recommendation based on the dataset's characteristics.")

    else:
        st.write(f"\t=> Your missing numerical data will be imputed using {choice_7}.")
    request.missing_values.impute.numerical_strategy = choice_7

st.write("---")
# TODO: distinguish this section (highlighting/shading/etc.)
st.header("Your preferences have been recorded")
display_synthesis_request_st(request)
#return request

col1, col2 = st.columns(2)
with col1:
    back = st.button("back")
    # TODO: add an " are you sure you want to go back button"

with col2:
    # TODO: make continue pop up once everything is done
    cont = st.button("continue")

if cont:
    st.write("to be continued...")
    st.switch_page("./pages/handle_dtypes.py")

if back:
    # TODO: Add a warning (are you sure you want to go back to x step?)
    st.switch_page("./pages/setup.py")
