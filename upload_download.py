import streamlit as st
import os

def save_uploaded_file(uploaded_file, save_path):
    with open(os.path.join(save_path, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())

def main():
    st.title("File Upload and Save")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload a file", type=["csv", "txt", "xlsx"])

    if uploaded_file is not None:
        # Specify the local path to save the file
        save_path = "Amzout/"

        # Save the uploaded file to the specified path
        save_uploaded_file(uploaded_file, save_path)

        st.success(f"File successfully saved to {save_path}/{uploaded_file.name}")

if __name__ == "__main__":
    main()
