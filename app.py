import streamlit as st
from langchain_community.llms import CTransformers

# -------- Cache model loading --------
@st.cache_resource
def load_model():
    llm = CTransformers(
        model="models/llama-2-7b-chat.Q8_0.gguf",   # Path to your .gguf file
        model_type="llama",
        config={
            "max_new_tokens": 256,
            "temperature": 0.01,
        }
    )
    return llm


# -------- Function to get Llama2 response --------
def getLLamaresponse(input_text, no_words, blog_style):
    llm = load_model()

    # Simple prompt (no need PromptTemplate separately)
    prompt = f"""
    Write a blog for a {blog_style} job profile on the topic "{input_text}" within {no_words} words.
    Make it clear, engaging, and informative.
    """

    # Generate response
    response = llm(prompt)
    print(response)
    return response

# -------- Streamlit App --------
st.set_page_config(page_title="Generate Blogs", page_icon='🤖', layout='centered')
st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the Blog Topic")

col1, col2 = st.columns(2)

with col1:
    no_words = st.text_input('Number of Words', value="300")

with col2:
    blog_style = st.selectbox('Writing the blog for', ('Researchers', 'Data Scientist', 'Common People'), index=0)

submit = st.button("Generate")

# Final response
if submit:
    if input_text.strip() == "":
        st.warning("Please enter a blog topic before generating!")
    else:
        with st.spinner("Generating blog... 🚀"):
            result = getLLamaresponse(input_text, no_words, blog_style)
            st.success("Here’s your blog ✨")
            st.write(result)
