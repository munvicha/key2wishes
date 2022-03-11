import streamlit as st
import streamlit_tags
from streamlit_tags import st_tags
import requests
import json


st.write('# KeytoWishes Demo')
st.markdown(
    """
    This is the initial version of the keytowishes model
    """
)
keywords = st_tags(label='## Enter Keywords:',
                    text='Press enter to add more',
                    value=['chúc mừng', 'vui vẻ', 'hạnh phúc'],
                    key='keywords')
data = {
    'kws': keywords
}

if st.button("Generate"):
    with st.spinner("Generating..."):
        res = requests.post('http://127.0.0.1:8000/api', json=data)
        d = json.loads(res.text)
    st.write("# Generated Sentence:")
    st.write("## {}".format(d.get('text')))
    st.write('# Time Generated:')
    st.write('## {} seconds.'.format(d.get('time')))
