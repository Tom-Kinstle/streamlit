import streamlit as st
import os

st.title("🔐 Streamlit Secrets Test")

st.subheader("Testing Streamlit Secrets Access")

try:
    # Show all available secrets (keys only, not values)
    st.write("Available secret keys:", list(st.secrets.keys()))
    
    # Test OPENAI_API_KEY specifically
    api_key = st.secrets.get("OPENAI_API_KEY")
    if api_key:
        st.success(f"✅ OPENAI_API_KEY found! Length: {len(api_key)} characters")
        st.write(f"Starts with: {api_key[:10]}...")
    else:
        st.error("❌ OPENAI_API_KEY not found in secrets")
        
except Exception as e:
    st.error(f"Error accessing secrets: {e}")

st.subheader("Testing Environment Variables")
env_key = os.getenv("OPENAI_API_KEY")
if env_key:
    st.success(f"✅ OPENAI_API_KEY found in environment! Length: {len(env_key)}")
else:
    st.warning("⚠️ OPENAI_API_KEY not found in environment variables")

st.subheader("Raw Secrets Object")
try:
    st.write("Type of st.secrets:", type(st.secrets))
    st.write("st.secrets content:", dict(st.secrets))
except Exception as e:
    st.error(f"Error showing secrets: {e}")