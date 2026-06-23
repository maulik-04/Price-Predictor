
import streamlit as st
import modal
import re

# Page config
st.set_page_config(
    page_title="AI Price Predictor",
    page_icon="💰",
    layout="centered"
)

# Header
st.title("💰 AI Price Predictor")
st.markdown("""
Predict the price of any product using a fine-tuned **Llama 3.2-3B** model
deployed on Modal cloud infrastructure.

**Model:** `maulik78/pricer-2026-06-10_06.40.40-lite`  
**Trained on:** 20,000 Amazon products using QLoRA on Google Colab T4 GPU
""")

st.divider()

# Input form
st.subheader("Enter Product Details")

col1, col2 = st.columns(2)

with col1:
    title    = st.text_input("Product Title", placeholder="Sony WH-1000XM4 Headphones")
    category = st.text_input("Category",      placeholder="Electronics")

with col2:
    brand       = st.text_input("Brand",       placeholder="Sony")
    description = st.text_input("Description", placeholder="Noise cancelling wireless headphones")

details = st.text_area(
    "Features/Details",
    placeholder="30 hour battery, touch controls, alexa built in",
    height=100
)

# Build summary
def build_summary(title, category, brand, description, details):
    parts = []
    if title:       parts.append(f"Title: {title}")
    if category:    parts.append(f"Category: {category}")
    if brand:       parts.append(f"Brand: {brand}")
    if description: parts.append(f"Description: {description}")
    if details:     parts.append(f"Details: {details}")
    return "\n".join(parts)

# Example products
st.divider()
st.subheader("Try an Example")

examples = {
    "Sony Headphones":     ("Sony WH-1000XM4 Wireless Headphones", "Electronics",   "Sony",      "Industry leading noise cancelling headphones",     "30 hour battery, touch controls, alexa built in"),
    "KitchenAid Mixer":    ("KitchenAid Stand Mixer 5Qt",           "Appliances",    "KitchenAid","Professional 5 quart stand mixer for home baking", "10 speed settings, includes dough hook and beater"),
    "Lego Technic":        ("Lego Technic Bugatti Chiron",          "Toys and Games","Lego",      "Advanced 3599 piece building set",                  "Working steering, W16 engine, top speed indicator"),
    "DeWalt Drill":        ("DeWalt 20V MAX Cordless Drill",        "Tools",         "DeWalt",    "Compact cordless drill for home and professional use","Variable speed, LED light, includes 2 batteries"),
    "iPhone Case":         ("Apple iPhone 15 Pro Leather Case",     "Electronics",   "Apple",     "Genuine leather case for iPhone 15 Pro",            "MagSafe compatible, card holder, drop protection"),
}

selected = st.selectbox("Choose an example:", ["-- Select --"] + list(examples.keys()))

if selected and selected != "-- Select --":
    ex = examples[selected]
    title       = ex[0]
    category    = ex[1]
    brand       = ex[2]
    description = ex[3]
    details     = ex[4]
    st.info(f"Loaded: {selected}")

# Predict button
st.divider()

if st.button("Predict Price", type="primary", use_container_width=True):
    if not title:
        st.error("Please enter at least a product title")
    else:
        summary = build_summary(title, category, brand, description, details)

        with st.spinner("Connecting to Modal cloud... (first call may take 30s)"):
            try:
                # Call Modal deployed service
                Pricer = modal.Cls.from_name("pricer-service", "Pricer")
                pricer = Pricer()
                price  = pricer.price.remote(summary)

                # Display result
                st.success(f"### Predicted Price: ${price:.2f}")

                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Price", f"${price:.2f}")
                with col2:
                    st.metric("Model MAE", "~$58.74")

                with st.expander("Product Summary Sent to Model"):
                    st.code(summary)

            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Make sure Modal service is deployed: `modal deploy pricer_service.py`")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px'>
Built by Maulik | Fine-tuned Llama 3.2-3B | QLoRA | Modal | HuggingFace<br>
<a href='https://huggingface.co/maulik78/pricer-2026-06-10_06.40.40-lite'>
🤗 View Model on HuggingFace
</a>
</div>
""", unsafe_allow_html=True)
