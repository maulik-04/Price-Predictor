import streamlit as st
import os

st.set_page_config(page_title="Price Predictor", page_icon="💰")

# ── Constants ─────────────────────────────────────────────────
MODEL_NAME  = "maulik78/pricer-2026-06-10_06.40.40-lite"
MODEL_MAE   = 58.74
XGBOOST_MAE = 68.23

EXAMPLES = {
    "Sony Headphones": {
        "title":       "Sony WH-1000XM4 Wireless Headphones",
        "category":    "Electronics",
        "brand":       "Sony",
        "description": "Industry-leading noise cancelling wireless headphones",
        "features":    "30 hour battery, touch sensor controls, multipoint connection",
    },
    "DeWalt Drill": {
        "title":       "DeWalt 20V MAX Cordless Drill Driver Kit",
        "category":    "Tools and Home Improvement",
        "brand":       "DeWalt",
        "description": "Compact and lightweight cordless drill",
        "features":    "High-speed transmission, LED work light, 2 batteries included",
    },
    "Nintendo Switch": {
        "title":       "Nintendo Switch OLED Model",
        "category":    "Video Games",
        "brand":       "Nintendo",
        "description": "Gaming console with vibrant 7-inch OLED screen",
        "features":    "TV, tabletop and handheld modes, 64GB storage",
    },
    "KitchenAid Mixer": {
        "title":       "KitchenAid Artisan Series 5-Qt Stand Mixer",
        "category":    "Appliances",
        "brand":       "KitchenAid",
        "description": "Professional stand mixer for home baking",
        "features":    "10 speed settings, dough hook, flat beater, wire whip",
    },
    "Fender Guitar": {
        "title":       "Fender Player Stratocaster Electric Guitar",
        "category":    "Musical Instruments",
        "brand":       "Fender",
        "description": "Classic Stratocaster with modern Player Series pickups",
        "features":    "Alder body, maple neck, 22 frets, 3 single-coil pickups",
    },
}


def build_summary(title, category, brand, description, features):
    parts = []
    if title:       parts.append(f"Title: {title}")
    if category:    parts.append(f"Category: {category}")
    if brand:       parts.append(f"Brand: {brand}")
    if description: parts.append(f"Description: {description}")
    if features:    parts.append(f"Details: {features}")
    return "\n".join(parts)


@st.cache_resource
def get_pricer():
    if hasattr(st, 'secrets') and 'modal' in st.secrets:
        os.environ['MODAL_TOKEN_ID']     = st.secrets['modal']['token_id']
        os.environ['MODAL_TOKEN_SECRET'] = st.secrets['modal']['token_secret']
    try:
        import modal
        Pricer = modal.Cls.from_name("pricer-service", "Pricer")
        return Pricer()
    except:
        return None


# ── Header ────────────────────────────────────────────────────
st.title("Amazon Price Predictor")
st.write("Fine-tuned Llama 3.2-3B · MAE $58.74 · 13.9% better than XGBoost")
st.caption(f"Model: [{MODEL_NAME}](https://huggingface.co/{MODEL_NAME})")

st.divider()

# ── Examples ──────────────────────────────────────────────────
st.write("**Try an example:**")
cols = st.columns(len(EXAMPLES))

for col, (label, data) in zip(cols, EXAMPLES.items()):
    if col.button(label, use_container_width=True):
        st.session_state['title']       = data['title']
        st.session_state['category']    = data['category']
        st.session_state['brand']       = data['brand']
        st.session_state['description'] = data['description']
        st.session_state['features']    = data['features']

st.divider()

# ── Input ─────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    title    = st.text_input("Title *",    value=st.session_state.get('title', ''),    key="title")
    category = st.text_input("Category",   value=st.session_state.get('category', ''), key="category")
with col2:
    brand    = st.text_input("Brand",      value=st.session_state.get('brand', ''),    key="brand")

description = st.text_input("Description", value=st.session_state.get('description', ''), key="description")
features    = st.text_area("Features",     value=st.session_state.get('features', ''),    key="features", height=80)

st.divider()

# ── Predict ───────────────────────────────────────────────────
if st.button("Predict Price", type="primary", use_container_width=True):

    if not st.session_state.get('title'):
        st.error("Please enter a product title.")
        st.stop()

    summary = build_summary(
        st.session_state.get('title', ''),
        st.session_state.get('category', ''),
        st.session_state.get('brand', ''),
        st.session_state.get('description', ''),
        st.session_state.get('features', ''),
    )

    with st.spinner("Running inference on Modal T4 GPU..."):
        pricer = get_pricer()
        if pricer is None:
            st.error("Could not connect to Modal. Make sure the service is deployed.")
            st.stop()
        try:
            price = float(pricer.price.remote(summary))
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    st.success(f"**Predicted Price: ${price:.2f}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted",  f"${price:.2f}")
    c2.metric("Model MAE",  f"±${MODEL_MAE}")
    c3.metric("vs XGBoost", f"-${XGBOOST_MAE - MODEL_MAE:.2f}", f"13.9% better")

    st.caption(f"Estimated range: ${max(0, price - MODEL_MAE):.2f} – ${price + MODEL_MAE:.2f}")

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption("Maulik Mathur · [GitHub](https://github.com/maulik-04/amazon-price-predictor) · [HuggingFace](https://huggingface.co/maulik78)")
