"""
streamlit_app.py — Amazon Price Predictor Demo
================================================

A simple web interface to predict product prices using
the fine-tuned Llama 3.2-3B model deployed on Modal.

HOW TO RUN LOCALLY:
    streamlit run app/streamlit_app.py
"""

import streamlit as st
import re

import os

# Set Modal credentials from Streamlit secrets
# Works both locally (from .env) and on Streamlit Cloud (from secrets)
if hasattr(st, 'secrets') and 'modal' in st.secrets:
    os.environ['MODAL_TOKEN_ID']     = st.secrets['modal']['token_id']
    os.environ['MODAL_TOKEN_SECRET'] = st.secrets['modal']['token_secret']
st.set_page_config(
    page_title = "Amazon Price Predictor",
    page_icon  = "💰",
    layout     = "centered",
)


MODEL_NAME  = "maulik78/pricer-2026-06-10_06.40.40-lite"
HF_USER     = "maulik78"
MODEL_MAE   = 58.74
XGBOOST_MAE = 68.23


EXAMPLES = {
    " Sony Headphones": {
        "title":       "Sony WH-1000XM4 Wireless Headphones",
        "category":    "Electronics",
        "brand":       "Sony",
        "description": "Industry-leading noise cancelling wireless headphones",
        "features":    "30 hour battery, touch sensor controls, multipoint connection",
    },
    " DeWalt Drill": {
        "title":       "DeWalt 20V MAX Cordless Drill Driver Kit",
        "category":    "Tools and Home Improvement",
        "brand":       "DeWalt",
        "description": "Compact and lightweight cordless drill",
        "features":    "High-speed transmission, LED work light, 2 batteries included",
    },
    " Nintendo Switch": {
        "title":       "Nintendo Switch OLED Model",
        "category":    "Video Games",
        "brand":       "Nintendo",
        "description": "Gaming console with vibrant 7-inch OLED screen",
        "features":    "TV, tabletop and handheld modes, 64GB storage",
    },
    " KitchenAid Mixer": {
        "title":       "KitchenAid Artisan Series 5-Qt Stand Mixer",
        "category":    "Appliances",
        "brand":       "KitchenAid",
        "description": "Professional stand mixer for home baking",
        "features":    "10 speed settings, dough hook, flat beater, wire whip",
    },
    " Fender Guitar": {
        "title":       "Fender Player Stratocaster Electric Guitar",
        "category":    "Musical Instruments",
        "brand":       "Fender",
        "description": "Classic Stratocaster with modern Player Series pickups",
        "features":    "Alder body, maple neck, 22 frets, 3 single-coil pickups",
    },
}


def build_summary(title, category, brand, description, features) -> str:
    """
    Format product fields into the model's training format.
    """
    parts = []
    if title: parts.append(f"Title: {title}")
    if category: parts.append(f"Category: {category}")
    if brand: parts.append(f"Brand: {brand}")
    if description: parts.append(f"Description: {description}")
    if features: parts.append(f"Details: {features}")
    return "\n".join(parts)


@st.cache_resource
def get_pricer():
    import os
    import modal

    # Authenticate Modal using Streamlit secrets
    # This is needed when running on Streamlit Cloud
    if hasattr(st, 'secrets') and 'modal' in st.secrets:
        os.environ['MODAL_TOKEN_ID']     = st.secrets['modal']['token_id']
        os.environ['MODAL_TOKEN_SECRET'] = st.secrets['modal']['token_secret']

    try:
        Pricer = modal.Cls.from_name("pricer-service", "Pricer")
        return Pricer()
    except Exception as e:
        st.error(f"Could not connect to Modal: {e}")
        return None


def predict_price(summary: str):
    """
    Call the fine-tuned Llama model via Modal.
    Returns (price_float, error_string) — one will always be None.
    """
    pricer = get_pricer()

    if pricer is None:
        return None, "Could not connect to Modal. Make sure the service is deployed."

    try:
        price = pricer.price.remote(summary)
        return float(price), None
    except Exception as e:
        return None, str(e)




st.title("💰 Amazon Price Predictor")
st.markdown(f"""
Predict the price of any Amazon product from its description.
Uses a fine-tuned **Llama 3.2-3B** model deployed on Modal.

**Model:** [`{MODEL_NAME}`](https://huggingface.co/{MODEL_NAME})  
**MAE:** ${MODEL_MAE} on test set &nbsp;|&nbsp; **XGBoost baseline:** ${XGBOOST_MAE} &nbsp;|&nbsp; **Improvement:** 13.9%
""")

st.divider()


st.subheader(" Try an Example")
st.caption("Click any example to pre-fill the form below.")

cols = st.columns(len(EXAMPLES))
selected_ex = None

for col, (label, data) in zip(cols, EXAMPLES.items()):
    if col.button(label, use_container_width=True):
        selected_ex = data

st.divider()


st.subheader(" Product Details")

ex = selected_ex or {}

col1, col2 = st.columns(2)
with col1:
    title    = st.text_input("Product Title *", value=ex.get("title", ""), placeholder="Sony WH-1000XM4")
    category = st.text_input("Category", value=ex.get("category", ""), placeholder="Electronics")
with col2:
    brand = st.text_input("Brand", value=ex.get("brand", ""), placeholder="Sony")

description = st.text_input(
    "Description",
    value = ex.get("description", ""),
    placeholder = "Industry leading noise cancelling headphones"
)

features = st.text_area(
    "Features / Details",
    value = ex.get("features", ""),
    placeholder = "30 hour battery, touch controls, Alexa built in",
    height = 80,
)

# Show what gets sent to the model
if title or description:
    summary = build_summary(title, category, brand, description, features)
    with st.expander("What the model receives"):
        st.code(summary, language=None)

st.divider()


predict = st.button(
    "Predict Price",
    type = "primary",
    use_container_width = True,
)


if predict:

    if not title:
        st.error("Please enter at least a Product Title.")
        st.stop()

    summary = build_summary(title, category, brand, description, features)

    with st.spinner("Connecting to Modal... (first call ~30s, then 2-3s)"):
        price, error = predict_price(summary)

    if error:
        st.error(f"Error: {error}")

    else:
        # Main result
        st.success(f"### Predicted Price: ${price:.2f}")

        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric(
            label = "Predicted Price",
            value = f"${price:.2f}"
        )
        m2.metric(
            label = "Model MAE",
            value = f"±${MODEL_MAE}",
            help  = "Average error on 200 held-out test products"
        )
        m3.metric(
            label = "vs XGBoost",
            value = f"-${XGBOOST_MAE - MODEL_MAE:.2f}",
            delta = f"{100*(XGBOOST_MAE - MODEL_MAE)/XGBOOST_MAE:.1f}% better",
        )

        # Confidence range
        st.info(
            f"**Estimated range:** "
            f"${max(0, price - MODEL_MAE):.2f} – ${price + MODEL_MAE:.2f}  "
            f"*(based on ±${MODEL_MAE} average model error)*"
        )

st.divider()

with st.expander("How does this work?"):
    st.markdown(f"""
    This model was fine-tuned on 20,000 Amazon product descriptions
    using QLoRA — a memory-efficient fine-tuning technique that runs
    on a free Google Colab T4 GPU.

    **Why fine-tuned LLM beats XGBoost:**

    XGBoost counts words independently. It sees "Sony", "noise",
    "cancelling" as three separate features.

    The LLM understands context — "Sony noise-cancelling headphones"
    means premium consumer electronics, not just three word counts.
    That semantic understanding is why it gets $58 MAE vs $68.

    **Results:**
    | Model | MAE ($) |
    |-------|---------|
    | Random baseline | ~$287 |
    | Constant (mean) | ~$141 |
    | XGBoost (800k items) | $68.23 |
    | **Fine-tuned Llama 3.2-3B** | **${MODEL_MAE}** |
    """)

with st.expander(" Technical details"):
    st.markdown(f"""
    **QLoRA = Quantization + LoRA**

    Quantization stores model weights in 4-bit NF4 format:
    6.4 GB → 2.2 GB

    LoRA trains only small adapter matrices instead of all 3B weights:
    3B trainable params → 18M trainable params (0.6%)

    Together: fine-tune a 3B model on a free T4 GPU in under 2 hours.

    | Parameter | Value |
    |-----------|-------|
    | Base Model | Llama 3.2-3B |
    | Training Data | 20,000 products |
    | LoRA Rank | 32 |
    | Trainable Params | 18M (0.6%) |
    | Training Time | 1h 54min |
    | Final Val Loss | 1.248 |
    | Test MAE | ${MODEL_MAE} |

    W&B training run: [wandb.ai/maulik04-lnmiit/pricer](https://wandb.ai/maulik04-lnmiit/pricer)
    """)

with st.expander(" Datasets"):
    st.markdown(f"""
    All datasets are public on HuggingFace:

    | Dataset | Items |
    |---------|-------|
    | [maulik78/items_raw_full](https://huggingface.co/datasets/maulik78/items_raw_full) | 820k |
    | [maulik78/items_full](https://huggingface.co/datasets/maulik78/items_full) | 820k |
    | [maulik78/items_prompts_full](https://huggingface.co/datasets/maulik78/items_prompts_full) | 820k |
    """)


st.divider()
st.markdown(
    f"""
    <div style='text-align:center; color:gray; font-size:12px'>
        Built by <b>Maulik Mathur</b> &nbsp;|&nbsp;
        <a href='https://huggingface.co/{HF_USER}' target='_blank'> HuggingFace</a>
        &nbsp;|&nbsp;
        <a href='https://github.com/maulik-04/amazon-price-predictor' target='_blank'>GitHub</a>
        <br>
        Fine-tuned Llama 3.2-3B · QLoRA · Modal · HuggingFace
    </div>
    """,
    unsafe_allow_html=True,
)
