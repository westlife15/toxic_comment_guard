import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import re
import time

# ==============================================================================
# 1. CONFIGURATION & ARCHITECTURE
# ==============================================================================

class RobertaMultiSampleDropoutHead(nn.Module):
    def __init__(self, hidden_size, num_labels, num_samples=5, dropout_rate=0.2):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(num_samples)])
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features):
        x = features[:, 0, :]
        x = self.dense(x)
        x = torch.tanh(x)
        logits = torch.mean(
            torch.stack([self.out_proj(dropout(x)) for dropout in self.dropouts], dim=0),
            dim=0
        )
        return logits

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

CUSTOM_THRESHOLDS = {
    'toxic': 0.50, 'severe_toxic': 0.66, 'obscene': 0.50, 
    'threat': 0.50, 'insult': 0.50, 'identity_hate': 0.50
}

SEVERITY_WEIGHTS = {
    'toxic': 1, 'severe_toxic': 5, 'obscene': 2, 
    'threat': 5, 'insult': 1, 'identity_hate': 4
}

# ==============================================================================
# 2. LOADING THE MODEL
# ==============================================================================

@st.cache_resource
def load_model():
    model_path = "." 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=6)
    model.classifier = RobertaMultiSampleDropoutHead(
        hidden_size=model.config.hidden_size, num_labels=6, num_samples=5
    )
    model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin", map_location=torch.device('cpu')))
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ==============================================================================
# 3. HELPER FUNCTIONS
# ==============================================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r'[^\x00-\x7F]+', '', text) 
    return text.strip()

def analyze_toxicity(text):
    clean_msg = clean_text(text)
    inputs = tokenizer(
        clean_msg, add_special_tokens=True, max_length=128,
        padding='max_length', truncation=True, return_tensors='pt'
    )
    with torch.no_grad():
        outputs = model(**inputs).logits
        probs = torch.sigmoid(outputs).numpy()[0]
        
    prob_dict = {LABELS[i]: probs[i] for i in range(len(LABELS))}
    active_flags = []
    score = 0
    for label, prob in prob_dict.items():
        if prob >= CUSTOM_THRESHOLDS[label]:
            active_flags.append(label)
            score += SEVERITY_WEIGHTS[label]
            
    final_score = min(5, score)
    if 'severe_toxic' in active_flags or 'threat' in active_flags or 'identity_hate' in active_flags:
        action = "AUTO-BAN"
        final_score = 5
    elif final_score >= 3:
        action = "MUTE (24h)"
    elif final_score >= 1:
        action = "WARNING"
    else:
        action = "ALLOW"
    return active_flags, final_score, action, prob_dict

# ==============================================================================
# 4.UI LAYOUT
# ==============================================================================

st.set_page_config(
    page_title="Toxic Comment Guard", 
    page_icon="🛡️",
    layout="centered"
)


st.markdown("""
<style>
    /* Make the title stand out */
    h1 {
        color: #FF4B4B;
        text-shadow: 2px 2px 4px #000000;
    }
    /* Style the analyze button */
    div.stButton > button:first-child {
        background-color: #FF4B4B;
        color: white;
        font-size: 20px;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 24px;
        border: 2px solid #FF4B4B;
    }
    div.stButton > button:first-child:hover {
        background-color: #333333;
        color: #FF4B4B;
        border: 2px solid #FF4B4B;
    }
    /* Add a subtle border to metrics */
    div[data-testid="stMetricValue"] {
        font-size: 3rem;
    }
</style>
""", unsafe_allow_html=True)

#SIDEBAR 
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png")
    st.title("System Specs")
    st.info("**Model:** RoBERTa-base")
    st.info("**Architecture:** Multi-Sample Dropout")
    st.info("**Loss Function:** Focal Loss")
    st.markdown("---")
    st.write("Designed for high-speed gaming chat moderation.")

#MAIN PAGE
st.title("🛡️ Gaming Chat Moderator")
st.markdown("##### *Live Toxicity Detection System powered by RoBERTa*")


with st.container():
    user_input = st.text_area(
        "📝 Enter Chat Message:", 
        height=100, 
        placeholder="Type a message here (e.g. 'gg everyone' or 'u suck')..."
    )

    if st.button("🚀 Analyze Message", use_container_width=True):
        if not user_input:
            st.warning("⚠️ Please type a message first.")
        else:
            
            with st.spinner("🔍 Scanning neural patterns..."):
                time.sleep(0.6) 
                flags, score, action, prob_dict = analyze_toxicity(user_input)
            
            
            st.markdown("---")
            
            
            if action == "AUTO-BAN":
                header_color = "red"
                emoji = "🛑"
            elif action == "MUTE (24h)":
                header_color = "orange"
                emoji = "🔇"
            elif action == "WARNING":
                header_color = "gold"
                emoji = "⚠️"
            else:
                header_color = "green"
                emoji = "✅"
            
            st.markdown(f"<h2 style='text-align: center; color: {header_color};'>{emoji} Decision: {action}</h2>", unsafe_allow_html=True)
            
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Risk Level", f"{score}/5", delta="High Risk" if score >= 4 else "Safe")
            
            with col2:
                
                st.write("**Flags Detected:**")
                if flags:
                    for f in flags:
                        st.error(f"🚩 {f.upper()}")
                else:
                    st.success("✨ CLEAN")
            
            with col3:
                
                max_prob = max(prob_dict.values())
                
                if action == "ALLOW":
                    
                    safety_score = 1.0 - max_prob
                    st.metric("Safety Score", f"{safety_score:.1%}")
                    st.progress(float(safety_score))
                else:
                    
                    st.metric("Toxicity Confidence", f"{max_prob:.1%}")
                    st.progress(float(max_prob))

            
            st.markdown("---")
            with st.expander("📊 View Detailed Probability breakdown"):
                st.write("The AI's raw confidence scores for each category:")
                
                for label, prob in prob_dict.items():
                    col_txt, col_bar = st.columns([1, 4])
                    with col_txt:
                        st.write(f"**{label.title()}**")
                    with col_bar:
                        
                        bar_color = ":red[" if prob > CUSTOM_THRESHOLDS[label] else ":green["
                        st.progress(float(prob))
                        st.caption(f"{prob:.4f}")