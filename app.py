import os
import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from openai import OpenAI

# Import from our new modules
from ui_styles import apply_custom_styles
from emergency_logic import (
    check_emergency_condition, 
    calculate_emergency_score, 
    get_emergency_advice,
    fahrenheit_to_celsius,
    celsius_to_fahrenheit
)
from ml_models import train_ml_models, map_risk_level
from llm_utils import build_llm_response, generate_summary_md

# =========================================================
#  CONFIG + ENV - FIXED VERSION
# =========================================================
api_key = None

# Try Streamlit secrets first
try:
    api_key = st.secrets.get("OPENAI_API_KEY")
except Exception:
    api_key = None

# If Streamlit secrets not available, try .env file
if not api_key:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
    except Exception:
        api_key = None

# FINAL CHECK
if not api_key:
    st.error("""
    ⚠ No OpenAI API key found! 
    
    Please set it in one of these ways:
    
    1. **Streamlit Secrets** (Recommended for deployment):
       - Create a file at: C:\\Users\\ISHAAN SHARMA\\.streamlit\\secrets.toml
       - Add: OPENAI_API_KEY = "your-api-key-here"
    
    2. **Environment Variable** (For local development):
       - Create a .env file in your project folder
       - Add: OPENAI_API_KEY=your-api-key-here
    
    3. **Direct Input** (Temporary):
       - Enter your API key in the sidebar
    """)
    
    # Allow manual API key input as fallback
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔑 Temporary API Key Setup")
        manual_api_key = st.text_input("Enter OpenAI API Key:", type="password")
        if manual_api_key:
            api_key = manual_api_key
            st.success("✅ Using manually entered API key")
    
    if not api_key:
        st.stop()

DOCTOR_NAME = "Dr. Kshitij Bhatnagar"
DATASET_PATH = "doctor_kshitij_cases.csv"

def test_openai():
    if not api_key:
        st.error("No API key available for testing")
        return
        
    client = OpenAI(api_key=api_key)
    try:
        models = client.models.list()
        st.success(f"✅ OpenAI key working. Total models: {len(models.data)}")
    except Exception as e:
        st.error(f"❌ OpenAI error: {e}")

# sidebar ya main me
if st.button("Test OpenAI Connection"):
    test_openai()

# =========================================================
#  STREAMLIT APP
# =========================================================
def main():
    st.set_page_config(
        page_title=f"{DOCTOR_NAME} – AI Assistant",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    apply_custom_styles()

    # ========== TRY LOADING + TRAINING MODELS ==========
    try:
        models_dict, metrics_dict, feature_cols = train_ml_models(DATASET_PATH)
    except Exception as e:
        st.error(f"Error loading/training models from {DATASET_PATH}: {e}")
        st.stop()

    # SESSION STATE INIT
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "patient_info" not in st.session_state:
        st.session_state.patient_info = {}
    if "prediction_info" not in st.session_state:
        st.session_state.prediction_info = {}
    if "consultation_done" not in st.session_state:
        st.session_state.consultation_done = False
    if "waiting_for_approval" not in st.session_state:
        st.session_state.waiting_for_approval = False
    if "approval_status" not in st.session_state:
        st.session_state.approval_status = None
    if "is_emergency" not in st.session_state:
        st.session_state.is_emergency = False
    if "emergency_conditions" not in st.session_state:
        st.session_state.emergency_conditions = []
    if "monitor_conditions" not in st.session_state:
        st.session_state.monitor_conditions = []

    # ----------- SIDEBAR -----------
    with st.sidebar:
        st.markdown(f"""
        <div style='text-align: center; padding: 1rem 0;'>
            <h2 style='color: #f8fafc; margin: 0;'>🩺 MEDICAL AI</h2>
            <p style='color: #94a3b8; margin: 0; font-size: 0.9rem;'>Enhanced ML Models · Feature Engineering</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        role = st.radio("**Select Role:**", ["Patient", "Doctor"])
        
        st.markdown("---")
        
        # Model Selection
        selected_model_name = st.selectbox(
            "🔧 Select ML Model for Prediction",
            list(models_dict.keys()),
            index=0,
            help="Choose which ML algorithm to use for risk prediction"
        )
        
        # Selected Model Metrics
        selected_metrics = metrics_dict[selected_model_name]
        with st.expander("📊 Selected Model Metrics", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("R² Score", f"{selected_metrics['r2']:.3f}")
                st.metric("Accuracy %", f"{selected_metrics['accuracy_percent']:.1f}%")
            with col2:
                st.metric("RMSE", f"{selected_metrics['rmse']:.2f}")
                st.metric("MAE", f"{selected_metrics['mae']:.2f}")
        
        # All Models Comparison
        with st.expander("📈 Model Comparison (All)", expanded=False):
            comp_data = []
            for model_name, metrics in metrics_dict.items():
                comp_data.append({
                    "Model": model_name,
                    "R²": f"{metrics['r2']:.3f}",
                    "Accuracy %": f"{metrics['accuracy_percent']:.1f}%",
                    "RMSE": f"{metrics['rmse']:.2f}",
                    "MAE": f"{metrics['mae']:.2f}"
                })
            comp_df = pd.DataFrame(comp_data)
            st.dataframe(comp_df, use_container_width=True)
            
            # GRAPHS FOR MODEL COMPARISON
            st.markdown("### 📊 Model Performance Graphs")
            
            models = list(metrics_dict.keys())
            r2_scores = [metrics_dict[m]['r2'] * 100 for m in models]  # Convert to % accuracy
            accuracy_scores = [metrics_dict[m]['accuracy_percent'] for m in models]
            rmse_scores = [metrics_dict[m]['rmse'] for m in models]
            
            # Bar Chart - Accuracy %
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            bars = ax1.bar(models, accuracy_scores, color=['#2563eb', '#059669', '#d97706', '#dc2626', '#7c3aed'])
            ax1.set_title("Model Accuracy % (Normalized from RMSE)", fontsize=12, fontweight='bold')
            ax1.set_ylabel("Accuracy %", fontweight='bold')
            ax1.set_xticklabels(models, rotation=45, ha='right')
            # Add value labels on bars
            for bar, value in zip(bars, accuracy_scores):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
            ax1.grid(axis='y', alpha=0.3)
            st.pyplot(fig1)
            
            # Line Chart - RMSE
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot(models, rmse_scores, marker='o', linewidth=2, markersize=8, color='#dc2626')
            ax2.set_title("RMSE Comparison (Lower = Better)", fontsize=12, fontweight='bold')
            ax2.set_ylabel("RMSE Value", fontweight='bold')
            ax2.set_xticklabels(models, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            # Add value labels on points
            for i, value in enumerate(rmse_scores):
                ax2.text(i, value + 0.1, f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
            st.pyplot(fig2)
        
        with st.expander("🆘 Enhanced Emergency Signs", expanded=True):
            st.markdown("""
            **Critical Emergencies:**
            - Fever ≥ 39.5°C (103.1°F)
            - Severe ear pain + fever
            - Severe/crushing chest pain
            - BP > 180/120 or < 90/60
            - Heart rate > 120 or < 40
            - SpO₂ < 90% (Emergency)
            - ML Risk Score ≥ 80
            
            **Monitor Closely:**
            - SpO₂ 90–94%
            - Moderate chest pain
            - Moderate fever (38.5-39.4°C)
            """)
        
        with st.expander("ℹ️ Quick Guide", expanded=False):
            st.markdown("""
            **For Patients:**
            - Fill health details accurately
            - Emergency symptoms get priority
            - AI provides initial guidance only
            
            **For Doctors:**
            - Review AI recommendations
            - Final decision always with doctor
            
            **ML Models:**
            - All models use enhanced feature engineering
            - Feature engineering improves accuracy
            - Random Forest: Best for complex patterns
            """)

    # ----------- MAIN HEADER -----------
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0 2rem 0;'>
        <h1 style='color: #f8fafc; margin: 0; font-size: 2.5rem;'>⚕ ENHANCED MEDICAL AI CONSULTATION</h1>
        <p style='color: #94a3b8; margin: 0.5rem 0 0 0; font-size: 1.1rem;'>
        Feature Engineering · Multiple ML Models · Enhanced Emergency Detection
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Main layout
    col_main, col_side = st.columns([2.5, 1], gap="large")

    # =====================================================
    #                 PATIENT PORTAL
    # =====================================================
    if role == "Patient":
        with col_main:
            # Emergency Alert if conditions detected
            if st.session_state.emergency_conditions:
                st.markdown(f"""
                <div class='professional-card emergency-card'>
                    <h3 style='color: #f8fafc; margin: 0 0 0.5rem 0;'>🚨 Emergency Alert</h3>
                    <p style='margin: 0; color: #f8fafc;'><strong>Critical conditions detected:</strong></p>
                    <ul style='margin: 0.5rem 0 0 0; color: #f8fafc;'>
                        {''.join([f'<li>{condition}</li>' for condition in st.session_state.emergency_conditions])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Monitor Conditions Alert
            if st.session_state.monitor_conditions and not st.session_state.emergency_conditions:
                st.markdown(f"""
                <div class='professional-card' style='border-left: 4px solid #d97706 !important;'>
                    <h3 style='color: #f8fafc; margin: 0 0 0.5rem 0;'>⚠️ Monitor Closely</h3>
                    <p style='margin: 0; color: #f8fafc;'><strong>Conditions requiring monitoring:</strong></p>
                    <ul style='margin: 0.5rem 0 0 0; color: #f8fafc;'>
                        {''.join([f'<li>{condition}</li>' for condition in st.session_state.monitor_conditions])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            # Consultation Status Card
            if st.session_state.waiting_for_approval:
                st.markdown(f"""
                <div class='status-card status-waiting'>
                    <h3 style='color: #fbbf24; margin: 0 0 1rem 0;'>⏳ Consultation Sent for Review</h3>
                    <p style='color: #e2e8f0; margin: 0.5rem 0; font-size: 1.1rem;'>
                    <strong>Your consultation has been sent to {DOCTOR_NAME}</strong>
                    </p>
                    <p style='color: #94a3b8; margin: 0;'>
                    The doctor will review your case and provide final approval. <br>
                    You will be notified when your prescription is ready.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            elif st.session_state.approval_status is True:
                st.markdown(f"""
                <div class='status-card status-approved'>
                    <h3 style='color: #10b981; margin: 0 0 1rem 0;'>✅ Consultation Approved by Doctor</h3>
                    <p style='color: #e2e8f0; margin: 0.5rem 0; font-size: 1.1rem;'>
                    <strong>Your prescription has been reviewed and approved!</strong>
                    </p>
                    <p style='color: #94a3b8; margin: 0;'>
                    You can now safely follow the medical advice below. <br>
                    This consultation is supervised and finalized by {DOCTOR_NAME}.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            elif st.session_state.approval_status is False:
                st.markdown(f"""
                <div class='status-card'>
                    <h3 style='color: #ef4444; margin: 0 0 1rem 0;'>❌ Offline Consultation Required</h3>
                    <p style='color: #e2e8f0; margin: 0.5rem 0; font-size: 1.1rem;'>
                    <strong>Please visit the clinic for physical examination</strong>
                    </p>
                    <p style='color: #94a3b8; margin: 0;'>
                    The doctor has recommended an in-person consultation <br>
                    for better diagnosis and treatment.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Patient Form
            with st.form("patient_form", clear_on_submit=False):
                # Personal Information
                st.markdown("<div class='professional-card'>", unsafe_allow_html=True)
                st.markdown("<div class='section-header'>👤 Personal Information</div>", unsafe_allow_html=True)
                
                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    name = st.text_input("**Full Name**", 
                                       value=st.session_state.patient_info.get("name", ""),
                                       placeholder="Enter your full name")
                with col_p2:
                    age = st.number_input("**Age**", min_value=1, max_value=120, value=30)
                with col_p3:
                    gender_display = st.selectbox("**Gender**", ["Male", "Female", "Other"])
                st.markdown("</div>", unsafe_allow_html=True)

                # Vitals & Measurements
                st.markdown("<div class='professional-card'>", unsafe_allow_html=True)
                st.markdown("<div class='section-header'>💓 Vitals & Measurements</div>", unsafe_allow_html=True)
                
                # Temperature with converter
                st.markdown("**Temperature**")
                temp_col1, temp_col2 = st.columns([2, 1])
                with temp_col1:
                    temp_value = st.number_input("Value", min_value=30.0, max_value=110.0, value=98.6, step=0.1, label_visibility="collapsed")
                with temp_col2:
                    temp_unit = st.selectbox("Unit", ["°F", "°C"], label_visibility="collapsed")
                
                # Convert temperature
                if temp_unit == "°F":
                    temp_celsius = fahrenheit_to_celsius(temp_value)
                    temp_fahrenheit = temp_value
                    st.markdown(f"""
                    <div class='temp-converter'>
                        <small>{temp_value}°F = {temp_celsius:.1f}°C</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    temp_celsius = temp_value
                    temp_fahrenheit = celsius_to_fahrenheit(temp_value)
                    st.markdown(f"""
                    <div class='temp-converter'>
                        <small>{temp_value}°C = {temp_fahrenheit:.1f}°F</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Other vitals
                col_v1, col_v2, col_v3, col_v4 = st.columns(4)
                with col_v1:
                    heart_rate = st.number_input("**Heart Rate (bpm)**", 40, 200, 80)
                with col_v2:
                    sys_bp = st.number_input("**Systolic BP**", 80, 220, 120)
                with col_v3:
                    dias_bp = st.number_input("**Diastolic BP**", 40, 140, 80)
                with col_v4:
                    spo2 = st.number_input("**SpO₂ (%)**", 70, 100, 98, 
                                         help="Oxygen saturation level")
                
                duration = st.number_input("**Days since symptoms started**", 0, 30, 1)
                st.markdown("</div>", unsafe_allow_html=True)

                # Symptoms
                st.markdown("<div class='professional-card'>", unsafe_allow_html=True)
                st.markdown("<div class='section-header'>🤒 Symptoms Checklist</div>", unsafe_allow_html=True)
                
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    cough = st.checkbox("Cough")
                    throat_pain = st.checkbox("Throat Pain")
                    headache = st.checkbox("Headache")
                    body_pain = st.checkbox("Body Pain")
                    
                with col_s2:
                    # Enhanced symptom severity selectors
                    chest_pain_severity = st.selectbox(
                        "**Chest Pain Severity**",
                        ["None", "Mild / occasional discomfort", "Moderate tightness", "Severe / crushing chest pain"],
                        help="Select the severity of chest pain"
                    )
                    
                    ear_pain_severity = st.selectbox(
                        "**Ear Pain Severity**",
                        ["None", "Mild / intermittent", "Moderate continuous pain", "Severe throbbing / discharge"],
                        help="Select the severity of ear pain"
                    )
                
                st.markdown("</div>", unsafe_allow_html=True)

                # Additional Information
                st.markdown("<div class='professional-card'>", unsafe_allow_html=True)
                st.markdown("<div class='section-header'>📝 Additional Information</div>", unsafe_allow_html=True)
                free_text = st.text_area(
                    "Describe your symptoms in detail:",
                    placeholder="Example: I've had fever since yesterday evening, with ear pain and headache...",
                    height=100,
                    label_visibility="collapsed"
                )
                st.markdown("</div>", unsafe_allow_html=True)

                # Submit Button
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    submitted = st.form_submit_button("**🔍 Get AI Consultation**", 
                                                    use_container_width=True,
                                                    type="primary")

            if submitted:
                # Save patient info
                st.session_state.patient_info = {
                    "name": name,
                    "age": age,
                    "gender_display": gender_display,
                }

                gender = 1 if gender_display == "Male" else 0

                # Encode severity levels for ML model
                chest_pain_encoded = 0
                if chest_pain_severity != "None":
                    chest_pain_encoded = 1  # Binary encoding for existing model
                
                ear_pain_encoded = 0
                if ear_pain_severity != "None":
                    ear_pain_encoded = 1  # Binary encoding for existing model

                # Build feature vector (using Celsius for model) with enhanced features
                feature_row = {
                    "age": age,
                    "gender": gender,
                    "temperature": temp_celsius,  # Use Celsius for model
                    "systolic_bp": sys_bp,
                    "diastolic_bp": dias_bp,
                    "heart_rate": heart_rate,
                    "cough": int(cough),
                    "throat_pain": int(throat_pain),
                    "ear_pain": ear_pain_encoded,  # Use encoded value
                    "chest_pain": chest_pain_encoded,  # Use encoded value
                    "headache": int(headache),
                    "body_pain": int(body_pain),
                    "duration_days": duration,
                    # Enhanced features
                    "pulse_pressure": sys_bp - dias_bp,
                    "bp_ratio": sys_bp / dias_bp if dias_bp > 0 else 1.5,
                    "symptom_count": int(cough) + int(throat_pain) + ear_pain_encoded + chest_pain_encoded + int(headache) + int(body_pain),
                    "temperature_flag": int(temp_celsius > 38),
                }

                # Check for emergency conditions (using Celsius) with enhanced rules
                symptoms_dict = {
                    'ear_pain_severity': ear_pain_severity,
                    'chest_pain_severity': chest_pain_severity,
                    'throat_pain': throat_pain,
                    'cough': cough
                }
                
                vitals_dict = {
                    'systolic_bp': sys_bp,
                    'diastolic_bp': dias_bp,
                    'heart_rate': heart_rate,
                    'spo2': spo2
                }
                
                emergency_conditions, monitor_conditions = check_emergency_condition(temp_celsius, symptoms_dict, vitals_dict)
                st.session_state.emergency_conditions = emergency_conditions
                st.session_state.monitor_conditions = monitor_conditions
                
                # Get currently selected model
                current_model = models_dict[selected_model_name]
                
                # ML prediction
                X_new = pd.DataFrame([feature_row])
                risk_score_raw = float(current_model.predict(X_new)[0])
                risk_score_raw = max(0.0, min(100.0, risk_score_raw))
                
                # Enhanced emergency detection: Combine ML risk score + rule-based conditions
                is_emergency = len(emergency_conditions) > 0 or risk_score_raw >= 80
                st.session_state.is_emergency = is_emergency
                
                # Adjust risk score based on emergency conditions
                final_risk_score = calculate_emergency_score(emergency_conditions, monitor_conditions, risk_score_raw)
                risk_level, risk_class = map_risk_level(final_risk_score)

                st.session_state.prediction_info = {
                    "risk_score": final_risk_score,
                    "risk_level": risk_level,
                    "features": feature_row,
                    "model_name": selected_model_name,
                }

                # Build structured description for LLM
                structured_text = (
                    f"Name: {name}, Age: {age}, Gender: {gender_display}\n"
                    f"Temperature: {temp_celsius:.1f}°C ({temp_fahrenheit:.1f}°F), "
                    f"BP: {sys_bp}/{dias_bp}, Heart Rate: {heart_rate} bpm, SpO₂: {spo2}%\n"
                    f"Chest Pain: {chest_pain_severity}, Ear Pain: {ear_pain_severity}\n"
                    f"Duration of symptoms: {duration} days\n"
                    f"Symptoms: "
                    f"{'cough, ' if cough else ''}"
                    f"{'throat pain, ' if throat_pain else ''}"
                    f"{'headache, ' if headache else ''}"
                    f"{'body pain, ' if body_pain else ''}".rstrip(", ")
                )

                if free_text.strip():
                    structured_text += f"\n\nPatient description: {free_text.strip()}"

                # Add emergency info if applicable
                if is_emergency:
                    structured_text += f"\n\n🚨 EMERGENCY CONDITIONS: {', '.join(emergency_conditions) if emergency_conditions else 'Very High Risk Score'}"
                if monitor_conditions:
                    structured_text += f"\n\n⚠️ MONITOR CONDITIONS: {', '.join(monitor_conditions)}"

                # LLM explanation
                with st.spinner("🔬 Analyzing with AI..." if not is_emergency else "🚨 Analyzing emergency situation..."):
                    ai_text = build_llm_response(
                        structured_text, 
                        final_risk_score, 
                        risk_level,
                        is_emergency,
                        emergency_conditions,
                        monitor_conditions,
                        api_key=api_key,
                        doctor_name=DOCTOR_NAME,
                    )

                # Update conversation
                st.session_state.conversation = []
                patient_msg = free_text if free_text.strip() else "Patient provided basic health information"
                st.session_state.conversation.append(("Patient", patient_msg, False))
                
                # Add emergency advice first if applicable
                if is_emergency or monitor_conditions:
                    emergency_advice = get_emergency_advice(emergency_conditions, monitor_conditions)
                    st.session_state.conversation.append(("AI Assistant - PRIORITY", emergency_advice, is_emergency))
                
                st.session_state.conversation.append(("AI Assistant", ai_text, is_emergency))
                st.session_state.consultation_done = False
                st.session_state.waiting_for_approval = True
                st.session_state.approval_status = None

                st.rerun()

            # Show conversation if available
            if st.session_state.conversation:
                st.markdown("---")
                st.markdown("## 💬 Consultation Results")
                
                # Risk Score Display
                if st.session_state.prediction_info:
                    rs = st.session_state.prediction_info["risk_score"]
                    rl, risk_class = map_risk_level(rs)
                    model_used = st.session_state.prediction_info.get("model_name", "Linear Regression")
                    
                    st.markdown(f"""
                    <div class='professional-card {risk_class}'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <div>
                                <h3 style='margin: 0; color: #f8fafc;'>Enhanced Risk Assessment</h3>
                                <p style='margin: 0.5rem 0 0 0; color: #e2e8f0;'>
                                <strong>Score:</strong> {rs:.1f}/100 | <strong>Level:</strong> {rl}<br>
                                <strong>ML Model:</strong> {model_used}<br>
                                <strong>Feature Engineering:</strong> Enabled
                                </p>
                            </div>
                            <div style='font-size: 1.5rem; color: #e2e8f0;'>
                                {'🚨' if rs >= 60 else '⚠️' if rs >= 30 else '✅'}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Conversation Display
                st.markdown("#### Consultation Dialogue")
                for speaker, msg, is_emergency in st.session_state.conversation:
                    chat_class = "chat-box chat-emergency" if is_emergency else "chat-box"
                    st.markdown(f"""
                    <div class="{chat_class}">
                        <strong>{speaker}:</strong><br>{msg}
                    </div>
                    """, unsafe_allow_html=True)

                # Summary and Download (only show when approved or if waiting)
                if st.session_state.prediction_info and (st.session_state.approval_status or st.session_state.waiting_for_approval):
                    summary_md = generate_summary_md(
                        st.session_state.conversation,
                        st.session_state.patient_info,
                        st.session_state.prediction_info,
                        st.session_state.is_emergency,
                        st.session_state.emergency_conditions,
                        st.session_state.monitor_conditions,
                        doctor_name=DOCTOR_NAME,
                    )
                    
                    with st.expander("📋 Detailed Summary", expanded=True):
                        st.markdown(summary_md)
                    
                    # Download button - only enable when approved
                    if st.session_state.approval_status:
                        st.download_button(
                            label="📥 Download Final Prescription",
                            data=summary_md,
                            file_name=f"prescription_{st.session_state.patient_info.get('name', 'patient')}.md",
                            mime="text/markdown",
                            use_container_width=True
                        )
                    else:
                        st.download_button(
                            label="📥 Download Draft Consultation",
                            data=summary_md,
                            file_name=f"consultation_draft_{st.session_state.patient_info.get('name', 'patient')}.md",
                            mime="text/markdown",
                            use_container_width=True,
                            disabled=False
                        )

        # SIDE COLUMN: Information
        with col_side:
            st.markdown("## 📋 Quick Reference")
            
            with st.expander("🆘 Enhanced Emergency Signs", expanded=True):
                st.markdown("""
                **Critical Emergencies:**
                - Fever ≥ 39.5°C (103.1°F)
                - Severe ear pain + fever
                - Severe/crushing chest pain
                - BP > 180/120 or < 90/60
                - Heart rate > 120 or < 40
                - SpO₂ < 90%
                - ML Risk Score ≥ 80
                
                **Monitor Closely:**
                - SpO₂ 90–94%
                - Moderate chest pain
                - Moderate fever
                - Elevated heart rate
                """)
            
            with st.expander("🌡️ Temperature Guide", expanded=False):
                st.markdown("""
                **Normal:** 36.5-37.5°C (97.7-99.5°F)
                **Fever:** ≥38.0°C (100.4°F)
                **High Fever:** ≥39.5°C (103.1°F)
                """)
            
            with st.expander("💊 Common Medicines", expanded=False):
                st.markdown("""
                **Fever/Pain:**
                - Paracetamol 650mg
                - Ibuprofen 400mg
                
                **Allergies:**
                - Cetirizine 10mg
                """)
            
            with st.expander("📈 ML Features Used", expanded=False):
                st.markdown("""
                **Enhanced Features:**
                - Pulse Pressure
                - BP Ratio
                - Symptom Count
                - Temperature Flag
                - All original vitals
                """)

    # =====================================================
    #                 DOCTOR PORTAL
    # =====================================================
    else:
        with col_main:
            st.markdown("## 👨‍⚕️ Doctor Portal")
            
            if st.session_state.waiting_for_approval:
                # Emergency Alert for Doctor
                if st.session_state.is_emergency:
                    st.markdown(f"""
                    <div class='professional-card emergency-card'>
                        <h3 style='color: #f8fafc; margin: 0 0 0.5rem 0;'>🚨 Emergency Case</h3>
                        <p style='margin: 0; color: #f8fafc;'>
                        <strong>Critical conditions:</strong> {', '.join(st.session_state.emergency_conditions) if st.session_state.emergency_conditions else 'Very High Risk Score'}
                        </p>
                        <p style='margin: 0.5rem 0 0 0; color: #f8fafc;'>
                        <strong>ML Model:</strong> {st.session_state.prediction_info.get('model_name', 'Linear Regression')}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Monitor Conditions Alert
                if st.session_state.monitor_conditions:
                    st.markdown(f"""
                    <div class='professional-card' style='border-left: 4px solid #d97706 !important;'>
                        <h3 style='color: #f8fafc; margin: 0 0 0.5rem 0;'>⚠️ Monitor Conditions</h3>
                        <p style='margin: 0; color: #f8fafc;'>
                        <strong>Require monitoring:</strong> {', '.join(st.session_state.monitor_conditions)}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                st.info(f"**Consultation awaiting approval** - Patient: {st.session_state.patient_info.get('name', 'Unknown')}")

                # Patient Summary Card
                if st.session_state.patient_info and st.session_state.prediction_info:
                    st.markdown("<div class='professional-card'>", unsafe_allow_html=True)
                    st.markdown("### Patient Overview")
                    
                    pi = st.session_state.patient_info
                    pred = st.session_state.prediction_info
                    rs = pred.get('risk_score', 0.0)
                    rl, risk_class = map_risk_level(rs)
                    model_used = pred.get('model_name', 'Linear Regression')
                    
                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                        st.metric("Patient Name", pi.get('name', '-'))
                        st.metric("Age/Gender", f"{pi.get('age', '-')}/{pi.get('gender_display', '-')}")
                    with col_d2:
                        st.metric("Risk Score", f"{rs:.1f}/100")
                        st.metric("ML Model", model_used)
                    
                    st.markdown(f"**Status:** {'🚨 EMERGENCY' if st.session_state.is_emergency else '⚠️ MONITOR' if st.session_state.monitor_conditions else '🟡 Routine'}")
                    st.markdown("</div>", unsafe_allow_html=True)

                # Conversation Review
                st.markdown("### Consultation Review")
                for speaker, msg, is_emergency in st.session_state.conversation:
                    chat_class = "chat-box chat-emergency" if is_emergency else "chat-box"
                    st.markdown(f"""
                    <div class="{chat_class}">
                        <strong>{speaker}:</strong><br>{msg}
                    </div>
                    """, unsafe_allow_html=True)

                # Approval Actions
                st.markdown("### Final Approval")
                col_approve, col_reject = st.columns(2)
                
                with col_approve:
                    if st.button("✅ Approve Consultation", 
                               use_container_width=True,
                               type="primary"):
                        st.session_state.approval_status = True
                        st.session_state.waiting_for_approval = False
                        st.success("✅ Consultation approved and finalized! Patient can now access the prescription.")
                        
                with col_reject:
                    if st.button("❌ Request Offline Visit", 
                               use_container_width=True):
                        st.session_state.approval_status = False
                        st.session_state.waiting_for_approval = False
                        st.warning("Consultation marked for offline evaluation. Patient will be notified.")

            else:
                st.info("No consultations currently waiting for approval.")
                
                # Show previous consultation if available
                if st.session_state.conversation:
                    st.markdown("### Last Consultation Summary")
                    summary_md = generate_summary_md(
                        st.session_state.conversation,
                        st.session_state.patient_info,
                        st.session_state.prediction_info,
                        st.session_state.is_emergency,
                        st.session_state.emergency_conditions,
                        st.session_state.monitor_conditions,
                        doctor_name=DOCTOR_NAME,
                    )
                    st.markdown(summary_md)
                    
                    # Show approval status
                    if st.session_state.approval_status is not None:
                        if st.session_state.approval_status:
                            st.success("✅ Last consultation was approved and finalized")
                        else:
                            st.warning("❌ Last consultation required offline visit")
                else:
                    st.markdown("""
                    <div class='professional-card' style='text-align: center; padding: 2rem;'>
                        <h3 style='color: #94a3b8;'>Welcome, Doctor</h3>
                        <p style='color: #94a3b8;'>No consultations in session history.</p>
                    </div>
                    """, unsafe_allow_html=True)

        with col_side:
            st.markdown("## 🎯 Clinical Notes")
            
            with st.expander("📊 ML Model Details", expanded=True):
                st.markdown(f"""
                **Selected Model:** {selected_model_name}
                **R² Score:** {selected_metrics['r2']:.3f}
                **Accuracy %:** {selected_metrics['accuracy_percent']:.1f}%
                **RMSE:** {selected_metrics['rmse']:.2f}
                **MAE:** {selected_metrics['mae']:.2f}
                **Total Features:** {len(feature_cols)} (Enhanced)
                """)
            
            with st.expander("🔬 Key Metrics", expanded=False):
                st.markdown("""
                **Normal Ranges:**
                - Temp: 36.5-37.5°C
                - BP: 120/80 mmHg
                - HR: 60-100 bpm
                - SpO₂: 95-100%
                
                **Risk Levels:**
                - Low: <30
                - Moderate: 30-60
                - High: 60-80
                - Very High: ≥80
                """)

if __name__ == "__main__":
    main()