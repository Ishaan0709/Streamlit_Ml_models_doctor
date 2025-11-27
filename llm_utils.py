from langchain_openai import ChatOpenAI

def build_llm_response(structured_data, risk_score, risk_level, is_emergency=False, 
                      emergency_conditions=None, monitor_conditions=None, api_key=None, doctor_name="Doctor"):
    """
    Uses OpenAI to generate natural language explanation
    """
    if not api_key:
        return "⚠️ OpenAI API key not available. Please configure your API key to get AI-generated responses."
    
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=0.5,
        )

        if is_emergency:
            system_prompt = f"""
🚨 **EMERGENCY MODE ACTIVATED** 🚨

You are an AI assistant working for {doctor_name}, detecting EMERGENCY medical conditions.
The patient shows critical symptoms requiring immediate attention.

CRITICAL CONDITIONS DETECTED: {emergency_conditions}
MONITOR CONDITIONS: {monitor_conditions}

Provide:
1. 🆘 IMMEDIATE first-aid measures with generic emergency medicines
2. 📋 Step-by-step emergency protocol
3. 🏥 When to go to emergency room vs urgent care
4. 💊 Emergency medicines that can be taken (generic names only)
5. 🚫 Important contraindications and warnings
6. Specific advice for oxygen issues if SpO₂ is low

EMPHASIZE: This is EMERGENCY guidance. Doctor consultation is URGENT.
"""
        else:
            system_prompt = f"""
You are an AI assistant working for {doctor_name}, a very experienced physician.
You NEVER replace the doctor, you only give preliminary guidance based on
doctor's past case data.

Provide:
1. A short summary of what might be going on (2-3 lines)
2. Precautions and lifestyle steps
3. Recommended OTC medicines (generic names only)
4. When to visit OPD (today/within 1-2 days/routine)
5. Final disclaimer that doctor will review
"""

        user_content = f"""
Patient information:
{structured_data}

Predicted risk score (0-100): {risk_score:.1f}
Risk level: {risk_level}

{'🚨 EMERGENCY SITUATION - Provide urgent medical guidance' if is_emergency else 'Provide routine medical guidance'}
{'⚠️ MONITOR CONDITIONS: ' + ', '.join(monitor_conditions) if monitor_conditions else ''}
"""

        response = llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
        )
        return response.content
    except Exception as e:
        return f"⚠️ Error generating AI response: {str(e)}. Please check your API key and internet connection."

def generate_summary_md(conversation, patient_info, prediction_info, is_emergency=False, 
                       emergency_conditions=None, monitor_conditions=None, doctor_name="Doctor"):
    """
    Creates a markdown consultation summary
    """
    patient_name = patient_info.get("name", "N/A")
    risk_score = prediction_info.get("risk_score", 0.0)
    risk_level = prediction_info.get("risk_level", "N/A")
    model_name = prediction_info.get("model_name", "Linear Regression")

    emergency_banner = "🚨 **EMERGENCY CONSULTATION** 🚨\n\n" if is_emergency else ""

    md = f"""
# 📝 Consultation Summary

{emergency_banner}
**Patient Name:** {patient_name}  
**Age:** {patient_info.get('age','-')}  
**Gender:** {patient_info.get('gender_display','-')}  

**Predicted Risk Score:** {risk_score:.1f} / 100  
**Risk Level:** {risk_level}  
**ML Model Used:** {model_name}

"""

    if emergency_conditions:
        md += "**Emergency Conditions:**\n"
        for condition in emergency_conditions:
            md += f"- {condition}\n"
        md += "\n"
    
    if monitor_conditions:
        md += "**Monitor Conditions:**\n"
        for condition in monitor_conditions:
            md += f"- {condition}\n"
        md += "\n"

    md += """
---

## Conversation Log
"""
    for speaker, msg, is_emergency_msg in conversation:
        if is_emergency_msg:
            md += f"\n**🚨 {speaker}:**\n{msg}\n"
        else:
            md += f"\n**{speaker}:**\n{msg}\n"

    md += """
---

**Note:** This is an AI-assisted draft based on historical data.  
Final decision and prescription will always be given by the real doctor.
"""
    return md