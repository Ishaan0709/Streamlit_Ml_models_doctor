def fahrenheit_to_celsius(f_temp):
    """Convert Fahrenheit to Celsius"""
    return (f_temp - 32) * 5/9

def celsius_to_fahrenheit(c_temp):
    """Convert Celsius to Fahrenheit"""
    return (c_temp * 9/5) + 32

def check_emergency_condition(temp, symptoms, vitals):
    """Check if patient condition requires emergency attention with enhanced rules"""
    emergency_conditions = []
    monitor_conditions = []
    
    # High fever emergency
    if temp >= 39.5:  # 103.1°F equivalent
        emergency_conditions.append(f"High fever ({temp}°C / {celsius_to_fahrenheit(temp):.1f}°F)")
    elif temp >= 38.5:
        monitor_conditions.append(f"Moderate fever ({temp}°C) - Monitor closely")
    
    # SpO2 emergency levels
    spo2 = vitals.get('spo2')
    if spo2 is not None:
        if spo2 < 90:
            emergency_conditions.append(f"Very low oxygen saturation (SpO₂ {spo2}% < 90%) – Emergency")
        elif spo2 < 95:
            monitor_conditions.append(f"Borderline oxygen saturation (SpO₂ {spo2}% 90–94%) – Monitor closely")
    
    # Chest pain severity based emergency
    chest_pain_severity = symptoms.get('chest_pain_severity', 'None')
    if chest_pain_severity == "Severe / crushing chest pain":
        emergency_conditions.append("Severe / crushing chest pain – possible cardiac emergency")
    elif chest_pain_severity == "Moderate tightness":
        # Check if combined with abnormal vitals
        if (vitals.get('systolic_bp', 120) > 160 or 
            vitals.get('heart_rate', 80) > 100 or 
            vitals.get('spo2', 98) < 95):
            emergency_conditions.append("Moderate chest pain with abnormal vitals – high-risk")
        else:
            monitor_conditions.append("Moderate chest pain – monitor and seek care if worsens")
    elif chest_pain_severity == "Mild / occasional discomfort":
        monitor_conditions.append("Mild chest discomfort – routine evaluation recommended")
    
    # Ear pain severity based emergency
    ear_pain_severity = symptoms.get('ear_pain_severity', 'None')
    if ear_pain_severity == "Severe throbbing / discharge" and temp >= 38.5:
        emergency_conditions.append("Severe ear pain with fever – possible acute otitis media (emergency ENT review)")
    elif ear_pain_severity == "Moderate continuous pain" and temp >= 39.0:
        emergency_conditions.append("Moderate ear pain with high fever – urgent ENT review needed")
    elif ear_pain_severity in ["Moderate continuous pain", "Severe throbbing / discharge"]:
        monitor_conditions.append(f"{ear_pain_severity} – ENT evaluation recommended")
    
    # Cardiovascular emergencies
    if vitals.get('systolic_bp', 120) > 180 or vitals.get('diastolic_bp', 80) > 120:
        emergency_conditions.append("Hypertensive crisis")
    elif vitals.get('systolic_bp', 120) > 160 or vitals.get('diastolic_bp', 80) > 100:
        monitor_conditions.append("Elevated blood pressure – monitor")
    
    if vitals.get('systolic_bp', 120) < 90 or vitals.get('diastolic_bp', 80) < 60:
        emergency_conditions.append("Hypotensive emergency")
    
    if vitals.get('heart_rate', 80) > 120:
        emergency_conditions.append("Tachycardia")
    elif vitals.get('heart_rate', 80) > 100:
        monitor_conditions.append("Elevated heart rate – monitor")
    
    if vitals.get('heart_rate', 80) < 40:
        emergency_conditions.append("Bradycardia")
    
    return emergency_conditions, monitor_conditions

def calculate_emergency_score(emergency_conditions, monitor_conditions, risk_score):
    """Calculate enhanced risk score based on emergency conditions"""
    if not emergency_conditions and risk_score < 80:
        return risk_score  # No adjustment needed
    
    base_emergency_score = 75
    emergency_weight = 0
    
    # Add weights based on severity of conditions
    for condition in emergency_conditions:
        if "SpO₂" in condition and "< 90" in condition:
            emergency_weight += 10
        elif "Severe / crushing chest pain" in condition:
            emergency_weight += 10
        elif "cardiac emergency" in condition:
            emergency_weight += 8
        elif "Hypertensive crisis" in condition or "Hypotensive emergency" in condition:
            emergency_weight += 7
        elif "Tachycardia" in condition or "Bradycardia" in condition:
            emergency_weight += 6
        elif "High fever" in condition:
            emergency_weight += 5
        elif "acute otitis media" in condition:
            emergency_weight += 5
        else:
            emergency_weight += 3  # Default weight for other emergencies
    
    # Add some weight for monitor conditions
    for condition in monitor_conditions:
        if "SpO₂" in condition and "90–94" in condition:
            emergency_weight += 2
        elif "Moderate chest pain" in condition and "high-risk" in condition:
            emergency_weight += 3
        else:
            emergency_weight += 1
    
    adjusted_emergency_score = base_emergency_score + emergency_weight
    final_score = max(risk_score, min(100, adjusted_emergency_score))
    
    return final_score

def get_emergency_advice(conditions, monitor_conditions=None):
    """Get emergency medical advice with enhanced guidance"""
    if monitor_conditions is None:
        monitor_conditions = []
        
    advice = []
    
    # Check for SpO2 conditions first
    spo2_emergency = any("SpO₂" in cond and "< 90" in cond for cond in conditions)
    spo2_monitor = any("SpO₂" in cond and "90–94" in cond for cond in monitor_conditions)
    
    if spo2_emergency:
        advice.extend([
            "🆘 **OXYGEN EMERGENCY:**",
            "• SIT UPRIGHT to ease breathing",
            "• Use supplemental oxygen if available",
            "• AVOID physical exertion completely",
            "• Seek IMMEDIATE emergency care",
            "• Monitor SpO₂ continuously until help arrives"
        ])
    elif spo2_monitor:
        advice.extend([
            "⚠️ **OXYGEN MONITORING NEEDED:**",
            "• Rest and avoid strenuous activity",
            "• Monitor SpO₂ every 30 minutes",
            "• Seek care if SpO₂ drops below 90%",
            "• Watch for bluish lips or difficulty breathing"
        ])
    
    if any("fever" in cond.lower() for cond in conditions):
        advice.extend([
            "🆘 **FEVER EMERGENCY:**",
            "• Take Paracetamol 650mg (if no allergies) for fever",
            "• Use cold compresses on forehead and armpits",
            "• Stay hydrated with electrolyte solutions",
            "• Remove excess clothing, keep room ventilated"
        ])
    
    if any("ear" in cond.lower() for cond in conditions):
        advice.extend([
            "🆘 **EAR PAIN EMERGENCY:**",
            "• Take Ibuprofen 400mg for pain and inflammation (if no stomach issues)",
            "• Avoid water entry in ears",
            "• Use warm compress on affected ear",
            "• Do NOT use eardrops without prescription"
        ])
    
    if any("chest" in cond.lower() and "Severe" in cond for cond in conditions):
        advice.extend([
            "🆘 **CHEST PAIN PROTOCOL:**",
            "• SIT UPRIGHT immediately, do not lie down",
            "• If available, take Aspirin 325mg (if no allergies)",
            "• Loosen tight clothing",
            "• CALL EMERGENCY SERVICES if pain radiates to arm/jaw",
            "• Do NOT drive yourself to hospital"
        ])
    
    if any("blood pressure" in cond.lower() for cond in conditions):
        advice.extend([
            "🆘 **BLOOD PRESSURE CRISIS:**",
            "• Sit quietly in a calm environment",
            "• Avoid any physical exertion",
            "• Do not take extra medication unless prescribed",
            "• Monitor BP every 15 minutes"
        ])
    
    # Add monitor condition advice
    if monitor_conditions and not advice:
        advice.extend([
            "⚠️ **MONITOR CLOSELY:**",
            "• Watch for symptom progression",
            "• Rest and avoid strenuous activity",
            "• Seek medical care if symptoms worsen",
            "• Follow up with doctor within 24 hours"
        ])
    
    advice.extend([
        "",
        "🚨 **URGENT: Proceed to nearest emergency department if:**",
        "• Symptoms worsen rapidly",
        "• Difficulty breathing occurs",
        "• Severe pain persists after medication",
        "• Altered mental state or confusion",
        "• SpO₂ drops below 90%"
    ])
    
    return "\n".join(advice)