from pathlib import Path

p = Path(r"c:\Users\gupta\Desktop\SAFE\src\dashboard\app.py")
lines = p.read_text(encoding="utf-8").splitlines()

new_lines = []
skip = False
for line in lines:
    if line.strip().startswith("# Simulated / Live history buffer"):
        # Insert rich dynamic demo curve generator and automatic hardware reset
        new_lines.append("# Dynamic Presentation Demo Curves & Live Hardware Auto-Reset")
        new_lines.append("def get_presentation_demo_curves():")
        new_lines.append("    now = datetime.now()")
        new_lines.append("    times = [now - timedelta(minutes=i*20) for i in range(24)][::-1]")
        new_lines.append("    temps = [23.5 + 7.0 * np.sin(i / 3.8) + np.random.normal(0, 0.2) for i in range(24)]")
        new_lines.append("    hums = [68.0 - 16.0 * np.sin(i / 3.8) + np.random.normal(0, 0.4) for i in range(24)]")
        new_lines.append("    soils = [60.0 - (i * 0.4) + np.random.normal(0, 0.2) for i in range(24)]")
        new_lines.append("    risks = [float(np.clip(0.18 + (0.35 if h > 68 else 0.05) + (0.22 if t > 28 else 0.02), 0.08, 0.88)) for t, h in zip(temps, hums)]")
        new_lines.append("    return pd.DataFrame({'timestamp': times, 'temperature': [round(x, 1) for x in temps], 'humidity': [round(x, 1) for x in hums], 'soil_moisture': [round(x, 1) for x in soils], 'risk_score': risks})")
        new_lines.append("")
        new_lines.append("if 'was_hardware' not in st.session_state:")
        new_lines.append("    st.session_state.was_hardware = False")
        new_lines.append("")
        new_lines.append("if is_real_esp32:")
        new_lines.append("    if not st.session_state.was_hardware or 'live_buf' not in st.session_state:")
        new_lines.append("        st.session_state.live_buf = pd.DataFrame(columns=['timestamp', 'temperature', 'humidity', 'soil_moisture', 'risk_score'])")
        new_lines.append("    st.session_state.was_hardware = True")
        new_lines.append("    now_time = datetime.now()")
        new_lines.append("    calc_risk = float(np.clip(0.12 + (0.35 if current_hum > 75 else 0.05) + (0.20 if current_temp > 28 else 0.0), 0.05, 0.95))")
        new_lines.append("    new_row = pd.DataFrame([{'timestamp': now_time, 'temperature': current_temp, 'humidity': current_hum, 'soil_moisture': current_soil, 'risk_score': calc_risk}])")
        new_lines.append("    st.session_state.live_buf = pd.concat([st.session_state.live_buf, new_row], ignore_index=True)")
        new_lines.append("    if len(st.session_state.live_buf) > 35:")
        new_lines.append("        st.session_state.live_buf = st.session_state.live_buf.iloc[-35:]")
        new_lines.append("    st.session_state.history = st.session_state.live_buf")
        new_lines.append("else:")
        new_lines.append("    st.session_state.was_hardware = False")
        new_lines.append("    if 'demo_buf' not in st.session_state:")
        new_lines.append("        st.session_state.demo_buf = get_presentation_demo_curves()")
        new_lines.append("    st.session_state.history = st.session_state.demo_buf")
        new_lines.append("    last_pt = st.session_state.history.iloc[-1]")
        new_lines.append("    current_temp = last_pt['temperature']")
        new_lines.append("    current_hum = last_pt['humidity']")
        new_lines.append("    current_soil = last_pt['soil_moisture']")
        new_lines.append("    last_update_str = last_pt['timestamp'].strftime('%H:%M:%S')")
        skip = True
        continue

    if skip and line.strip().startswith("if \"latest_prediction\" not in st.session_state:"):
        skip = False

    if not skip:
        new_lines.append(line)

p.write_text("\n".join(new_lines), encoding="utf-8")
print("Updated dashboard with presentation curves and auto-reset!")
