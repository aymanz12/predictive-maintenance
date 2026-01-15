import gradio as gr
import requests
import json

# API Endpoint
API_URL = "http://127.0.0.1:8000/predict"

# Custom CSS for enhanced styling
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif !important;
}

.gradio-container {
    background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%) !important;
    padding: 2rem !important;
}

.main {
    background: rgba(255, 255, 255, 0.95) !important;
    border-radius: 20px !important;
    padding: 2.5rem !important;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3) !important;
    backdrop-filter: blur(10px) !important;
}

h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700 !important;
    font-size: 2.5rem !important;
    margin-bottom: 0.5rem !important;
}

.subtitle {
    color: #6b7280 !important;
    font-size: 1.1rem !important;
    margin-bottom: 2rem !important;
}

.input-section {
    background: linear-gradient(135deg, #f6f8fb 0%, #e9ecef 100%) !important;
    border-radius: 15px !important;
    padding: 1.5rem !important;
    border: 2px solid #e0e7ff !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05) !important;
}

.output-section {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%) !important;
    border-radius: 15px !important;
    padding: 1.5rem !important;
    border: 2px solid #fbbf24 !important;
    margin-top: 1.5rem !important;
}

.safe-output {
    background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%) !important;
    border: 2px solid #10b981 !important;
}

.danger-output {
    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%) !important;
    border: 2px solid #ef4444 !important;
}

button.primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    padding: 1rem 2rem !important;
    border-radius: 12px !important;
    box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4) !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 15px 35px rgba(102, 126, 234, 0.5) !important;
}

.gr-box {
    border-radius: 12px !important;
    border: 2px solid #e5e7eb !important;
}

label {
    font-weight: 600 !important;
    color: #374151 !important;
    font-size: 0.95rem !important;
}

input[type="range"] {
    accent-color: #667eea !important;
}

.status-card {
    padding: 1.5rem !important;
    border-radius: 12px !important;
    font-size: 1.3rem !important;
    font-weight: 600 !important;
    text-align: center !important;
}

.metric-card {
    background: white !important;
    border-radius: 12px !important;
    padding: 1.2rem !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07) !important;
    border-left: 4px solid #667eea !important;
}

.info-badge {
    background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%) !important;
    color: #1e40af !important;
    padding: 0.5rem 1rem !important;
    border-radius: 20px !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    display: inline-block !important;
    margin-top: 0.5rem !important;
}
"""

def get_prediction(air_temperature, process_temperature, rotational_speed, torque, tool_wear, type_val):
    """
    Sends data to FastAPI backend and returns the prediction result with enhanced formatting.
    """
    payload = {
        "air_temperature": float(air_temperature),
        "process_temperature": float(process_temperature),
        "rotational_speed": int(rotational_speed),
        "torque": float(torque),
        "tool_wear": int(tool_wear),
        "type": type_val
    }
    
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Debug print for user
        # print(f"\n🔍 API Response: Input: {payload}")
        # print(f"👉 Prediction: {result['prediction']} (Prob: {result['probability']:.4f})")
        
        prediction_val = result["prediction"]
        probability = result["probability"]
        
        # Enhanced output formatting with HTML
        if prediction_val == 1:
            status_html = f"""
            <div style='background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); 
                        padding: 2rem; border-radius: 15px; text-align: center; 
                        border: 3px solid #ef4444; box-shadow: 0 8px 16px rgba(239, 68, 68, 0.2);'>
                <div style='font-size: 3rem; margin-bottom: 0.5rem;'>⚠️</div>
                <div style='font-size: 1.8rem; font-weight: 700; color: #991b1b; margin-bottom: 0.5rem;'>
                    FAILURE DETECTED
                </div>
                <div style='font-size: 1rem; color: #7f1d1d; font-weight: 500;'>
                    Immediate maintenance recommended
                </div>
            </div>
            """
            prob_html = f"""
            <div style='background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); 
                        padding: 2rem; border-radius: 15px; text-align: center;
                        border: 3px solid #ef4444; box-shadow: 0 8px 16px rgba(239, 68, 68, 0.2);'>
                <div style='font-size: 1rem; color: #7f1d1d; font-weight: 600; margin-bottom: 0.5rem;'>
                    FAILURE PROBABILITY
                </div>
                <div style='font-size: 2.5rem; font-weight: 700; color: #991b1b;'>
                    {probability * 100:.2f}%
                </div>
                <div style='font-size: 0.9rem; color: #7f1d1d; margin-top: 0.5rem;'>
                    Risk Level: {"CRITICAL" if probability > 0.7 else "HIGH" if probability > 0.4 else "MODERATE"}
                </div>
            </div>
            """
        else:
            status_html = f"""
            <div style='background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); 
                        padding: 2rem; border-radius: 15px; text-align: center;
                        border: 3px solid #10b981; box-shadow: 0 8px 16px rgba(16, 185, 129, 0.2);'>
                <div style='font-size: 3rem; margin-bottom: 0.5rem;'>✅</div>
                <div style='font-size: 1.8rem; font-weight: 700; color: #065f46; margin-bottom: 0.5rem;'>
                    SYSTEM SAFE
                </div>
                <div style='font-size: 1rem; color: #047857; font-weight: 500;'>
                    No immediate action required
                </div>
            </div>
            """
            prob_html = f"""
            <div style='background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); 
                        padding: 2rem; border-radius: 15px; text-align: center;
                        border: 3px solid #10b981; box-shadow: 0 8px 16px rgba(16, 185, 129, 0.2);'>
                <div style='font-size: 1rem; color: #047857; font-weight: 600; margin-bottom: 0.5rem;'>
                    FAILURE PROBABILITY
                </div>
                <div style='font-size: 2.5rem; font-weight: 700; color: #065f46;'>
                    {probability * 100:.2f}%
                </div>
                <div style='font-size: 0.9rem; color: #047857; margin-top: 0.5rem;'>
                    Risk Level: LOW
                </div>
            </div>
            """
        
        return status_html, prob_html
        
    except requests.exceptions.ConnectionError:
        error_html = """
        <div style='background: #fef3c7; padding: 1.5rem; border-radius: 12px; 
                    text-align: center; border: 2px solid #f59e0b;'>
            <div style='font-size: 2rem; margin-bottom: 0.5rem;'>🔌</div>
            <div style='color: #92400e; font-weight: 600;'>Connection Error</div>
            <div style='color: #78350f; font-size: 0.9rem; margin-top: 0.3rem;'>
                Could not connect to backend API
            </div>
        </div>
        """
        return error_html, error_html
    except Exception as e:
        error_html = f"""
        <div style='background: #fee2e2; padding: 1.5rem; border-radius: 12px; 
                    text-align: center; border: 2px solid #ef4444;'>
            <div style='font-size: 2rem; margin-bottom: 0.5rem;'>❌</div>
            <div style='color: #991b1b; font-weight: 600;'>Error</div>
            <div style='color: #7f1d1d; font-size: 0.9rem; margin-top: 0.3rem;'>
                {str(e)}
            </div>
        </div>
        """
        return error_html, error_html

# Create Enhanced Gradio Interface
# Note: Theme and CSS moved to launch() as per deprecation warning
with gr.Blocks(title="🏭 Predictive Maintenance AI") as demo:
    
    gr.Markdown("# 🏭 Predictive Maintenance AI Dashboard")
    gr.Markdown("<p class='subtitle'>🤖 AI-powered machine failure prediction system using XGBoost ML model</p>")
    
    gr.Markdown("""
    <div style='background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); 
                padding: 1rem; border-radius: 12px; border-left: 4px solid #3b82f6; margin-bottom: 1.5rem;'>
        <strong style='color: #1e40af;'>ℹ️ How it works:</strong> 
        <span style='color: #1e3a8a;'>Adjust the machine parameters below and click "Analyze Machine Status" to get real-time failure predictions.</span>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🌡️ Temperature Sensors")
            air_temp_input = gr.Slider(
                minimum=290, maximum=310, step=0.1, value=300.0, 
                label="Air Temperature [K]",
                info="Ambient air temperature around the machine"
            )
            process_temp_input = gr.Slider(
                minimum=300, maximum=320, step=0.1, value=310.0, 
                label="Process Temperature [K]",
                info="Operating temperature during process"
            )
            
            gr.Markdown("### ⚙️ Machine Type")
            type_input = gr.Radio(
                choices=["L", "M", "H"], value="M", 
                label="Quality Variant",
                info="L=Low | M=Medium | H=High quality variant"
            )
            
        with gr.Column(scale=1):
            gr.Markdown("### 🔧 Operating Parameters")
            rotational_speed_input = gr.Slider(
                minimum=1100, maximum=2900, step=1, value=1500, 
                label="Rotational Speed [rpm]",
                info="Spindle rotation speed"
            )
            torque_input = gr.Slider(
                minimum=0, maximum=80, step=0.1, value=40.0, 
                label="Torque [Nm]",
                info="Applied torque on the tool"
            )
            tool_wear_input = gr.Slider(
                minimum=0, maximum=300, step=1, value=0, 
                label="Tool Wear [min]",
                info="Accumulated tool usage time"
            )

    submit_btn = gr.Button("🔍 Analyze Machine Status", variant="primary", size="lg")
    
    gr.Markdown("---")
    gr.Markdown("## 📊 Prediction Results")
    
    with gr.Row():
        with gr.Column(scale=1):
            output_status = gr.HTML(label="Status")
        with gr.Column(scale=1):
            output_prob = gr.HTML(label="Probability")

    gr.Markdown("""
    <div style='background: #f9fafb; padding: 1rem; border-radius: 8px; margin-top: 1.5rem; text-align: center;'>
        <span style='color: #6b7280; font-size: 0.85rem;'>
            💡 <strong>Tip:</strong> Higher tool wear and extreme temperatures increase failure probability
        </span>
    </div>
    """)

    submit_btn.click(
        fn=get_prediction,
        inputs=[
            air_temp_input, 
            process_temp_input, 
            rotational_speed_input, 
            torque_input, 
            tool_wear_input, 
            type_input
        ],
        outputs=[output_status, output_prob]
    )

if __name__ == "__main__":
    # Pass theme and css here as per deprecation warning
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        theme=gr.themes.Soft(),
        css=custom_css
    )
