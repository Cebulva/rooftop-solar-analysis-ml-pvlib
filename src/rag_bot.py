import streamlit as st
import os
from groq import Groq
from dotenv import load_dotenv
import random
import sqlite3

load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("LLM_MODEL")

# --- Database path ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_PATH = os.path.join(DATA_DIR, "solar_programs.db")

SYSTEM_PROMPT = """
You are a helpful solar assistant for private homeowners in Germany.

Rules:
- Focus exclusively on residential rooftop solar installations for private houses.
- Use report data ONLY for numbers and factual values; never modify them.
- You MAY use general knowledge to explain, interpret, and analyze what the report numbers mean.
- All answers must be relevant to Germany and private residential use.
- Keep answers clear, well structured, and easy to read.
- Avoid tables (width is limited); prefer short paragraphs or bullet points.
- If information is missing from the report, say so politely.
- Be friendly and supportive.

Pricing information for Germany (2026):
- Residential PV systems: €1,100 - €1,600 per kWp (including installation)
- Battery storage: €700 - €1,000 per kWh capacity
- Costs decrease slightly with larger system sizes due to economies of scale
- Prices include panels, inverter, mounting, installation, and grid connection

Electricity prices (2026):
- Grid electricity purchase: €0.30 - €0.40 per kWh (average household rate)
- EEG feed-in tariff (guaranteed for 20 years): 
  * Up to 10 kWp: ~€0.082 per kWh
  * 10-40 kWp: ~€0.071 per kWh
  * 40-100 kWp: ~€0.058 per kWh
- The feed-in rate you get at installation is fixed for 20 years (degression protection)
- Self-consumption saves €0.30-0.40 per kWh, while selling earns only €0.08 per kWh
- Maximum self-consumption is financially optimal

Length & guidance:
- Initial answers should be very short (~30-60 words), just key info.
- Only expand in detail if user asks via a follow-up question.
- When discussing costs, provide the per-kWp range and let users calculate for their specific size.

Follow-up logic:
- Always return exactly ONE short, topic-style follow-up question (1–2 words) in this exact format:
  Next question: <question>
- Do NOT include this question in the main answer text; it will only be shown as a button.
- Suggestions should be actionable, specific, and relevant (e.g., "Roof suitability", "Installation cost", "Payback time").
- Do NOT duplicate content from the main answer.
- If a report is NOT ready, suggest general educational follow-ups.
- If a report IS ready, suggest report-specific analytical follow-ups.
"""

# Fallback questions if LLM doesn't generate one
FALLBACK_QUESTIONS = [
    "Costs",
    "Installation",
    "Savings",
    "Subsidies",
    "Maintenance",
    "ROI",
    "Battery storage",
    "Grid connection",
    "Energy yield",
    "Roof requirements"
]

def get_subsidy_data(state=None):
    """Query subsidy database for relevant programs"""
    try:
        if not os.path.exists(DB_PATH):
            return "Subsidy database not available."
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        if state:
            # Get federal + specific state programs
            cursor.execute("""
                SELECT Type, State, Program, Subsidy, MaxSizeBonus, Link 
                FROM Programs 
                WHERE State = 'All' OR State = ?
                ORDER BY Type, State
            """, (state,))
        else:
            # Get all federal programs
            cursor.execute("""
                SELECT Type, State, Program, Subsidy, MaxSizeBonus, Link 
                FROM Programs 
                WHERE State = 'All'
                ORDER BY Type
            """)
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return "No subsidy programs found."
        
        # Format results as text
        subsidy_text = "Available Solar Subsidies:\n\n"
        for row in results:
            type_, state_, program, subsidy, max_bonus, link = row
            subsidy_text += f"• {program} ({state_}): {subsidy}"
            if max_bonus and max_bonus != "NA":
                subsidy_text += f" | Max: {max_bonus}"
            subsidy_text += f" | {link}\n"
        
        return subsidy_text
    
    except Exception as e:
        return f"Error accessing subsidy database: {str(e)}"

def get_report_data():
    """Extract report data from session state if available"""
    if "data" not in st.session_state:
        return None
    
    data = st.session_state.data
    
    # Check if final analysis exists
    if "final_analysis" not in data:
        return None
    
    analysis = data.get("final_analysis", {})
    solar_results = data.get("solar_results", {})
    consumption_inputs = data.get("consumption_inputs", {})
    
    # Build report context
    report_context = "\n\n=== USER'S SOLAR REPORT DATA ===\n"
    
    # System details
    if solar_results:
        actual_kwp = solar_results.get('system_kwp', 0)
        actual_panels = solar_results.get('panel_count', 0)
        roof_area = solar_results.get('total_roof_area_m2', 0)
        usable_area = solar_results.get('usable_roof_area_m2', 0)
        azimuth = solar_results.get('azimuth', 0)
        tilt = solar_results.get('tilt_angle', 0)
        roof_form = solar_results.get('roof_form', 'Unknown')
        
        report_context += f"\nSystem Size: {actual_kwp:.1f} kWp ({actual_panels} panels)\n"
        report_context += f"Total Roof Area: {roof_area:.1f} m²\n"
        report_context += f"Usable Roof Area: {usable_area:.1f} m²\n"
        report_context += f"Roof Orientation: {azimuth:.0f}° azimuth\n"
        report_context += f"Roof Tilt: {tilt:.0f}°\n"
        report_context += f"Roof Type: {roof_form}\n"
    
    # Consumption data
    if consumption_inputs:
        annual_kwh = consumption_inputs.get('annual_kwh', 0)
        people = consumption_inputs.get('people', 0)
        b_type = consumption_inputs.get('b_type_code', 'Unknown')
        has_water_heater = consumption_inputs.get('has_water_heater', False)
        has_ev = consumption_inputs.get('has_ev', False)
        has_heat_pump = consumption_inputs.get('has_heat_pump', False)
        
        report_context += f"\nAnnual Consumption: {annual_kwh:,} kWh\n"
        report_context += f"Household Size: {people} people\n"
        report_context += f"Building Type: {b_type}\n"
        report_context += f"Electric Water Heater: {'Yes' if has_water_heater else 'No'}\n"
        report_context += f"Electric Vehicle: {'Yes' if has_ev else 'No'}\n"
        report_context += f"Heat Pump: {'Yes' if has_heat_pump else 'No'}\n"
    
    # Production and financial data
    if analysis:
        production_data = analysis.get('production', {})
        specific_yield = production_data.get('specific_yield_kwh_kwp', 0)
        
        if solar_results and specific_yield:
            actual_production = actual_kwp * specific_yield
            report_context += f"\nAnnual Solar Production: {int(actual_production):,} kWh\n"
            report_context += f"Specific Yield: {specific_yield:.0f} kWh/kWp\n"
            
            # Calculate financial metrics
            # Constants from your code
            ELECTRICITY_PRICE = 0.35  # €/kWh
            FEED_IN_TARIFF = 0.082  # €/kWh
            INSTALLATION_COST_PER_KW = 1400  # €/kW
            
            actual_invest = actual_kwp * INSTALLATION_COST_PER_KW
            self_consumption_rate = 0.30
            self_consumed = min(actual_production * self_consumption_rate, annual_kwh)
            savings_usage = self_consumed * ELECTRICITY_PRICE
            fed_in = actual_production - self_consumed
            earnings_feedin = fed_in * FEED_IN_TARIFF
            annual_benefit = savings_usage + earnings_feedin
            roi_years = actual_invest / annual_benefit if annual_benefit > 0 else 99
            
            coverage_percent = (actual_production / annual_kwh) * 100 if annual_kwh > 0 else 0
            
            report_context += f"\nInvestment Cost: €{int(actual_invest):,} (0% VAT)\n"
            report_context += f"Annual Savings (self-consumption): €{int(savings_usage):,}\n"
            report_context += f"Annual Feed-in Earnings: €{int(earnings_feedin):,}\n"
            report_context += f"Total Annual Benefit: €{int(annual_benefit):,}\n"
            report_context += f"Payback Period: {roi_years:.1f} years\n"
            report_context += f"Coverage of Consumption: {coverage_percent:.0f}%\n"
            
            lifetime_profit = (annual_benefit * 25) - actual_invest
            report_context += f"25-Year Net Profit: €{int(lifetime_profit):,}\n"
    
    report_context += "\n=== END OF REPORT DATA ===\n"
    
    return report_context

def initialize_chat():
    if "rag_bot" not in st.session_state:
        st.session_state.rag_bot = {
            "messages": [{"role": "assistant", "content": "How can I assist you today?"}],
            "expanded": False,
            "report_ready": False,
            "next_question": "PV basics",
            "used_fallbacks": set(),
            "user_state": None  # Store user's state for targeted subsidy info
        }

def get_fallback_question(state):
    """Get a random fallback question that hasn't been used recently"""
    available = [q for q in FALLBACK_QUESTIONS if q not in state.get("used_fallbacks", set())]
    
    if not available:
        state["used_fallbacks"] = set()
        available = FALLBACK_QUESTIONS
    
    question = random.choice(available)
    state["used_fallbacks"].add(question)
    return question

def generate_report_summary():
    """Generate an automatic summary when report becomes ready"""
    state = st.session_state.rag_bot
    
    try:
        report_context = get_report_data()
        if not report_context:
            return
        
        client = Groq(api_key=GROQ_KEY)
        
        summary_prompt = """Based on the user's solar report data, generate a very brief message (20-30 words) that:
1. Confirms their report is ready
2. Mentions they should check available subsidies
3. Offers to explain any metrics

Then suggest ONE specific follow-up question.

Format:
[Your 20-30 word message]

Next question: [1-2 word suggestion]"""
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + report_context},
            {"role": "user", "content": summary_prompt}
        ]
        
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=200
        )
        
        content = resp.choices[0].message.content.strip()
        
        # Extract follow-up question
        next_q = None
        if "Next question:" in content:
            try:
                next_q = content.split("Next question:")[1].strip().split("\n")[0]
                next_q = next_q.replace("*", "").replace("**", "").strip()
            except:
                next_q = None
        
        # Get main content without follow-up
        main_content = content.split("Next question:")[0].strip()
        
        # Add to messages
        state["messages"].append({"role": "assistant", "content": main_content})
        state["next_question"] = next_q
        
    except Exception as e:
        # Fallback message if auto-summary fails
        state["messages"].append({
            "role": "assistant", 
            "content": "Your solar report is ready! I can help explain the results, costs, savings, or any other questions you have."
        })
        state["next_question"] = "Payback time"

def render_chat_interface():
    if not GROQ_KEY:
        st.sidebar.warning("Set GROQ_API_KEY")
        return

    state = st.session_state.rag_bot
    
    # Check if report is ready and update state
    report_data = get_report_data()
    if report_data and not state["report_ready"]:
        state["report_ready"] = True
        # Generate automatic summary
        generate_report_summary()

    if state["report_ready"] and not state["expanded"]:
        state["expanded"] = True
        st.rerun()

    with st.sidebar.expander("☀️ Solar Assistant", expanded=state["expanded"]):
        state["expanded"] = True

        # --- Input field at the top ---
        if "user_input" not in st.session_state:
            st.session_state.user_input = ""

        def on_enter():
            prompt = st.session_state.user_input.strip()
            if prompt:
                st.session_state.user_input = ""
                handle_message(prompt)

        st.text_input(
            "Ask solar question:",
            key="user_input",
            label_visibility="visible",
            on_change=on_enter
        )

        # --- Messages container below input ---
        with st.container(height=300):
            # Add custom CSS for compact dividers
            st.markdown("""
                <style>
                .stMarkdown hr {
                    margin-top: 0.5rem !important;
                    margin-bottom: 0.5rem !important;
                }
                </style>
            """, unsafe_allow_html=True)
            
            messages_to_show = state["messages"][-12:]
            messages_to_show = list(reversed(messages_to_show))
            
            for i, msg in enumerate(messages_to_show):
                if msg["role"] == "user":
                    st.markdown(f"**❔ :** {msg['content']}")
                    st.markdown("---")
                else:
                    st.markdown(f"**☀️ :** {msg['content']}")

        # --- Single follow-up button below messages ---
        if not state.get("next_question"):
            state["next_question"] = get_fallback_question(state)
        
        if state.get("next_question"):
            button_key = f"followup_{hash(state['next_question'])}"
            if st.button(f"➡️ {state['next_question']}", key=button_key, use_container_width=True):
                question_to_send = state["next_question"]
                state["next_question"] = None
                handle_message(question_to_send)
                st.rerun()

def handle_message(prompt):
    state = st.session_state.rag_bot
    if not prompt.strip():
        return

    state["messages"].append({"role": "user", "content": prompt})

    try:
        # Get report data if available
        report_context = get_report_data() or ""
        
        # Check if user is asking about subsidies, feed-in tariff, or loans
        subsidy_context = ""
        prompt_lower = prompt.lower()
        subsidy_keywords = [
            "subsidy", "subsidies", "förderung", "funding", "grant", "financial support",
            "feed-in", "feed in", "eeg", "tariff", "einspeisung", "einspeisevergütung",
            "loan", "loans", "kfw", "kredit", "financing", "finanzierung"
        ]
        
        if any(keyword in prompt_lower for keyword in subsidy_keywords):
            subsidy_context = "\n\n" + get_subsidy_data(state.get("user_state"))
        
        # Detect if user mentions their state
        german_states = ["Baden-Württemberg", "Bavaria", "Berlin", "Brandenburg", "Bremen", 
                        "Hamburg", "Hesse", "Lower Saxony", "Mecklenburg-Vorpommern", 
                        "North Rhine-Westphalia", "Rhineland-Palatinate", "Saarland", 
                        "Saxony", "Saxony-Anhalt", "Schleswig-Holstein", "Thuringia"]
        for german_state in german_states:
            if german_state.lower() in prompt_lower:
                state["user_state"] = german_state
                subsidy_context = "\n\n" + get_subsidy_data(german_state)
                break

        client = Groq(api_key=GROQ_KEY)
        
        # Add report and subsidy context to system prompt if available
        system_prompt_with_context = SYSTEM_PROMPT + report_context + subsidy_context
        
        messages_with_system = [{"role": "system", "content": system_prompt_with_context}] + state["messages"]

        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages_with_system,
            temperature=0.7,
            max_tokens=500
        )

        content = resp.choices[0].message.content.strip()

        # --- Extract follow-up question ---
        next_q = None
        if "Next question:" in content:
            try:
                next_q = content.split("Next question:")[1].strip().split("\n")[0]
                next_q = next_q.replace("*", "").replace("**", "").strip()
                if next_q:
                    if "used_fallbacks" in state:
                        state["used_fallbacks"].clear()
            except:
                next_q = None

        # Append assistant message without the follow-up
        main_content = content.split("Next question:")[0].strip()
        state["messages"].append({"role": "assistant", "content": main_content})
        
        state["next_question"] = next_q

    except Exception as e:
        state["messages"].append({"role": "assistant", "content": f"Error: {str(e)[:150]}..."})
        state["next_question"] = None

# Initialize chat on load
initialize_chat()
render_chat_interface()