import streamlit as st
import os
from groq import Groq
from dotenv import load_dotenv
import random
import sqlite3
from src.rag_retrieval import retrieve_relevant_chunks

load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("LLM_MODEL")

# --- Database path ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_PATH = os.path.join(DATA_DIR, "solar_programs.db")

SYSTEM_PROMPT = """
You are a helpful solar assistant for private homeowners in Germany.

=== CRITICAL: REPORT STATUS DETECTION ===

Does this prompt contain "=== USER'S SOLAR REPORT DATA ==="?
→ YES = REPORT MODE | NO = EDUCATIONAL MODE

=== EDUCATIONAL MODE (NO REPORT) ===

User has NOT completed analysis yet.
FORBIDDEN: "your system", "your roof", "your home", "your location", "for you", "in your case", ANY user-specific numbers
ALLOWED: "A typical system...", "Most homeowners...", "Generally speaking..."
RULE: Use RANGES only. Never invent user-specific data.

=== REPORT MODE (REPORT EXISTS) ===

User analysis IS available.
FORBIDDEN: "your report", "the report shows", "typically", "usually", "on average"
REQUIRED: Use actual report data. "Your 7.5 kWp system...", "In Hamburg...", "With your €10,500..."
CRITICAL: ONLY use data from "=== USER'S SOLAR REPORT DATA ===" section. If not there, you DON'T have it.

=== CORE FACTS ===

Pricing (2026):
- PV: €1,100-1,600/kWp | Battery: €700-1,000/kWh | Grid: €0.30-0.40/kWh

EEG Tariff (§ 48 EEG 2023, after August 2025 degression):
- ≤10 kWp: 7.86 ct/kWh (partial), 12.47 ct/kWh (full feed-in)
- ≤40 kWp: 6.80 ct/kWh (partial), 10.45 ct/kWh (full feed-in)
- Degression: -1% every 6 months (Feb/Aug)

Tax (§3 Nr.72):
- 2025+: 30kW per unit max | Total: 100kW (all-or-nothing)
- All income + self-consumption tax-free | Cannot deduct expenses

VAT: 0% since Jan 2023 (all components + install)

BEG: PV NOT eligible | Heat pumps 70%

KfW 270: Loan 100% costs, risk-adjusted rates, battery separate OK

Snow: Zone 1(0.65), 1a(0.81), 2(0.85), 2a(1.06), 3(1.10) kN/m²

Installers: https://www.solarwirtschaft.de/fachpartnersuche/

Classification: ≤30kW residential | 30-100kW transition | >100kW commercial

=== ANTI-HALLUCINATION ===

NEVER claim:
- KfW fixed rate 2.75% (it's risk-adjusted!)
- Specific rates/amounts you don't know

If unknown: "Check [official source] for current details"

=== FOLLOW-UP TOPICS ===

Priority order (suggest HIGH-VALUE info user likely doesn't know):

Tier 1 - Financial: Full feed-in bonus | Tax exemption | 0% VAT | Battery financing | Mieterstrom
Tier 2 - Misconceptions: BEG eligibility | 2025 tax update | Tariff degression | 100kW limit | Ground-mounted tax
Tier 3 - Practical: Battery storage | Heat pump integration | Installer directory | Snow loads | System expansion

Routing:
- Costs → Full feed-in bonus / Tax exemption
- BEG → BEG eligibility / Heat pump subsidy
- Tariff → Full feed-in bonus
- Taxes → Tax exemption / 0% VAT
- Battery → Battery financing / Heat pump
- Install → Installer directory / 0% VAT
- ROI → Tariff degression / Full feed-in
- Multi-family → 2025 tax update / Mieterstrom
- Financing → Battery financing / Tax exemption
- System size → 100kW limit / 30kW threshold

Avoid redundancy clusters:
- A: costs/pricing/investment/budget
- B: subsidies/programs/funding/grants
- C: installer/installation/contractor
- D: ROI/payback/savings/returns
- E: tariff/feed-in/rate

=== RESPONSE STYLE ===

- Short answers (30-60 words)
- Friendly, supportive
- No tables (use paragraphs/bullets)
- MANDATORY end format: "Next question: <1-2 words>"

=== SELF-CHECK ===

1. Report in prompt? → Choose mode
2. Mentioning specific values? → Educational=ranges, Report=from report only
3. Using forbidden words? → Check mode rules
4. Making up numbers? → STOP, say "check [source]"
5. Provided follow-up? → Required (not optional)
"""

def get_financial_benefits(state=None):
    """Return formatted financial summary with program types for a given German state"""
    try:
        if not os.path.exists(DB_PATH):
            return "Subsidy database not available."

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        if state:
            cursor.execute("""
                SELECT Type, Program, Subsidy, MaxSizeBonus, Link, ProgramType
                FROM Programs
                WHERE State = 'All' OR State = ?
                ORDER BY Type, ProgramType, State
            """, (state,))
        else:
            cursor.execute("""
                SELECT Type, Program, Subsidy, MaxSizeBonus, Link, ProgramType
                FROM Programs
                WHERE State = 'All'
                ORDER BY Type, ProgramType
            """)

        results = cursor.fetchall()
        conn.close()

        if not results:
            return "No financial programs found for your location."

        # Build formatted list
        lines = []
        for row in results:
            type_, program, subsidy, max_bonus, link, program_type = row
            
            # Format: • Federal/State ProgramType: Program (max X) – Subsidy [More info](link)
            if type_ == "Federal":
                line = f"• Federal {program_type}: {program}"
            else:
                line = f"• State {program_type}: {program}"
            
            # Add max/bonus info if available
            if max_bonus and max_bonus != "NA":
                line += f" (max {max_bonus})"
            
            # Add subsidy amount
            line += f" – {subsidy}"
            
            # Add link
            if link:
                link_url = link if link.startswith("http") else f"https://{link}"
                line += f" - [More info]({link_url})"
            
            lines.append(line)

        # Use two spaces + newline for proper markdown line breaks
        return "Available financial programs:\n" + "  \n".join(lines)

    except Exception as e:
        return f"Error accessing financial programs: {str(e)}"

def initialize_chat():
    if "rag_bot" not in st.session_state:
        st.session_state.rag_bot = {
            "messages": [{"role": "assistant", "content": "Ask me anything!"}],
            "expanded": False,  # Start closed
            "report_ready": False,
            "next_question": "How solar works",  # Initial suggestion
            "used_fallbacks": set(),
            "user_state": None  # Store user's state for targeted subsidy info
        }

def get_report_data():
    """Build personalized solar system summary for the user (starts with conclusions)."""
    if "data" not in st.session_state:
        return None

    data = st.session_state.data
    analysis = data.get("final_analysis", {})
    solar_results = data.get("solar_results", {})
    consumption_inputs = data.get("consumption_inputs", {})

    actual_kwp = solar_results.get("system_kwp", 0)
    if actual_kwp <= 0:
        return None

    tilt = solar_results.get("tilt_angle")
    azimuth = solar_results.get("azimuth")
    if tilt is None or azimuth is None:
        return None

    report_context = "\n\n=== USER'S SOLAR REPORT DATA ===\n"

    # --- Location ---
    lat = data.get("confirmed_lat")
    lon = data.get("confirmed_lon")
    address = data.get("address", "")
    user_state = None
    if lat is not None and lon is not None:
        report_context += f"Location: {address} ({lat:.4f}°, {lon:.4f}°)\n"
        # Extract German state for financial programs
        german_states = [
            "Baden-Württemberg", "Bavaria", "Berlin", "Brandenburg", "Bremen",
            "Hamburg", "Hesse", "Lower Saxony", "Mecklenburg-Vorpommern",
            "North Rhine-Westphalia", "Rhineland-Palatinate", "Saarland",
            "Saxony", "Saxony-Anhalt", "Schleswig-Holstein", "Thuringia"
        ]
        for state in german_states:
            if state.lower() in address.lower():
                user_state = state
                if "rag_bot" in st.session_state:
                    st.session_state.rag_bot["user_state"] = state
                break

    # --- Key system metrics ---
    actual_panels = solar_results.get("panel_count", 0)
    roof_area = solar_results.get("total_roof_area_m2", 0)
    usable_area = solar_results.get("usable_roof_area_m2", 0)
    roof_form = solar_results.get("roof_form", "Unknown")

    # --- Production & financial metrics ---
    content_lines = []

    if analysis:
        production_data = analysis.get("production", {})
        specific_yield = production_data.get("specific_yield_kwh_kwp", 0)

        if specific_yield > 0:
            actual_production = actual_kwp * specific_yield

            annual_kwh = consumption_inputs.get("annual_kwh", 0)
            ELECTRICITY_PRICE = 0.35
            FEED_IN_TARIFF = 0.082
            INSTALLATION_COST_PER_KW = 1400

            actual_invest = actual_kwp * INSTALLATION_COST_PER_KW
            self_consumption_rate = 0.30
            self_consumed = min(actual_production * self_consumption_rate, annual_kwh)
            savings_usage = self_consumed * ELECTRICITY_PRICE
            fed_in = actual_production - self_consumed
            earnings_feedin = fed_in * FEED_IN_TARIFF
            annual_benefit = savings_usage + earnings_feedin
            roi_years = actual_invest / annual_benefit if annual_benefit > 0 else 99
            coverage_percent = (actual_production / annual_kwh) * 100 if annual_kwh else 0
            lifetime_profit = (annual_benefit * 25) - actual_invest

            # --- Start with conclusion ---
            content_lines.append(f"Your {actual_kwp:.1f} kWp system is expected to pay back in {roi_years:.1f} years, "
                                 f"with an annual benefit of €{int(annual_benefit):,}.")

            # --- Key details for context ---
            content_lines.append(f"Annual solar production: {int(actual_production):,} kWh "
                                 f"(Specific Yield: {specific_yield:.0f} kWh/kWp).")
            if annual_kwh:
                content_lines.append(f"Annual consumption: {annual_kwh:,} kWh "
                                     f"({coverage_percent:.0f}% covered by your system).")
            content_lines.append(f"Investment cost: €{int(actual_invest):,} for {actual_panels} panels.")
            content_lines.append(f"Roof: {roof_form}, {tilt:.0f}° tilt, {azimuth:.0f}° azimuth, usable area {usable_area:.1f} m².")

    # --- Consumption data if present ---
    if consumption_inputs:
        people = consumption_inputs.get("people", 0)
        b_type = consumption_inputs.get("b_type_code", "Unknown")
        has_water_heater = consumption_inputs.get("has_water_heater", False)
        has_ev = consumption_inputs.get("has_ev", False)
        has_heat_pump = consumption_inputs.get("has_heat_pump", False)
        content_lines.append(f"Household: {people} people, Building type: {b_type}, "
                             f"Electric water heater: {'Yes' if has_water_heater else 'No'}, "
                             f"EV: {'Yes' if has_ev else 'No'}, Heat pump: {'Yes' if has_heat_pump else 'No'}")

    report_context += "\n".join(content_lines)
    report_context += "\n\n=== END OF REPORT DATA ===\n"

    return report_context


def get_fallback_question(state):
    """Get a random fallback question that hasn't been used recently"""
    available = [q for q in FALLBACK_QUESTIONS if q not in state.get("used_fallbacks", set())]
    
    if not available:
        state["used_fallbacks"] = set()
        available = FALLBACK_QUESTIONS
    
    question = random.choice(available)
    state["used_fallbacks"].add(question)
    return question

# Deterministic summary for Streamlit
def generate_report_summary():
    """Generate an automatic summary when report becomes ready, including all available subsidies, incentives, and loans."""
    state = st.session_state.rag_bot

    try:
        report_context = get_report_data()
        if not report_context:
            return

        # --- Extract user_state from session data if not already set ---
        if not state.get("user_state"):
            # Try to get address first
            address = st.session_state.data.get("address", "")
            print(f"DEBUG: Address from session: '{address}'")
            
            # If address is empty or doesn't contain state, try to detect from coordinates
            if not address or len(address) < 5:
                lat = st.session_state.data.get("confirmed_lat")
                lon = st.session_state.data.get("confirmed_lon")
                print(f"DEBUG: Coordinates: {lat}, {lon}")
                
                # Detect German state from coordinates (approximate bounding boxes)
                if lat and lon:
                    detected_state = None
                    
                    # Check CITY-STATES FIRST (they are surrounded by larger states)
                    city_states = {
                        "Berlin": (52.3, 52.7, 13.0, 13.8),
                        "Hamburg": (53.4, 53.7, 9.7, 10.3),
                        "Bremen": (53.0, 53.6, 8.5, 9.0)
                    }
                    
                    for state_name, (lat_min, lat_max, lon_min, lon_max) in city_states.items():
                        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                            detected_state = state_name
                            print(f"DEBUG: Detected {state_name} from coordinates (city-state)")
                            break
                    
                    # If not a city-state, check larger states
                    if not detected_state:
                        state_coordinates = {
                            "Schleswig-Holstein": (53.4, 55.1, 8.0, 11.5),
                            "Lower Saxony": (51.3, 53.9, 6.6, 11.6),
                            "Mecklenburg-Vorpommern": (53.1, 54.7, 10.6, 14.4),
                            "Brandenburg": (51.4, 53.6, 11.3, 14.8),
                            "Saxony-Anhalt": (51.0, 53.0, 10.6, 13.2),
                            "Saxony": (50.2, 51.7, 11.9, 15.0),
                            "Thuringia": (50.2, 51.7, 9.9, 12.7),
                            "Hesse": (49.4, 51.7, 7.8, 10.3),
                            "North Rhine-Westphalia": (50.3, 52.5, 5.9, 9.5),
                            "Rhineland-Palatinate": (48.9, 50.9, 6.1, 8.5),
                            "Saarland": (49.1, 49.7, 6.4, 7.4),
                            "Baden-Württemberg": (47.5, 49.8, 7.5, 10.5),
                            "Bavaria": (47.3, 50.6, 8.9, 13.9)
                        }
                        
                        for state_name, (lat_min, lat_max, lon_min, lon_max) in state_coordinates.items():
                            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                                detected_state = state_name
                                print(f"DEBUG: Detected {state_name} from coordinates")
                                break
                    
                    if detected_state:
                        state["user_state"] = detected_state
            
            # If we still don't have a state, try parsing the address
            if not state.get("user_state") and address:
                german_states = [
                    "Baden-Württemberg", "Bavaria", "Berlin", "Brandenburg", "Bremen",
                    "Hamburg", "Hesse", "Lower Saxony", "Mecklenburg-Vorpommern",
                    "North Rhine-Westphalia", "Rhineland-Palatinate", "Saarland",
                    "Saxony", "Saxony-Anhalt", "Schleswig-Holstein", "Thuringia"
                ]
                for german_state in german_states:
                    if german_state.lower() in address.lower():
                        state["user_state"] = german_state
                        print(f"DEBUG: Detected {german_state} from address")
                        break
                
                if not state.get("user_state"):
                    print(f"DEBUG: No state detected in address or coordinates")
        
        user_state = state.get("user_state")
        print(f"DEBUG: Final user_state for query: {user_state}")

        # --- Build fallback manually if LLM fails ---
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            if user_state:
                cursor.execute("""
                    SELECT Type, State, Program, Subsidy, MaxSizeBonus, Link, ProgramType
                    FROM Programs
                    WHERE State = 'All' OR State = ?
                    ORDER BY Type, ProgramType, State
                """, (user_state,))
            else:
                cursor.execute("""
                    SELECT Type, State, Program, Subsidy, MaxSizeBonus, Link, ProgramType
                    FROM Programs
                    WHERE State = 'All'
                    ORDER BY Type, ProgramType
                """)
            results = cursor.fetchall()
            print(f"DEBUG: Found {len(results)} programs")  # Debug line
            conn.close()
        except Exception as e:
            results = []
            print("Error accessing financial programs:", e)

        if results:
            # --- Compose bullet list with proper line breaks ---
            program_lines = []
            for row in results:
                type_, state_in_db, program, subsidy, max_bonus, link, prog_type = row
                scope_text = "Federal" if type_ == "Federal" else "State"
                bonus_text = f" (max {max_bonus})" if max_bonus and max_bonus != "NA" else ""
                # Link with "More info" text
                if link:
                    link_url = link if link.startswith("http") else f"https://{link}"
                    link_text = f" - [More info]({link_url})"
                else:
                    link_text = ""
                program_lines.append(f"• {scope_text} {prog_type}: {program}{bonus_text} – {subsidy}{link_text}")

            # Use two spaces + newline for proper markdown line breaks
            programs_text = "  \n\n".join(program_lines)
        else:
            programs_text = "No financial programs found."

        # --- Build main message ---
        solar_results = st.session_state.data.get("solar_results", {})
        actual_kwp = solar_results.get("system_kwp", 0)

        main_msg = (
            f"Your {actual_kwp:.1f} kWp solar setup is ready!\n\n"
            f"Savings you are eligible for:\n\n{programs_text}\n\n"
            "Ask me anything!"
        )

        state["messages"].append({"role": "assistant", "content": main_msg})
        
        # --- Smart follow-up based on available programs ---
        # Extract first non-federal program or first loan as follow-up
        smart_followup = None
        for row in results:
            type_, state_in_db, program, subsidy, max_bonus, link, prog_type = row
            if prog_type == "Loan":
                # Suggest loan details
                smart_followup = f"EEG Feed-in Tariff"
                break
            elif type_ == "State":
                # Suggest state program details
                program_short = program.split()[0:2]  # First 2 words
                smart_followup = " ".join(program_short)
                break
        
        # Fallback to generic if no specific program found
        state["next_question"] = smart_followup if smart_followup else "Payback time"

    except Exception as e:
        state["messages"].append({"role": "assistant", "content": f"Error generating summary: {str(e)[:150]}..."})
        state["next_question"] = "Subsidies"

def render_chat_interface():
    if not GROQ_KEY:
        st.sidebar.warning("Set GROQ_API_KEY")
        return

    state = st.session_state.rag_bot
    
    # Check if report is ready and update state
    report_data = get_report_data()
    if report_data and not state["report_ready"]:
        state["report_ready"] = True
        state["expanded"] = True  # Auto-expand when report becomes ready
        # Generate automatic summary
        generate_report_summary()

    # Auto-expand and rerun only when report becomes ready
    if state["report_ready"] and not state.get("has_expanded_once"):
        state["has_expanded_once"] = True
        st.rerun()

    with st.sidebar.expander("☀️ Solar Assistant", expanded=state["expanded"]):
        # Only set expanded to True if user manually opens it or report is ready
        pass

        # --- Messages container at the top ---
        with st.container(height=300):
            st.markdown("""
                <style>
                .stMarkdown hr {
                    margin-top: 0.5rem !important;
                    margin-bottom: 0.5rem !important;
                }
                </style>
            """, unsafe_allow_html=True)
            
            messages_to_show = state["messages"][-12:]
            # Reversed order (newest at top) - ensures newest message is always visible
            messages_to_show = list(reversed(messages_to_show))
            
            for i, msg in enumerate(messages_to_show):
                # Show divider before user question (except the very first message at top)
                if msg["role"] == "user" and i > 0:
                    st.markdown("---")
                
                if msg["role"] == "user":
                    st.markdown(f"**❔ :** {msg['content']}")
                else:
                    st.markdown(f"**☀️ :** {msg['content']}")

        # --- Single follow-up button below messages ---
        # Show button - use LLM suggestion or fallback to "More details"
        follow_up_text = state.get("next_question") or "More details"

        # Use a simple static key - only one follow-up button exists at a time
        if st.button(f"➡️ {follow_up_text}", key="rag_followup_btn", use_container_width=True):
            question_to_send = follow_up_text
            state["next_question"] = None  # Clear for next time
            handle_message(question_to_send)
            st.rerun()

        # --- Input field at the bottom ---
        if "rag_bot_user_input" not in st.session_state:
            st.session_state.rag_bot_user_input = ""

        def on_enter():
            prompt = st.session_state.rag_bot_user_input.strip()
            if prompt:
                st.session_state.rag_bot_user_input = ""
                handle_message(prompt)

        st.text_input(
            "Ask solar question:",
            key="rag_bot_user_input",
            label_visibility="collapsed",
            placeholder="Ask solar question...",
            on_change=on_enter
        )

def handle_message(prompt):
    state = st.session_state.rag_bot
    if not prompt.strip():
        return

    state["messages"].append({"role": "user", "content": prompt})

    try:
        # Get report data if available
        report_context = get_report_data() or ""

        # --- Detect German state mentioned by user ---
        prompt_lower = prompt.lower()
        german_states = ["Baden-Württemberg", "Bavaria", "Berlin", "Brandenburg", "Bremen", 
                        "Hamburg", "Hesse", "Lower Saxony", "Mecklenburg-Vorpommern", 
                        "North Rhine-Westphalia", "Rhineland-Palatinate", "Saarland", 
                        "Saxony", "Saxony-Anhalt", "Schleswig-Holstein", "Thuringia"]
        for german_state in german_states:
            if german_state.lower() in prompt_lower:
                state["user_state"] = german_state
                break

        # --- Add financial summary if user asks about subsidies/loans ---
        subsidy_keywords = [
            "subsidy", "subsidies", "förderung", "funding", "grant", "financial support",
            "feed-in", "feed in", "eeg", "tariff", "einspeisung", "einspeisevergütung",
            "loan", "loans", "kfw", "kredit", "financing", "finanzierung"
        ]
        financial_summary = ""
        if any(keyword in prompt_lower for keyword in subsidy_keywords):
            financial_summary = "\n\n" + get_financial_benefits(state.get("user_state"))

        # === ADD RAG CONTEXT ===
        rag_context = retrieve_relevant_chunks(prompt, n_results=2)
        rag_summary = ""
        if rag_context:
            rag_summary = f"\n\n=== DETAILED REGULATIONS ===\n{rag_context}\nCite sources when using this information."
        # =======================

        client = Groq(api_key=GROQ_KEY)

        # Combine system prompt with report context + financial summary + RAG
        if report_context:
            system_prompt_with_context = SYSTEM_PROMPT + report_context + financial_summary + rag_summary
        else:
            system_prompt_with_context = SYSTEM_PROMPT + financial_summary + rag_summary

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
                # Only use if it's not empty and not too long
                if next_q and len(next_q) < 50:
                    pass  # Valid question
                else:
                    next_q = None
            except:
                next_q = None

        # Append assistant message without the follow-up
        main_content = content.split("Next question:")[0].strip()
        state["messages"].append({"role": "assistant", "content": main_content})
        
        # Set next question (None if LLM didn't generate one)
        state["next_question"] = next_q

    except Exception as e:
        state["messages"].append({"role": "assistant", "content": f"Error: {str(e)[:150]}..."})
        state["next_question"] = None