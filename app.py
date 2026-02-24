import streamlit as st
from streamlit_js_eval import streamlit_js_eval
import pandas as pd
import json
import os
from PIL import Image
from datetime import datetime
import google.generativeai as genai

# --- CONFIGURATION ---
DB_FILE = "balloon_inventory.json"
SETTINGS_FILE = "settings.json"

LATEX_SIZES = ["5in", "11in", "17in", "24in", "32in"]
DEFAULT_THRESHOLDS = {
    "5in": {"low": 2, "medium": 5},
    "11in": {"low": 2, "medium": 5},
    "17in": {"low": 1, "medium": 3},
    "24in": {"low": 1, "medium": 2},
    "32in": {"low": 1, "medium": 2},
}

# --- 1. BACKEND FUNCTIONS ---

def load_settings():
    """Loads thresholds from JSON file or creates it with defaults."""
    if not os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(DEFAULT_THRESHOLDS, f)
        return DEFAULT_THRESHOLDS
    
    with open(SETTINGS_FILE, 'r') as f:
        return json.load(f)

def save_settings(settings_data):
    """Saves the thresholds back to JSON."""
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings_data, f)

def load_data():
    """Loads inventory and handles migration for new fields (Foils, Usage, Barcodes, Open Bags)."""
    if not os.path.exists(DB_FILE):
        # Initial dummy data
        initial_data = [
            # Latex Examples
            {"id": 1, "category": "latex", "brand": "Tuftex", "color": "Burnt Orange", "hex": "#CC5500", "5in": {"full": 2, "open": 0}, "11in": {"full": 5, "open": 0}, "17in": {"full": 1, "open": 0}, "24in": {"full": 0, "open": 0}, "32in": {"full": 0, "open": 0}, "barcodes": {}, "monthly_usage": {}},
            {"id": 2, "category": "latex", "brand": "Sempertex", "color": "White Sand", "hex": "#E8E3D9", "5in": {"full": 10, "open": 0}, "11in": {"full": 8, "open": 0}, "17in": {"full": 4, "open": 0}, "24in": {"full": 2, "open": 0}, "32in": {"full": 1, "open": 0}, "barcodes": {}, "monthly_usage": {"2025-02": 5}},
            # Foil Examples
            {"id": 3, "category": "foil", "foil_type": "Number", "design": "1", "color": "Gold", "hex": "#D4AF37", "small": {"full": 2, "open": 0}, "large": {"full": 4, "open": 0}, "barcodes": {}, "monthly_usage": {"2026-01": 1}},
            {"id": 4, "category": "foil", "foil_type": "Shape", "design": "Dinosaur", "color": "Green", "hex": "#228B22", "small": {"full": 0, "open": 0}, "large": {"full": 2, "open": 0}, "barcodes": {}, "monthly_usage": {}},
        ]
        with open(DB_FILE, 'w') as f:
            json.dump(initial_data, f)
    
    with open(DB_FILE, 'r') as f:
        try:
            data = json.load(f)
        except:
            return pd.DataFrame()
            
    # --- MIGRATION: UPGRADE OLD DATA TO SUPPORT FOILS, USAGE, BARCODES & OPEN BAGS ---
    if data and isinstance(data, list) and len(data) > 0:
        needs_save = False
        current_month_str = datetime.now().strftime("%Y-%m")
        for entry in data:
            # 1. Add Category if missing (assume Latex)
            if "category" not in entry:
                entry["category"] = "latex"
                needs_save = True
            
            # 2. Add Foil Fields if missing
            if "foil_type" not in entry:
                entry["foil_type"] = "" # Number, Letter, Shape
                entry["design"] = ""    # "1", "A", "Star"
                entry["small"] = 0      # Will migrate to dict below
                entry["large"] = 0      
                needs_save = True
            
            # 3. Add 32in size if missing
            if "32in" not in entry:
                entry["32in"] = 0
                needs_save = True
            
            # 4. Migrate from 'usage' to 'monthly_usage'
            if "monthly_usage" not in entry:
                needs_save = True
                usage_val = entry.get("usage", 0)
                entry["monthly_usage"] = {}
                if usage_val > 0:
                    entry["monthly_usage"][current_month_str] = usage_val
                if "usage" in entry:
                    del entry["usage"]
                    
            # 5. Add barcodes dictionary mapping size -> list of barcodes
            if "barcodes" not in entry:
                needs_save = True
                entry["barcodes"] = {}
                
            # 6. Migrate sizes from ints to dicts {"full": X, "open": Y}
            sizes_to_check = LATEX_SIZES if entry["category"] == "latex" else ["small", "large"]
            for size in sizes_to_check:
                if size in entry and isinstance(entry[size], int):
                    needs_save = True
                    val = entry[size]
                    entry[size] = {"full": val, "open": 0}
                elif size not in entry:
                    needs_save = True
                    entry[size] = {"full": 0, "open": 0}
                
        if needs_save:
            with open(DB_FILE, 'w') as f:
                json.dump(data, f)
            
    return pd.DataFrame(data)

def save_data(df):
    data = df.to_dict(orient="records")
    with open(DB_FILE, 'w') as f:
        json.dump(data, f)

def analyze_image_with_gemini(image):
    # Check for the API key in Streamlit secrets
    if "API_KEY" not in st.secrets or not st.secrets["API_KEY"]:
        st.error("Gemini API Key not found. Please add it to your Streamlit secrets.")
        st.info("You can get a free API key from Google AI Studio and add it to the secrets of this app.")
        return []
        
    genai.configure(api_key=st.secrets["API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Updated prompt to handle Foils
    prompt = """
    Analyze this image of balloon packaging. 
    1. Determine if it is LATEX (standard round) or FOIL (mylar shapes/numbers).
    2. Extract Brand, Color.
    3. If FOIL: Extract the Shape/Design (e.g., "Number 1", "Letter A", "Star") and Type (Number/Letter/Shape).
    4. If LATEX: Extract Size (5in, 11in, etc).

    Return JSON list. Examples:
    [{"category": "latex", "brand": "Tuftex", "color": "Cocoa", "size": "11in"}]
    [{"category": "foil", "brand": "Northstar", "color": "Gold", "foil_type": "Number", "design": "1", "size": "large"}]
    """
    
    with st.spinner('🤖 AI is scanning...'):
        try:
            response = model.generate_content([prompt, image])
            text = response.text.strip()
            if text.startswith("```json"): text = text[7:-3]
            elif text.startswith("```"): text = text[3:-3]
            return json.loads(text)
        except Exception as e:
            st.error(f"AI Error: {e}")
            return []

# --- 2. FRONTEND UI ---

st.set_page_config(page_title="PopStock", page_icon="🎈", layout="wide")

# --- AUTHENTICATION ---
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if "APP_PASSWORD" in st.secrets:
            if st.session_state.get("password", "") == st.secrets["APP_PASSWORD"]:
                st.session_state["password_correct"] = True
                if "password" in st.session_state:
                    del st.session_state["password"]  # don't store password
            else:
                st.session_state["password_correct"] = False
        else:
            # If no password is set in secrets, allow access (or you could choose to block)
            st.warning("No 'APP_PASSWORD' found in Streamlit secrets. App is open to public.")
            st.session_state["password_correct"] = True

    def render_login_ui(show_error=False):
        st.markdown(
            """
            <style>
            /* Make the password input look like a PIN pad */
            div[data-baseweb="input"] input[type="password"] {
                text-align: center !important;
                font-size: 2rem !important;
                letter-spacing: 0.5em !important;
                padding: 10px !important;
            }
            /* Hide the password reveal toggle to keep it clean */
            div[data-baseweb="input"] button {
                display: none !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown("<h1 style='text-align: center; margin-top: 10vh;'>🎈 PopStock</h1>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center; color: #888; font-weight: normal;'>Enter PIN to unlock</h4>", unsafe_allow_html=True)
        
        _, col, _ = st.columns([1, 2, 1])
        with col:
            st.text_input(
                "PIN", 
                type="password", 
                on_change=password_entered, 
                key="password", 
                label_visibility="collapsed",
                placeholder="••••"
            )
            
            import streamlit.components.v1 as components
            components.html(
                """
                <script>
                // We use an interval to keep checking until Streamlit renders the input field
                const interval = setInterval(() => {
                    const pinInput = window.parent.document.querySelector('div[data-baseweb="input"] input[type="password"]');
                    if (pinInput) {
                        pinInput.setAttribute('inputmode', 'numeric');
                        pinInput.setAttribute('pattern', '[0-9]*');
                        clearInterval(interval);
                    }
                }, 100);
                </script>
                """,
                height=0,
                width=0
            )
            
            if show_error:
                st.error("😕 PIN incorrect")

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        render_login_ui(show_error=False)
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        render_login_ui(show_error=True)
        return False
    else:
        # Password correct.
        return True

if not check_password():
    st.stop()  # Do not continue if check_password is not True.

# Sidebar
st.sidebar.title("🎈 PopStock")
page = st.sidebar.radio("Go to", ["Inventory", "Scanner Hub", "Scan Shipment", "Add Manually", "Analytics", "Settings"])
st.sidebar.markdown("---")

# Auto-detect screen width to set view mode
screen_width = streamlit_js_eval(js_expressions='window.innerWidth', key='SCR_WIDTH')

# Default to Desktop if width is unknown (first load), otherwise use 768px threshold
if screen_width is None or screen_width > 768:
    view_mode = "💻 Desktop"
else:
    view_mode = "📱 Mobile"

# For debugging or manual override, we can hide this or keep it disabled
st.sidebar.markdown(f"*Auto-detected view: {view_mode}*")
st.sidebar.markdown("---")
st.sidebar.markdown("🛒 **[Open Supplier Site](https://bargainballoons.com)**")

latex_thresholds = load_settings()
df = load_data()

# Initialize a render key counter for mobile inputs
if "render_key" not in st.session_state:
    st.session_state.render_key = 0

# --- PAGE: INVENTORY ---
if page == "Inventory":
    st.title("Current Inventory")
    
    if view_mode == "📱 Mobile":
        st.markdown("""
        <style>
        /* 1. Target EXACTLY the stVerticalBlock container of the card, preventing bleeding to parent tabs */
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .header-row-marker) {
            position: relative !important;
        }
        /* 2. Float the entire layout wrapper containing the popover to top right */
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .header-row-marker) > div[data-testid="stLayoutWrapper"]:has(div[data-testid="stPopover"]) {
            position: absolute !important;
            top: 10px !important;
            right: 10px !important;
            width: auto !important;
            z-index: 10;
        }
        /* 3. Make the popover button a compact circle */
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .header-row-marker) div[data-testid="stPopover"] button {
            padding: 0 !important;
            width: 38px !important;
            height: 38px !important;
            min-height: 0 !important;
            border-radius: 50% !important;
            line-height: 1 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        /* 4. Hide the chevron icon container so the gear fits perfectly */
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .header-row-marker) div[data-testid="stPopover"] button > div > div:last-child {
            display: none !important;
        }
        /* 5. Force the inner button container and markdown to be perfectly centered */
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .header-row-marker) div[data-testid="stPopover"] button > div > div:first-child {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 100% !important;
            height: 100% !important;
        }
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .header-row-marker) div[data-testid="stPopover"] button div[data-testid="stMarkdownContainer"] {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 100% !important;
            height: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .header-row-marker) div[data-testid="stPopover"] button p {
            margin: 0 !important;
            padding: 0 !important;
            font-size: 1.2em !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .header-row-marker) div[data-testid="stPopover"] button span {
            margin: 0 !important;
            padding: 0 !important;
        }
        /* Safely force size columns to stay side-by-side on mobile without stacking */
        div[data-testid="stHorizontalBlock"]:has(.mobile-grid-marker) {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            width: 100% !important;
            gap: 10px !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.mobile-grid-marker) > div[data-testid="stColumn"] {
            width: calc(50% - 5px) !important;
            flex: 1 1 calc(50% - 5px) !important;
            min-width: 0 !important;
            padding: 0 !important;
        }
        /* Make number inputs more compact */
        input[type="number"] {
            text-align: center !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
    # TABS for Latex vs Foil
    tab_latex, tab_foil = st.tabs(["🔵 Latex Balloons", "✨ Foil Balloons"])
    
    # --- TAB 1: LATEX ---
    with tab_latex:
        # Filter for Latex
        latex_df = df[df['category'] == 'latex'].copy()
        
        # Search Latex
        search = st.text_input("🔍 Search Latex (Color)", "", key="search_latex")
        if search:
            latex_df = latex_df[latex_df['color'].str.contains(search, case=False) | latex_df['brand'].str.contains(search, case=False)]

        for index, row in latex_df.iterrows():
            if view_mode == "💻 Desktop":
                with st.container():
                    st.markdown(f"### {row['brand']} - {row['color']}")
                    c1, c2 = st.columns([1, 6])
                    with c1:
                        st.markdown(f'<div style="background-color:{row["hex"]}; width:60px; height:60px; border-radius:50%; border: 2px solid #ddd;"></div>', unsafe_allow_html=True)
                    with c2:
                        cols = st.columns(len(LATEX_SIZES))
                        for i, size in enumerate(LATEX_SIZES):
                            qty_dict = row[size]
                            full_qty = qty_dict.get('full', 0)
                            open_qty = qty_dict.get('open', 0)
                            
                            thresholds = latex_thresholds[size]
                            if full_qty <= thresholds["low"]:
                                color_alert = "red"
                            elif full_qty <= thresholds["medium"]:
                                color_alert = "orange"
                            else:
                                color_alert = "green"

                            cols[i].markdown(f"**{size}**")
                            cols[i].markdown(f":{color_alert}[**{full_qty} Full**] | **{open_qty} Open**")
                            
                            btn_full_c1, btn_full_c2 = cols[i].columns(2)
                            if btn_full_c1.button("➖ Full", key=f"d_l_f_sub_{row['id']}_{size}", help="Remove a full bag"):
                                if full_qty > 0:
                                    df.at[index, size]['full'] = full_qty - 1
                                    current_month_str = datetime.now().strftime("%Y-%m")
                                    usage_dict = df.at[index, 'monthly_usage']
                                    usage_dict[current_month_str] = usage_dict.get(current_month_str, 0) + 1
                                    save_data(df)
                                    st.rerun()
                            if btn_full_c2.button("➕ Full", key=f"d_l_f_add_{row['id']}_{size}", help="Add a full bag"):
                                df.at[index, size]['full'] = full_qty + 1
                                save_data(df)
                                st.rerun()
                                
                            btn_open_c1, btn_open_c2 = cols[i].columns(2)
                            if btn_open_c1.button("➖ Open", key=f"d_l_o_sub_{row['id']}_{size}", help="Trash an empty open bag"):
                                if open_qty > 0:
                                    df.at[index, size]['open'] = open_qty - 1
                                    save_data(df)
                                    st.rerun()
                            if btn_open_c2.button("➕ Open", key=f"d_l_o_add_{row['id']}_{size}", help="Open a full bag"):
                                if full_qty > 0:
                                    df.at[index, size]['full'] = full_qty - 1
                                    df.at[index, size]['open'] = open_qty + 1
                                    save_data(df)
                                    st.rerun()
                    
                    with st.popover("⚙️ Edit / Delete"):
                        st.markdown(f"**Edit {row['brand']} - {row['color']}**")
                        new_brand = st.text_input("Brand", value=row['brand'], key=f"d_edit_brand_l_{row['id']}")
                        new_color = st.text_input("Color Name", value=row['color'], key=f"d_edit_color_l_{row['id']}")
                        new_hex = st.color_picker("Color Match", value=row['hex'], key=f"d_edit_hex_l_{row['id']}")
                        
                        if st.button("Save Changes", key=f"d_save_l_{row['id']}"):
                            df.at[index, 'brand'] = new_brand
                            df.at[index, 'color'] = new_color
                            df.at[index, 'hex'] = new_hex
                            save_data(df)
                            st.rerun()
                        
                        st.divider()
                        if st.checkbox("Confirm Delete", key=f"d_confirm_delete_l_{row['id']}"):
                            if st.button("❌ Delete Permanently", type="primary", key=f"d_delete_l_{row['id']}"):
                                df.drop(index, inplace=True)
                                save_data(df)
                                st.rerun()

                    st.divider()
            else: # Mobile
                with st.container(border=True):
                    st.markdown(
                        f"""
                        <div class="header-row-marker" style="display: flex; align-items: center; gap: 10px; margin-top: 5px; padding-right: 40px;">
                            <div style="background-color:{row['hex']}; width:35px; height:35px; border-radius:50%; border: 1px solid #ddd; flex-shrink: 0;"></div>
                            <div style="line-height: 1.2;">
                                <strong>{row['brand']}</strong><br/>
                                <span style="font-size: 0.9em;">{row['color']}</span>
                            </div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    with st.popover("⚙️"):
                        st.markdown(f"**Edit {row['brand']} - {row['color']}**")
                        new_brand = st.text_input("Brand", value=row['brand'], key=f"m_edit_brand_l_{row['id']}")
                        new_color = st.text_input("Color Name", value=row['color'], key=f"m_edit_color_l_{row['id']}")
                        new_hex = st.color_picker("Color Match", value=row['hex'], key=f"m_edit_hex_l_{row['id']}")
                        
                        if st.button("Save Changes", key=f"m_save_l_{row['id']}", use_container_width=True):
                            df.at[index, 'brand'] = new_brand
                            df.at[index, 'color'] = new_color
                            df.at[index, 'hex'] = new_hex
                            save_data(df)
                            st.rerun()
                        
                        st.divider()
                        if st.checkbox("Confirm Delete", key=f"m_confirm_delete_l_{row['id']}"):
                            if st.button("❌ Delete Permanently", type="primary", key=f"m_delete_l_{row['id']}", use_container_width=True):
                                df.drop(index, inplace=True)
                                save_data(df)
                                st.rerun()

                    st.markdown("<hr style='margin: 10px 0; border: none; border-top: 1px solid #eee;'>", unsafe_allow_html=True)

                    # Sizes grid mobile
                    for i in range(0, len(LATEX_SIZES), 2):
                        chunk = LATEX_SIZES[i:i+2]
                        # Always create 2 columns so an odd item out (like 32in) doesn't expand to full width
                        cols = st.columns(2)
                        for j, size in enumerate(chunk):
                            qty_dict = row[size]
                            full_qty = qty_dict.get('full', 0)
                            open_qty = qty_dict.get('open', 0)
                            
                            thresholds = latex_thresholds[size]
                            
                            indicator = "🔴" if full_qty <= thresholds["low"] else "🟠" if full_qty <= thresholds["medium"] else "🟢"
                                
                            with cols[j]:
                                st.markdown('<div class="mobile-grid-marker" style="display:none;"></div>', unsafe_allow_html=True)
                                new_full_qty = st.number_input(
                                    f"{indicator} {size} (Full)",
                                    min_value=0,
                                    value=int(full_qty),
                                    step=1,
                                    key=f"m_qty_l_full_{row['id']}_{size}_{st.session_state.render_key}"
                                )
                                if new_full_qty != full_qty:
                                    if new_full_qty < full_qty:
                                        current_month_str = datetime.now().strftime("%Y-%m")
                                        usage_dict = df.at[index, 'monthly_usage']
                                        usage_dict[current_month_str] = usage_dict.get(current_month_str, 0) + (full_qty - new_full_qty)
                                    df.at[index, size]['full'] = new_full_qty
                                    save_data(df)
                                    st.session_state.render_key += 1
                                    st.rerun()
                                
                                # Open bags controller
                                new_open_qty = st.number_input(
                                    f"{size} (Open)",
                                    min_value=0,
                                    value=int(open_qty),
                                    step=1,
                                    key=f"m_qty_l_open_{row['id']}_{size}_{st.session_state.render_key}"
                                )
                                if new_open_qty != open_qty:
                                    if new_open_qty > open_qty:
                                        # They are opening a bag. Subtract from full.
                                        if full_qty > 0:
                                            df.at[index, size]['full'] = full_qty - 1
                                            df.at[index, size]['open'] = new_open_qty
                                            save_data(df)
                                            st.session_state.render_key += 1
                                            st.rerun()
                                        else:
                                            # They tried to open a bag but none are full. Reject the change.
                                            st.toast(f"No full bags of {size} to open!")
                                            st.session_state.render_key += 1
                                            st.rerun()
                                    else:
                                        # They are just removing/trashing an open bag.
                                        df.at[index, size]['open'] = new_open_qty
                                        save_data(df)
                                        st.session_state.render_key += 1
                                        st.rerun()

    # --- TAB 2: FOIL ---
    with tab_foil:
        # Filter for Foil
        foil_df = df[df['category'] == 'foil'].copy()
        
        # Search Foil
        c_search, c_filter = st.columns([3, 1])
        search_foil = c_search.text_input("🔍 Search Foils (e.g. 'Gold 1')", "", key="search_foil")
        type_filter = c_filter.selectbox("Filter Type", ["All", "Number", "Letter", "Shape"])
        
        if search_foil:
            foil_df = foil_df[foil_df['color'].str.contains(search_foil, case=False) | foil_df['design'].str.contains(search_foil, case=False)]
        if type_filter != "All":
            foil_df = foil_df[foil_df['foil_type'] == type_filter]
            
        # Sort so Numbers 0-9 appear in order
        if not foil_df.empty:
            foil_df = foil_df.sort_values(by=['foil_type', 'design'])

        for index, row in foil_df.iterrows():
            if view_mode == "💻 Desktop":
                with st.container():
                    # Foil Header: "Gold Number 1"
                    st.markdown(f"### {row['color']} - {row['design']} ({row['foil_type']})")
                    
                    c1, c2 = st.columns([1, 6])
                    with c1:
                        # Square icon for foils
                        st.markdown(f'<div style="background-color:{row["hex"]}; width:60px; height:60px; border-radius:10%; border: 2px solid #ddd;"></div>', unsafe_allow_html=True)
                    
                    with c2:
                        cols = st.columns(3)
                        # FOIL SIZES: Small vs Large
                        foil_sizes = [("small", "Small (16in/Air)"), ("large", "Large (40in/Helium)")]
                        
                        for i, (field, label) in enumerate(foil_sizes):
                            qty_dict = row[field]
                            full_qty = qty_dict.get('full', 0)
                            open_qty = qty_dict.get('open', 0)
                            color_alert = "red" if full_qty == 0 else "green"
                            cols[i].markdown(f"**{label}**")
                            cols[i].markdown(f":{color_alert}[**{full_qty} Full**] | **{open_qty} Open**")
                            
                            btn_full_c1, btn_full_c2 = cols[i].columns(2)
                            if btn_full_c1.button("➖ Full", key=f"d_f_f_sub_{row['id']}_{field}"):
                                if full_qty > 0:
                                    df.at[index, field]['full'] = full_qty - 1
                                    current_month_str = datetime.now().strftime("%Y-%m")
                                    usage_dict = df.at[index, 'monthly_usage']
                                    usage_dict[current_month_str] = usage_dict.get(current_month_str, 0) + 1
                                    save_data(df)
                                    st.rerun()
                            if btn_full_c2.button("➕ Full", key=f"d_f_f_add_{row['id']}_{field}"):
                                df.at[index, field]['full'] = full_qty + 1
                                save_data(df)
                                st.rerun()
                                
                            btn_open_c1, btn_open_c2 = cols[i].columns(2)
                            if btn_open_c1.button("➖ Open", key=f"d_f_o_sub_{row['id']}_{field}"):
                                if open_qty > 0:
                                    df.at[index, field]['open'] = open_qty - 1
                                    save_data(df)
                                    st.rerun()
                            if btn_open_c2.button("➕ Open", key=f"d_f_o_add_{row['id']}_{field}"):
                                if full_qty > 0:
                                    df.at[index, field]['full'] = full_qty - 1
                                    df.at[index, field]['open'] = open_qty + 1
                                    save_data(df)
                                    st.rerun()
                    
                    with st.popover("⚙️ Edit / Delete"):
                        st.markdown(f"**Edit {row['color']} - {row['design']}**")
                        new_brand = st.text_input("Brand", value=row['brand'], key=f"d_edit_brand_f_{row['id']}")
                        new_color = st.text_input("Color Name", value=row['color'], key=f"d_edit_color_f_{row['id']}")
                        new_design = st.text_input("Design", value=row['design'], key=f"d_edit_design_f_{row['id']}")
                        foil_types = ["Number", "Letter", "Shape"]
                        current_type_index = foil_types.index(row['foil_type']) if row['foil_type'] in foil_types else 0
                        new_foil_type = st.selectbox("Foil Type", foil_types, index=current_type_index, key=f"d_edit_type_f_{row['id']}")
                        new_hex = st.color_picker("Color Match", value=row['hex'], key=f"d_edit_hex_f_{row['id']}")

                        if st.button("Save Changes", key=f"d_save_f_{row['id']}"):
                            df.at[index, 'brand'] = new_brand
                            df.at[index, 'color'] = new_color
                            df.at[index, 'design'] = new_design
                            df.at[index, 'foil_type'] = new_foil_type
                            df.at[index, 'hex'] = new_hex
                            save_data(df)
                            st.rerun()
                        
                        st.divider()
                        if st.checkbox("Confirm Delete", key=f"d_confirm_delete_f_{row['id']}"):
                            if st.button("❌ Delete Permanently", type="primary", key=f"d_delete_f_{row['id']}"):
                                df.drop(index, inplace=True)
                                save_data(df)
                                st.rerun()
                                
                    st.divider()
            else: # Mobile
                with st.container(border=True):
                    st.markdown(
                        f"""
                        <div class="header-row-marker" style="display: flex; align-items: center; gap: 10px; margin-top: 5px; padding-right: 40px;">
                            <div style="background-color:{row['hex']}; width:35px; height:35px; border-radius:10%; border: 1px solid #ddd; flex-shrink: 0;"></div>
                            <div style="line-height: 1.2;">
                                <strong>{row['color']} - {row['design']}</strong><br/>
                                <span style="font-size: 0.9em;">({row['foil_type']})</span>
                            </div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    with st.popover("⚙️"):
                        st.markdown(f"**Edit {row['color']} - {row['design']}**")
                        new_brand = st.text_input("Brand", value=row['brand'], key=f"m_edit_brand_f_{row['id']}")
                        new_color = st.text_input("Color Name", value=row['color'], key=f"m_edit_color_f_{row['id']}")
                        new_design = st.text_input("Design", value=row['design'], key=f"m_edit_design_f_{row['id']}")
                        foil_types = ["Number", "Letter", "Shape"]
                        current_type_index = foil_types.index(row['foil_type']) if row['foil_type'] in foil_types else 0
                        new_foil_type = st.selectbox("Foil Type", foil_types, index=current_type_index, key=f"m_edit_type_f_{row['id']}")
                        new_hex = st.color_picker("Color Match", value=row['hex'], key=f"m_edit_hex_f_{row['id']}")

                        if st.button("Save Changes", key=f"m_save_f_{row['id']}", use_container_width=True):
                            df.at[index, 'brand'] = new_brand
                            df.at[index, 'color'] = new_color
                            df.at[index, 'design'] = new_design
                            df.at[index, 'foil_type'] = new_foil_type
                            df.at[index, 'hex'] = new_hex
                            save_data(df)
                            st.rerun()
                        
                        st.divider()
                        if st.checkbox("Confirm Delete", key=f"m_confirm_delete_f_{row['id']}"):
                            if st.button("❌ Delete Permanently", type="primary", key=f"m_delete_f_{row['id']}", use_container_width=True):
                                df.drop(index, inplace=True)
                                save_data(df)
                                st.rerun()
                    
                    st.markdown("<hr style='margin: 10px 0; border: none; border-top: 1px solid #eee;'>", unsafe_allow_html=True)
                    
                    # FOIL SIZES grid mobile
                    foil_sizes = [("small", "Small (16in)"), ("large", "Large (40in)")]
                    cols = st.columns(2)
                    for j, (field, label) in enumerate(foil_sizes):
                        qty_dict = row[field]
                        full_qty = qty_dict.get('full', 0)
                        open_qty = qty_dict.get('open', 0)
                        
                        indicator = "🔴" if full_qty == 0 else "🟢"
                        
                        with cols[j]:
                            st.markdown('<div class="mobile-grid-marker" style="display:none;"></div>', unsafe_allow_html=True)
                            new_full_qty = st.number_input(
                                f"{indicator} {label} (Full)",
                                min_value=0,
                                value=int(full_qty),
                                step=1,
                                key=f"m_qty_f_full_{row['id']}_{field}_{st.session_state.render_key}"
                            )
                            if new_full_qty != full_qty:
                                if new_full_qty < full_qty:
                                    current_month_str = datetime.now().strftime("%Y-%m")
                                    usage_dict = df.at[index, 'monthly_usage']
                                    usage_dict[current_month_str] = usage_dict.get(current_month_str, 0) + (full_qty - new_full_qty)
                                df.at[index, field]['full'] = new_full_qty
                                save_data(df)
                                st.session_state.render_key += 1
                                st.rerun()
                                
                            # Open bags controller
                            new_open_qty = st.number_input(
                                f"{label} (Open)",
                                min_value=0,
                                value=int(open_qty),
                                step=1,
                                key=f"m_qty_f_open_{row['id']}_{field}_{st.session_state.render_key}"
                            )
                            if new_open_qty != open_qty:
                                if new_open_qty > open_qty:
                                    if full_qty > 0:
                                        df.at[index, field]['full'] = full_qty - 1
                                        df.at[index, field]['open'] = new_open_qty
                                        save_data(df)
                                        st.session_state.render_key += 1
                                        st.rerun()
                                    else:
                                        st.toast(f"No full bags of {label} to open!")
                                        st.session_state.render_key += 1
                                        st.rerun()
                                else:
                                    df.at[index, field]['open'] = new_open_qty
                                    save_data(df)
                                    st.session_state.render_key += 1
                                    st.rerun()

# --- PAGE: ADD MANUALLY ---
elif page == "Add Manually":
    st.title("➕ Add New Inventory")
    
    type_choice = st.radio("What are you adding?", ["🔵 Latex Balloon", "✨ Foil Balloon"], horizontal=True)
    
    with st.form("add_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            brand = st.text_input("Brand", "Tuftex")
            color = st.text_input("Color Name", "Gold")
            hex_code = st.color_picker("Color Match", "#D4AF37")
            
        with col2:
            if type_choice == "🔵 Latex Balloon":
                category = "latex"
                foil_type = ""
                design = ""
                st.info(f"Creating a standard color family ({'/'.join(LATEX_SIZES)})")
            else:
                category = "foil"
                foil_type = st.selectbox("Foil Type", ["Number", "Letter", "Shape"])
                design = st.text_input("Design/Value (e.g. '1', 'A', 'Star')")
                st.info("Creating a foil entry (Small/Large)")

        submitted = st.form_submit_button("Create Entry")
        
        if submitted:
            new_id = df['id'].max() + 1 if not df.empty else 1
            new_row = {
                "id": new_id, 
                "category": category,
                "brand": brand, 
                "color": color, 
                "hex": hex_code,
                "foil_type": foil_type,
                "design": design,
                "5in": {"full": 0, "open": 0}, "11in": {"full": 0, "open": 0}, "17in": {"full": 0, "open": 0}, "24in": {"full": 0, "open": 0}, "32in": {"full": 0, "open": 0}, # Latex fields
                "small": {"full": 0, "open": 0}, "large": {"full": 0, "open": 0}, # Foil fields
                "barcodes": {},
                "monthly_usage": {}
            }
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            save_data(df)
            st.success(f"Added {color} {design} to {category} list!")

# --- PAGE: ANALYTICS ---
elif page == "Analytics":
    st.title("📊 Usage Trends")
    st.markdown("Analyze balloon usage by month and compare year-over-year data.")
    
    if not df.empty:
        # --- Data Preparation ---
        # Get all available year-month keys from the data
        all_months = set()
        for usage_dict in df['monthly_usage']:
            if isinstance(usage_dict, dict):
                all_months.update(usage_dict.keys())
        
        if not all_months:
            st.info("No usage data recorded yet. Start using balloons (clicking ➖) to see trends here.")
        else:
            sorted_months = sorted(list(all_months), reverse=True)
            
            # --- UI Filters ---
            selected_month_str = st.selectbox("Select Month to Analyze", sorted_months)
            
            # --- Chart for Selected Month ---
            st.header(f"Usage for {datetime.strptime(selected_month_str, '%Y-%m').strftime('%B %Y')}")
            
            # Create a dataframe for the chart
            df['current_month_usage'] = df['monthly_usage'].apply(lambda x: x.get(selected_month_str, 0) if isinstance(x, dict) else 0)
            chart_df = df[df['current_month_usage'] > 0].copy()
            chart_df = chart_df.sort_values(by='current_month_usage', ascending=False).head(15)
            
            if chart_df.empty:
                st.write("No balloons were used in this month.")
            else:
                chart_df['Label'] = chart_df.apply(lambda x: f"{x['color']} {x['design']}" if x['category'] == 'foil' else x['color'], axis=1)
                st.bar_chart(chart_df.set_index('Label')['current_month_usage'])

                # --- Year-over-Year Comparison ---
                st.header("Year-over-Year Comparison")
                try:
                    current_date = datetime.strptime(selected_month_str, '%Y-%m')
                    last_year_date = current_date.replace(year=current_date.year - 1)
                    last_year_month_str = last_year_date.strftime('%Y-%m')

                    if last_year_month_str in all_months:
                        df['last_year_usage'] = df['monthly_usage'].apply(lambda x: x.get(last_year_month_str, 0) if isinstance(x, dict) else 0)
                        
                        # Combine data for comparison view
                        comparison_df = df[['category', 'color', 'design', 'current_month_usage', 'last_year_usage']].copy()
                        comparison_df = comparison_df[(comparison_df['current_month_usage'] > 0) | (comparison_df['last_year_usage'] > 0)]
                        comparison_df['Label'] = comparison_df.apply(lambda x: f"{x['color']} {x['design']}" if x['category'] == 'foil' else x['color'], axis=1)
                        comparison_df['Change'] = comparison_df['current_month_usage'] - comparison_df['last_year_usage']
                        
                        st.write(f"Comparing {selected_month_str} with {last_year_month_str}")
                        st.dataframe(comparison_df[['Label', 'current_month_usage', 'last_year_usage', 'Change']].rename(columns={
                            'current_month_usage': f'Usage ({selected_month_str})',
                            'last_year_usage': f'Usage ({last_year_month_str})'
                        }).set_index('Label'))

                    else:
                        st.info(f"No data available for {last_year_month_str} to compare.")
                except Exception as e:
                    st.error(f"Could not perform year-over-year comparison: {e}")

    else:
        st.info("No inventory data found.")

# --- PAGE: SETTINGS ---
elif page == "Settings":
    st.title("⚙️ Settings")
    st.header("Latex Stock Thresholds")
    st.write("Set the bag count at which stock is considered 'Low' (red) or 'Medium' (orange).")

    # Use a copy of the loaded settings to allow for changes
    updated_thresholds = latex_thresholds.copy()

    for size in LATEX_SIZES:
        st.subheader(f"{size} Balloons")
        col1, col2 = st.columns(2)
        low_val = col1.number_input(
            "Low Stock Threshold (<=)", 
            min_value=0, 
            value=updated_thresholds[size]["low"], 
            key=f"low_{size}"
        )
        medium_val = col2.number_input(
            "Medium Stock Threshold (<=)", 
            min_value=low_val,
            value=updated_thresholds[size]["medium"], 
            key=f"medium_{size}"
        )
        # Update the dictionary with new values from the UI
        updated_thresholds[size] = {"low": low_val, "medium": medium_val}

    if st.button("Save Settings"):
        save_settings(updated_thresholds)
        st.success("Settings saved successfully!")
        st.rerun()

# --- PAGE: SCANNER HUB ---
elif page == "Scanner Hub":
    st.title("🎯 Scanner Hub")
    
    st.markdown("""
    <style>
    /* Make the radio buttons huge for easy tapping */
    div.row-widget.stRadio > div{
        flex-direction:row;
        align-items: stretch;
    }
    div.row-widget.stRadio > div > label {
        padding: 20px !important;
        background-color: #f0f2f6;
        border-radius: 10px;
        margin-right: 10px;
        text-align: center;
        flex: 1;
        cursor: pointer;
    }
    div.row-widget.stRadio > div > label[data-checked="true"] {
        background-color: #4CAF50;
        color: white;
    }
    /* Hide the actual radio circle */
    div.row-widget.stRadio > div > label > div:first-child {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state for manual link
    if "unknown_barcode" not in st.session_state:
        st.session_state.unknown_barcode = None

    # Scanner Modes
    mode = st.radio(
        "Select Action", 
        ["🔵 RECEIVING\n(Add Full)", "🟡 OPENING\n(Full ➔ Open)", "🔴 TRASHING\n(Use Open)"],
        label_visibility="collapsed"
    )
    
    st.divider()

    # If we have an unknown barcode, show the linking UI
    if st.session_state.unknown_barcode:
        st.warning(f"Barcode **{st.session_state.unknown_barcode}** not recognized!")
        st.write("Please link this barcode to an existing item:")
        
        with st.form("link_barcode_form"):
            # Create a selection list of all items and sizes
            item_options = []
            for index, row in df.iterrows():
                label_base = f"{row['brand']} - {row['color']}" if row['category'] == 'latex' else f"{row['color']} {row['design']} ({row['foil_type']})"
                sizes = LATEX_SIZES if row['category'] == 'latex' else ["small", "large"]
                for size in sizes:
                    item_options.append({"label": f"{label_base} - {size}", "id": row['id'], "size": size})
            
            selected_item_label = st.selectbox("Select Item", [opt["label"] for opt in item_options])
            
            c1, c2 = st.columns(2)
            with c1:
                if st.form_submit_button("🔗 Link & Process", type="primary", use_container_width=True):
                    # Find the selected item details
                    selected_opt = next(opt for opt in item_options if opt["label"] == selected_item_label)
                    item_id = selected_opt["id"]
                    item_size = selected_opt["size"]
                    
                    # Update database
                    idx = df.index[df['id'] == item_id][0]
                    
                    # Ensure barcodes dict exists and the size list exists
                    if 'barcodes' not in df.at[idx]:
                        df.at[idx, 'barcodes'] = {}
                    if not isinstance(df.at[idx, 'barcodes'], dict):
                        df.at[idx, 'barcodes'] = {}
                        
                    barcodes_dict = df.at[idx, 'barcodes']
                    if item_size not in barcodes_dict:
                        barcodes_dict[item_size] = []
                    
                    barcodes_dict[item_size].append(st.session_state.unknown_barcode)
                    df.at[idx, 'barcodes'] = barcodes_dict
                    
                    # Also process the scan based on current mode!
                    qty_dict = df.at[idx, item_size]
                    if mode.startswith("🔵"): # Receiving
                        qty_dict['full'] += 1
                        action_msg = f"Added 1 Full bag to {selected_item_label}"
                    elif mode.startswith("🟡"): # Opening
                        if qty_dict['full'] > 0:
                            qty_dict['full'] -= 1
                            qty_dict['open'] += 1
                            action_msg = f"Opened 1 bag of {selected_item_label}"
                        else:
                            action_msg = f"Cannot open: No full bags of {selected_item_label} in stock."
                    else: # Trashing
                        if qty_dict['open'] > 0:
                            qty_dict['open'] -= 1
                            action_msg = f"Trashed 1 open bag of {selected_item_label}"
                        elif qty_dict['full'] > 0:
                            qty_dict['full'] -= 1
                            action_msg = f"Trashed 1 full bag of {selected_item_label}"
                        else:
                            action_msg = f"Cannot trash: No stock of {selected_item_label}."
                            
                        # Log usage
                        current_month_str = datetime.now().strftime("%Y-%m")
                        usage_dict = df.at[idx, 'monthly_usage']
                        usage_dict[current_month_str] = usage_dict.get(current_month_str, 0) + 1
                        df.at[idx, 'monthly_usage'] = usage_dict
                    
                    df.at[idx, item_size] = qty_dict
                    save_data(df)
                    
                    st.session_state.unknown_barcode = None
                    st.toast(action_msg)
                    st.rerun()
            with c2:
                if st.form_submit_button("Cancel", use_container_width=True):
                    st.session_state.unknown_barcode = None
                    st.rerun()

    else:
        # Standard Scanning UI
        def handle_scan():
            scanned_code = st.session_state.barcode_input.strip()
            if not scanned_code:
                return
                
            # Search for barcode
            found = False
            for index, row in df.iterrows():
                barcodes_dict = row.get('barcodes', {})
                if not isinstance(barcodes_dict, dict): continue
                
                for size, code_list in barcodes_dict.items():
                    if scanned_code in code_list:
                        # FOUND IT!
                        found = True
                        label_base = f"{row['brand']} - {row['color']}" if row['category'] == 'latex' else f"{row['color']} {row['design']} ({row['foil_type']})"
                        item_label = f"{label_base} - {size}"
                        
                        qty_dict = df.at[index, size]
                        
                        if mode.startswith("🔵"): # Receiving
                            qty_dict['full'] += 1
                            action_msg = f"✅ Added 1 Full bag to {item_label}"
                        elif mode.startswith("🟡"): # Opening
                            if qty_dict['full'] > 0:
                                qty_dict['full'] -= 1
                                qty_dict['open'] += 1
                                action_msg = f"✅ Opened 1 bag of {item_label}"
                            else:
                                action_msg = f"❌ Cannot open: No full bags of {item_label} in stock."
                        else: # Trashing
                            if qty_dict['open'] > 0:
                                qty_dict['open'] -= 1
                                action_msg = f"🗑️ Trashed 1 open bag of {item_label}"
                            elif qty_dict['full'] > 0:
                                qty_dict['full'] -= 1
                                action_msg = f"🗑️ Trashed 1 full bag of {item_label}"
                            else:
                                action_msg = f"❌ Cannot trash: No stock of {item_label}."
                                
                            # Log usage if trashed
                            if "🗑️" in action_msg:
                                current_month_str = datetime.now().strftime("%Y-%m")
                                usage_dict = df.at[index, 'monthly_usage']
                                usage_dict[current_month_str] = usage_dict.get(current_month_str, 0) + 1
                                df.at[index, 'monthly_usage'] = usage_dict

                        df.at[index, size] = qty_dict
                        save_data(df)
                        st.toast(action_msg)
                        break
                if found:
                    break
            
            if not found:
                st.session_state.unknown_barcode = scanned_code
                
            # Clear input for next scan
            st.session_state.barcode_input = ""

        # The actual input field
        st.text_input(
            "Scan Barcode Here", 
            key="barcode_input", 
            on_change=handle_scan,
            help="Ensure this box is selected before pulling the scanner trigger."
        )
        st.info("💡 Keep the text box above selected. When the scanner beeps, it will automatically process and clear itself for the next scan.")

# --- PAGE: SCAN SHIPMENT ---
elif page == "Scan Shipment":
    st.title("📷 Scan New Shipment")
    st.write("Upload a photo of the balloon bags. The AI will identify them and update your inventory.")

    img_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    if img_file:
        image = Image.open(img_file)
        st.image(image, caption="Uploaded Image", width=500)

        if st.button("Analyze Image"):
            detected_items = analyze_image_with_gemini(image)

            if detected_items:
                st.success(f"AI Found {len(detected_items)} item(s)!")
                st.subheader("Processing Results:")
                
                updated_count = 0
                new_item_warnings = []

                for item in detected_items:
                    category = item.get('category')
                    
                    if category == 'latex':
                        brand = item.get('brand', 'Unknown')
                        color = item.get('color', 'Unknown')
                        size = item.get('size')
                        
                        if size not in LATEX_SIZES:
                            st.warning(f"Skipping '{color}' - AI returned unknown size '{size}'.")
                            continue

                        # Case-insensitive matching
                        mask = (df['category'] == 'latex') & (df['brand'].str.lower() == brand.lower()) & (df['color'].str.lower() == color.lower())
                        if mask.any():
                            idx = df[mask].index[0]
                            df.at[idx, size] += 1
                            st.write(f"✅ Added 1 bag to **{df.at[idx, 'brand']} {df.at[idx, 'color']} ({size})**.")
                            updated_count += 1
                        else:
                            new_item_warnings.append(f"Latex: **{brand} {color}**. Please add it via 'Add Manually'.")

                    elif category == 'foil':
                        brand = item.get('brand', 'Unknown')
                        color = item.get('color', 'Unknown')
                        design = item.get('design', 'Unknown')
                        size_field = "large" if item.get('size') == "large" else "small"
                        
                        mask = (df['category'] == 'foil') & (df['brand'].str.lower() == brand.lower()) & (df['color'].str.lower() == color.lower()) & (df['design'].str.lower() == design.lower())
                        if mask.any():
                            idx = df[mask].index[0]
                            df.at[idx, size_field] += 1
                            st.write(f"✅ Added 1 to **{df.at[idx, 'color']} {df.at[idx, 'design']} ({size_field})**.")
                            updated_count += 1
                        else:
                            new_item_warnings.append(f"Foil: **{brand} {color} {design}**. Please add it via 'Add Manually'.")
                
                if updated_count > 0:
                    save_data(df)
                    st.toast(f"Successfully updated {updated_count} inventory item(s)!")
                
                if new_item_warnings:
                    st.warning("Some items are new and could not be added automatically:")
                    for warning in set(new_item_warnings):
                        st.markdown(f"- {warning}")
            else:
                st.warning("The AI could not detect any balloon bags in the image. Please try another photo.")
