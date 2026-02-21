import streamlit as st
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
    """Loads inventory and handles migration for new Foil fields."""
    if not os.path.exists(DB_FILE):
        # Initial dummy data
        initial_data = [
            # Latex Examples
            {"id": 1, "category": "latex", "brand": "Tuftex", "color": "Burnt Orange", "hex": "#CC5500", "5in": 2, "11in": 5, "17in": 1, "24in": 0, "32in": 0, "monthly_usage": {}},
            {"id": 2, "category": "latex", "brand": "Sempertex", "color": "White Sand", "hex": "#E8E3D9", "5in": 10, "11in": 8, "17in": 4, "24in": 2, "32in": 1, "monthly_usage": {"2025-02": 5}},
            # Foil Examples
            {"id": 3, "category": "foil", "foil_type": "Number", "design": "1", "color": "Gold", "hex": "#D4AF37", "small": 2, "large": 4, "monthly_usage": {"2026-01": 1}},
            {"id": 4, "category": "foil", "foil_type": "Shape", "design": "Dinosaur", "color": "Green", "hex": "#228B22", "small": 0, "large": 2, "monthly_usage": {}},
        ]
        with open(DB_FILE, 'w') as f:
            json.dump(initial_data, f)
    
    with open(DB_FILE, 'r') as f:
        try:
            data = json.load(f)
        except:
            return pd.DataFrame()
            
    # --- MIGRATION: UPGRADE OLD DATA TO SUPPORT FOILS ---
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
                entry["small"] = 0      # 16 inch / Air
                entry["large"] = 0      # 40 inch / Helium
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
            if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
                st.session_state["password_correct"] = True
                del st.session_state["password"]  # don't store password
            else:
                st.session_state["password_correct"] = False
        else:
            # If no password is set in secrets, allow access (or you could choose to block)
            st.warning("No 'APP_PASSWORD' found in Streamlit secrets. App is open to public.")
            st.session_state["password_correct"] = True

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Please enter the app password to access PopStock", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Please enter the app password to access PopStock", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if not check_password():
    st.stop()  # Do not continue if check_password is not True.

# Sidebar
st.sidebar.title("🎈 PopStock")
page = st.sidebar.radio("Go to", ["Inventory", "Scan Shipment", "Add Manually", "Analytics", "Settings"])
st.sidebar.markdown("---")
view_mode = st.sidebar.radio("View Mode", ["💻 Desktop", "📱 Mobile"])
st.sidebar.markdown("---")
st.sidebar.markdown("🛒 **[Open Supplier Site](https://bargainballoons.com)**")

latex_thresholds = load_settings()
df = load_data()

# --- PAGE: INVENTORY ---
if page == "Inventory":
    st.title("Current Inventory")
    
    if view_mode == "📱 Mobile":
        st.markdown("""
        <style>
        /* Card container relative for absolute positioning */
        div[data-testid="stVerticalBlock"]:has(.header-row-marker) {
            position: relative !important;
        }
        /* Float the entire layout wrapper containing the popover to top right */
        div[data-testid="stVerticalBlock"]:has(.header-row-marker) > div[data-testid="stLayoutWrapper"]:has(div[data-testid="stPopover"]) {
            position: absolute !important;
            top: 5px !important;
            right: 5px !important;
            width: auto !important;
            z-index: 10;
        }
        /* Make the popover button a compact circle */
        div[data-testid="stVerticalBlock"]:has(.header-row-marker) div[data-testid="stPopover"] button {
            padding: 0 !important;
            width: 32px !important;
            height: 32px !important;
            min-height: 0 !important;
            border-radius: 50% !important;
            line-height: 1 !important;
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
                            qty = row[size]
                            
                            thresholds = latex_thresholds[size]
                            if qty <= thresholds["low"]:
                                color_alert = "red"
                            elif qty <= thresholds["medium"]:
                                color_alert = "orange"
                            else:
                                color_alert = "green"

                            cols[i].markdown(f"**{size}**")
                            cols[i].markdown(f":{color_alert}[**{qty} bags**]")
                            
                            if cols[i].button("➖", key=f"d_l_sub_{row['id']}_{size}"):
                                if qty > 0:
                                    df.at[index, size] = qty - 1
                                    current_month_str = datetime.now().strftime("%Y-%m")
                                    usage_dict = df.at[index, 'monthly_usage']
                                    usage_dict[current_month_str] = usage_dict.get(current_month_str, 0) + 1
                                    save_data(df)
                                    st.rerun()
                            if cols[i].button("➕", key=f"d_l_add_{row['id']}_{size}"):
                                df.at[index, size] = qty + 1
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
                        cols = st.columns(len(chunk))
                        for j, size in enumerate(chunk):
                            qty = row[size]
                            thresholds = latex_thresholds[size]
                            
                            indicator = "🔴" if qty <= thresholds["low"] else "🟠" if qty <= thresholds["medium"] else "🟢"
                                
                            with cols[j]:
                                new_qty = st.number_input(
                                    f"{indicator} {size}",
                                    min_value=0,
                                    value=int(qty),
                                    step=1,
                                    key=f"m_qty_l_{row['id']}_{size}"
                                )
                                if new_qty != qty:
                                    if new_qty < qty:
                                        current_month_str = datetime.now().strftime("%Y-%m")
                                        usage_dict = df.at[index, 'monthly_usage']
                                        usage_dict[current_month_str] = usage_dict.get(current_month_str, 0) + (qty - new_qty)
                                    df.at[index, size] = new_qty
                                    save_data(df)
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
                            qty = row[field]
                            color_alert = "red" if qty == 0 else "green"
                            cols[i].markdown(f"**{label}**")
                            cols[i].markdown(f":{color_alert}[**{qty}**]")
                            
                            if cols[i].button("➖", key=f"d_f_sub_{row['id']}_{field}"):
                                if qty > 0:
                                    df.at[index, field] = qty - 1
                                    # Update monthly usage
                                    current_month_str = datetime.now().strftime("%Y-%m")
                                    usage_dict = df.at[index, 'monthly_usage']
                                    usage_dict[current_month_str] = usage_dict.get(current_month_str, 0) + 1
                                    save_data(df)
                                    st.rerun()
                            if cols[i].button("➕", key=f"d_f_add_{row['id']}_{field}"):
                                df.at[index, field] = qty + 1
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
                        qty = row[field]
                        indicator = "🔴" if qty == 0 else "🟢"
                        
                        with cols[j]:
                            new_qty = st.number_input(
                                f"{indicator} {label}",
                                min_value=0,
                                value=int(qty),
                                step=1,
                                key=f"m_qty_f_{row['id']}_{field}"
                            )
                            if new_qty != qty:
                                if new_qty < qty:
                                    current_month_str = datetime.now().strftime("%Y-%m")
                                    usage_dict = df.at[index, 'monthly_usage']
                                    usage_dict[current_month_str] = usage_dict.get(current_month_str, 0) + (qty - new_qty)
                                df.at[index, field] = new_qty
                                save_data(df)
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
                "5in": 0, "11in": 0, "17in": 0, "24in": 0, "32in": 0, # Latex fields
                "small": 0, "large": 0, # Foil fields
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
