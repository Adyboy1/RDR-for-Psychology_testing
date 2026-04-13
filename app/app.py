#app.py
import streamlit as st
import os
import json
import sys
import pickle
import graphviz
import textwrap
import docx

# --- PATH CONFIGURATION ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

# Define storage paths locally for UI access
STORAGE_DIR = os.path.join(CURRENT_DIR, '..', 'storage')
TREE_STORAGE_FILE = os.path.join(STORAGE_DIR, "rdr_tree_summary.pkl")
LOG_FILE = os.path.join(STORAGE_DIR, "rdr_event_log.csv")

if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

import backend
    
from rdr_engine import RDREngine, Node, Rule, Vertex
from llm_api import (
    llm_check_condition, 
    llm_get_differentiating_conditions, 
    llm_generate_summary, 
    llm_merge_summaries,
    llm_verify_manual_condition 
)

# --- PAGE SETUP ---
st.set_page_config(page_title="RDR Clinical Engine", layout="wide")

# --- LOAD ORGANIZATION ID ---
try:
    # This reads from the [general] section in secrets.toml
    ORG_ID = st.secrets["general"]["app_id"]
except Exception:
    ORG_ID = "default_rdr_app" # Fallback for local dev if secret missing

st.title(f"🧠 Clinical RDR Knowledge Base ({ORG_ID})")

# --- PERSISTENCE FUNCTIONS ---

def sync_from_cloud():
    """Attempts to pull latest state from Cloud to Local."""
    if 'cloud_synced' not in st.session_state:
        # 1. Load Engine (PASS ORG_ID)
        cloud_engine = backend.load_engine_from_cloud(app_id=ORG_ID)
        if cloud_engine:
            # Update Session State
            st.session_state.engine = cloud_engine
            # Update Local File (for other scripts)
            with open(TREE_STORAGE_FILE, "wb") as f:
                pickle.dump(cloud_engine, f)
            st.toast(f"✅ Knowledge Base synced from Cloud ({ORG_ID})")
        
        # 2. Load Logs (PASS ORG_ID)
        cloud_logs = backend.load_logs_from_cloud(app_id=ORG_ID)
        if cloud_logs:
            with open(LOG_FILE, "w", encoding='utf-8') as f:
                f.write(cloud_logs)
            st.toast("✅ Event Logs synced from Cloud")
            
        st.session_state.cloud_synced = True

def sync_to_cloud():
    """Pushes current local state to Cloud."""
    # 1. Push Engine (PASS ORG_ID)
    if 'engine' in st.session_state:
        backend.save_engine_to_cloud(st.session_state.engine, app_id=ORG_ID)
    
    # 2. Push Logs (PASS ORG_ID)
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding='utf-8') as f:
            csv_content = f.read()
            backend.save_logs_to_cloud(csv_content, app_id=ORG_ID)
    
    st.toast("☁️ State persisted to Cloud", icon="💾")

# --- SESSION STATE INITIALIZATION ---

# 1. First Run: Sync from Cloud
sync_from_cloud()

# 2. Load/Init Engine (Local Fallback)
if 'engine' not in st.session_state:
    if os.path.exists(TREE_STORAGE_FILE):
        try:
            with open(TREE_STORAGE_FILE, "rb") as f:
                st.session_state.engine = pickle.load(f)
        except:
            st.session_state.engine = RDREngine()
    else:
        st.session_state.engine = RDREngine()

if not hasattr(st.session_state.engine, 'history'):
    st.session_state.engine.history = []

if 'current_summary' not in st.session_state:
    st.session_state.current_summary = None
if 'interpretation_result' not in st.session_state:
    st.session_state.interpretation_result = None
if 'diff_conditions' not in st.session_state:
    st.session_state.diff_conditions = []
if 'current_source_file' not in st.session_state:
    st.session_state.current_source_file = "Unknown"

# --- VISUALIZATION LOGIC ---
def build_graph(root):
    dot = graphviz.Digraph()
    dot.attr(rankdir='TB')        
    dot.attr(nodesep='0.5')       
    dot.attr(ranksep='1.0')       
    dot.attr('node', shape='Mrecord', style='filled', fillcolor='#F0F2F6', 
             fontname='Arial', fontsize='10', margin='0.1')
    
    if not root:
        dot.node("empty", "Empty Tree", style='dashed')
        return dot

    def format_label(node):
        try:
            conds = json.loads(node.vertex.rule.conditions)
            if isinstance(conds, list):
                cond_list = [f"• {c}" for c in conds]
                cond_text = "\n".join(cond_list)
            else:
                cond_text = str(conds)
        except:
            cond_text = node.vertex.rule.conditions
            
        concl_text = node.vertex.rule.conclusions
        wrapper = textwrap.TextWrapper(width=25, break_long_words=False, replace_whitespace=False)
        
        wrapped_cond_lines = wrapper.wrap(cond_text)
        final_cond = "\\n".join(wrapped_cond_lines)
        
        wrapped_concl_lines = wrapper.wrap(concl_text)
        final_concl = "\\n".join(wrapped_concl_lines)
        return f"{{ IF:\\n{final_cond} | THEN:\\n{final_concl} }}"

    def add_nodes_recursive(node, node_id):
        label = format_label(node)
        dot.node(node_id, label)
        
        if node.left:
            left_id = f"{node_id}L"
            add_nodes_recursive(node.left, left_id)
            dot.edge(node_id, left_id, label=" False", color="#D32F2F", fontcolor="#D32F2F")
            
        if node.right:
            right_id = f"{node_id}R"
            add_nodes_recursive(node.right, right_id)
            dot.edge(node_id, right_id, label=" True", color="#388E3C", fontcolor="#388E3C")

    add_nodes_recursive(root, "root")
    return dot

# --- SIDEBAR ---
with st.sidebar:
    st.header("Tree Controls")
    
    # 1. UNDO FUNCTION
    if st.button("↩️ Undo Last Rule"):
        success, msg = st.session_state.engine.undo_last_addition()
        if success:
            with open(TREE_STORAGE_FILE, "wb") as f:
                pickle.dump(st.session_state.engine, f)
            
            sync_to_cloud() # <-- PUSH TO CLOUD
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)
            
    # 2. FLUSH FUNCTION
    if "confirm_flush" not in st.session_state:
        st.session_state.confirm_flush = False

    if st.button("🗑️ Flush Entire Tree"):
        st.session_state.confirm_flush = True

    if st.session_state.confirm_flush:
        st.error(f"⚠️ **CONFIRM DELETION: {ORG_ID}** ⚠️")
        st.write("This will permanently delete this organization's knowledge base.")
        
        col_conf1, col_conf2 = st.columns(2)
        with col_conf1:
            if st.button("✅ YES, DELETE"):
                st.session_state.engine.flush_tree()
                
                # Delete Cloud Artifacts (Manually calling init to delete specific docs)
                try:
                    db = backend.init_firebase()
                    db.collection("artifacts").document(ORG_ID).collection("public").document("tree_data").delete()
                    db.collection("artifacts").document(ORG_ID).collection("public").document("event_logs").delete()
                except:
                    pass
                
                st.session_state.confirm_flush = False
                st.success("Tree flushed and file deleted.")
                st.rerun()
        with col_conf2:
            if st.button("❌ NO, CANCEL"):
                st.session_state.confirm_flush = False
                st.rerun()
    
    cloud_log_content = backend.load_logs_from_cloud(app_id=ORG_ID)
     # 3. DOWNLOAD LOG FUNCTION
    st.markdown("---")
    st.header("Logs")
    # Step 1: Attempt to fetch from Cloud (Source of Truth)
    # We pass ORG_ID to ensure we get the correct organization's logs
    cloud_log_content = backend.load_logs_from_cloud(app_id=ORG_ID)
    
    # Step 2: Prepare download data
    download_data = None
    source_label = ""
    if cloud_log_content:
        download_data = cloud_log_content
        source_label = "(Cloud)"
    elif os.path.exists(LOG_FILE):
        # Fallback: Read local file if cloud is empty/unreachable
        with open(LOG_FILE, "r", encoding='utf-8') as f:
            download_data = f.read()
        source_label = "(Local Cache)"
        
    # Step 3: Render Button
    if download_data:
        import time
        timestamp = int(time.time())
        st.download_button(
            label=f"📥 Download Event Log {source_label}",
            data=download_data,
            file_name=f"{ORG_ID}_log_{timestamp}.csv",
            mime="text/csv"
        )
    else:
        st.caption("No logs available yet.")


    # 4. Render Tree logic
    st.markdown("---")
    # added to fix tree not being visible issue 
    st.header("Visualization")
    if st.button("Refresh Tree"):
        pass 
    if st.session_state.engine.root:
        graph = build_graph(st.session_state.engine.root)
        st.graphviz_chart(graph, use_container_width=True)
    else:
        st.info("Tree is empty. Process a case to start.")

# --- 1. MAIN CONTENT AREA ---
st.subheader("Input Transcript")
    
uploaded_file = st.file_uploader("Upload Patient Transcript (DOCX)", type=["docx"])
    
transcript_text = None
source_name = "Unknown"

if uploaded_file:
    source_name = uploaded_file.name
    try:
        doc = docx.Document(uploaded_file)
        # Combine all paragraphs into one string
        transcript_text = "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"Error reading file: {e}")

if st.button("Analyze Case") and transcript_text:
    st.session_state.current_source_file = source_name
        
    with st.spinner("Generating Clinical Summary..."):
        summary = llm_generate_summary(transcript_text)
        st.session_state.current_summary = summary
            
    with st.spinner("Interpreting Rules..."):
        n1, n2 = st.session_state.engine.interpret(summary)
        st.session_state.interpretation_result = (n1, n2)
        st.session_state.diff_conditions = [] 

# --- 2. RESULTS AREA ---
# This section only appears once a summary exists in session state
if st.session_state.get("current_summary"):
    st.divider()
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("### 📄 Clinical Summary")
        st.info(st.session_state.current_summary)
        
    with col_b:
        st.markdown("### 🤖 System Conclusion")
        # Ensure we have the result before unpacking
        if st.session_state.get("interpretation_result"):
            n1, n2 = st.session_state.interpretation_result
            
            if n2:
                st.success(f"**{n2.vertex.rule.conclusions}**")
                st.markdown("#### 🔍 Matching Conditions:")
              # Logic to handle both string-encoded JSON or direct lists
                import json
                raw_conds = n2.vertex.rule.conditions
                
                try:
                    # Try to parse if it's a JSON string
                    conditions = json.loads(raw_conds)
                except (ValueError, TypeError):
                    # Fallback if it's already a list or a plain string
                    conditions = raw_conds if isinstance(raw_conds, list) else [raw_conds]
                    
                # Render the list
                if conditions:
                    for c in conditions:
                        st.write(f"• {c}")
                else:
                    st.caption("No specific conditions listed for this node.")
            else:
                st.warning("**No Conclusion (Empty Tree or Fallback)**")
        
        st.markdown("#### Clinician Review")
        
        if st.button("✅ Agree (Correct)"):
            if n2:
                with st.spinner("Merging profile..."):
                    updated = llm_merge_summaries(n2.vertex.summary, st.session_state.current_summary)
                    n2.vertex.summary = updated
                    
                    st.session_state.engine.log_merge(
                        st.session_state.current_source_file,
                        n1, 
                        n2
                    )
                    
                    with open(TREE_STORAGE_FILE, "wb") as f:
                        pickle.dump(st.session_state.engine, f)
                    
                    sync_to_cloud() # <-- PUSH TO CLOUD
                    st.success("Knowledge base updated (Profile Merged)!")
                    st.rerun()

        if st.button("❌ Disagree (Revise)"):
            st.session_state.show_revision_form = True

# --- 3. REVISION AREA ---
# This is also in the main flow, appearing only if 'Disagree' was clicked
if st.session_state.get("show_revision_form"):
    st.divider()
    st.subheader("🔧 Knowledge Acquisition (Revise)")
    
    new_conclusion = st.text_input("What is the CORRECT conclusion?")
    
    if st.button("🔍 Find Differences (Ask AI)"):
        n1, n2 = st.session_state.interpretation_result
        ref_sum = n2.vertex.summary if n2 else ""
        with st.spinner("Comparing cases..."):
            conds = llm_get_differentiating_conditions(st.session_state.current_summary, ref_sum)
            st.session_state.diff_conditions = conds
    
    final_conditions = []
    if st.session_state.get("diff_conditions"):
        st.write("Select distinguishing conditions:")
        for cond in st.session_state.diff_conditions:
            if st.checkbox(cond, key=cond):
                final_conditions.append(cond)


    # adding multiple manual conditions
    manual_conds_text = st.text_area(
        "Or enter manual conditions (one per line):",
        height=120,
        placeholder="Example:\nPatient reports suicidal ideation\nSleep disturbance present"
    )

    manual_conditions = []
    if manual_conds_text.strip():
        manual_conditions = [
            c.strip() for c in manual_conds_text.split("\n") if c.strip()
        ]
        manual_conditions = list(dict.fromkeys(manual_conditions))
        final_conditions.extend(manual_conditions)
        
    if st.button("💾 Save New Rule"):
        if not new_conclusion or not final_conditions:
            st.error("Please provide a conclusion and at least one condition.")
        else:
            n1, n2 = st.session_state.interpretation_result
            ref_sum = n2.vertex.summary if n2 else ""
            
            condition_type = "AI_GENERATED"
            if manual_conditions:
                condition_type = "MANUAL"
                with st.spinner("Verifying Manual Conditions..."):
                    #Adding a loop to check all manual conditions for consistency
                    for cond in manual_conditions:
                        valid_new = llm_verify_manual_condition(st.session_state.current_summary, cond)
                        if not valid_new:
                            st.error(f"⚠️ Error: The condition '*{cond}*' is **FALSE** for the current patient.")
                            st.stop()

                        if ref_sum:
                            valid_ref = llm_verify_manual_condition(ref_sum, cond)
                            if valid_ref:
                                st.error(f"⚠️ **Differentiation Failed**: The condition '*{cond}*' is ALSO TRUE for the Reference Case.")
                                st.stop()
            
            import json
            cond_json = json.dumps(final_conditions)
            new_rule = Rule(cond_json, new_conclusion)
            new_vertex = Vertex(new_rule, st.session_state.current_summary)
            new_node = Node(new_vertex)
            
            source_file = st.session_state.current_source_file
            
            if st.session_state.engine.root is None:
                st.session_state.engine.add_node_to_tree(None, new_node, "ROOT", source_file, n1, n2, condition_type)
            elif n1 == n2:
                st.session_state.engine.add_node_to_tree(n1, new_node, "RIGHT", source_file, n1, n2, condition_type)
            else:
                st.session_state.engine.add_node_to_tree(n1, new_node, "LEFT", source_file, n1, n2, condition_type)
                
            with open(TREE_STORAGE_FILE, "wb") as f:
                pickle.dump(st.session_state.engine, f)
            
            sync_to_cloud() # <-- PUSH TO CLOUD
            
            st.success("New rule added successfully!")
            st.session_state.show_revision_form = False
            st.rerun()