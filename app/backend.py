import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st
import pickle
import base64
import os
import csv
import logging

# --- CONFIGURATION ---
# Secrets must be in .streamlit/secrets.toml
# [firebase]
# type = "service_account"
# project_id = "..."
# ...

def init_firebase():
    """Initializes Firebase Admin SDK."""
    if not firebase_admin._apps:
        try:
            # Check if secrets exist
            if "firebase" in st.secrets:
                key_dict = dict(st.secrets["firebase"])
                cred = credentials.Certificate(key_dict)
                firebase_admin.initialize_app(cred)
            else:
                logging.warning("Firebase secrets not found. Persistence disabled.")
                return None
        except Exception as e:
            logging.error(f"Firebase Init Error: {e}")
            return None
    return firestore.client()

# --- TREE PERSISTENCE (.pkl) ---

def save_engine_to_cloud(engine, app_id="default_rdr_app"):
    """
    Pickles engine and saves to Firestore under the specific app_id collection.
    Path: artifacts/{app_id}/public/tree_data
    """
    db = init_firebase()
    if not db: return False

    try:
        # Serialize
        bytes_data = pickle.dumps(engine)
        b64_string = base64.b64encode(bytes_data).decode('utf-8')
        
        # Save to specific App ID collection
        doc_ref = db.collection("artifacts").document(app_id).collection("public").document("tree_data")
        doc_ref.set({
            "blob": b64_string,
            "updated_at": firestore.SERVER_TIMESTAMP
        })
        return True
    except Exception as e:
        logging.error(f"Cloud Save Error: {e}")
        return False

def load_engine_from_cloud(app_id="default_rdr_app"):
    """Loads engine from Firestore for the specific app_id."""
    db = init_firebase()
    if not db: return None

    try:
        doc_ref = db.collection("artifacts").document(app_id).collection("public").document("tree_data")
        doc = doc_ref.get()
        
        if doc.exists:
            data = doc.to_dict()
            b64_string = data.get("blob")
            if b64_string:
                bytes_data = base64.b64decode(b64_string)
                return pickle.loads(bytes_data)
        return None
    except Exception as e:
        logging.error(f"Cloud Load Error: {e}")
        return None

# --- LOG PERSISTENCE (.csv) ---

def save_logs_to_cloud(csv_content, app_id="default_rdr_app"):
    """Saves CSV string to Firestore for the specific app_id."""
    db = init_firebase()
    if not db: return False

    try:
        doc_ref = db.collection("artifacts").document(app_id).collection("public").document("event_logs")
        doc_ref.set({
            "csv_data": csv_content,
            "updated_at": firestore.SERVER_TIMESTAMP
        })
        return True
    except Exception as e:
        logging.error(f"Log Cloud Save Error: {e}")
        return False

def load_logs_from_cloud(app_id="default_rdr_app"):
    """Returns CSV string from Firestore for the specific app_id."""
    db = init_firebase()
    if not db: return None

    try:
        doc_ref = db.collection("artifacts").document(app_id).collection("public").document("event_logs")
        doc = doc_ref.get()
        
        if doc.exists:
            return doc.to_dict().get("csv_data")
        return None
    except Exception as e:
        logging.error(f"Log Cloud Load Error: {e}")
        return None

def get_tree_file_bytes(app_id="default_rdr_app"):
    """Fetches raw bytes for download."""
    db = init_firebase()
    if not db: return None
    try:
        doc_ref = db.collection("artifacts").document(app_id).collection("public").document("tree_data")
        doc = doc_ref.get()
        if doc.exists:
            b64 = doc.to_dict().get("blob")
            if b64: return base64.b64decode(b64)
        return None
    except: return None