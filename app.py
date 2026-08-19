import os
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
from sqlalchemy import text

from model import get_response
from db import db, User, ChatHistory, init_db

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "yojanagpt-super-secret-key-change-in-production")

# Database Configuration
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///yojanagpt.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize Database
init_db(app)

def get_firebase_config():
    """Retrieve Firebase client config securely from environment variables"""
    return {
        "apiKey": os.environ.get("FIREBASE_API_KEY", ""),
        "authDomain": os.environ.get("FIREBASE_AUTH_DOMAIN", ""),
        "projectId": os.environ.get("FIREBASE_PROJECT_ID", ""),
        "storageBucket": os.environ.get("FIREBASE_STORAGE_BUCKET", ""),
        "messagingSenderId": os.environ.get("FIREBASE_MESSAGING_SENDER_ID", ""),
        "appId": os.environ.get("FIREBASE_APP_ID", ""),
        "measurementId": os.environ.get("FIREBASE_MEASUREMENT_ID", "")
    }

@app.route("/")
def home():
    """Render index page with dynamic Firebase configuration"""
    return render_template("index.html", firebase_config=get_firebase_config())

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint validating server and database connection"""
    db_status = "healthy"
    try:
        db.session.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"error: {str(e)}"
        
    return jsonify({
        "status": "healthy" if db_status == "healthy" else "degraded",
        "service": "YojanaGPT Flask API",
        "database": db_status
    }), 200

@app.route("/signup", methods=["POST"])
def signup():
    """Handle user registration and store user in SQLite database"""
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    password = data.get("password", "").strip()
    
    if not name or not email or not password:
        return jsonify({"success": False, "message": "All fields are required"}), 400
        
    if len(password) < 6:
        return jsonify({"success": False, "message": "Password must be at least 6 characters"}), 400

    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        return jsonify({"success": False, "message": "Email address already registered"}), 409

    try:
        new_user = User(name=name, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        
        session["user_id"] = new_user.id
        session["user_email"] = new_user.email
        session["user_name"] = new_user.name
        
        return jsonify({
            "success": True,
            "message": "User registered successfully",
            "user": new_user.to_dict()
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": f"Database error: {str(e)}"}), 500

@app.route("/login", methods=["POST"])
def login():
    """Handle user login against SQLite database"""
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip().lower()
    password = data.get("password", "").strip()
    
    if not email or not password:
        return jsonify({"success": False, "message": "Email and password are required"}), 400
        
    user = User.query.filter_by(email=email).first()
    if user and user.check_password(password):
        session["user_id"] = user.id
        session["user_email"] = user.email
        session["user_name"] = user.name
        return jsonify({
            "success": True,
            "message": "Login successful",
            "user": user.to_dict()
        })
    else:
        return jsonify({"success": False, "message": "Invalid email or password"}), 401

@app.route("/logout", methods=["POST"])
def logout():
    """Handle user logout"""
    session.clear()
    return jsonify({"success": True, "message": "Logged out successfully"})

@app.route("/ask", methods=["POST"])
def ask():
    """Handle chat messages and persist conversation history"""
    data = request.get_json(silent=True) or {}
    user_msg = data.get("message", "").strip()
    
    if not user_msg:
        return jsonify({"success": False, "reply": "Please provide a message"}), 400
    
    bot_reply = get_response(user_msg)
    
    # Save conversation to database if user session exists
    user_id = session.get("user_id")
    try:
        chat_entry = ChatHistory(
            user_id=user_id,
            user_message=user_msg,
            bot_reply=bot_reply
        )
        db.session.add(chat_entry)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print("Failed to save chat history:", e)
        
    return jsonify({
        "success": True,
        "reply": bot_reply
    })

@app.route("/history", methods=["GET"])
def history():
    """Retrieve chat history for the currently authenticated user"""
    user_id = session.get("user_id")
    if not user_id:
        return jsonify({"success": False, "message": "User not authenticated"}), 401
        
    chats = ChatHistory.query.filter_by(user_id=user_id).order_by(ChatHistory.timestamp.asc()).all()
    return jsonify({
        "success": True,
        "history": [chat.to_dict() for chat in chats]
    })

@app.errorhandler(404)
def not_found(e):
    if request.path.startswith("/ask") or request.path.startswith("/api") or request.is_json:
        return jsonify({"success": False, "error": "Endpoint not found"}), 404
    return render_template("index.html", firebase_config=get_firebase_config()), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"success": False, "error": "Internal server error"}), 500

# ------------------ RUN SERVER ------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    debug = os.environ.get("FLASK_DEBUG", "True").lower() in ["true", "1", "t"]
    app.run(host="0.0.0.0", port=port, debug=debug)
