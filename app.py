import os
from flask import Flask, render_template, request, jsonify, session
from dotenv import load_dotenv
from model import get_response

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "your-secret-key-change-in-production")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "YojanaGPT Flask API"
    }), 200

@app.route("/login", methods=["POST"])
def login():
    """Handle user login"""
    data = request.get_json(silent=True) or {}
    email = data.get("email", "").strip()
    password = data.get("password", "").strip()
    
    # Simple validation - accept any non-empty email and password
    if email and password:
        session["user_email"] = email
        return jsonify({"success": True, "message": "Login successful"})
    else:
        return jsonify({"success": False, "message": "Invalid credentials"}), 401

@app.route("/logout", methods=["POST"])
def logout():
    """Handle user logout"""
    session.clear()
    return jsonify({"success": True, "message": "Logged out successfully"})

@app.route("/ask", methods=["POST"])
def ask():
    """Handle chat messages"""
    data = request.get_json(silent=True) or {}
    user_msg = data.get("message", "").strip()
    
    if not user_msg:
        return jsonify({"success": False, "reply": "Please provide a message"}), 400
    
    bot_reply = get_response(user_msg)
    return jsonify({"success": True, "reply": bot_reply})

@app.errorhandler(404)
def not_found(e):
    if request.path.startswith("/ask") or request.is_json:
        return jsonify({"success": False, "error": "Endpoint not found"}), 404
    return render_template("index.html"), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"success": False, "error": "Internal server error"}), 500

# ------------------ RUN SERVER ------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    debug = os.environ.get("FLASK_DEBUG", "True").lower() in ["true", "1", "t"]
    app.run(host="0.0.0.0", port=port, debug=debug)
