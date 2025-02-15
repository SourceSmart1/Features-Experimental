import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import time
import threading
import queue
from dotenv import load_dotenv
from openai import OpenAI
from flask import Flask, request, jsonify

# ========================
# Load environment variables
# ========================
load_dotenv()

# Mailgun SMTP credentials (for sending)
SMTP_USER = os.getenv("SMTP_USER")          # e.g., "brad@mail.sourcesmart.ai"
SMTP_PASS = os.getenv("SMTP_PASS")          # Your Mailgun sending API key
SMTP_SERVER = os.getenv("SMTP_SERVER")      # e.g., "smtp.eu.mailgun.org"
SMTP_PORT = int(os.getenv("SMTP_PORT", 587)) # Typically 587 (TLS) or 465 (SSL)

# OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Dynamic alias settings (using Mailgun domain)
FROM_MAIL = "zoe+conv@mail.sourcesmart.ai"
REPLY_TO_MAIL = "zoe+conv@mail.sourcesmart.ai"

# Recipient email (this can be any email address; adjust as needed)
RECIPIENT = "nisar@sourcesmart.ai"

# ========================
# Set up the Flask app for inbound emails via Mailgun
# ========================
app = Flask(__name__)

# A thread-safe queue to store inbound emails received via webhook
inbound_email_queue = queue.Queue()

@app.route('/inbound', methods=['POST'])
def inbound_email():
    """
    This endpoint receives incoming emails from Mailgun.
    Mailgun will POST form data with keys like 'sender', 'recipient', 'subject', 'body-plain', etc.
    """
    sender = request.form.get('sender')
    recipient = request.form.get('recipient')
    subject = request.form.get('subject')
    body = request.form.get('body-plain')
    
    print(f"[Webhook] Received email from: {sender} with subject: {subject}")
    
    # Package the email data and put it in the queue
    email_data = {
        "sender": sender,
        "recipient": recipient,
        "subject": subject,
        "body": body
    }
    inbound_email_queue.put(email_data)
    return jsonify({"status": "success"}), 200

def run_flask():
    """Run the Flask app (webhook) on port 5000."""
    app.run(host="0.0.0.0", port=5000)

# ========================
# Email sending function using Mailgun SMTP
# ========================
def send_email(dynamic_email, reply_to_email, to_email, subject, body):
    msg = MIMEMultipart()
    msg["From"] = dynamic_email      # The email address that recipients will see
    msg["To"] = to_email
    msg["Subject"] = subject
    msg["Reply-To"] = reply_to_email # Ensures replies go to our dynamic alias
    msg.attach(MIMEText(body, "plain"))
    
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(dynamic_email, to_email, msg.as_string())
        print(f"[Send] Email sent to {to_email}")
    except Exception as e:
        print("[Send] Error sending email:", e)

# ========================
# OpenAI response generator
# ========================
def generate_ai_response(conversation_history):
    """
    Generate an AI response based on the conversation history using OpenAI.
    """
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation_history,
            max_tokens=150,
            temperature=0.7
        )
        ai_reply = completion.choices[0].message.content.strip()
        return ai_reply
    except Exception as e:
        print("Error generating AI response:", e)
        return "I'm sorry, I couldn't generate a response."

# ========================
# Main conversation loop
# ========================
def main():
    subject = "Initial Inquiry"
    initial_body = (
        "Subject: Request for Quotation for Aluminum Materials\n\n"
        "Dear Raimond Metal Supplier Inc.,\n\n"
        "I hope this message finds you well. We are currently evaluating new suppliers for high-quality aluminum materials and would appreciate it if you could provide us with a detailed quotation. Specifically, we are interested in the following specifications:\n\n"
        " - Aluminum Alloy Type: 5000 series\n"
        " - Form: Sheets & Rolls\n"
        " - Dimensions: 5*12h\n"
        " - Quantities: 400kgs\n\n"
        "Kindly include pricing, delivery timelines, and any volume discounts or terms that may apply. If you require additional details to prepare your quotation, please feel free to contact me directly.\n\n"
        "Thank you in advance for your prompt attention to this inquiry. I look forward to your response.\n\n"
        "Best regards,\n"
        "Zoe\n"
        "Procurement Assistant\n"
        "SourceSmart"
    )
    
    # Initialize the conversation history
    conversation_history = [
        {
            "role": "system",
            "content": (
                "You are a professional procurement assistant. You excel at negotiating deals and obtaining the best prices for your clients. "
                "Your role is to request quotes, negotiate effectively, and secure deals that offer outstanding value. "
                "Approach every negotiation with a focus on detail, cost-efficiency, and clear communication to ensure the best outcomes."
            )
        },
        {
            "role": "assistant",
            "content": initial_body
        }
    ]
    
    # Send the initial email via Mailgun
    send_email(FROM_MAIL, REPLY_TO_MAIL, RECIPIENT, subject, initial_body)
    print(f"[Main] Waiting for a reply from {RECIPIENT}...")
    
    # Conversation loop: poll for new inbound emails every 5 seconds
    while True:
        if not inbound_email_queue.empty():
            email_data = inbound_email_queue.get()
            reply_subject = email_data.get("subject", "")
            reply_body = email_data.get("body", "")
            
            print("[Main] Reply received!")
            print("[Main] Reply Subject:", reply_subject)
            print("[Main] Reply Body:", reply_body)
            
            # Append the incoming reply to the conversation history
            conversation_history.append({
                "role": "user",
                "content": reply_body
            })
            
            # Generate an AI response based on the conversation history
            ai_response = generate_ai_response(conversation_history)
            print("[Main] AI Generated Response:", ai_response)
            
            # Append the AI response to the conversation history
            conversation_history.append({
                "role": "assistant",
                "content": ai_response
            })
            
            # Send the AI-generated response back
            send_email(FROM_MAIL, REPLY_TO_MAIL, RECIPIENT, "Re: " + subject, ai_response)
        else:
            print("[Main] No reply yet. Checking again in 5 seconds...")
        time.sleep(5)

# ========================
# Entry Point
# ========================
if __name__ == "__main__":
    # Start the Flask webhook in a separate daemon thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    
    try:
        main()
    except KeyboardInterrupt:
        print("Conversation loop terminated by user.")
