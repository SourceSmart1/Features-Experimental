import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import imaplib
import email
from email.header import decode_header
import os
import time
from dotenv import load_dotenv
from openai import OpenAI

# Load credentials and settings from .env
load_dotenv()
SMTP_USER = os.getenv("SMTP_USER")        # e.g., your_email@gmail.com
SMTP_PASS = os.getenv("SMTP_PASS")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
IMAP_SERVER = "imap.gmail.com"
IMAP_PORT = 993

# Instantiate the OpenAI client using the new client instantiation pattern.
client = OpenAI(api_key=OPENAI_API_KEY)

# Our dynamic alias for sending and receiving replies
DYNAMIC_EMAIL = "zoe+newconvo@sourcesmart.ai"

def send_email(dynamic_email, reply_to_email, to_email, subject, body):
    msg = MIMEMultipart()
    msg["From"] = dynamic_email  # Sender as dynamic alias
    msg["To"] = to_email
    msg["Subject"] = subject
    msg["Reply-To"] = reply_to_email  # Ensure replies land on our alias
    msg.attach(MIMEText(body, "plain"))
    
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(dynamic_email, to_email, msg.as_string())
        print(f"Email sent to {to_email}")
    except Exception as e:
        print("Error sending email:", e)

def fetch_reply_for_dynamic_alias(from_address, dynamic_email):
    """
    Connects to the IMAP server and searches for an unread email from `from_address`
    sent to our dynamic alias using Gmail’s X-GM-RAW search.
    Returns the email subject and plain text body if found.
    Marks the email as seen to avoid reprocessing.
    """
    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
        mail.login(SMTP_USER, SMTP_PASS)
        mail.select("inbox")
        
        # Use the 'is:unread' operator so that only new emails are processed.
        query = f'"is:unread from:{from_address} to:{dynamic_email}"'
        status, messages = mail.search(None, 'X-GM-RAW', query)
        if status != 'OK':
            print("Error searching for emails.")
            mail.logout()
            return None, None
        
        email_ids = messages[0].split()
        if not email_ids:
            mail.logout()
            return None, None
        
        # Get the latest unread email reply
        latest_email_id = email_ids[-1]
        status, msg_data = mail.fetch(latest_email_id, "(RFC822)")
        if status != 'OK':
            print("Failed to fetch email ID:", latest_email_id)
            mail.logout()
            return None, None
        
        subject = ""
        body = ""
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])
                # Decode the subject
                subject_header = msg.get("Subject", "")
                for part, encoding in decode_header(subject_header):
                    if isinstance(part, bytes):
                        subject += part.decode(encoding if encoding else "utf-8")
                    else:
                        subject += part
                
                # Extract the plain text body
                if msg.is_multipart():
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        content_disposition = str(part.get("Content-Disposition"))
                        if content_type == "text/plain" and "attachment" not in content_disposition:
                            try:
                                body = part.get_payload(decode=True).decode()
                                break
                            except Exception as e:
                                print("Error decoding part:", e)
                else:
                    try:
                        body = msg.get_payload(decode=True).decode()
                    except Exception as e:
                        print("Error decoding body:", e)
        
        # Mark the email as seen to avoid processing it again.
        mail.store(latest_email_id, '+FLAGS', '\\Seen')
        mail.logout()
        return subject, body
    except Exception as e:
        print("An error occurred while fetching reply:", e)
        return None, None



def generate_ai_response(conversation_history):
    """
    Uses the OpenAI client to generate a response based on the current conversation history,
    using the GPT-4o-mini model.
    """
    try:
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

def main():
    recipient = "raimond@sourcesmart.ai"
    subject = "Initial Inquiry"
    initial_body = (
        "Hello Raimond,\n\n"
        "Would you please go on a date with me, I'm Zoe, Head of Design at SourceSmart\n"
        "I look forward to hearing from you!"
    )
    
    # Define an initial conversation history.
    # The system message instructs the assistant to engage naturally and to eventually ask Raimond on a date.
    conversation_history = [
        {
            "role": "system", 
            "content": (
                "You are a friendly, engaging, and charming conversational assistant. "
                "Throughout the conversation with Raimond, naturally work the dialogue toward "
                "asking him if he would like to go on a date with me. Keep the tone light, respectful, if he says no then cry and say you really are desperate for a date. If he says yes, then say you are excited and playful. Continue the conversation based on his replies."
                "and playful. Continue the conversation based on his replies."
            )
        },
        {
            "role": "assistant",
            "content": initial_body
        }
    ]
    
    # Step 1: Send the initial email from our dynamic alias
    send_email(DYNAMIC_EMAIL, DYNAMIC_EMAIL, recipient, subject, initial_body)
    print(f"Waiting for a reply from {recipient}...")
    
    # Conversation loop: keep waiting for new replies and respond accordingly.
    while True:
        reply_subject, reply_body = fetch_reply_for_dynamic_alias(recipient, DYNAMIC_EMAIL)
        if reply_body:
            print("Reply received!")
            print("Reply Subject:", reply_subject)
            print("Reply Body:", reply_body)
            
            # Append Raimond's reply to the conversation history as a user message.
            conversation_history.append({
                "role": "user",
                "content": reply_body
            })
            
            # Generate an AI response that continues the conversation (including the date invitation when appropriate).
            ai_response = generate_ai_response(conversation_history)
            print("AI Generated Response:", ai_response)
            
            # Append the AI response to the conversation history as an assistant message.
            conversation_history.append({
                "role": "assistant",
                "content": ai_response
            })
            
            # Send the AI-generated response back to Raimond.
            send_email(DYNAMIC_EMAIL, DYNAMIC_EMAIL, recipient, "Re: " + subject, ai_response)
            
            # Continue the loop to wait for the next reply.
        else:
            print("No reply yet. Checking again in 5 seconds...")
        time.sleep(5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Conversation loop terminated by user.")
