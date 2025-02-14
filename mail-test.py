import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import os
load_dotenv()
email_user=os.getenv("SMTP_USER")
email_pass=os.getenv("SMTP_PASS")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = email_user
SMTP_PASS = email_pass

def send_email(dynamic_email, reply_to_email, to_email, subject, body):
    msg = MIMEMultipart()
    msg["From"] = dynamic_email  # Using dynamically generated alias
    msg["To"] = to_email
    msg["Subject"] = subject
    msg["Reply-To"] = reply_to_email  # Set the reply-to email

    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(dynamic_email, to_email, msg.as_string())

# Send an email using a plus-alias with a custom Reply-To address
send_email(
    dynamic_email="zoe+conversationreak@sourcesmart.ai", 
    reply_to_email="zoe+conversationreak@sourcesmart.ai",  # Replies will go here
    to_email="nisarvskp@gmail.com", 
    subject="Test Email with Reply-To", 
    body="This is a test email with a custom Reply-To."
)

