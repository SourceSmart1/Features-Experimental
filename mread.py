import imaplib
import email
from email.header import decode_header
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# IMAP server settings and credentials
IMAP_USER = os.getenv("SMTP_USER")  # e.g., your_email@gmail.com
IMAP_PASS = os.getenv("SMTP_PASS")
IMAP_SERVER = "imap.gmail.com"
IMAP_PORT = 993  # IMAP over SSL

def fetch_emails_for_plus_alias():
    try:
        # Connect securely to the IMAP server
        mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
        # Log in to your account
        mail.login(IMAP_USER, IMAP_PASS)
        # Select the inbox
        mail.select("inbox")
        
        # Search for emails addressed to the plus alias using Gmail's IMAP extension.
        # The 'X-GM-RAW' search key lets you use Gmail search operators.
        status, messages = mail.search(None, 'X-GM-RAW', 'to:zoe+conversationreak@sourcesmart.ai')
        if status != 'OK':
            print("Error searching for emails.")
            return
        
        email_ids = messages[0].split()
        print(f"Found {len(email_ids)} emails addressed to zoe+conversationreak@sourcesmart.ai.")
        
        for email_id in email_ids:
            status, msg_data = mail.fetch(email_id, "(RFC822)")
            if status != 'OK':
                print("Failed to fetch email ID:", email_id)
                continue
            
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    # Parse the email message from bytes
                    msg = email.message_from_bytes(response_part[1])
                    
                    # Decode the subject header
                    subject_header = msg.get("Subject", "")
                    subject_parts = decode_header(subject_header)
                    subject = ""
                    for part, encoding in subject_parts:
                        if isinstance(part, bytes):
                            subject += part.decode(encoding if encoding else "utf-8")
                        else:
                            subject += part
                    
                    from_ = msg.get("From")
                    print("-----")
                    print("Subject:", subject)
                    print("From:", from_)
                    
                    # Extract the plain text body from the email
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            content_type = part.get_content_type()
                            content_disposition = str(part.get("Content-Disposition"))
                            if content_type == "text/plain" and "attachment" not in content_disposition:
                                try:
                                    body = part.get_payload(decode=True).decode()
                                    break  # Stop at the first plain text part
                                except Exception as e:
                                    print("Error decoding part:", e)
                    else:
                        try:
                            body = msg.get_payload(decode=True).decode()
                        except Exception as e:
                            print("Error decoding body:", e)
                    print("Body:", body)
            
            # Optionally, mark the email as seen so it won't be processed again
            mail.store(email_id, '+FLAGS', '\\Seen')
        
        mail.logout()
    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    fetch_emails_for_plus_alias()