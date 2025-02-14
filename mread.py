import imaplib
import email
from email.header import decode_header
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# IMAP server settings and credentials
IMAP_USER = os.getenv("IMAP_USER")  # e.g., your_email@gmail.com
IMAP_PASS = os.getenv("IMAP_PASS")
IMAP_SERVER = "imap.gmail.com"
IMAP_PORT = 993  # IMAP over SSL

def fetch_and_process_replies():
    try:
        # Connect securely to the IMAP server
        mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
        # Log in to your account
        mail.login(IMAP_USER, IMAP_PASS)
        # Select the inbox (or another folder if desired)
        mail.select("inbox")
        
        # Search for unseen emails in the inbox
        status, messages = mail.search(None, 'UNSEEN')
        email_ids = messages[0].split()
        print(f"Found {len(email_ids)} unseen emails.")
        
        for email_id in email_ids:
            # Fetch the full RFC822 message for each email ID
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
                    
                    # Process only emails that appear to be replies (subject starts with "Re:")
                    if subject.strip().lower().startswith("re:"):
                        from_ = msg.get("From")
                        print("-----")
                        print("Reply Subject:", subject)
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
                                        break  # Stop at the first found plain text part
                                    except Exception as e:
                                        print("Error decoding body:", e)
                        else:
                            try:
                                body = msg.get_payload(decode=True).decode()
                            except Exception as e:
                                print("Error decoding body:", e)
                        print("Body:", body)
                        
                        # TODO: Here you can store email details into a database or file
                        
                    # Mark the email as seen so it won't be reprocessed next time
                    mail.store(email_id, '+FLAGS', '\\Seen')
        
        # Logout from the IMAP server
        mail.logout()
    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    fetch_and_process_replies()
