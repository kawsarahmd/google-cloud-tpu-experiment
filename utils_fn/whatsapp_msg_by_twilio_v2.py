import os
from twilio.rest import Client
import argparse
import sys

def send_whatsapp_message(phone_no, body, account_sid=None, auth_token=None):
    
    account_sid = account_sid or os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = auth_token or os.getenv("TWILIO_AUTH_TOKEN")
    from_phone_no = 'whatsapp:+14155238886'
    
    if not account_sid or not auth_token:
        print("Error: Twilio account_sid and auth_token are required. Provide them as arguments or set them as environment variables.")
        sys.exit(1)
    
    client = Client(account_sid, auth_token)

    try:
        message = client.messages.create(
            from_= from_phone_no, 
            body= body,
            to=f'whatsapp:{phone_no}'
        )
        print(f"Message sent successfully. SID: {message.sid}")
    except Exception as e:
        print(f"Failed to send message: {e}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Send WhatsApp messages via Twilio.")
    parser.add_argument("--phone_no", required=True, help="Recipient phone number in international format (e.g., +8801748717173).")
    parser.add_argument("--message_body", required=True, help="Message body to send.")
    parser.add_argument("--account_sid", help="Twilio Account SID. Optional if set in environment variables.")
    parser.add_argument("--auth_token", help="Twilio Auth Token. Optional if set in environment variables.")
    
    args = parser.parse_args()
    
    send_whatsapp_message(
        phone_no=args.phone_no,
        body=args.message_body,
        account_sid=args.account_sid,
        auth_token=args.auth_token
    )


## Sample Command
# python ~/whatsapp_msg_by_twilio_v2.py \
#   --phone_no +8801748717173 \
#   --message_body "Hi there" \
#   --account_sid "" \
#   --auth_token ""