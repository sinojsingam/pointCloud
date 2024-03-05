import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ['SENDGRID_API_KEY']
to_email = 'sinoj.singam@gmail.com'
def sendNotification(msg):
    message = Mail(
        from_email='sinojsingam1@gmail.com',
        to_emails=to_email,
        subject='Script has finished',
        html_content=f'<strong>{msg}</strong>')
    try:
        sg = SendGridAPIClient(api_key)
        response = sg.send(message)
        print(response.status_code)
        print(f'Email sent to {to_email}.')
    except Exception as e:
        print(e.message)