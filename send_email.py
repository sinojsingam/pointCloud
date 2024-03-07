from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

try:
    import sendGridCreds
    api_key = sendGridCreds.SENDGRID_API_KEY
    """
    #Add functionality for CLI to change to mail
    #by default mail will be sent to sinoj 
    """
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
            print(f'Email sent to {to_email}. Status code: {response.status_code}')
        except Exception as e:
            print('API import worked but, something else went wrong')
            print(e.message)
except:
    print('API key does not exist')
    



