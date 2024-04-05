from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

try:
    import sendGridCreds

    api_key = sendGridCreds.SENDGRID_API_KEY
    to_email = sendGridCreds.email
    from_email = 'sinojsingam1@gmail.com'
    def sendNotification(msg):
        message = Mail(
            from_email=from_email,
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
    def sendUpdate(update_msg):
        message = Mail(
            from_email=from_email,
            to_emails=to_email,
            subject='Update from script',
            html_content=f'{update_msg}')
        try:
            sg = SendGridAPIClient(api_key)
            response = sg.send(message)
            print(f'Update sent to {to_email}. Status code: {response.status_code}')
        except Exception as e:
            print('API import worked but, something else went wrong')
            print(e.message)
except:
    print('API key does not exist')
    



