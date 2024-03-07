import send_email
import sys


foo = [1,2,3,4,5,6]
print_message = 'This email is a test mail to see if the API is functioning.'
try:
    if len(sys.argv) >1:
        if sys.argv[1]=='mailme':
            send_email.sendNotification(f'Process finished. {print_message}')
except:
    print("mail was not send, due to API key error")
print('hello world')

def sq(x):
    return x**2

print(list(map(sq,foo)))