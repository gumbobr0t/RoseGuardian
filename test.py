import random, string
import base64

def hello():
   
    global var
    
    var = 'hii' # hi
    
    print(var)

print(random.choices(string.ascii_letters, k=5))

hello()

print(base64.b64encode(var.encode('utf-8')))

class banana(): # banana
    ss = b'\nPEWPEW'

    print(ss)

def counter(num):

    for i in range(num):

        print('dogs are cool!')

counter(5)

# lalala