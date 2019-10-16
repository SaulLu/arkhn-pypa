import os

if __name__ == '__main__':
    print('Hello world')

    # test to create a dummy file on the host filesystem
    data = 'done' if os.path.isdir('/data') else 'error'
    print(f"Recognising mounted directory : {data}")
    with  open('/data/test.txt', 'w') as fp:
        fp.write('Hello world')
