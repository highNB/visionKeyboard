#!/usr/bin/env python3
import serial
import time

keyset = [ b'a',b's', b'd',b'f']

a = 0
if __name__ == '__main__':
    ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
    ser.close()
    ser.open()
    ser.flush()

    while True:
        ser.write(keyset[a%4])            
        #line = ser.readline().decode('utf-8').rstrip()
        print(a)
        a = a+1
        time.sleep(1)