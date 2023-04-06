import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('noisy_sine_wave', 'rb') as file:
    data_from_file = pickle.load(file)
"""
the above few lines makes an array called data_from_file which contains
a noisy sine wave as long as you downloaded the file "noisy_sine_wave" 
and put it in the same directory as this python file

pickle is a Python package which nicely saves data to files. it can be
a little tricky when you save lots of data, but this file only has one
object (an array) saved so it is pretty easy
"""

# print(data_from_file)

plt.plot(data_from_file)
xmax = 300
plt.xlim(0, xmax)
plt.xlabel('time t(s)')
plt.ylabel('y(t)')
plt.title('Pickle Example - Noisy Sine Wave\n (Position-Time)')
plt.show()

number = len(data_from_file)
message = "There are " + \
          str(number) + \
          " data points in total, only drawing the first " + \
          str(xmax)
print(message)

# calculate frequency and period of the wave
time = np.arange(number)
data = data_from_file[:300]
z2 = np.fft.fft(data_from_file)
# print(data)

freqss = np.fft.fftfreq(len(data_from_file), 1)

index1 = np.argmax(np.abs(z2))
index2 = np.argmax(np.abs(z2[:170]))
index3 = np.argmax(np.abs(z2[:140]))
print(index1, index2, index3)

freq1 = freqss[index1]
freq2 = freqss[index2]
freq3 = freqss[index3]

print(freq1, freq2, freq3)
freq = np.arange(number)  # frequency values, like time is the time values
width = 3  # width=2*sigma**2 where sigma is the standard deviation ,,,8
# ideal value is approximately N/T1 ,,,12.3
filter_function1 = (np.exp(-(freq - index1) ** 2 / width) + np.exp(
    -(freq + index1 - number) ** 2 / width))
filter_function2 = (np.exp(-(freq - index2) ** 2 / width) + np.exp(
    -(freq + index2 - number) ** 2 / width))
filter_function3 = (np.exp(-(freq - index3) ** 2 / width) + np.exp(
    -(freq + index3 - number) ** 2 / width))
z2_filtered = z2 * (filter_function1 + filter_function2 + filter_function3)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='col')
ax1.plot(np.abs(z2))
ax2.plot((filter_function1 + filter_function2 + filter_function3))
ax3.plot(np.abs(z2_filtered))

fig.subplots_adjust(hspace=0.2)
ax1.set_xlim(0, 300)
ax2.set_xlim(0, 300)
ax3.set_xlim(0, 300)
ax1.set_ylabel('Noisy FFT\n\n Y(w)')
ax2.set_ylabel('Filter Function\n\n Y(w)')
ax3.set_ylabel('Filtered FFT\n\n Y(w)')
ax3.set_xlabel('Frequency w (s^-1)')
ax1.set_title('Absolute value of FFT of Position-Time\n(Amplitude-Frequency)')
plt.show()

w1 = (index3 * 2 * np.pi) / number
w2 = (index2 * 2 * np.pi) / number
w3 = (index1 * 2 * np.pi) / number

T1 = 2 * np.pi / w1
T2 = 2 * np.pi / w2
T3 = 2 * np.pi / w3

A1 = 1
A2 = 2.136
A3 = 4.046

y1 = A1 * np.sin(2 * np.pi * time / T1)
y2 = A2 * np.sin(2 * np.pi * time / T2)
y3 = A3 * np.sin(2 * np.pi * time / T3)

y = 3 * (y1 + y2 + y3)

cleaned = np.fft.ifft(z2_filtered)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='col')

ax1.plot(data_from_file)
ax1.set_xlim(0, xmax)
ax2.plot(cleaned)
ax2.set_xlim(0, xmax)
ax3.plot(y - np.real(cleaned))

ax1.set_ylabel('Original Data\n\n y(t)')
ax2.set_ylabel('Filtered Data\n\n y(t)')
ax3.set_ylabel('Ideal Result\n\n y(t)')
ax3.set_xlabel('time t(s)')
ax1.set_title('Propagating Signal Wave\n (Position-Time)')
plt.show()

