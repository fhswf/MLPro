import control as ct 
import numpy as np
import matplotlib.pyplot as plt



K = 3
T= 4


num = np.array([K])
den = np.array([4,1])

# tranfer function
H = ct.tf(num,den)

#get Answer of tranfer function
t,y = ct.step_response(H)

plt.plot(t,y)
plt.title("Step Response")
plt.grid()
plt.show()

