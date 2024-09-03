import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import random


from mlpro.bf.math.basics import Log
from mlpro.bf.mt import Async, Log, Task, Workflow
from mlpro.bf.streams import *

from mlpro.bf.math import *
from mlpro.bf.streams.basics import InstDict, StreamShared
from mlpro.bf.various import *
from datetime import datetime, timedelta
from mlpro.bf.control.controllers.pid_controller import PIDController
from mlpro.bf.control.basics import CTRLError



env = gym.make("Pendulum-v1",render_mode="human")

def Random_games():

    action_size = env.action_space.shape[0]

    for episode in range(10):
        env.reset()

        while True:
            env.render()

            action = np.random.uniform(-1.0,1.0,size=action_size)

            next_state,reward, done, info,_ = env.step(action)
            print(f'Next state{len(next_state)}\n',f'reward:{reward}')

            if done:
                break
def dpg_algorithm():
        # The noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
    model.learn(total_timesteps=10000, log_interval=10)
    vec_env = model.get_env()

    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        env.render()
def main():   

    # Parameter
    setpoint = 22.0          # Sollwert in °C
    initial_temp = 15.0      # Starttemperatur in °C
    ambient_temp = 15.0      # Außentemperatur in °C
    time_step = 1            # Zeitintervall in Minuten
    total_time =5000       # Gesamte Simulationszeit in Minuten

    # PID-Koeffizienten
    Kp = 10.0   # Proportionaler Koeffizient
    Ti =  100.01  # Integraler Koeffizient
    Td = 250.0  # Differenzieller Koeffizient

    # Heizwendel Parameter
    coil_mass = 10.0          # Masse der Heizwendel (kg)
    specific_heat_coil = 0.5  # Spezifische Wärmekapazität der Heizwendel (J/(kg*K))
    coil_temp = initial_temp   # Anfangstemperatur der Heizwendel
    heat_transfer_coeff = 0.1  # Wärmeübertragungskoeffizient (W/K)

    # Initialisierung
    time = np.arange(0, total_time + time_step, time_step)
    temperature = np.zeros_like(time,dtype=float)
    setpoints = np.zeros_like(time,dtype=float)
    temperature[0] = initial_temp
    setpoints[0]= setpoint
    # PID-Regler Variablen
    integral = 0
    previous_error = 0


    # Simulation
    for i in range(1, len(time)):

        #if i%600 ==0:
        #    setpoint+= random.randint(-2,2)
        error = setpoint - temperature[i - 1]
        integral += error * time_step
        derivative = (error - previous_error) / time_step
        
        # PID-Regler Berechnung
        control_signal = Kp * error + (Kp/Ti)*integral + Kp*Td * derivative
        
        # Begrenzung der Steuergröße und Normierung auf 0 bis 1
        control_signal = np.clip(control_signal, 0, 100) / 100
        
        # Heizwendel-Erwärmung
        power_input = control_signal * 100  # z.B. in Watt
        coil_temp += (power_input - heat_transfer_coeff * (coil_temp - ambient_temp)) / (coil_mass * specific_heat_coil) * time_step
        print(f"difference:{coil_temp-ambient_temp}",f"Power:{power_input}")
        
        # Wärmeübertragung zum Raum
        heating_power = heat_transfer_coeff * (coil_temp - temperature[i - 1])

        
        # Temperaturänderung des Raums
        delta_temp = heating_power * time_step / 60
        temperature[i] = temperature[i - 1] + delta_temp
        
        # Temperaturveränderung durch Umgebung
        temperature[i] += (ambient_temp - temperature[i]) * 0.01
        
        # Update der PID-Variablen
        previous_error = error
        setpoints[i]+=setpoint

    # Plotten der Ergebnisse
    plt.plot(time, temperature, label='Raumtemperatur')
    plt.plot(time,setpoints, color='r', linestyle='--', label='Sollwert')
    plt.xlabel('Zeit (Minuten)')
    plt.ylabel('Temperatur (°C)')
    plt.title('Temperaturregelung mit normiertem PID-Algorithmus')
    plt.legend()
    plt.grid(True)
    plt.show()

def create_instance():
    p_set = Set()
    elem = Element(p_set=p_set)
    elem.set_values([1,2,3,4])
    inst = Instance(p_feature_data=elem)
    print(inst.get_feature_data().get_values())


pid = PIDController(12,1,2)
p_set = Set()
elem = Element(p_set=p_set)
elem.set_values([12])
error = CTRLError(elem)
print(pid.compute_action(error))


