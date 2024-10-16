from mlpro.bf.control.controllers.pid_controller import PIDController
from mlpro.bf.control.basics import ControlError, SetPoint
from mlpro.bf.ml.basics import *
from mlpro.oa.control.controllers import RLPID,wrapper_rl
import gymnasium as gym
from stable_baselines3 import A2C, PPO, DQN, DDPG, SAC
from mlpro_int_sb3.wrappers import WrPolicySB32MLPro
from mlpro.rl.models import *
from mlpro.rl.models_env import Reward
import datetime
from mlpro.bf.control import ControlVariable, ControlledVariable

import math
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def radians_to_degrees(radians):
    degrees = radians * (180 / math.pi)
    return degrees



class MyReward(FctReward):

    def __init__(self, p_logging=Log.C_LOG_ALL):
        self._reward = Reward(p_value=0)
        self._reward_value =0
        self.error_streak_counter=0
        super().__init__(p_logging)


    
    def _compute_reward(self, p_state_old: ControlledVariable = None, p_state_new: ControlledVariable= None) -> Reward:

        #get old error
        error_old = p_state_old.get_feature_data().get_values()[0]
        
        #get new error
        error_new = p_state_new.get_feature_data().get_values()[0]    
        
        
                # Berechnung der Ableitung des Fehlers (Fehleränderungsrate)
        error_derivative = error_new - error_old
        
        # Inverser Reward basierend auf dem neuen Fehler (kleiner Fehler = größerer Reward)
        epsilon = 1e-6  # Kleine Zahl, um Division durch Null zu vermeiden
        reward = 20 / (abs(error_new) + epsilon)
        
        # 1. Bestrafung für große Fehler (Abweichung vom Sollwert)
        penalty_threshold = 3.0  # Schwellenwert für zu große Fehler
        if abs(error_new) > penalty_threshold:
            reward -= (abs(error_new) - penalty_threshold)  # Bestrafung für große Fehler
        
        # 2. Bestrafung für starkes Überschwingen (Wenn der Fehler das Sollwert überschreitet)
        if (error_old > 0 and error_new < 0) or (error_old < 0 and error_new > 0):
            reward -= 5.0  # Bestrafung für Überschwingen (besonders stark bestraft)
        
        # 3. Bestrafung für starke Schwankungen in der Fehleränderung (Oszillationen)
        derivative_threshold = 3  # Schwellenwert für starke Fehleränderungen
        if abs(error_derivative) > derivative_threshold:
            reward -= 2.0 * abs(error_derivative)  # Bestrafung für starke Änderungen
        
        # 4. Exponentielle Belohnung für kleine Fehler (schnelles Erreichen des Sollwerts)
        k = 20  # Tuning-Parameter für exponentielle Bestrafung
        reward += math.exp(-0.3 * (error_new*error_new))*k

        # 5. Zusätzliche Belohnung für konstant kleinen Fehler über Zeit
        # Schwellenwert für "kleinen Fehler"
        small_error_threshold = 1  # Definiere, was als "kleiner Fehler" gilt
        streak_threshold = 10  # Anzahl der aufeinanderfolgenden Schritte für zusätzliche Belohnung

        # Prüfe, ob der Fehler klein ist
        if abs(error_new) < small_error_threshold:
            self.error_streak_counter += 1  # Zähler erhöhen, wenn der Fehler klein ist
        else:
            self.error_streak_counter = 0  # Zähler zurücksetzen, wenn der Fehler größer wird

        # Wenn der Fehler für mehrere Schritte konstant klein war, zusätzliche Belohnung
        if self.error_streak_counter >= streak_threshold:
            reward += 10  # Zusätzliche Belohnung, wenn der Fehler konstant klein bleibt
            # Optional: Zähler zurücksetzen, wenn der maximale Bonus erreicht ist
            # self.error_streak_counter = 0
        reward =min(reward,30)         

        self._reward.set_overall_reward(reward)
        
        return self._reward
       
action_space = MSpace()
p_obs = MSpace()
dim1 = Dimension('Kp',p_boundaries=[0.1,100])
dim2 = Dimension('Tn',p_unit='second',p_boundaries=[0,100])
dim3= Dimension('Tv',p_unit='second',p_boundaries=[0,100]) 
action_space.add_dim(dim1)        
action_space.add_dim(dim2)
action_space.add_dim(dim3)
observation_space = MSpace()
error_dim = Dimension('error',p_boundaries=[-100,100])
setpoint_dim = Dimension('setpoint',p_boundaries=[-100,100])
observation_space.add_dim(dim1)
pid = PIDController(Kp=13,p_input_space=MSpace(),p_output_space=MSpace(),Ti=23,Tv=34)
# PPO
policy_sb3 = PPO(
    policy="MlpPolicy",
    n_steps=100,
    env=None,
    _init_setup_model=False,
    device="cpu",learning_rate=0.003,seed=42)
cycle_limit=200
poliy_wrapper = WrPolicySB32MLPro(p_sb3_policy=policy_sb3,
                                  p_cycle_limit=cycle_limit,
                                  p_observation_space=observation_space,
                                  p_action_space=action_space)

rl_pid_policy = RLPID(p_observation_space=observation_space,
                      p_action_space=action_space,
                      pid_controller = pid,
                      policy=poliy_wrapper)


setpoint_space = Set()
setpoint_space.add_dim(p_dim=setpoint_dim)
setpoint = SetPoint(p_id=0,p_value_space=setpoint_space,p_values=[0],p_tstamp=datetime.datetime.now())



error_space = Set()
error_space.add_dim(p_dim=error_dim)
control_error = ControlError(p_id=0,p_value_space=error_space,p_values=[0],p_tstamp=datetime.datetime.now())
oa_controller=wrapper_rl.OAControllerRL(p_input_space=MSpace(),p_output_space=MSpace(),p_rl_policy=rl_pid_policy,p_rl_fct_reward=MyReward())



# Daten für das Plotten sammeln
angles = []
actions = []
rewards = []
gain_values=[]
integral_values=[]
deritave_values=[]
errors =[]
total_reward = 0




#training loop
for k in range(5):
    env = gym.make('CartPole-v1', render_mode="human")
    observation = env.reset()[0]
    for t in range(cycle_limit):

        env.render()
        # get obs values
        cart_position, cart_velocity, pole_angle, pole_velocity = observation
        #convert rad in °
        actual_angle = radians_to_degrees(pole_angle)
        #calculate error     
        control_error.get_feature_data().set_value(error_space.get_dim_ids()[0],actual_angle-setpoint.get_feature_data().get_values()[0])
        control_error.set_tstamp(datetime.datetime.now())

       
        oa_controller._adapt(p_ctrl_error=control_error,p_ctrl_var=ControlVariable(p_id=0,p_value_space=MSpace()))
     
        
        control_variable=oa_controller.compute_output(control_error)
        output= control_variable._get_values()[0]

        # Aktion umsetzen (nach links oder rechts)
        if output > 0:
            output = 1  # Bewegung nach rechts
        else:
            output = 0  # Bewegung nach links

        # Führe die Aktion in der Umgebung aus
        observation, reward, done, *_ = env.step(output)
        total_reward += oa_controller._rl_fct_reward._reward.get_overall_reward()
        # Daten sammeln
        angles.append(actual_angle)
        actions.append(output)
        rewards.append(oa_controller._rl_fct_reward._reward.get_overall_reward())
        gain,integral,deritave = tuple(oa_controller._rl_policy._pid_controller.get_parameter_values())
        gain_values.append(gain)
        integral_values.append(integral)
        deritave_values.append(deritave)
        errors.append(control_error._get_values()[0])
        
        if done:
             break

    env.close()

# Plotten der Ergebnisse
time_steps = range(len(angles))

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(time_steps, angles, label='Pole Angle')
plt.xlabel('Time Step')
plt.ylabel('Angle (radians)')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time_steps, actions, label='ControlVariable')
plt.xlabel('Time Step')
plt.ylabel('ControlVariable (0=left, 1=right)')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time_steps, rewards, label='Reward')
plt.xlabel('Time Step')
plt.ylabel('Reward')
plt.legend()

plt.subplot(3, 1, 1)
plt.plot(time_steps, gain_values, label='Gain')
plt.xlabel('Time Step')
plt.ylabel('-')
plt.legend()

plt.subplot(3, 1, 1)
plt.plot(time_steps,integral_values, label='integral')
plt.xlabel('Time Step')
plt.ylabel('seconds')
plt.legend()

plt.subplot(3, 1, 1)
plt.plot(time_steps, deritave_values, label='deritives')
plt.xlabel('Time Step')
plt.ylabel('seconds')
plt.legend()

plt.subplot(3, 1, 1)
plt.plot(time_steps, errors, label='errors')
plt.xlabel('Time Step')
plt.ylabel('°')
plt.legend()

plt.tight_layout()
plt.show()

print(f'Total Reward: {total_reward}')


    





