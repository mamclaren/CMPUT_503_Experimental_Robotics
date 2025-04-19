#!/usr/bin/env python3
import math

# PID controller variables
simple_pid = {
    "kp": -0.025, #-0.025
    "ki": 0,
    "kd": -0.0125, #-0.0125
    "previous_error": 0,
    "integral": 0
}

flow_pid = {
    "kp": -0.5, #-0.025
    "ki": -0.125, #-0.125,
    "kd": -0.2, #-1.0, #-0.0125
    "previous_error": 0,
    "integral": 0
}

sparse_flow_pid = {
    "kp": -0.5, #-0.025
    "ki": 0, #-0.125,
    "kd": 0, #-1.0, #-0.0125
    "previous_error": 0,
    "integral": 0
}

def pid_controller(pid, error, dt, reset=False):
    '''
    The method to get PID controls.
    For P/PD, just set ki and/or kd to 0
    use the reset flag when the desired value changes a lot
    need to tune the kp, ki, kd values for different tasks (keep a note of them)
    '''
    if reset:
        pid['integral'] = 0
        pid['previous_error'] = 0
    # error = desired_value - measured_value
    pid['integral'] += error * dt
    derivative = (error - pid['previous_error']) / dt if dt > 0 else 0
    
    output = (pid['kp'] * error) + (pid['ki'] * pid['integral']) + (pid['kd'] * derivative)
    pid['previous_error'] = error
    
    return output
    
# Feed the error into the pid function to get the amount to turn the bot
# and perform clamping to limit large overcorrections    
def pid_controller_v_omega(error, pid_values, rate, reset=False):
    dt = 1 / rate
    omega = None

    if error is not None:
        omega = pid_controller(pid_values, error, dt, reset=reset)
        clamp_value = math.pi * 1.25
        omega = max(-clamp_value, min(omega, clamp_value))

    # Return the calculated v and omega
    if error is None:
        return 0, 0
    else:
        return 0.23, omega
