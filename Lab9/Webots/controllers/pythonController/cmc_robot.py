"""CMC robot"""

import numpy as np
from network import SalamanderNetwork
from experiment_logger import ExperimentLogger
from robot_parameters import RobotParameters


class SalamanderCMC(object):
    """Salamander robot for CMC"""

    N_BODY_JOINTS = 10
    N_LEGS = 4
    X_HIGH_POS = 1.0
    X_LOW_POS = 0.25
    SWIM = True

    def __init__(self, robot, n_iterations, parameters, logs="logs/log.npz"):
        super(SalamanderCMC, self).__init__()
        self.robot = robot
        timestep = int(robot.getBasicTimeStep())
        self.network = SalamanderNetwork(1e-3*timestep, parameters)

        # Position sensors
        self.position_sensors = [
            self.robot.getPositionSensor('position_sensor_{}'.format(i+1))
            for i in range(self.N_BODY_JOINTS)
        ] + [
            self.robot.getPositionSensor('position_sensor_leg_{}'.format(i+1))
            for i in range(self.N_LEGS)
        ]
        for sensor in self.position_sensors:
            sensor.enable(timestep)

        # GPS
        self.enable = False
        self.gps = robot.getGPS("fgirdle_gps")
        self.gps.enable(timestep)

        # Get motors
        self.motors_body = [
            self.robot.getMotor("motor_{}".format(i+1))
            for i in range(self.N_BODY_JOINTS)
        ]
        self.motors_legs = [
            self.robot.getMotor("motor_leg_{}".format(i+1))
            for i in range(self.N_LEGS)
        ]

        # Set motors
        for motor in self.motors_body:
            motor.setPosition(0)
            motor.enableForceFeedback(timestep)
            motor.enableTorqueFeedback(timestep)
        for motor in self.motors_legs:
            motor.setPosition(-np.pi/2)
            motor.enableForceFeedback(timestep)
            motor.enableTorqueFeedback(timestep)

        # Iteration counter
        self.iteration = 0

        # Logging
        self.log = ExperimentLogger(
            n_iterations,
            n_links=1,
            n_joints=self.N_BODY_JOINTS+self.N_LEGS,
            filename=logs,
            timestep=1e-3*timestep,
            **parameters
        )

    def log_iteration(self):
        """Log state"""
        self.log.log_link_positions(self.iteration, 0, self.gps.getValues())
        for i, motor in enumerate(self.motors_body):
            # Position
            self.log.log_joint_position(
                self.iteration, i,
                self.position_sensors[i].getValue()
            )
            # Command
            self.log.log_joint_cmd(
                self.iteration, i,
                motor.getTargetPosition()
            )
            # Torque
            self.log.log_joint_torque(
                self.iteration, i,
                motor.getTorqueFeedback()
            )
            # Torque feedback
            self.log.log_joint_torque_feedback(
                self.iteration, i,
                motor.getTorqueFeedback()
            )
        for i, motor in enumerate(self.motors_legs):
            # Position
            self.log.log_joint_position(
                self.iteration, 10+i,
                self.position_sensors[10+i].getValue()
            )
            # Command
            self.log.log_joint_cmd(
                self.iteration, 10+i,
                motor.getTargetPosition()
            )
            # Torque
            self.log.log_joint_torque(
                self.iteration, 10+i,
                motor.getTorqueFeedback()
            )
            # Torque feedback
            self.log.log_joint_torque_feedback(
                self.iteration, 10+i,
                motor.getTorqueFeedback()
            )

    def step(self):
        """Step"""
        # Increment iteration
        self.iteration += 1

        # Update network
        self.network.step()
        positions = self.network.get_motor_position_output()

        # Update control
        for i in range(self.N_BODY_JOINTS):
            self.motors_body[i].setPosition(positions[i])
        for i in range(self.N_LEGS):
            self.motors_legs[i].setPosition(
                positions[self.N_BODY_JOINTS+i] - np.pi/2
            )

        # Log data
        self.log_iteration()

        # Retrieve GPS to change from walking to swimming
        if self.iteration == 1:
            self.enable = True

        pos = self.gps.getValues()

        if self.network.parameters.enable_transitions:
            if pos[0] > self.X_HIGH_POS:
                if not self.SWIM:
                    self.SWIM = True

                self.network.parameters.drive_left = 4.0
                self.network.parameters.drive_right = 4.0
            elif pos[0] < self.X_LOW_POS:
                self.network.parameters.drive_left = 2.0
                self.network.parameters.drive_right = 2.0

                if self.SWIM:
                    self.network.state.phases = 1e-4 * np.random.ranf(self.network.parameters.n_oscillators)
                    self.SWIM = False

            self.network.parameters.set_saturation_params(self.network.parameters)
            self.network.parameters.saturate_params()
