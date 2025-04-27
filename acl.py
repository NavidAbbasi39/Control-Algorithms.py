import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple

np.random.seed(0)  # For reproducibility

@dataclass
class Man2DOF:
    dt: float = 0.01 #
    state: np.ndarray = field(default_factory=lambda: np.zeros(4))  #output of joints[q1, q2, q1_dot, q2_dot]

    def dynamics(self, u: np.ndarray) -> None:
        
        q = self.state[:2]     #q_dot = velocity
        q_dot = self.state[2:] #control input signal (torque)

        # State-Space equation of Manipulator
        q_ddot = u  # assuming no friction 
        q_dot_new = q_dot + q_ddot * self.dt
        q_new = q + q_dot * self.dt
        #new velocity and input
        self.state[:2] = q_new
        self.state[2:] = q_dot_new

    def get_state(self) -> np.ndarray:
        return self.state.copy()


@dataclass
class BacksteppingController:
    k1: float = 5.0
    k2: float = 3.0

    def control(self, state: np.ndarray, qd: np.ndarray, qd_dot: np.ndarray, qd_ddot: np.ndarray) -> np.ndarray:
        """
        Backstepping control for tracking:
        state: [q1, q2, q1_dot, q2_dot]
        qd, qd_dot, qd_ddot: desired position, velocity, acceleration
        """
        q = state[:2]
        q_dot = state[2:]

        e = q - qd
        e_dot = q_dot - qd_dot

        # Virtual control law
        v = - self.k1 * e_dot - self.k2 * e + qd_ddot 

        # Control input (torque) assuming double integrator
        u = v

        return u


@dataclass
class ACL:
    state_dim: int = 4
    action_dim: int = 2
    actor_lr: float = 0.001
    critic_lr: float = 0.005
    gamma: float = 0.95

    actor_weights: np.ndarray = field(init=False)
    critic_weights: np.ndarray = field(init=False)

    def __post_init__(self):
        # Initialize weights small random
        self.actor_weights = np.random.randn(self.state_dim, self.action_dim) * 0.1
        self.critic_weights = np.random.randn(self.state_dim) * 0.1

    def actor(self, state: np.ndarray) -> np.ndarray:
        # Linear policy: action = W^T * state
        return state @ self.actor_weights

    def critic(self, state: np.ndarray) -> float:
        # State value estimate: V(s) = w^T * state
        return float(state @ self.critic_weights)

    def update(self, state: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Update actor and critic weights using TD error
        """
        value = self.critic(state)
        next_value = 0.0 if done else self.critic(next_state)
        td_error = reward + self.gamma * next_value - value

        # Critic update (gradient descent)
        self.critic_weights += self.critic_lr * td_error * state

        # Actor update (policy gradient)
        action = self.actor(state)
        self.actor_weights += self.actor_lr * td_error * np.outer(state, action)


@dataclass
class Simulator:
    dt: float = 0.01
    total_time: float = 10.0

    manipulator: Man2DOF = field(default_factory=Man2DOF)
    backstepping: BacksteppingController = field(default_factory=BacksteppingController)
    agent: ACL = field(default_factory=ACL)

    time_steps: int = field(init=False)
    history: dict = field(init=False)

    def __post_init__(self):
        self.time_steps = int(self.total_time / self.dt)
        self.history = {
            "time": [],
            "q1": [],
            "q2": [],
            "qd1": [],
            "qd2": [],
            "u1": [],
            "u2": [],
            "error_norm": [],
        }

    def reference(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reference trajectory and derivatives: sinusoids
        """
        qd = np.array([np.sin(t), np.sin(0.5 * t)])
        qd_dot = np.array([np.cos(t), 0.5 * np.cos(0.5 * t)])
        qd_ddot = np.array([-np.sin(t), -0.25 * np.sin(0.5 * t)])
        return qd, qd_dot, qd_ddot

    def run(self) -> None:
        for step in range(self.time_steps):
            t = step * self.dt
            state = self.manipulator.get_state()
            qd, qd_dot, qd_ddot = self.reference(t)

            # Backstepping control
            u_bs = self.backstepping.control(state, qd, qd_dot, qd_ddot)

            # Actor control adjustment
            u_actor = self.agent.actor(state)

            # Total control input
            u = u_bs + u_actor

            # Apply control to manipulator
            self.manipulator.dynamics(u)

            # Calculate reward (negative squared error)
            q = state[:2]
            error = q - qd
            reward = -np.sum(error**2)

            next_state = self.manipulator.get_state()
            done = step == self.time_steps - 1

            # Update agent
            self.agent.update(state, reward, next_state, done)

            # Log data
            self.history["time"].append(t)
            self.history["q1"].append(q[0])
            self.history["q2"].append(q[1])
            self.history["qd1"].append(qd[0])
            self.history["qd2"].append(qd[1])
            self.history["u1"].append(u[0])
            self.history["u2"].append(u[1])
            self.history["error_norm"].append(np.linalg.norm(error))

        self.plot_results()

    def plot_results(self) -> None:
        time = self.history["time"]
        plt.figure(figsize=(12, 10))

        # Joint 1 position tracking
        plt.subplot(3, 1, 1)
        plt.plot(time, self.history["q1"], label="q1")
        plt.plot(time, self.history["qd1"], label="qd1", linestyle="--")
        plt.title("Joint 1 Position Tracking")
        plt.xlabel("Time [s]")
        plt.ylabel("Position")
        plt.legend()
        plt.grid(True)

        # Joint 2 position tracking
        plt.subplot(3, 1, 2)
        plt.plot(time, self.history["q2"], label="q2")
        plt.plot(time, self.history["qd2"], label="qd2", linestyle="--")
        plt.title("Joint 2 Position Tracking")
        plt.xlabel("Time [s]")
        plt.ylabel("Position")
        plt.legend()
        plt.grid(True)

        # Control inputs
        plt.subplot(3, 1, 3)
        plt.plot(time, self.history["u1"], label="Control u1")
        plt.plot(time, self.history["u2"], label="Control u2")
        plt.title("Control Inputs")
        plt.xlabel("Time [s]")
        plt.ylabel("Torque")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    sim = Simulator()
    sim.run()
