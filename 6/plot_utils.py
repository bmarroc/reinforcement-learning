import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from matplotlib import animation
from IPython.display import HTML

def plot_phasespace_matrix(policy_evaluation, n, m):

    fig, axes = plt.subplots(figsize=(30,30),nrows=n, ncols=m)

    for i in range(n):
        for j in range(m):
            x = np.array(policy_evaluation.agent.episodes[i*m+j])[:,0] 
            y = np.array(policy_evaluation.agent.episodes[i*m+j])[:,1]

            axes[i,j].set_xlabel('position')
            axes[i,j].set_ylabel('velocity')
            axes[i,j].set_xlim([-1.3, 0.6])
            axes[i,j].set_ylim([-0.08, 0.08])
            axes[i,j].plot(x, y, marker='.', color='#0F00FF', markersize=12, linestyle='-')
            
def plot_policy_actions(policy):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis([-1.2-0.1, 0.5+0.1, -0.07-0.1, 0.07+0.1])
    
    u = np.linspace(-1.2-0.1, 0.5+0.1, 100)
    v = np.linspace(-0.07-0.1, 0.07+0.1, 100)
    z = []
    for i in range(100):
        for j in range(100):
            z.append([u[i], v[j]])
    z = np.array(z)

    w = np.argmax(policy.predict(z), axis=1)

    full_throttle_reverse = np.where(w == 0)[0]
    zero_throttle = np.where(w == 1)[0]
    full_throttle_forward = np.where(w == 2)[0]
    
    ax.plot(z[full_throttle_reverse,0], z[full_throttle_reverse,1], marker='.', color='#FF00AE', markersize=10, linestyle='', label='FULL THROTTLE REVERSE') 
    ax.plot(z[zero_throttle,0], z[zero_throttle,1], marker='.', color='#00BA7F', markersize=10, linestyle='', label='ZERO THROTTLE') 
    ax.plot(z[full_throttle_forward,0], z[full_throttle_forward,1], marker='.', color='#0F00FF', markersize=10, linestyle='', label='FULL THROTTLE FORWARD') 
    
    ax.legend()
    
def plot_valuefunction(value_function):
    grid = []
    for x in np.round(np.linspace(-1.2, 0.5, 100), 3):
        for v in np.round(np.linspace(-0.07, 0.07, 100), 3):
            grid.append([x,v])
    grid = np.array(grid)

    J_vals = value_function.predict(grid)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('-Value Function')
    ax.set_xlabel('position')
    ax.set_ylabel('velocity')
    ax.set_xticklabels([])
    ax.set_yticklabels([]) 

    im = ax.imshow(-np.rot90(J_vals.reshape((100,100))), cmap='hot', interpolation='nearest')
    fig.colorbar(mappable=im, ax=ax, shrink=0.75)
    
def plot_valuefunction3d(value_function):
    theta0_vals = np.round(np.linspace(-1.2, 0.5, 50), 3)
    theta1_vals = np.round(np.linspace(-0.07, 0.07, 50), 3)

    X,Y = np.meshgrid(theta0_vals, theta1_vals)

    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

    values = []
    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            values.append([theta0_vals[i], theta1_vals[j]])
    values = np.array(values)

    J_vals = value_function.predict(values).reshape((len(theta0_vals),len(theta1_vals)))

    J_vals = J_vals.T

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('-Value Function')
    ax.plot_surface(X, Y, -J_vals, rstride=1, cstride=1, cmap='jet', edgecolor='black', linewidth=0.5, antialiased=True)
    
    ax.azim = -20
    ax.dist = 10
    ax.elev = 20
    
def plot_history(history, window_size):    
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_title('History')
    ax1.set_xlabel('episodes')
    ax1.set_ylabel('steps')
    ax1.plot(history, marker='.', color='#0F00FF', markersize=1, linestyle='-')
    
    moving_average = []
    for i in range(history.shape[0]-window_size):
        moving_average.append(history[i:i+window_size])
    moving_average = np.array(moving_average, dtype='float32')
    mean_history = np.mean(moving_average, axis=1)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_title('Mean average history over the last {} episodes'.format(window_size))
    ax2.set_xlabel('episodes')
    ax2.set_ylabel('steps')
    ax2.plot(mean_history, marker='.', color='#0F00FF', markersize=1, linestyle='-')
    
def plot_history_histogram(history):    
    fig = plt.figure(figsize=(10, 10)) 
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('History Distribution')
    n, bins, patches = ax.hist(history, bins=100, color='#6A89CC', edgecolor='#0F00FF', density=True)

    mu = np.mean(history)
    sigma = np.std(history)
    
    x = np.linspace(mu-4*sigma, mu+4*sigma, 100)
    y = st.norm.pdf(x, mu, sigma)

    ax.plot(x, y, marker='.', color='#FF00AE', markersize=1, linestyle='--')
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    label = 'mu = {}\nsigma = {}'.format(np.round(mu,2), np.round(sigma,2))
    ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    
def plot_animation(model, save_name, repeat=False, interval=40):
    
    def render_policy_net(model, n_max_steps=200, seed=42):
        frames = []
        base_env = gym.make('MountainCar-v0', render_mode="rgb_array")
        env = TimeLimit(base_env.env, max_episode_steps=n_max_steps)
        obs, _ = env.reset(seed=seed)
        np.random.seed(seed)

        for step in range(n_max_steps):
            frames.append(env.render())
            action = np.argmax(model.predict(obs[np.newaxis])[0])
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        env.close()
        return frames
    
    def update_scene(num, frames, patch, step_text):
        patch.set_data(frames[num])
        step_text.set_text("Step: {}".format(num))
        return [patch, step_text]
    
    frames = render_policy_net(model)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    step_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    
    patch = ax.imshow(frames[0], animated=True)
    plt.axis('off')
    anim = animation.FuncAnimation(fig, 
                                   update_scene, 
                                   fargs=(frames, patch, step_text),
                                   frames=len(frames), 
                                   repeat=repeat, 
                                   interval=interval)
    plt.close(fig)
    anim.save(save_name, writer='pillow')
    return HTML(anim.to_jshtml())