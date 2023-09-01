import numpy as np
import matplotlib.pyplot as plt
import hcipy as hp
from gym.spaces import Box

# Global variables
DIAMETER = 8  # meter
WAVELENGTH = 1e-6  # meter
RESOLUTION = 256  # pixels
F_NUMBER = 40
OVERSAMPLING = 4  # pixels
N_AIRY = 12
N_PHOTONS = 1e10
N_MODES = 1000
N_ACT_ACROSS = 20
MODE_BASIS = 'actuators'
WF_RMS = 0.2
DT = 1  # s
DECOR_TIME = 100  # s
IWA = 2  # lambda/D
OWA = 6  # lambda/D
N_PROBES = 5 #  actuators

class Darkhole_AO_system():
    def __init__(self):
        # Initialize physical parameters
        self.diameter = DIAMETER
        self.wavelength = WAVELENGTH
        self.iwa = IWA * F_NUMBER * WAVELENGTH
        self.owa = OWA * F_NUMBER * WAVELENGTH
        self.pupil_grid = hp.make_pupil_grid(RESOLUTION, self.diameter)
        self.focal_length = F_NUMBER*self.diameter
        self.focal_grid = hp.make_focal_grid(q=OVERSAMPLING, num_airy=N_AIRY,
                                             pupil_diameter=self.diameter,
                                             reference_wavelength=self.wavelength, 
                                             f_number=F_NUMBER)

        self.propagator = hp.FraunhoferPropagator(self.pupil_grid,
                                                  self.focal_grid,
                                                  focal_length=self.focal_length)
        self.spatial_resolution = self.focal_length * self.wavelength / self.diameter
        self.aperture = hp.evaluate_supersampled(hp.make_vlt_aperture(),
                                                 self.pupil_grid, 4)
        self.wf_in = hp.Wavefront(self.aperture, self.wavelength)

        # Get reference measurements
        self.wf_in.total_power = 1
        self.ref_image = self.get_image(self.wf_in, dt=1, ref=True)
        self.Inorm = self.ref_image.sum()
        self.num_photons = N_PHOTONS
        self.Ipeak = self.get_image(self.wf_in, dt=1, noiseless=True).max()
        self.cent_pixel = np.argmax(self.ref_image)

        # Initialize mirror and coronagraph
        self.make_dm(N_MODES, MODE_BASIS)
        r_grid = np.sqrt(self.focal_grid.x**2 + self.focal_grid.y**2)
        self.dark_zone = (self.focal_grid.x > self.iwa) * (r_grid < self.owa)
        self.dark_zone_pixels = np.sum(self.dark_zone)
        self.make_ideal_coronagraph()
        self.action_space = Box(low=-0.3, high=0.3, shape=(self.num_modes,),
                                dtype=np.float32)
        self.observation_space = Box(low=0, high=1.,
                                     shape=(N_PROBES, self.dark_zone_pixels),
                                     dtype=np.float32)
        self.probes = np.linspace(int(self.num_modes/N_PROBES),
                                  int(N_PROBES * (1 - 1./N_PROBES)), N_PROBES)
        print("probes:", self.probes)
        self.iteration = 0
        self.episode = 0
        self.tot_rewards = []
        self.reward_range = (0, 1)
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8))

    def step(self, action):
        self.deformable_mirror.actuators += action / (2 * np.pi) *\
            self.wavelength
        field_in = self.wf_in.copy()
        self.update_aberration()
        field_in.electric_field *= np.exp(1j*self.abb)
        self.observation = self.pw_probing(field_in)/self.Ipeak
        self.image_out = self.get_image(field_in)/self.Ipeak
        self.tot_image += self.observation
        self.dark_zone_I = np.nanmean(self.image_out[self.dark_zone])
        self.reward = -1*np.log10(self.dark_zone_I)
        self.ep_reward += self.reward
        self.terminated = False
        self.truncated = self.dark_zone_I > 0.01
        self.iteration += 1
        info = {}
        return self.observation, self.reward, self.terminated, \
            self.truncated, info

    def reset(self):
        self.reset_actuators()
        if hasattr(self, 'ep_reward'):
            self.tot_rewards.append(self.ep_reward)
        self.ep_reward = 0
        self.tot_image = 0
        self.strehls = []
        observation = self.step(np.zeros(self.num_modes))[0]
        self.iteration = 0
        self.episode += 1
        return observation

    def render(self):
        for ax in self.axes.ravel():
            ax.cla()
        
        plt.sca(self.axes[0, 0])
        plt.axis('off')
        plt.title(f'Image, log10 contrast: {np.log10(self.dark_zone_I):.2f}')
        im1 = hp.imshow_field(self.image_out, cmap='viridis', vmin=0)
        if self.episode == 1 and self.iteration == 1:
            self.cbar1 = plt.colorbar(im1)
        else:
            self.cbar1.update_normal(im1)
        
        plt.sca(self.axes[0, 1])
        im2 = hp.imshow_field(np.log10(self.image_out), vmax=-2,
                              vmin=-8, cmap='inferno')
        plt.plot([self.iwa, ] * 100, np.linspace(-self.owa, self.owa, 100), 
                 ls='--', color='blue')
        t = np.linspace(-np.pi/2, np.pi/2, 100)
        plt.plot(self.iwa + (self.owa - self.iwa) * np.cos(t), 
                 self.owa * np.sin(t), ls='--', color='blue')
        plt.axis('off')
        plt.title(f'log10 Image')
        if self.episode == 1 and self.iteration == 1:
            self.cbar2 = plt.colorbar(im2)
        else:
            self.cbar2.update_normal(im2)
        
        # Plot mirror shape
        plt.sca(self.axes[1, 0])
        dm_phase = self.deformable_mirror.phase_for(self.wavelength) * self.aperture
        vmax = np.max(np.abs(dm_phase))
        im3 = hp.imshow_field(dm_phase, cmap='bwr', vmin=-vmax, vmax=vmax)
        plt.axis('off')
        plt.title('Deformable mirror shape')
        if self.episode == 1 and self.iteration == 1:
            self.cbar3 = plt.colorbar(im3)
        else:
            self.cbar3.update_normal(im3)
        plt.sca(self.axes[1, 1])
        #plt.plot(np.arange(self.iteration+1), self.strehls)
        if self.episode > 1:
            plt.plot(np.arange(self.episode-1), self.tot_rewards, marker='o',
                     color='black')
            plt.ylabel('Episode reward')
            plt.xlabel('Episode')
        plt.suptitle(f'Episode: {self.episode}, iteration: {self.iteration}')
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
    
    def make_ideal_coronagraph(self, order=2):
        self.coronagraph = hp.PerfectCoronagraph(self.aperture, order)
    
    def close(self):
        plt.close()
    
    def get_random_aberration(self, rms):
        abb = hp.make_power_law_error(self.pupil_grid, 1., self.diameter, -2.5)
        abb = (abb - np.mean(abb[self.aperture > 0])) / \
            np.std(abb[self.aperture > 0]) * rms
        return abb

    def update_aberration(self):
        if self.iteration == 0:
            self.abb1 = self.get_random_aberration(WF_RMS)
            self.abb2 = self.get_random_aberration(WF_RMS)
        if (self.iteration * DT) % DECOR_TIME == 0:
            self.abb1 = self.abb2
            self.abb2 = self.get_random_aberration(WF_RMS)
        frac = ((self.iteration * DT) % DECOR_TIME)/DECOR_TIME
        self.abb = np.sqrt((1.-frac)) * self.abb1 + np.sqrt(frac) * self.abb2

    def get_image(self, wf, dt=1, ref=False, noiseless=True):
        im_out = self.get_focal_field(wf).intensity*dt
        if ref:
            return im_out
        else:
            if noiseless:
                return im_out*self.num_photons/self.Inorm
            else:
                return hp.Field(np.random.poisson(im_out*self.num_photons/self.Inorm), im_out.grid)
    
    def get_focal_field(self, wf, coro=True):
        if hasattr(self, 'deformable_mirror'):
            wf = self.deformable_mirror.forward(wf)

        if coro and hasattr(self, 'coronagraph'):
            wf = self.coronagraph.forward(wf)
        return self.propagator.forward(wf)

    def make_dm(self, num_act, mode_basis='actuators'):
        if mode_basis=='actuators':
            num_act_across =  2* int(np.sqrt(num_act/np.pi))
            num_act_across =  N_ACT_ACROSS
            actuator_spacing = self.diameter / num_act_across
            influence_functions = hp.make_gaussian_influence_functions(
                self.pupil_grid, num_act_across, actuator_spacing)
            self.deformable_mirror = hp.DeformableMirror(influence_functions)
            self.num_modes = self.deformable_mirror.num_actuators
            self.influence_matrix = np.array(
                self.deformable_mirror.influence_functions.transformation_matrix.todense())
        elif mode_basis=='zernike':
            influence_functions = hp.make_zernike_basis(
                num_act**2, self.diameter, self.pupil_grid)
            self.deformable_mirror = hp.DeformableMirror(influence_functions)
            self.num_modes = self.deformable_mirror.num_actuators
            self.influence_matrix = np.array(
                self.deformable_mirror.influence_functions.transformation_matrix)
        else:
            raise ValueError('Unknown modal basis {mode_basis}')

    def make_ideal_coronagraph(self, order=2):
        self.coronagraph = hp.PerfectCoronagraph(self.aperture, order)

    def reset_actuators(self):
        self.deformable_mirror.actuators = np.zeros(self.num_modes)

    def pw_probing(self, field_in, probe_amp=0.01):
        orig_actuators = self.deformable_mirror.actuators.copy()
        deltas = []
        for probe in self.probes:
            delta_I = 0
            for amp in [1, -1]:
                new_actuators = orig_actuators.copy()
                new_actuators[int(probe)] += amp * probe_amp * self.wavelength
                self.deformable_mirror.actuators = new_actuators
                delta_I += amp * self.get_image(field_in)[self.dark_zone]
            deltas.append(delta_I)
        self.deformable_mirror.actuators = orig_actuators
        z = np.array(deltas)
        print(z)
        return z


def run():
    env = Focal_AO_system()
    N_iter = 50
    N_episode = 10

    for episode in range(N_episode):
        o = env.reset()
        print('Episode:', env.episode)
        for i in range(N_iter):
            a = 0.1 * env.action_space.sample()
            o, r, t, trunc, info = env.step(a)
            if trunc:
                break
            env.render()

    env.close()


if __name__ == '__main__':
    run() 
