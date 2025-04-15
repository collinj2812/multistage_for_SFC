import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

import sys

sys.path.append('../do-crystal/pbe_sol')
import PBE
import cryst

from pressure_drop import pressure_drop


class SFCModel:
    def __init__(self):
        self.Q_g = None
        self.output = None
        self.T_TM_in = None
        self.mf_PM = None
        self.mf_TM = None
        self.liquid_slugs = None
        self.z_TM_center = None
        self.z_TM_outer = None
        self.dz = None
        self.N_discr = None
        self.L = None
        self.dt = None
        self.v_0 = None
        self.param = None
        self.T_env = 273.15 + 20

    def setup(self, L, dt, N_discr, param, MC_param, T_TM_init=None):
        self.L = L  # length of crystallizer
        self.dt = dt  # time step
        self.N_discr = N_discr  # number of discretization points for outer tube (PM)
        self.dz = L / N_discr  # discretization interval
        self.z_TM_center = np.linspace(self.dz / 2, L - self.dz / 2,
                                       N_discr)  # center of finite volumes for outer tube (PM)
        self.z_TM_outer = np.linspace(0, L, N_discr + 1)  # outer points of finite volumes for outer tube (PM)

        # initialize liquid slugs
        # liquid slugs are dictionarys with keys: position 'z', concentration 'c', temperature 'T', mass 'm', surface area 'dA', ratio to real slug 'ratio_to_real', and Monte Carlo model 'MC_model'
        self.liquid_slugs = []

        # parameters
        self.param = param  # dictionary with parameters for the model
        self.MC_param = MC_param  # dictionary with parameters for Monte Carlo simulation
        self.check_param()

        # initialize outer tube (TM)
        if T_TM_init is not None and len(T_TM_init) == N_discr:
            self.T_TM = T_TM_init
        else:
            # print('No initial temperature profile for TM given or wrong length. Using 325 K as initial temperature.')
            self.T_TM = np.ones((N_discr)) * 330

        # setup output dictionary
        # output are c, T, d50, and d90-d10 of last liquid slug, T_TM at the end of the crystallizer as well as time and inputs
        self.output = {'time': [], 'c': [], 'T_PM': [], 'd50': [], 'd90': [], 'd10': [], 'T_TM': [], 'mf_PM': [],
                       'mf_TM': [], 'Q_g': [], 'w_crystal': [], 'c_in': [], 'T_PM_in': [], 'T_TM_in': []}

    def v_profile(self, z):
        # calculate velocity profile using the ideal gas law (assume constant temperature for now)
        p_out = 1.01325e5  # atmospheric pressure [bar]
        Q_PM = self.mf_PM / self.param['rho_PM'] * 1000 * 1000 * 60  # liquid flow rate [mL/min]
        Q_TM = self.mf_TM / self.param['rho_TM'] * 1000 * 1000 * 60
        Q_g = self.Q_g * 1000 * 1000 * 60  # gas flow rate [mL/min]
        delta_p = pressure_drop(self.L, Q_PM, Q_TM, Q_g)
        p_in = p_out + delta_p

        return p_in / (p_in - z * (p_in - p_out) / self.L) * self.v_0

    def advance_liquid_slugs(self):
        # go through all liquid slugs and advance them, z_new = z_old + v(z_old)*dt
        # save information where each slug started and where it ended
        slug_position_0 = [slug['z'] for slug in self.liquid_slugs]
        for slug in self.liquid_slugs:
            slug['z'] += self.v_profile(slug['z']) * self.dt

        self.liquid_slugs = [slug for slug in self.liquid_slugs if slug['z'] <= self.L]
        # # check if any liquid slug has reached the end of the crystallizer
        # if slug['z'] > self.L:
        #     self.liquid_slugs.remove(slug)
        slug_position_1 = [slug['z'] for slug in self.liquid_slugs]
        return slug_position_0, slug_position_1

    def add_liquid_slug(self, c, T_PM, m, dA, w_crystal):
        # initialize Monte Carlo population for this liquid slug
        # calculate mu_3 for this liquid slug
        mu_3 = w_crystal * m / self.param['rho_cryst'] / self.param['kv']
        MC_model = PBE.MC_simulation(self.MC_param['n_init'], self.MC_param['G'], self.MC_param['B'],
                                     self.MC_param['beta'], self.MC_param['Dil'], self.MC_param['domain'],
                                     self.MC_param['coordinate'])
        MC_model.init_particles(mu_3=mu_3, silence=True)

        # calculate ratio of real to model slug
        L_real = cryst.slug_length(self.param['rho_PM'], self.v_0, self.param['eta_l'], self.param['d_iPM'], self.param['sigma_l'], self.eps)  # length of real slug
        L_sim = dA / (np.pi * self.param['d_iPM'])  # length of simulated slug

        ratio = L_real / L_sim

        # add a new liquid slug at the beginning of the crystallizer
        new_slug = {'z': 0, 'c': c, 'T_PM': T_PM, 'm': m, 'dA': dA, 'ratio_to_real' : ratio, 'MC_model': MC_model}
        self.liquid_slugs.insert(0, new_slug)

    def simulate_states(self, slug_position_0, slug_position_1):

        # initialize vector for heat transfer to each element
        Q_dot_PM_TM_full = np.zeros(self.N_discr)
        # go through all liquid slugs and simulate states
        for i, slug in enumerate(self.liquid_slugs):
            # calculate at which element of TM the liquid slug starts and ends
            start_element = np.floor(slug_position_0[i] / self.dz).astype(int)
            end_element = np.floor(slug_position_1[i] / self.dz).astype(int)

            c_star = cryst.solubility(slug['T_PM'])
            rel_S = slug['c'] / c_star - 1
            G = ca.fmax(0, cryst.G(rel_S))
            N = 0
            beta = self.MC_param['beta_0']*G**self.MC_param['beta_1']*self.v_profile(slug['z'])**self.MC_param['beta_2']  # fit parameters

            mu_2 = slug['MC_model'].mu[2]

            # advance crystal population in time
            def kernel_function(d1, d2):
                return ((d1**3 - d2**3)**2) / (d1**3 + d2**3)

            slug['MC_model'].simulate(G=G, B=N, beta=beta, t=self.dt, dt=self.dt, silence=True, kernel_function=kernel_function)

            slug['c'] += self.dt * (-3 * self.param['kv'] * self.param['rho_cryst'] * G * mu_2 / slug['m'])

            # advance temperature of liquid slug in time using backward Euler
            T_PM_old = slug['T_PM']
            slug['T_PM'] = (self.param['U_PM_TM'] * slug['dA'] / (slug['m'] * self.param['c_p_PM']) * self.dt *
                            self.T_TM[end_element] + slug['T_PM']) / (
                                   1 + self.param['U_PM_TM'] * slug['dA'] * self.dt / (
                                   slug['m'] * self.param['c_p_PM']))
            # calculate transfer of heat from liquid slug to outer tube based on difference in temperature
            Q_dot_PM = self.param['U_PM_TM'] * slug['dA'] * (slug['T_PM'] - T_PM_old)

            # # calculate total heat transfer from this liquid slug to the outer tube
            # Q_dot_PM = self.dt * (-self.param['U_PM_TM'] * slug['dA'] * (slug['T_PM'] - self.T_TM[end_element]))
            #
            # # advance temperature of liquid slug in time
            # slug['T_PM'] += Q_dot_PM / (self.param['c_p_PM'] * slug['m'])

            # calculate heat transfer to outer tube single elements
            # length that liquid slug has traveled in this time step
            dz_travel = slug_position_1[i] - slug_position_0[i]
            # number of elements that liquid slug has traveled
            N_travel = dz_travel / self.dz
            # calculate heat transfer to each element
            Q_dot_PM_element = Q_dot_PM / N_travel
            # if liquid slug has traveled along a discrete element add corresponding portion of heat transfer to this element
            for slug_i in range(int(start_element), int(end_element + 1)):
                # check how much the liquid slug traveled in this element
                dz_travel_element = min(slug_position_1[i], self.z_TM_outer[slug_i + 1]) - max(slug_position_0[i],
                                                                                               self.z_TM_outer[slug_i])
                Q_dot_PM_TM_full[slug_i] += Q_dot_PM_element * dz_travel_element / self.dz

        # calculate temperature profile in outer tube
        m_TM_cp = self.param['c_p_TM'] * self.param['rho_TM'] * self.dz * self.param['A_TM_cross']
        dA_TM_env = self.param['A_TM_env'] / self.N_discr
        T_TM_in = np.concatenate((np.array([self.T_TM_in]), self.T_TM[:-1]))

        self.T_TM += self.dt * (1 / m_TM_cp * (
                self.mf_TM * self.param['c_p_TM'] * (T_TM_in - self.T_TM) - Q_dot_PM_TM_full - self.param[
            'U_TM_env'] * dA_TM_env * (self.T_TM - self.T_env))).squeeze()


    def make_step(self, c_in, T_PM_in, T_TM_in, mf_PM, mf_TM, Q_g, w_crystal):
        self.v_0 = 4 * (Q_g + mf_PM / self.param['rho_PM']) / (np.pi * self.param['d_iPM'] ** 2)
        self.mf_PM = mf_PM
        self.mf_TM = mf_TM
        self.Q_g = Q_g
        self.T_TM_in = T_TM_in
        self.eps = (mf_PM / self.param['rho_PM']) / (Q_g + mf_PM / self.param['rho_PM'])
        # if w_crystal is array squeeze
        if isinstance(w_crystal, np.ndarray):
            w_crystal = w_crystal.squeeze()
        self.w_crystal = w_crystal  # weight fraction of crystals in the feed

        # advance all liquid slugs
        slug_position_0, slug_position_1 = self.advance_liquid_slugs()

        # simulate states
        self.simulate_states(slug_position_0, slug_position_1)

        # add new liquid slug
        m = self.dt * self.mf_PM
        dA = 4 * (m / self.param['rho_PM']) / self.param['d_iPM']
        self.add_liquid_slug(c_in, T_PM_in, m, dA, w_crystal)

        # save output
        self.output['c'].append(self.liquid_slugs[-1]['c'])
        self.output['T_PM'].append(self.liquid_slugs[-1]['T_PM'])

        # incorporate real to model ratio, delete appropriate amount of particles
        # only delete particles for calculation here since the last slug might still be in the crystallizer for the next time step
        n_particles = len(self.liquid_slugs[-1]['MC_model'].particles)

        n_particles_to_delete = int(np.round((1 - self.liquid_slugs[-1]['ratio_to_real']) * n_particles))

        indices_to_delete = np.random.choice(n_particles, n_particles_to_delete, replace=False)

        reduced_particle_list = np.delete(self.liquid_slugs[-1]['MC_model'].particles, indices_to_delete)

        self.output['d50'].append(np.quantile(reduced_particle_list, 0.5))
        self.output['d90'].append(np.quantile(reduced_particle_list, 0.9))
        self.output['d10'].append(np.quantile(reduced_particle_list, 0.1))


        self.output['T_TM'].append(self.T_TM[-1])
        # add up time
        if len(self.output['time']) == 0:
            self.output['time'].append(self.dt)
        else:
            self.output['time'].append(self.output['time'][-1] + self.dt)
        self.output['mf_PM'].append(mf_PM)
        self.output['mf_TM'].append(mf_TM)
        self.output['Q_g'].append(Q_g)
        self.output['w_crystal'].append(w_crystal)
        self.output['c_in'].append(c_in)
        self.output['T_PM_in'].append(T_PM_in)
        self.output['T_TM_in'].append(T_TM_in)

    def plot_states_over_length(self):
        # plot concentration and temperature over length
        z = np.array([slug['z'] for slug in self.liquid_slugs])
        c = np.array([slug['c'] for slug in self.liquid_slugs])
        T_PM = np.array([slug['T_PM'] for slug in self.liquid_slugs])
        d10 = np.array([slug['MC_model'].d10 for slug in self.liquid_slugs])
        d50 = np.array([slug['MC_model'].d50 for slug in self.liquid_slugs])
        d90 = np.array([slug['MC_model'].d90 for slug in self.liquid_slugs])

        # calculate solubility line
        c_star = cryst.solubility(T_PM)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 6), sharex=True)

        # First subplot
        ax1.plot(z, c, 'x', label='c')
        ax1.plot(z, c_star, label='c*')
        ax1.set_ylabel('c')
        ax1.legend()

        # Second subplot
        ax2.plot(z, T_PM, 'x', label='T')
        ax2.plot(self.z_TM_center, self.T_TM, label='T_TM')
        ax2.set_ylabel('T')
        ax2.legend()

        # Third subplot
        for i in range(len(z)):
            ax3.fill_between([z[i], z[i]], d10[i], d90[i], color='blue', alpha=0.2)
        ax3.plot(z, d50, 'x', label='$d_{50}$')
        ax3.set_xlabel('z')
        ax3.set_ylabel('$d_{50}$')
        ax3.legend()

        plt.tight_layout()
        plt.show()

        return fig

    def check_param(self):
        # check if all necessary parameters are in param
        necessary_param = ['rho_PM', 'rho_cryst', 'delta_H_cryst', 'c_p_PM', 'U_PM_TM', 'rho_TM', 'c_p_TM', 'kv',
                           'U_TM_env', 'L', 'd_iPM', 'd_aPM', 'D_iTM', 'D_aTM']
        for param_name in necessary_param:
            if param_name not in self.param:
                raise ValueError(f'Parameter {param_name} is missing in param')


def generate_input_sequence(bounds, sequence_length, counter_params=[1, 10]):
    # takes dictionary bounds with keys as input names and values as list with two elements [min, max]
    input_sequence = {}
    for key in bounds.keys():
        sequence = []
        counter = 0

        for t in range(sequence_length):
            if counter <= 0:
                # set new value from uniform distribution within bounds
                val = np.random.uniform(bounds[key][0], bounds[key][1])
                sequence.append(val)
                # set new counter
                counter = np.random.randint(counter_params[0], counter_params[1])
            else:
                # keep value the same
                sequence.append(val)
                counter -= 1
        input_sequence[key] = sequence
    return input_sequence
