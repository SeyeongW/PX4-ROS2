"""Measured 20cm physics inputs. No missing value falls back to Iris parameters."""
import json
import math
from pathlib import Path


def number(value, name, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or positive and value <= 0:
        raise ValueError('finite'+('_positive' if positive else '')+'_required: '+name)
    return float(value)


def xyz(values, name):
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError('XYZ_required: '+name)
    return [number(v, name) for v in values]


def inertia(values, name):
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError('symmetric_positive_inertia_required: '+name)
    a = [xyz(row, name) for row in values]
    if any(abs(a[i][j]-a[j][i]) > 1e-12 for i in range(3) for j in range(3)):
        raise ValueError('symmetric_positive_inertia_required: '+name)
    # Sylvester criterion for a real symmetric 3x3 tensor.
    det = (a[0][0]*(a[1][1]*a[2][2]-a[1][2]**2)
           - a[0][1]*(a[0][1]*a[2][2]-a[1][2]*a[0][2])
           + a[0][2]*(a[0][1]*a[1][2]-a[1][1]*a[0][2]))
    if a[0][0] <= 0 or a[0][0]*a[1][1]-a[0][1]**2 <= 0 or det <= 0:
        raise ValueError('symmetric_positive_inertia_required: '+name)
    # A valid rigid-body tensor also obeys principal-moment triangle inequalities.
    half_trace = sum(a[i][i] for i in range(3))/2
    second = [[(half_trace if i == j else 0)-a[i][j] for j in range(3)] for i in range(3)]
    import numpy as np
    if np.linalg.eigvalsh(second).min() < -1e-12:
        raise ValueError('nonphysical_principal_inertia: '+name)
    return a


def motor_fit(rows, max_rpm):
    """Fit T=kT*omega² and Q=kQ*omega² to actual motor/propeller bench data."""
    if not isinstance(rows, list) or len(rows) < 3:
        raise ValueError('at_least_three_RPM_thrust_torque_samples_required')
    samples = [xyz(row, 'RPM_thrust_N_torque_Nm') for row in rows]
    if any(rpm <= 0 or thrust <= 0 or torque < 0 for rpm, thrust, torque in samples):
        raise ValueError('positive_RPM_thrust_and_nonnegative_torque_required')
    if len({r[0] for r in samples}) < 3 or max_rpm > max(r[0] for r in samples):
        raise ValueError('distinct_RPM_samples_must_cover_declared_max_RPM')
    omega2 = [(rpm*math.pi/30)**2 for rpm, _, _ in samples]
    denominator = sum(x*x for x in omega2)
    kt = sum(x*row[1] for x, row in zip(omega2, samples))/denominator
    kq = sum(x*row[2] for x, row in zip(omega2, samples))/denominator
    if kq <= 0:
        raise ValueError('nonzero_measured_reaction_torque_required_for_yaw_control')
    error = lambda column, coefficient: math.sqrt(sum((row[column]-coefficient*x)**2
                                    for x, row in zip(omega2, samples))/len(samples))
    return dict(thrust_coefficient=kt, torque_coefficient=kq,
                moment_constant_m=kq/kt, max_omega_rad_s=max_rpm*math.pi/30,
                thrust_fit_rmse_N=error(1, kt), torque_fit_rmse_Nm=error(2, kq))


def audit(config, requested_speed=1.5):
    """Input availability is separate from predicted feasibility and flight proof."""
    if not isinstance(config, dict):
        raise ValueError('physics_JSON_object_required')
    missing = [key for key in ('mass_kg', 'center_of_mass_body_m', 'inertia_body_kg_m2',
                              'rotor_mass_kg', 'rotor_inertia_kg_m2', 'rotor_positions_body_m',
                              'rotor_axes_body', 'rotor_spin_directions') if config.get(key) is None]
    motor = config.get('motor') or {}
    aero = config.get('aerodynamics') or {}
    sources = config.get('sources') or {}
    if any(not isinstance(section, dict) for section in (motor, aero, sources)):
        raise ValueError('motor_aerodynamics_sources_must_be_JSON_objects')
    for section, values, keys in (
        ('motor', motor, ('max_rpm', 'time_constant_up_s', 'time_constant_down_s', 'rpm_thrust_torque_samples')),
        ('aerodynamics', aero, ('wing_area_m2', 'mean_chord_m', 'center_of_pressure_body_m',
           'CL0', 'CLa_per_rad', 'CL_max', 'CD0', 'aspect_ratio', 'efficiency', 'Cm0', 'Cma_per_rad',
           'alpha_stall_rad', 'CLa_stall_per_rad', 'CDa_stall_per_rad', 'Cma_stall_per_rad'))):
        missing += [section+'.'+key for key in keys if values.get(key) is None]
    missing += ['sources.'+key for key in ('mass', 'inertia', 'rotor', 'motor', 'aerodynamics', 'geometry')
                if not isinstance(sources.get(key), str) or not sources[key].strip()]
    result = dict(data_complete=False, flight_verified=False, missing=missing,
                  data_status=config.get('data_status', 'not_declared'),
                  geometry_scope='visual-derived geometry until checked against physical hardware')
    if missing:
        return result
    if type(config.get('schema_version')) is not int or config['schema_version'] != 1:
        raise ValueError('unsupported_20cm_physics_schema')
    mass = number(config['mass_kg'], 'mass_kg', True)
    xyz(config['center_of_mass_body_m'], 'center_of_mass_body_m')
    inertia(config['inertia_body_kg_m2'], 'inertia_body_kg_m2')
    rotor_mass = number(config['rotor_mass_kg'], 'rotor_mass_kg', True)
    if 4*rotor_mass >= mass:
        raise ValueError('total_mass_must_include_four_rotors_and_body')
    inertia(config['rotor_inertia_kg_m2'], 'rotor_inertia_kg_m2')
    import numpy as np
    positions = config['rotor_positions_body_m']
    axes = config['rotor_axes_body']
    spins = config['rotor_spin_directions']
    if (not isinstance(positions, list) or len(positions) != 4 or not isinstance(axes, list)
            or len(axes) != 4 or not isinstance(spins, list) or len(spins) != 4
            or any(spin not in ('cw', 'ccw') for spin in spins)):
        raise ValueError('four_measured_rotor_positions_axes_and_cw_ccw_required')
    positions = np.array([xyz(p, 'rotor_positions_body_m') for p in positions])
    axes = np.array([xyz(a, 'rotor_axes_body') for a in axes])
    if np.max(np.abs(np.linalg.norm(axes, axis=1)-1)) > 1e-6:
        raise ValueError('unit_rotor_thrust_axes_required')
    for key in ('max_rpm', 'time_constant_up_s', 'time_constant_down_s'):
        number(motor[key], 'motor.'+key, True)
    fit = motor_fit(motor['rpm_thrust_torque_samples'], motor['max_rpm'])
    signs = np.array([-1 if spin == 'ccw' else 1 for spin in spins])
    torques = np.cross(positions-np.array(config['center_of_mass_body_m']), axes)
    torques += fit['moment_constant_m']*signs[:, None]*axes
    allocation = np.vstack((axes[:, 2], torques.T))
    if np.linalg.matrix_rank(allocation) != 4:
        raise ValueError('rotor_geometry_cannot_independently_control_hover_and_three_torques')
    for key in ('wing_area_m2', 'mean_chord_m', 'CLa_per_rad', 'CL_max', 'CD0', 'aspect_ratio',
                'efficiency', 'alpha_stall_rad'):
        number(aero[key], 'aerodynamics.'+key, True)
    for key in ('CL0', 'Cm0', 'Cma_per_rad', 'CLa_stall_per_rad', 'CDa_stall_per_rad', 'Cma_stall_per_rad'):
        number(aero[key], 'aerodynamics.'+key)
    xyz(aero['center_of_pressure_body_m'], 'center_of_pressure_body_m')
    if aero['efficiency'] > 1:
        raise ValueError('aerodynamic_efficiency_must_be_at_most_one')
    density = number(config['air_density_kg_m3'], 'air_density_kg_m3', True)
    speed = number(requested_speed, 'requested_speed', True)
    stall = math.sqrt(2*mass*9.80665/(density*aero['wing_area_m2']*aero['CL_max']))
    thrust_ratio = float(axes[:, 2].sum()*fit['thrust_coefficient']*fit['max_omega_rad_s']**2/(mass*9.80665))
    result.update(data_complete=True, motor_fit=fit, stall_speed_estimate_m_s=stall,
        geometry_scope='supplied_rotor_geometry; not inferred from display-mesh dimensions',
        minimum_forward_speed_with_margin_m_s=1.3*stall, max_static_thrust_to_weight=thrust_ratio,
        requested_forward_speed_m_s=speed,
        forward_speed_above_stall_margin=speed >= 1.3*stall,
        static_hover_thrust_available=thrust_ratio > 1,
        hover_allocation_rank=4,
        limitations=['Stall-speed estimate uses supplied CL_max; does not certify transition.',
                     'Propwash, Reynolds effects, lateral/dynamic derivatives and control tuning require separate validation.',
                     'A source label records provenance; it does not authenticate a measurement.'])
    return result


def load(path):
    config = json.loads(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError('physics_JSON_object_required')
    return config
