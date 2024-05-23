import openpnm.models as mods


def equal_value(target: any, prop):
    return target[prop]


spheres_and_cylinders = {
    'pore.max_size': {
        'model': mods.geometry.pore_size.largest_sphere,
        'iters': 10,
    },
    'pore.diameter': {
        'model': equal_value,
        'prop': 'pore.max_size',
    },
    'pore.volume': {
        'model': mods.geometry.pore_volume.sphere,
        'pore_diameter': 'pore.diameter',
    },
    'throat.max_size': {
        'model': mods.misc.from_neighbor_pores,
        'mode': 'min',
        'prop': 'pore.diameter',
    },
    'throat.diameter': {
        'model': mods.misc.scaled,
        'factor': 0.5,
        'prop': 'throat.max_size',
    },
    'throat.length': {
        'model': mods.geometry.throat_length.spheres_and_cylinders,
        'pore_diameter': 'pore.diameter',
        'throat_diameter': 'throat.diameter',
    },
    'throat.cross_sectional_area': {
        'model': mods.geometry.throat_cross_sectional_area.cylinder,
        'throat_diameter': 'throat.diameter',
    },
    'throat.total_volume': {
        'model': mods.geometry.throat_volume.cylinder,
        'throat_diameter': 'throat.diameter',
        'throat_length': 'throat.length',
    },
    'throat.lens_volume': {
        'model': mods.geometry.throat_volume.lens,
        'throat_diameter': 'throat.diameter',
        'pore_diameter': 'pore.diameter',
    },
    'throat.volume': {
        'model': mods.misc.difference,
        'props': ['throat.total_volume', 'throat.lens_volume'],
    },
    'throat.diffusive_size_factors': {
        'model': mods.geometry.diffusive_size_factors.spheres_and_cylinders,
        'pore_diameter': 'pore.diameter',
        'throat_diameter': 'throat.diameter',
    },
    'throat.hydraulic_size_factors': {
        'model': mods.geometry.hydraulic_size_factors.spheres_and_cylinders,
        'pore_diameter': 'pore.diameter',
        'throat_diameter': 'throat.diameter',
    },
}