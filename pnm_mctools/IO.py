import numpy as np
import vtk
import pandas as pd
from xml.etree import ElementTree as ET
from openpnm.io import _parse_filename
from openpnm.utils import NestedDict

_TEMPLATE = """
<?xml version="1.0" ?>
<VTKFile byte_order="LittleEndian" type="PolyData" version="0.1">
    <PolyData>
        <Piece NumberOfLines="0" NumberOfPoints="0">
            <Points>
            </Points>
            <Lines>
            </Lines>
            <PointData>
            </PointData>
            <CellData>
            </CellData>
        </Piece>
    </PolyData>
</VTKFile>
""".strip()


def _array_to_element(name: str, array: np.ndarray, n: int = 1, component_names: list = None):
    r"""
    prepares an xml node for the data set

    Parameters
    ----------
    name: str
        name of the data set
    array: np.ndarray
        the data set
    n: int
        number of components
    component_names: list
        optional list of component names, as default '0', '1' and so on are chosen
    Returns
    -------
    An xml node for printing with vtk
    """
    dtype_map = {
        "int8": "Int8",
        "int16": "Int16",
        "int32": "Int32",
        "int64": "Int64",
        "uint8": "UInt8",
        "uint16": "UInt16",
        "uint32": "UInt32",
        "uint64": "UInt64",
        "float32": "Float32",
        "float64": "Float64",
        "str": "String",
    }
    element = None
    component_names = [] if component_names is None else component_names
    if str(array.dtype) in dtype_map.keys():
        element = ET.Element("DataArray")
        element.set("Name", name)
        element.set("NumberOfComponents", str(n))
        element.set("type", dtype_map[str(array.dtype)])
        if n > 1:
            for i in range(n):
                comp_key = 'ComponentName' + str(i)
                comp_str = str(i)
                if len(component_names) == n:
                    comp_str = str(component_names[i])
                element.set(comp_key, comp_str)
        element.text = "\t".join(map(str, array.ravel()))
    return element


def network_to_dict(network, additional_data=None, categorize_by=['name'], flatten=False, element=None,
                    delim=' | '):
    r"""
    Returns a single dictionary object containing data from the given
    OpenPNM project, with the keys organized differently depending on
    optional arguments.

    Parameters
    ----------
    network
        An OpenPNM network
    categorize_by : str or list[str]
        Indicates how the dictionaries should be organized.  The list can
        contain any, all or none of the following strings:

        **'object'** : If specified the dictionary keys will be stored
        under a general level corresponding to their type (e.g.
        'network/net_01/pore.all').

        **'name'** : If specified, then the data arrays are additionally
        categorized by their name.  This is enabled by default.

        **'data'** : If specified the data arrays are additionally
        categorized by ``label`` and ``property`` to separate *boolean*
        from *numeric* data.

        **'element'** : If specified the data arrays are
        additionally categorized by ``pore`` and ``throat``, meaning
        that the propnames are no longer prepended by a 'pore.' or
        'throat.'

    Returns
    -------
    A dictionary with the data stored in a hierarchical data structure, the
    actual format of which depends on the arguments to the function.

    """

    if flatten:
        d = {}
    else:
        d = NestedDict(delimiter=delim)

    def build_network_path(obj, key):
        propname = key
        name = ''
        prefix = ''
        datatype = ''
        arr = obj[key]
        if 'object' in categorize_by:
            if hasattr(obj, 'coords'):
                prefix = 'network' + delim
            else:
                prefix = 'phase' + delim
        if 'element' in categorize_by:
            propname = key.replace('.', delim)
        if 'data' in categorize_by:
            if arr.dtype == bool:
                datatype = 'labels' + delim
            else:
                datatype = 'properties' + delim
        if 'name' in categorize_by:
            name = obj.name + delim
        path = prefix + name + datatype + propname
        return path

    def build_data_path(data_set, key: str):
        propname = key
        prefix = 'data' + delim
        if data_set.shape[0] == network.Np:
            location = 'pore' + delim
        else:
            location = 'throat' + delim
        if data_set.dtype == bool:
            datatype = 'labels' + delim
        else:
            datatype = 'properties' + delim
        path = prefix + location + datatype + propname
        return path

    for key in network.props(element=element) + network.labels(element=element):
        path = build_network_path(obj=network, key=key)
        d[path] = network[key]

    if additional_data is not None:
        for key, data in additional_data.items():
            if isinstance(data, np.ndarray):
                path = build_data_path(data_set=data, key=key)
            else:
                data_set = data[0] if isinstance(data[0], np.ndarray) else data[1]
                path = build_data_path(data_set=data_set, key=key)
            d[path] = data

    return d


def network_to_vtk(network, filename: str, additional_data: dict = None, fill_nans=None, fill_infs=None) -> None:
    r"""
    Prints the network and data to a VTK file

    Parameters
    ----------
    network: OpenPNM network
        network with geometrical information
    filename: str
        file name of the VTK file
    additional_data: dict
        additional data for output in form of a dict, where each key value pair has to take
        following forms:
        str : np.ndarray    -> just a regular array
        or
        str : [np.ndarray, [str]]   -> an array with a list of component names
        or
        str : [[str], np.ndarray]   -> an list of component names and an array
    fill_nans
        value to use for filling up NaN values
    fill_infs
        value to use for filling up Inf values

    Notes
    -----
    This function is adapted from OpenPNMs project_to_vtk to provided flexibility
    """
    # Check if any of the phases has time series
    if filename == "":
        raise ValueError("no filename provided!")
    filename = _parse_filename(filename=filename, ext="vtp")

    am = network_to_dict(network=network,
                         additional_data=additional_data,
                         categorize_by=["object", "data"])
    am = pd.json_normalize(am, sep='.').to_dict(orient='records')[0]
    for k in list(am.keys()):
        am[k.replace('.', ' | ')] = am.pop(k)
    key_list = list(sorted(am.keys()))

    points = network["pore.coords"]
    pairs = network["throat.conns"]
    num_points = np.shape(points)[0]
    num_throats = np.shape(pairs)[0]

    root = ET.fromstring(_TEMPLATE)
    piece_node = root.find("PolyData").find("Piece")
    piece_node.set("NumberOfPoints", str(num_points))
    piece_node.set("NumberOfLines", str(num_throats))
    points_node = piece_node.find("Points")
    coords = _array_to_element("coords", points.T.ravel("F"), n=3)
    points_node.append(coords)
    lines_node = piece_node.find("Lines")
    connectivity = _array_to_element("connectivity", pairs)
    lines_node.append(connectivity)
    offsets = _array_to_element("offsets", 2 * np.arange(len(pairs)) + 2)
    lines_node.append(offsets)

    point_data_node = piece_node.find("PointData")
    cell_data_node = piece_node.find("CellData")
    for key in key_list:
        data_set = am[key]
        array = None
        comp_str = None
        if isinstance(data_set, list):
            if len(data_set) != 2:
                raise ValueError('if the names are provided for the components,\
                                  it needs to be provided in the form [data, [names]]')
            array, comp_str = data_set if isinstance(data_set[0], np.ndarray) else [data_set[1], data_set[0]]
            if not isinstance(comp_str, list):
                raise TypeError(f'The component names for {key} are not provided as list')
        else:
            array = data_set

        if array.dtype == bool:
            array = array.astype(int)
        if np.any(np.isnan(array)):
            if fill_nans is None:
                print(key + " has nans," + " will not write to file")
                continue
            else:
                array[np.isnan(array)] = fill_nans
        if np.any(np.isinf(array)):
            if fill_infs is None:
                print(key + " has infs," + " will not write to file")
                continue
            else:
                array[np.isinf(array)] = fill_infs
        if len(array.shape) == 2:
            n = array.shape[1]
        else:
            n = 1
        element = _array_to_element(key, array, n=n, component_names=comp_str)
        if array.shape[0] == num_points:
            point_data_node.append(element)
        elif array.shape[0] == num_throats:
            cell_data_node.append(element)

    tree = ET.ElementTree(root)
    tree.write(filename)

    with open(filename, "r+") as f:
        string = f.read()
        string = string.replace("</DataArray>", "</DataArray>\n\t\t\t")
        f.seek(0)
        # consider adding header: '<?xml version="1.0"?>\n'+
        f.write(string)


def _writeToVTK(particles: vtk.vtkAppendPolyData, out_file: str) -> None:
    r"""
    writes polydata to vtk file

    Parameters
    ----------
    particles: vtk.vtkAppendPolyData
        a list of polydata to be printed
    out_file: str
        file name for the file to generate
    """
    particles.Update()
    writer = vtk.vtkPolyDataWriter()
    writer.SetInputData(particles.GetOutput())
    writer.SetFileName(out_file)
    writer.Update()


def _vtkPolyDataSpheres(coords,
                        radii,
                        quality: int,
                        polydata: vtk.vtkAppendPolyData = None,
                        filter=None) -> vtk.vtkAppendPolyData:
    r"""
    Appends spheres to a VTK AppendPolyData object

    Parameters
    ----------
    coords: np.ndarray
        set of coordinates with size [Np, 3]
    radii: np.ndarray
        array of pore radii of size [Np, 1]
    quality: int
        parameter given to VTK to determine the number of facets on the surface of the sphere
    polydata: vtk.vtkAppendPolyData:
        Data set the spheres are appended to, will be created if not provided
    filter: Callable
        function object that allows filtering of the pores by their coordinates, signature
        of the object needs to be (list[float]) -> bool

    Returns
    -------
    A vtkAppendPolyData object with the provided spheres
    """
    if polydata is None:
        polydata = vtk.vtkAppendPolyData()
    radii_l = np.asarray(radii)

    for i in range(radii_l.size):
        if not (filter(coords[i, :]) if filter is not None else True):
            continue
        sphere = vtk.vtkSphereSource()
        sphere.SetThetaResolution(quality)
        sphere.SetPhiResolution(quality)
        sphere.SetRadius(radii[i])
        sphere.SetCenter(coords[i, 0], coords[i, 1], coords[i, 2])
        sphere.Update()
        polydata.AddInputData(sphere.GetOutput())

    return polydata


def _vtkPolyDataCylinders(coords,
                          conns,
                          radii,
                          quality: int,
                          polydata: vtk.vtkAppendPolyData = None,
                          capping_on: bool = False,
                          filter=None) -> vtk.vtkAppendPolyData:
    r"""
    Appends cylinders to a VTK AppendPolyData object

    Parameters
    ----------
    coords: np.ndarray
        set of coordinates with size [Np, 3]
    conns: np.ndarray
        array of connecting pores [Nt, 2]
    radii: np.ndarray
        array of throat radii of size [Nt, 1]
    quality: int
        parameter given to VTK to determine the number of facets on the surface of the sphere
    polydata: vtk.vtkAppendPolyData:
        Data set the cylinders are appended to, will be created if not provided
    capping_on: bool
        add capping to cylinders
    filter: Callable
        function object that allows filtering of the pores by their coordinates, signature
        of the object needs to be (list[float]) -> bool

    Returns
    -------
    A vtkAppendPolyData object with the provided cylinders
    """
    if polydata is None:
        polydata = vtk.vtkAppendPolyData()
    radii_l = np.asarray(radii)

    if conns.shape[0] != radii_l.shape[0]:
        raise Exception('radii and throats are incompatible')

    for i in range(radii_l.size):
        line = vtk.vtkLineSource()
        p1 = conns[i, 0]
        p2 = conns[i, 1]
        if not (filter(coords[p1, :], coords[p2, :]) if filter is not None else True):
            continue

        line.SetPoint1(coords[p1, 0], coords[p1, 1], coords[p1, 2])
        line.SetPoint2(coords[p2, 0], coords[p2, 1], coords[p2, 2])
        line.SetResolution(1)
        line.Update()

        tubefilter = vtk.vtkTubeFilter()
        tubefilter.SetInputData(line.GetOutput())
        tubefilter.SetRadius(radii_l[i])
        tubefilter.SetNumberOfSides(quality)
        if capping_on:
            tubefilter.CappingOn()
        tubefilter.Update()
        polydata.AddInputData(tubefilter.GetOutput())

    return polydata


def WritePoresToVTK(coords, radii, filename: str, quality: int, filter=None) -> None:
    r"""
    writes pores as spheres to a VTK file

    Parameters
    ----------
    coords: np.ndarray
        set of coordinates with size [Np, 3]
    radii: np.ndarray
        array of pore radii of size [Np, 1]
    filename: str
        name of the file to write to
    quality: int
        parameter given to VTK to determine the number of facets on the surface of the sphere
    filter: Callable
        function object that allows filtering of the pores by their coordinates, signature
        of the object needs to be (list[float]) -> bool

    Notes
    -----
    This is an extremely heavy function and the resulting VTK files are usually quite resource
    intensive for Paraview, use the quality measure with care!
    """
    if coords.shape[0] != radii.shape[0]:
        raise Exception('coordinates and pore incompatible')
    if (len(coords) == 0):
        print('No pores provided for writing, skipping writing of ' + filename)
        return

    all_pores = _vtkPolyDataSpheres(coords, radii=radii, quality=quality, filter=filter)

    _writeToVTK(all_pores, filename)


def WriteThroatsToVTK(coords, conns, radii, filename: str, quality: int, filter=None):
    r"""
    writes throats as cylinders to a VTK file

    Parameters
    ----------
    coords: np.ndarray
        set of coordinates with size [Np, 3]
    conns: np.ndarray
        array of connecting pores [Nt, 2]
    radii: np.ndarray
        array of throat radii of size [Nt, 1]
    filename: str
        name of the file to write to
    quality: int
        parameter given to VTK to determine the number of facets on the surface of the cylinder
    filter: Callable
        function object that allows filtering of the pores by their coordinates, signature
        of the object needs to be (list[float]) -> bool

    Notes
    -----
    This is an extremely heavy function and the resulting VTK files are usually quite resource
    intensive for Paraview, use the quality measure with care!
    """
    if conns.shape[0] != radii.shape[0]:
        raise Exception('radii and throats are incompatible')
    if len(coords) == 0 or len(radii) == 0:
        print('No coordinates provided for writing, skipping writing of ' + filename)
        return

    all_cylinders = _vtkPolyDataCylinders(coords=coords, conns=conns, radii=radii, quality=quality, filter=filter)

    _writeToVTK(all_cylinders, filename)
