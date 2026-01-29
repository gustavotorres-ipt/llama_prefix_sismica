import numpy as np
import segyio
from tqdm import tqdm

def carregar_arquivo(nome_arquivo, dims=None):
    if "sgy" in nome_arquivo or "segy" in nome_arquivo:
        return segy2numpy(nome_arquivo)

    elif ".dat" in nome_arquivo:
        volume = np.fromfile(nome_arquivo, dtype=np.single)
        if dims is not None:
            volume = volume.reshape(dims)

    elif ".npy" in nome_arquivo:
        volume = np.load(nome_arquivo)
        if dims is not None:
            volume = volume.reshape(dims)

    else:
        raise Exception("Erro. Formato de arquivo inválido")
    return volume.astype(np.single)


def carregar_segy(filename, dims=None):
    with segyio.open(filename, "r", strict=False) as f:
        # Access the trace data
        trace_data = f.trace.raw[:]

        if dims is not None:
            length = dims[0] * dims[1]
            trace_data = trace_data[:length, :].reshape(dims)

        # Convert trace data to numpy array for further processing
        return np.array(trace_data)


def segy2numpy(filename):
    # with segyio.open(filename, xline=181) as segyfile:
    #     return segyio.tools.cube(segyfile)

    with segyio.open(filename) as f:
        # assemble with dimension order: inline x crossline x depth
        X = len(f.ilines)
        Y = len(f.xlines)
        Z = len(f.samples)
        volume = np.empty((X, Y, Z), dtype=float)

        # load section by section along axis X
        with tqdm(total=X, desc="Loading data volume") as pbar:
            for i,section in enumerate(f.iline):
                volume[i] = section
                pbar.update()
    return volume
