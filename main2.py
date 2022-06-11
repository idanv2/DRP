# CONDA_SUBDIR=osx-arm64 conda create -n ml python=3.9 -c conda-forge
# pip3 install -U --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
def run_p():
    import numpy as np

    import time
    import math

    nx = 20  # number of points in the x direction
    ny = 20  # number of points in the y direction
    xmin, xmax = 0.0, 1.0  # limits in the x direction
    ymin, ymax = 0.0, 1.0
    x = np.linspace(xmin, xmax, nx)
    y  = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    # dt=0.0002
    T = 10
    time_steps = 8000
    dt = T / time_steps  # limits in the y direction
    lx = xmax - xmin  # domain length in the x direction
    ly = ymax - ymin  # domain length in the y direction
    dx = lx / (nx - 1)  # grid spacing in the x direction
    dy = ly / (ny - 1)  # grid spacing in the y direction
    E = np.zeros((nx, ny))  # E_z at t=0
    E_a = np.zeros((nx, ny))
    Hx = np.zeros((nx, ny))  # at t=1/2
    Hy = np.zeros((nx, ny))  # at t=1/2
    k1, k2 = 1., 1.
    c = math.pi * (np.sqrt(k1 ** 2 + k2 ** 2))
    P = math.pi
    for i in range(0, nx):
        for j in range(0, ny):
            Hx[i, j] = math.sin(c * dt / 2) * (
                        -P * k2 * math.sin(P * k1 * dx * i) * math.cos(P * k2 * dy * (j + 1 / 2)) - P * k1 * math.sin(
                    P * k2 * dx * i) * math.cos(P * k1 * dy * (j + 1 / 2)))
            Hy[i, j] = math.sin(c * dt / 2) * (
                        P * k1 * math.cos(P * k1 * dx * i) * math.sin(P * k2 * dy * (j + 1 / 2)) + P * k2 * math.cos(
                    P * k2 * dx * i) * math.sin(P * k1 * dy * (j + 1 / 2)))

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            E[i, j] = c * (math.sin(P * k1 * dx * i) * math.sin(P * k2 * dy * j) + math.sin(P * k2 * dx * i) * math.sin(
                P * k1 * dy * j))

    E[:, -1] = 0
    E[-1, :] = 0

    Z = 1.

    E_tot = E
    Hx_tot = Hx
    Hy_tot = Hy

    #weights = torch.tensor([[-1., 1.]],device=device).unsqueeze(0)  # for 2D add more squieeze

    #weights.requires_grad = True
    #conv = torch.nn.Conv1d(1, 1, kernel_size=(1, 2), padding=0, bias=False)

    start_time = time.time()
    #with torch.no_grad():
    #    conv.weight = torch.nn.Parameter(weights)
    # print(conv.weight.device)
    for n in range(400):
        E[1:nx - 1, 1:ny - 1] = amper(E,Hx,Hy,Z,dt,dx,dy,nx,ny)


        Hx[1:nx - 1, 0:ny - 1] = faraday(E,Hx,Hy,Z,dt,dx,dy,nx,ny)[0]
        Hy[0:nx - 1, 1:ny - 1] = faraday(E,Hx,Hy,Z,dt,dx,dy,nx,ny)[1]
        if n % 100 == 0:
            E_b=c * np.cos(c * (n+1) * dt) * (
                                np.sin(P * k1 * X) * np.sin(P * k2 * Y) + np.sin(
                            P * k2 * X) * np.sin(P * k1 * Y))


            print("{:.6f}".format((np.square(E-E_b)).mean()))



    print("--- %s seconds ---" % (time.time() - start_time))
    #print(torch.has_mps)

    return 1
def amper(E,Hnx,Hny,Z,dt,dx,dy,nx,ny):
    return E[1:nx-1,1:ny-1]+(Z*dt/dx)*(Hny[1:nx-1,1:ny-1]-Hny[0:nx-2,1:ny-1])-(Z*dt/dy)*(Hnx[1:nx-1,1:ny-1]-Hnx[1:nx-1,0:ny-2])
def faraday(E,Hnx,Hny,Z,dt,dx,dy,nx,ny):
    Ax= Hnx[1:nx - 1, 0:ny - 1] - (dt / (Z * dy)) * (E[1:nx - 1, 1:ny] - E[1:nx - 1, 0:ny - 1])
    Ay= Hny[0:nx - 1, 1:ny - 1] + (dt / (Z * dx)) * (E[1:nx, 1:ny - 1] - E[0:nx - 1, 1:ny - 1])
    return [Ax,Ay]

