import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, CheckButtons
import numpy as np

def read_image(filename,gray=False):
    image = plt.imread(filename)
    if image.dtype == np.uint8:
        image = image / 255.0
    if gray:
        image = image.mean(2)
    return image

def lDFT(image):
    M, N = image.shape
    dft = np.zeros((M,N),dtype=complex)
    
    for k in range(M):
        for l in range(N):
            for m in range(M):
                for n in range(N):
                    dft[k,l] += image[m,n]*np.exp(-2j * np.pi * (k*m/M+l*n/N))
    return dft

def lIDFT(dft):
    M, N = image.shape
    idft = np.zeros((M,N),dtype=complex)
    
    for k in range(M):
        for l in range(N):
            for m in range(M):
                for n in range(N):
                    idft[k,l] += dft[m,n]*np.exp(2j * np.pi * (k*m/M+l*n/N))
    return idft            

def mDFT(image):
    M, N = image.shape
    
    m = np.arange(M).reshape(M,1)
    n = np.arange(N).reshape(N,1)

    exp_m = np.exp(-2j * np.pi * np.dot(m,m.T)/M)
    exp_n = np.exp(-2j * np.pi * np.dot(n,n.T)/N)

    dft = np.dot(exp_m, np.dot(image,exp_n))
    return dft / np.sqrt(M*N)

def mIDFT(dft):
    M, N = dft.shape
    
    m = np.arange(M).reshape(M,1)
    n = np.arange(N).reshape(N,1)

    exp_m = np.exp(2j * np.pi * np.dot(m,m.T)/M)
    exp_n = np.exp(2j * np.pi * np.dot(n,n.T)/N)

    idft = np.dot(exp_m,np.dot(dft,exp_n))
    return idft / np.sqrt(M*N)

def DFT(image):
    c = image.shape
    if len(c) == 2:
        return mDFT(image)
    elif len(c) == 3:
        dft = np.zeros(c, dtype=complex)
        for i in range(c[2]):
            dft[:,:,i] = mDFT(image[:,:,i])
        return dft
    else:
        print("Bild hat nicht die richtige Dimension")

def IDFT(dft):
    c = dft.shape
    if len(c) == 2:
        return mIDFT(dft)
    elif len(c) == 3:
        idft = np.zeros(c, dtype=complex)
        for i in range(c[2]):
            idft[:,:,i] = mIDFT(dft[:,:,i])
        return idft
    else:
        print("DFT hat nicht die richtige Dimension")
    
def high_pass_filter(dft,k):
    M, N = dft.shape[:2]
    dft_filtered = np.zeros_like(dft, dtype=complex)
    
    for m in range(M):
        for n in range(N):
            if np.sqrt((2*min(m, M-m)/M)**2 + (2*min(n, N-n)/N)**2) >= k:
                dft_filtered[m, n] = dft[m, n]
    
    return dft_filtered

def low_pass_filter(dft, k):
    M, N = dft.shape[:2]
    dft_filtered = np.zeros_like(dft, dtype=complex)
    
    for m in range(M):
        for n in range(N):
            if np.sqrt((2*min(m, M-m)/M)**2 + (2*min(n, N-n)/N)**2) <= k:
                dft_filtered[m, n] = dft[m, n]
    
    return dft_filtered

def vertical_filter(dft, k):
    M, N = dft.shape[:2]
    dft_filtered = np.zeros_like(dft)
    
    for m in range(M):
        for n in range(N):
            if 2*min(m, M-m)/M <= k:
                dft_filtered[m, n] = dft[m, n]
    
    return dft_filtered

def horizontal_filter(dft, k):
    M, N = dft.shape[:2]
    dft_filtered = np.zeros_like(dft)
    
    for m in range(M):
        for n in range(N):
            if 2*min(n, N-n)/N <= k:
                dft_filtered[m, n] = dft[m, n]
    
    return dft_filtered

def plot_images(original, dft_image, idft_image):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title('Originalbild')
    if original.ndim == 2:
        plt.imshow(original,cmap='gray')
    else:
        plt.imshow(original)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('DFT Betragsquadrat')
    h,w = dft_image.shape[:2]
    dft_rolled = np.abs(np.roll(dft_image, shift=(h//2,w//2), axis=(0, 1)))
    dft_rolled = np.log1p(dft_rolled)
    dft_norm = (dft_rolled-dft_rolled.min())/(dft_rolled.max()-dft_rolled.min())
    if dft_image.ndim == 2:
        plt.imshow(dft_norm,cmap='gray')
    else:
        plt.imshow(dft_norm)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('IDFT')
    if idft_image.ndim == 2:
        plt.imshow(np.abs(idft_image), cmap='gray')
    else:
        plt.imshow(np.abs(idft_image))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def plot_DFT_with_filter(image):
    log = False
    filters = {f.__name__ : [f, False] for f in [high_pass_filter, low_pass_filter, vertical_filter, horizontal_filter]}
    
    fig, ax = plt.subplots(1,3,figsize=(12,6))
    fig.subplots_adjust(bottom=0.25)    
    
    ax[0].set_title('Originalbild')
    if image.ndim == 2:
        ax[0].imshow(image, cmap='gray')
    else:
        ax[0].imshow(image)
    ax[0].axis('off')

    dft_image = DFT(image)

    h, w = dft_image.shape[:2]
    dft_rolled = np.abs(np.roll(dft_image, shift=(h//2, w//2), axis=(0, 1)))
    dft_norm = (dft_rolled-dft_rolled.min())/(dft_rolled.max()-dft_rolled.min())

    ax[1].set_title('DFT Betragsquadrat')
    if dft_norm.ndim == 2:
        im = ax[1].imshow(dft_norm, cmap='gray')
    else:
        im = ax[1].imshow(dft_norm)
    ax[1].axis('off')

    idft_image = IDFT(dft_image)

    ax[2].set_title('IDFT')
    if idft_image.ndim == 2:
        idft = ax[2].imshow(np.abs(idft_image), cmap='gray')
    else:
        idft = ax[2].imshow(np.abs(idft_image))
    ax[2].axis('off')

    axbox = fig.add_axes([0.55,0.05,0.05,0.075])
    text_box = TextBox(axbox, 'Filter Parameter k = ', initial='0.1')

    def submit(text):
        k = float(text)
        filtered_dft = dft_image
        for filter in filters:
            if filters[filter][1]:
                filtered_dft = filters[filter][0](filtered_dft, k)
        h, w = filtered_dft.shape[:2]
        dft_rolled = np.abs(np.roll(filtered_dft, shift=(h//2, w//2), axis=(0, 1)))
        if log:
            dft_rolled = np.log1p(dft_rolled)
        dft_norm = (dft_rolled-dft_rolled.min())/(dft_rolled.max()-dft_rolled.min())
        im.set_data(dft_norm)
        idft.set_data(np.abs(IDFT(filtered_dft)))
        plt.draw()
    text_box.on_submit(submit)

    checkax = fig.add_axes([0.0, 0.2, 0.12, 0.1])
    logcheck = CheckButtons(
        ax=checkax,
        labels=["Log1P"]
    )
    
    def act_log(val):
        nonlocal log
        log = not log
        submit(text_box.text)
    logcheck.on_clicked(act_log)

    filterax = fig.add_axes([0.0, 0.0, 0.12, 0.2])
    filtercheck = CheckButtons(
        ax = filterax,
        labels = filters.keys()
    )

    def change_filter(label):
        filters[label] = [filters[label][0],not filters[label][1]]
        submit(text_box.text)
    filtercheck.on_clicked(change_filter)

    plt.show()

filename = 'railroad.webp'
#filename = 'Gitter.png'
image = read_image(filename, True)

#dft_image = DFT(image)
#dft_image = low_pass_filter(dft_image,0.5)
#idft_image = IDFT(dft_image)

#plot_images(image, dft_image, idft_image)

plot_DFT_with_filter(image)