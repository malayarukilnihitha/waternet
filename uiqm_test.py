import numpy as np
import cv2
from scipy.ndimage import convolve

def calculate_uicm(image):
    # Extract RGB channels
    R, G, B = image[..., 2], image[..., 1], image[..., 0]
    RG = R - G
    YB = 0.5 * (R + G) - B
    # Flatten and sort RG and YB
    RG = np.sort(RG.flatten())
    YB = np.sort(YB.flatten())

    # Trim outliers
    alphaL, alphaR = 0.1, 0.1
    K = R.shape[0] * R.shape[1]
    RG = RG[int(alphaL * K): int(K * (1 - alphaR))]
    YB = YB[int(alphaL * K): int(K * (1 - alphaR))]

    # Calculate mean and standard deviation
    meanRG = np.mean(RG)
    deltaRG = np.sqrt(np.mean((RG - meanRG) ** 2))

    meanYB = np.mean(YB)
    deltaYB = np.sqrt(np.mean((YB - meanYB) ** 2))

    # Compute UICM
    uicm = -0.0268 * np.sqrt(meanRG ** 2 + meanYB ** 2) + 0.1586 * np.sqrt(deltaRG ** 2 + deltaYB ** 2)
    return uicm

    return uicm

def calculate_uism(image, patch_size=5):
    # Sobel kernels
    hx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    hy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # Split the image into RGB channels
    Ir, Ig, Ib = image[..., 0], image[..., 1], image[..., 2]

    # Apply Sobel filter to each channel
    SobelR = np.abs(convolve(Ir, hx, mode='reflect') + convolve(Ir, hy, mode='reflect'))
    SobelG = np.abs(convolve(Ig, hx, mode='reflect') + convolve(Ig, hy, mode='reflect'))
    SobelB = np.abs(convolve(Ib, hx, mode='reflect') + convolve(Ib, hy, mode='reflect'))

    # Resize if necessary to make dimensions divisible by patch size
    m, n = Ir.shape
    if m % patch_size != 0 or n % patch_size != 0:
        new_m = m - m % patch_size + patch_size
        new_n = n - n % patch_size + patch_size
        SobelR = resize(SobelR, (new_m, new_n), mode='reflect', anti_aliasing=True)
        SobelG = resize(SobelG, (new_m, new_n), mode='reflect', anti_aliasing=True)
        SobelB = resize(SobelB, (new_m, new_n), mode='reflect', anti_aliasing=True)
        m, n = new_m, new_n

    # Calculate k1 and k2 (number of patches)
    k1, k2 = m // patch_size, n // patch_size

    # Helper function to calculate EME for a channel
    def calculate_eme(SobelChannel):
        EME = 0
        for i in range(0, m, patch_size):
            for j in range(0, n, patch_size):
                patch = SobelChannel[i:i+patch_size, j:j+patch_size]
                patch_max = patch.max()
                patch_min = patch.min()
                if patch_max != 0 and patch_min != 0:
                    EME += np.log(patch_max / patch_min)
        return 2 / (k1 * k2) * abs(EME)

    # Calculate EME for each channel
    EMER = calculate_eme(SobelR)
    EMEG = calculate_eme(SobelG)
    EMEB = calculate_eme(SobelB)

    # Weighted combination of EME values
    lambdaR, lambdaG, lambdaB = 0.299, 0.587, 0.114
    uism = lambdaR * EMER + lambdaG * EMEG + lambdaB * EMEB

    return uism
def calculate_uiconm(image, patch_size=5):

    # Split the image into RGB channels
    R, G, B = image[..., 0], image[..., 1], image[..., 2]

    # Resize if necessary to make dimensions divisible by patch size
    m, n = R.shape
    if m % patch_size != 0 or n % patch_size != 0:
        new_m = m - m % patch_size + patch_size
        new_n = n - n % patch_size + patch_size
        R = resize(R, (new_m, new_n), mode='reflect', anti_aliasing=True)
        G = resize(G, (new_m, new_n), mode='reflect', anti_aliasing=True)
        B = resize(B, (new_m, new_n), mode='reflect', anti_aliasing=True)
        m, n = new_m, new_n

    # Calculate k1 and k2 (number of patches)
    k1, k2 = m // patch_size, n // patch_size

    # Helper function to calculate EME for a channel
    def calculate_eme(CHANNEL):
        AMEE = 0
        for i in range(0, m, patch_size):
            for j in range(0, n, patch_size):
                patch = CHANNEL[i:i+patch_size, j:j+patch_size]
                patch_max = patch.max()
                patch_min = patch.min()
                if patch_max != 0 and patch_min != 0:
                    AMEE += np.log(patch_max / patch_min)
        return 1 / (k1 * k2) * abs(AMEE)

    # Calculate EME for each channel
    AMEER = calculate_eme(R)
    AMEEG = calculate_eme(G)
    AMEEB = calculate_eme(B)

    # Weighted combination
    uiconm = AMEER +  AMEEG +  AMEEB

    return uiconm
    '''
def calculate_uiconm(image):
    Y = 0.299 * image[..., 2] + 0.587 * image[..., 1] + 0.114 * image[..., 0]
    hist, _ = np.histogram(Y, bins=256, range=(0, 256), density=True)
    p = hist[hist > 0]
    uiconm = -np.sum(p * np.log2(p))
    return uiconm
'''
def calculate_uiqm(image):
    alpha, beta, gamma = 0.0282, 0.2953, 3.5753
    uicm = calculate_uicm(image)
    uism = calculate_uism(image)
    uiconm = calculate_uiconm(image)
    uiqm = alpha * uicm + beta * uism + gamma * uiconm
    return uiqm


