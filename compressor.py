from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def compress(original_image, rank):
    mypic = np.array(original_image)  # Convert original image to numpy array
    mypic_red = mypic[:, :, 0]
    mypic_green = mypic[:, :, 1]
    mypic_blue = mypic[:, :, 2]
    rank = int(rank)
    U_red, s_red, V_red = np.linalg.svd(mypic_red)
    U_green, s_green, V_green = np.linalg.svd(mypic_green)
    U_blue, s_blue, V_blue = np.linalg.svd(mypic_blue)
    
    compress_red = np.round(U_red[:, :rank] @ np.diag(s_red[:rank]) @ V_red[:rank, :]).astype(int)
    compress_green = np.round(U_green[:, :rank] @ np.diag(s_green[:rank]) @ V_green[:rank, :]).astype(int)
    compress_blue = np.round(U_blue[:, :rank] @ np.diag(s_blue[:rank]) @ V_blue[:rank, :]).astype(int)
    
    compress_red = np.clip(compress_red, 0, 255).astype(int)
    compress_green = np.clip(compress_green, 0, 255).astype(int)
    compress_blue = np.clip(compress_blue, 0, 255).astype(int)



    compressed_array = np.stack((compress_red, compress_green, compress_blue), axis=2)
    
    rows, cols = np.shape(mypic_red)
    compression_ratio = 100 * (1 - (rank * (rows + cols + rank) / (rows * cols)))
    print("Compression percent", compression_ratio)
    
    # Convert numpy array to matplotlib image
    compressed_image = Image.fromarray(compressed_array.astype(np.uint8))
    # plt.imshow(compressed_image)
    # plt.show()
    compressed_image.save(f"temp.png")
    return compressed_image
