import random

def show_before_after(path):
    """
    Menampilkan:
    - Citra asli (RGB)
    - Citra grayscale
    - Hasil segmentasi Otsu
    """
    img = cv2.imread(path)
    if img is None:
        print("Gagal membaca citra:", path)
        return

    rgb_norm, gray = preprocess_image(img)
    otsu_bin = segment_otsu(gray)

    plt.figure(figsize=(9, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(otsu_bin, cmap="gray")
    plt.title("Otsu Segmentation")
    plt.axis("off")

    plt.suptitle(os.path.basename(path))
    plt.tight_layout()
    plt.show()


print("Contoh visualisasi before-after per kelas:\n")
for folder_name in CLASS_MAP.keys():
    folder_path = os.path.join(BASE_DIR, folder_name)
    if not os.path.isdir(folder_path):
        continue
    files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        continue
    sample_path = os.path.join(folder_path, random.choice(files))
    print(f"Kelas: {folder_name}, contoh file: {sample_path}")
    show_before_after(sample_path)
