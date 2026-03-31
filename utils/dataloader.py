import os
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import albumentations as A
import numpy as np

# =========================
# Utilidades
# =========================
def _to_tensor_chw_uint8_01(img_np):
    """
    Convierte un numpy array HxWxC (uint8) o HxW (uint8) a tensor float en [0,1] con shape CxHxW.
    """
    if img_np.ndim == 2:
        t = torch.from_numpy(img_np).unsqueeze(0).float() / 255.0  # 1xHxW
    elif img_np.ndim == 3:
        t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0  # CxHxW
    else:
        raise ValueError(f"Formato de imagen no soportado con shape {img_np.shape}")
    return t

def _normalize_inplace(tensor, mean, std):
    """
    Normaliza un tensor CxHxW en el lugar con mean/std (tuplas o listas) por canal.
    """
    assert tensor.ndim == 3, "Se esperaba tensor CxHxW"
    c = tensor.shape[0]
    if len(mean) != c or len(std) != c:
        raise ValueError(f"Mean/Std no coinciden con canales: C={c}, mean={mean}, std={std}")
    for ch in range(c):
        tensor[ch] = (tensor[ch] - mean[ch]) / std[ch]
    return tensor

# =========================
# Dataset de Entrenamiento
# =========================
class CODataset(data.Dataset):
    def __init__(
        self,
        image_root,
        thermal_root,
        gt_root,
        trainsize,
        augmentations=True,
        thermal_norm_mean=(0.5, 0.5, 0.5),
        thermal_norm_std=(0.5, 0.5, 0.5),
    ):
        self.trainsize = trainsize
        self.augmentations = augmentations
        self.thermal_norm_mean = thermal_norm_mean
        self.thermal_norm_std = thermal_norm_std

        # Contar imágenes originales
        original_images = [f for f in os.listdir(image_root) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        original_gts = [f for f in os.listdir(gt_root) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        original_thermals = [f for f in os.listdir(thermal_root) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        print(f"📊 ESTADÍSTICAS DEL DATASET:")
        print(f" • Imágenes RGB encontradas: {len(original_images)}")
        print(f" • Ground truths encontradas: {len(original_gts)}")
        print(f" • Thermals encontradas: {len(original_thermals)}")

        self.images = sorted([os.path.join(image_root, f) for f in original_images])
        self.gts = sorted([os.path.join(gt_root, f) for f in original_gts])
        self.thermals = sorted([os.path.join(thermal_root, f) for f in original_thermals])

        # Preprocesar y redimensionar UNA SOLA VEZ
        self.preprocess_and_resize()
        self.size = len(self.processed_data)

        print(f"• Pares procesados exitosamente: {self.size}")

        # Mostrar información sobre augmentaciones
        if self.augmentations:
            print(f"• 🔄 AUGMENTACIONES ACTIVADAS:")
            print(f" - Las mismas transformaciones aleatorias para RGB y Thermal")
            print(f" - GT recibe solo geométricas (interpolación de máscara)")
        else:
            print(f"• ❌ AUGMENTACIONES DESACTIVADAS:")
            print(f" - Solo resize y normalización")

        # Configurar transformaciones (geométricas + fotométricas aplicables)
        self.aug_transform = self.get_augmentations()

        # Normalización: parámetros
        self.rgb_mean = (0.485, 0.456, 0.406)
        self.rgb_std = (0.229, 0.224, 0.225)

    def preprocess_and_resize(self):
        """
        Preprocesa y redimensiona las imágenes UNA SOLA VEZ al inicializar:
        - RGB como 3 canales.
        - Thermal como 3 canales (RGB).
        - GT como 1 canal (L) y redimensionada con NEAREST.
        """
        self.processed_data = []
        resized_count = 0
        error_count = 0

        for img_path, thermal_path, gt_path in zip(self.images, self.thermals, self.gts):
            try:
                # Cargar imagen, térmica y máscara
                image = Image.open(img_path).convert('RGB')     # 3 canales
                thermal = Image.open(thermal_path).convert('RGB')  # 3 canales (solicitado)
                gt = Image.open(gt_path).convert('L')           # 1 canal (máscara binaria)

                # Verificar y alinear tamaños al de la RGB
                if (image.size != gt.size) or (image.size != thermal.size):
                    resized_count += 1
                    target_size = image.size  # tamaño de referencia

                    print(f" 📏 Redimensionando a {target_size}: "
                          f"{os.path.basename(gt_path)} {gt.size} → {target_size}, "
                          f"{os.path.basename(thermal_path)} {thermal.size} → {target_size}")

                    # Máscara: SIEMPRE NEAREST para no crear grises
                    gt = gt.resize(target_size, Image.NEAREST)
                    # Thermal: bilineal/bicúbica
                    thermal = thermal.resize(target_size, Image.BICUBIC)

                # Convertir a numpy arrays y guardar
                image_np = np.array(image)     # HxWx3
                thermal_np = np.array(thermal) # HxWx3
                gt_np = np.array(gt)           # HxW

                self.processed_data.append((image_np, thermal_np, gt_np))

            except Exception as e:
                error_count += 1
                print(f" ❌ Error procesando {os.path.basename(img_path)}: {str(e)}")

        print(f"\n📈 RESUMEN DEL PREPROCESAMIENTO:")
        print(f" • Pares procesados: {len(self.processed_data)}")
        print(f" • Pares redimensionados: {resized_count}")
        print(f" • Errores encontrados: {error_count}")
        print(f" • ✅ Todas las imágenes están listas para augmentación")

    def get_augmentations(self):
        """
        Devuelve un A.Compose con augmentaciones geométricas y fotométricas
        sincronizadas entre RGB y Thermal. La GT solo recibe transformaciones
        geométricas (Albumentations ya maneja la máscara correctamente).
        """
        if self.augmentations:
            print('\n🎨 CONFIGURANDO AUGMENTACIONES UNIFICADAS (RGB y Thermal):')
            transform = A.Compose(
                [
                    # Aseguramos tamaño final exacto desde el inicio.
                    A.Resize(self.trainsize, self.trainsize),

                    # Flips y rotaciones sencillas.
                    A.OneOf([
                        A.HorizontalFlip(p=1.0),
                        A.VerticalFlip(p=1.0),
                        A.RandomRotate90(p=1.0),
                    ], p=0.7),

                    # Afinaciones ligeras (rotación, escala, traslación, shear).
                    A.Affine(
                        scale=(0.85, 1.15),
                        translate_percent=(-0.1, 0.1),
                        rotate=(-20, 20),
                        shear=(-10, 10),
                        p=0.5,
                        #cval=0,
                    ),

                    # Distorsiones suaves para variar la geometría.
                    A.OneOf([
                        A.ElasticTransform(alpha=40, sigma=6, approximate=True, p=1.0), #alpha_affine=8, 
                        A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                        A.OpticalDistortion(distort_limit=0.35, p=1.0), #shift_limit=0.03,
                    ], p=0.3),

                    # Cambios de brillo/contraste, gamma o CLAHE (fotometría).
                    A.OneOf([
                        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
                        A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),
                    ], p=0.6),

                    # Ruido multiplicativo o gaussiano.
                    A.OneOf([
                        A.GaussNoise(p=1.0), #var_limit=(10.0, 60.0), 
                        A.MultiplicativeNoise(multiplier=(0.95, 1.05), per_channel=True, p=1.0),
                    ], p=0.4),

                    # Blur ligero (simula desenfoques o movimiento).
                    A.OneOf([
                        A.MotionBlur(blur_limit=3, p=1.0),
                        A.MedianBlur(blur_limit=3, p=1.0),
                    ], p=0.2),
                ],
                additional_targets={
                    'thermal': 'image',
                    'mask': 'mask',
                }
            )
            print(' ✅ Augmentaciones configuradas (sin Normalize/ToTensor):')
            print(' - Geométricas: flips, rotación, affine, distorsiones')
            print(' - Oclusiones: CoarseDropout')
            print(' - Fotométricas: brillo/contraste, gamma/CLAHE, ruido, blur')
            return transform
        else:
            print('\n➡️  CONFIGURANDO TRANSFORMACIONES BÁSICAS (solo Resize):')
            transform = A.Compose(
                [
                    A.Resize(self.trainsize, self.trainsize),
                ],
                additional_targets={
                    'thermal': 'image',
                    'mask': 'mask',
                }
            )
            print(' ✅ Solo resize configurado (sin fotométricas)')
            return transform
            
    def get_augmentations_ant(self):
        """
        Devuelve un A.Compose que aplica:
        - Geométricas a image/thermal/mask de forma sincronizada.
        - Fotométricas aplicables (brillo/contraste y ruido) a image y thermal.
        NOTA: No incluimos Normalize/ToTensor aquí para poder normalizar
        diferente RGB vs Thermal más adelante.
        """
        if self.augmentations:
            print('\n🎨 CONFIGURANDO AUGMENTACIONES UNIFICADAS (RGB y Thermal):')
            transform = A.Compose(
                [
                    A.Resize(self.trainsize, self.trainsize),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    #A.ShiftScaleRotate(
                    #    shift_limit=0.1,
                    #    scale_limit=0.1,
                    #    rotate_limit=45,
                    #    p=0.5,
                    #    border_mode=0,   # cv2.BORDER_CONSTANT
                    #    value=0,         # relleno para image/thermal
                    #    mask_value=0     # relleno para máscaras
                    #),
                    # Fotométricas aplicables a RGB y Thermal (evitamos HSV para térmica)
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2,
                        p=0.5
                    ),
                    #A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                    A.GaussNoise(p=0.3),
                ],
                additional_targets={
                    'thermal': 'image',  # aplica mismo pipeline de "image" a thermal
                    'mask': 'mask',      # aplica pipeline de máscara
                }
            )
            print(' ✅ Augmentaciones configuradas (sin Normalize/ToTensor):')
            print(' - Geométricas: Flip H/V, Rotación 90°, Shift/Scale/Rotate')
            print(' - Fotométricas: Brillo/Contraste, Ruido Gaussiano (también a Thermal)')
            return transform
        else:
            print('\n➡️  CONFIGURANDO TRANSFORMACIONES BÁSICAS (solo Resize):')
            transform = A.Compose(
                [
                    A.Resize(self.trainsize, self.trainsize),
                ],
                additional_targets={
                    'thermal': 'image',
                    'mask': 'mask',
                }
            )
            print(' ✅ Solo resize configurado (sin fotométricas)')
            return transform

    def __getitem__(self, index):
        # Obtener datos preprocesados (ya redimensionados una vez)
        image_np, thermal_np, gt_np = self.processed_data[index]

        # 1) Augmentaciones (geométricas + fotométricas aplicables) para los tres
        aug = self.aug_transform(image=image_np, thermal=thermal_np, mask=gt_np)
        image_np = aug['image']     # HxWx3, uint8
        thermal_np = aug['thermal'] # HxWx3, uint8
        gt_np = aug['mask']         # HxW, uint8 (0/255)

        # 2) Convertir a tensores en [0,1]
        image = _to_tensor_chw_uint8_01(image_np)     # 3xHxW
        thermal = _to_tensor_chw_uint8_01(thermal_np) # 3xHxW
        gt = torch.from_numpy(gt_np).unsqueeze(0).float() / 255.0  # 1xHxW (0/1)

        # 3) Normalización (separada)
        _normalize_inplace(image, self.rgb_mean, self.rgb_std)
        _normalize_inplace(thermal, self.thermal_norm_mean, self.thermal_norm_std)

        return image, thermal, gt

    def __len__(self):
        return self.size


def get_loader(
    image_root,
    thermal_root,
    gt_root,
    batchsize,
    trainsize,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    augmentation=True,
    thermal_norm_mean=(0.5, 0.5, 0.5),
    thermal_norm_std=(0.5, 0.5, 0.5),
):
    print(f"\n🚀 CREANDO DATALOADER:")
    print(f" • Directorio de imágenes: {image_root}")
    print(f" • Directorio de thermals: {thermal_root}")
    print(f" • Directorio de ground truths: {gt_root}")
    print(f" • Tamaño de entrenamiento: {trainsize}x{trainsize}")
    print(f" • Batch size: {batchsize}")
    print(f" • Augmentaciones: {'✅ Activadas' if augmentation else '❌ Desactivadas'}")

    dataset = CODataset(
        image_root,
        thermal_root,
        gt_root,
        trainsize,
        augmentations=augmentation,
        thermal_norm_mean=thermal_norm_mean,
        thermal_norm_std=thermal_norm_std,
    )

    # Calcular información adicional del entrenamiento
    total_batches = len(dataset) // batchsize
    remaining_samples = len(dataset) % batchsize

    print(f"\n📈 INFORMACIÓN DE ENTRENAMIENTO:")
    print(f" • Total de imágenes para entrenamiento: {len(dataset)}")
    print(f" • Batches por época: {total_batches}")
    if remaining_samples > 0:
        print(f" • Muestras en el último batch: {remaining_samples}")

    if augmentation:
        print(f" • 🎲 Variaciones por época: INFINITAS (transformaciones aleatorias sincronizadas)")
    else:
        print(f" • 📊 Muestras fijas por época: {len(dataset)}")

    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return data_loader

# =========================
# Dataset de Prueba
# =========================
class test_dataset:
    """
    Devuelve tensores consistentes:
    - image: 1x3xHxW (normalizada ImageNet)
    - thermal: 1x3xHxW (normalizada con mean/std térmico)
    - gt: 1xHxW (0/1)
    """
    def __init__(self, image_root, thermal_root, gt_root, testsize,
                 thermal_norm_mean=(0.5, 0.5, 0.5),
                 thermal_norm_std=(0.5, 0.5, 0.5)):
        self.testsize = testsize
        self.thermal_norm_mean = thermal_norm_mean
        self.thermal_norm_std = thermal_norm_std

        self.images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root)
                              if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        self.gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root)
                           if f.lower().endswith(('.tif', '.png', '.jpg', '.jpeg'))])
        self.thermals = sorted([os.path.join(thermal_root, f) for f in os.listdir(thermal_root)
                                if f.lower().endswith(('.tif', '.png', '.jpg', '.jpeg'))])

        self.rgb_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),  # [0,1]
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
        self.thermal_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),  # bilinear
            transforms.ToTensor(),  # [0,1]
            transforms.Normalize(list(self.thermal_norm_mean), list(self.thermal_norm_std)),
        ])
        self.gt_resize = transforms.Resize((self.testsize, self.testsize), interpolation=Image.NEAREST)

        self.size = min(len(self.images), len(self.gts), len(self.thermals))
        self.index = 0

    def load_data(self):
        image_pil = self.rgb_loader(self.images[self.index])       # RGB
        gt_pil = self.binary_loader(self.gts[self.index])          # L
        thermal_pil = self.rgb_loader(self.thermals[self.index])   # RGB (3 canales)

        # Resize
        image = self.rgb_transform(image_pil).unsqueeze(0)         # 1x3xHxW
        gt = torch.from_numpy(np.array(self.gt_resize(gt_pil))).unsqueeze(0).unsqueeze(0).float() / 255.0  # 1x1xHxW
        thermal = self.thermal_transform(thermal_pil).unsqueeze(0) # 1x3xHxW

        # Nombre
        name = os.path.basename(self.images[self.index])
        if name.lower().endswith('.jpg'):
            name = name[:-4] + '.png'
        self.index += 1
        return image, thermal, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


class My_test_dataset:
    """
    Variante que retorna en el mismo orden (image, thermal, gt, name) y mismos tipos que test_dataset.
    """
    def __init__(self, image_root, thermal_root, gt_root, testsize,
                 thermal_norm_mean=(0.5, 0.5, 0.5),
                 thermal_norm_std=(0.5, 0.5, 0.5)):
        self.testsize = testsize
        self.thermal_norm_mean = thermal_norm_mean
        self.thermal_norm_std = thermal_norm_std

        self.images = sorted([os.path.join(image_root, f) for f in os.listdir(image_root)
                              if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        self.gts = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root)
                           if f.lower().endswith(('.tif', '.png', '.jpg', '.jpeg'))])
        self.thermals = sorted([os.path.join(thermal_root, f) for f in os.listdir(thermal_root)
                                if f.lower().endswith(('.tif', '.png', '.jpg', '.jpeg'))])

        self.rgb_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.thermal_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize(list(self.thermal_norm_mean), list(self.thermal_norm_std))])
        self.gt_resize = transforms.Resize((self.testsize, self.testsize), interpolation=Image.NEAREST)

        self.size = min(len(self.images), len(self.gts), len(self.thermals))
        self.index = 0

    def load_data(self):
        image_pil = self.rgb_loader(self.images[self.index])
        thermal_pil = self.rgb_loader(self.thermals[self.index])   # RGB
        gt_pil = self.binary_loader(self.gts[self.index])          # L

        image = self.rgb_transform(image_pil).unsqueeze(0)         # 1x3xHxW
        thermal = self.thermal_transform(thermal_pil).unsqueeze(0) # 1x3xHxW
        gt = torch.from_numpy(np.array(self.gt_resize(gt_pil))).unsqueeze(0).unsqueeze(0).float() / 255.0  # 1x1xHxW

        name = os.path.basename(self.images[self.index])
        if name.lower().endswith('.jpg'):
            name = name[:-4] + '.png'
        self.index += 1
        return image, thermal, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size


# =========================
# Utilidades de Resumen y Verificación
# =========================
def show_dataset_summary(train_loader, test_dataset=None):
    """
    Muestra un resumen completo del dataset con información de augmentaciones
    """
    print(f"\n" + "="*70)
    print(f"📋 RESUMEN COMPLETO DEL DATASET")
    print(f"="*70)

    # Información del dataset de entrenamiento
    train_dataset = train_loader.dataset
    print(f"🏋️  ENTRENAMIENTO:")
    print(f"   • Imágenes base: {len(train_dataset)}")
    print(f"   • Batch size: {train_loader.batch_size}")
    print(f"   • Batches por época: {len(train_loader)}")
    print(f"   • Preprocesamiento: ✅ Una sola vez al inicializar")
    print(f"   • Thermal como 3 canales: ✅")

    if train_dataset.augmentations:
        print(f"   • Augmentaciones: ✅ ACTIVADAS")
        print(f"     - Idénticas (sincronizadas) para RGB y Thermal")
        print(f"     - GT solo geométricas")
    else:
        print(f"   • Augmentaciones: ❌ DESACTIVADAS")
        print(f"     - Imágenes fijas por época: {len(train_dataset)}")

    if test_dataset:
        print(f"\n🧪 PRUEBA:")
        print(f"   • Imágenes de prueba: {len(test_dataset)}")
        print(f"   • Normalización: RGB (ImageNet), Thermal (media 0.5, std 0.5)")

    print(f"\n💾 CONFIGURACIÓN:")
    print(f"   • Tamaño de imagen: {train_dataset.trainsize}x{train_dataset.trainsize}")
    print(f"   • Shuffle: {'✅' if train_loader.sampler is None else '❌'}")
    print(f"   • Num workers: {train_loader.num_workers}")
    print(f"   • Pin memory: {'✅' if train_loader.pin_memory else '❌'}")
    print(f"="*70)

def demonstrate_augmentations(dataset, num_samples=3):
    """
    Demuestra cómo las augmentaciones crean diferentes versiones de la misma imagen
    """
    if not dataset.augmentations:
        print("❌ Las augmentaciones están desactivadas")
        return

    print(f"\n🎨 DEMOSTRACIÓN DE AUGMENTACIONES:")
    print(f"Mostrando {num_samples} transformaciones de la primera imagen...")

    for i in range(num_samples):
        image, thermal, gt = dataset[0]  # Siempre la misma imagen base
        print(f" Muestra {i+1}: image {tuple(image.shape)}, thermal {tuple(thermal.shape)}, gt {tuple(gt.shape)}")
        print(f" - Min/Max image: {float(image.min()):.3f}/{float(image.max()):.3f}")
        print(f" - Min/Max thermal: {float(thermal.min()):.3f}/{float(thermal.max()):.3f}")

    print("✅ Cada llamada produce una transformación diferente (sincronizadas entre RGB y Thermal)!")

def verify_dataset_integrity(image_root, thermal_root, gt_root):
    """
    Verifica la integridad del dataset antes del preprocesamiento
    """
    print(f"\n🔍 VERIFICANDO INTEGRIDAD DEL DATASET...")

    images = sorted([f for f in os.listdir(image_root) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    gts = sorted([f for f in os.listdir(gt_root) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif'))])
    thermals = sorted([f for f in os.listdir(thermal_root) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif'))])

    size_stats = {}
    mismatches = 0

    for img_name, thermal_name, gt_name in zip(images, thermals, gts):
        try:
            img = Image.open(os.path.join(image_root, img_name)).convert('RGB')
            gt = Image.open(os.path.join(gt_root, gt_name)).convert('L')
            thermal = Image.open(os.path.join(thermal_root, thermal_name)).convert('RGB')

            img_size = img.size
            gt_size = gt.size
            thermal_size = thermal.size

            if (img_size != gt_size) or (img_size != thermal_size):
                mismatches += 1

            # Estadísticas de tamaños
            if img_size not in size_stats:
                size_stats[img_size] = 0
            size_stats[img_size] += 1

        except Exception as e:
            print(f" ❌ Error con {img_name}: {e}")

    print(f"\n📊 ESTADÍSTICAS DE TAMAÑOS:")
    for size, count in sorted(size_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"   • {size}: {count} imágenes RGB (y supuestamente alineadas)")

    print(f"\n📈 RESUMEN:")
    print(f"   • Total de pares (por zip): {min(len(images), len(thermals), len(gts))}")
    print(f"   • Pares con tamaños diferentes: {mismatches}")
    print(f"   • 🔧 Se redimensionarán térmicas y GT UNA SOLA VEZ al tamaño RGB")

    return min(len(images), len(thermals), len(gts)), mismatches