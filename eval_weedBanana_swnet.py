import os
import cv2
import numpy as np
from tqdm import tqdm
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

dataset = 'WeedBanana'
tecnique_param = 'swnet'

if tecnique_param == 'basnet':
    tecnique = 'BASNet-master'    
    method = f'BASNet_{dataset}'
elif tecnique_param == 'dgnet':
    tecnique = 'DGNet-main'  
    method = f'DGNet_{dataset}'  
elif tecnique_param == 'bgnet':
    tecnique = 'BGNet-master'  
    method = f'BGNet_{dataset}'  
elif tecnique_param == 'hitnet':
    tecnique = 'HitNet-main'  
    method = f'Hitnet_{dataset}'  
elif tecnique_param == 'sinet-v2':
    tecnique = 'SINet-V2-main'  
    method = f'SINet-V2_{dataset}'  
elif tecnique_param == 'pcnet':
    tecnique = 'PlantCamo-main'  
    method = f'PCNet_{dataset}'  
elif tecnique_param == 'c2fnet':
    tecnique = 'C2FNet-master'  
    method = f'C2FNet_{dataset}'  
elif tecnique_param == 'ocenet':
    tecnique = 'OCENet-main'  
    method = f'OCENet_{dataset}'
elif tecnique_param == 'eamnet':
    tecnique = 'EAMNet-main'  
    method = f'EAMNet_{dataset}'
elif tecnique_param == 'ctf-net':
    tecnique = 'CTF-Net-main'  
    method = f'CTF-Net_{dataset}'
elif tecnique_param == 'arnet':
    tecnique = 'ARNet-main'  
    method = f'ARNet_{dataset}' 
elif tecnique_param == 'chnet':
    tecnique = 'CHNet-main'  
    method = f'CHNet_{dataset}'     
elif tecnique_param == 'arnet-v2':
    tecnique = 'ARNet-v2-main'  
    method = f'ARNet-v2_{dataset}'         
elif tecnique_param == 'swnet':
    tecnique = 'SWNet-main'  
    method = f'SWNet_{dataset}'


def get_file_with_extension(directory, filename_without_ext):  
    #Busca un archivo con cualquier extensión en el directorio dado
    for ext in ['.png', '.jpg', '.jpeg']:  # Añade más extensiones si es necesario  
        potential_file = filename_without_ext + ext  
        if os.path.exists(os.path.join(directory, potential_file)):  
            return potential_file  
    return None  


def resize_to_match(img1, img2):  
    #Redimensiona img2 para que coincida con las dimensiones de img1
    if img1.shape != img2.shape:  
        return cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_NEAREST)  
        
    return img2  
    
def get_metrics_tracker():
    """Retorna un diccionario con una instancia de cada métrica"""
    return {
        "FM": Fmeasure(),
        "WFM": WeightedFmeasure(),
        "SM": Smeasure(),
        "EM": Emeasure(),
        "M": MAE()
    }

def calculate_final_results(trackers):
    """Extrae y formatea los resultados de los trackers"""
    fm = trackers["FM"].get_results()["fm"]
    wfm = trackers["WFM"].get_results()["wfm"]
    sm = trackers["SM"].get_results()["sm"]
    em = trackers["EM"].get_results()["em"]
    mae = trackers["M"].get_results()["mae"]
    
    return {
        "Smeasure": sm,
        "wFmeasure": wfm,
        "MAE": mae,
        "adpEm": em["adp"],
        "meanEm": em["curve"].mean(),
        "maxEm": em["curve"].max(),  
        "adpFm": fm["adp"],
        "meanFm": fm["curve"].mean(),
        "maxFm": fm["curve"].max(),  
    }
    

# Inicializar trackers por duplicado
metrics_with_weeds = get_metrics_tracker()
metrics_no_weeds = get_metrics_tracker()

mask_root = f'C:/Respaldo/Henry/Proyecto Camuflaje/Datasets/{dataset}/test/GT'
#pred_root = f'./{tecnique}/results/{method}/{dataset}/'

if tecnique == 'DGNet-main':  
    pred_root = f'./{tecnique}/lib_pytorch/results/{method}/{dataset}/'  
else:  
    pred_root = f'./{tecnique}/results/{method}/{dataset}/'  
        
mask_names = [os.path.splitext(f)[0] for f in sorted(os.listdir(mask_root))]

count_with = 0
count_no = 0

for base_name in tqdm(mask_names, total=len(mask_names)):
    mask_file = get_file_with_extension(mask_root, base_name)
    pred_file = get_file_with_extension(pred_root, base_name)

    if mask_file and pred_file:
        mask = cv2.imread(os.path.join(mask_root, mask_file), cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(os.path.join(pred_root, pred_file), cv2.IMREAD_GRAYSCALE)
        
        mask = resize_to_match(pred, mask)
        
        # --- LÓGICA DE SEPARACIÓN ---
        # Si el valor máximo es 0, es una imagen "No Weed"
        is_empty = (np.max(mask) == 0)
        current_tracker = metrics_no_weeds if is_empty else metrics_with_weeds
        
        if is_empty: count_no += 1
        else: count_with += 1

        # Actualizar métricas correspondientes
        for m in current_tracker.values():
            m.step(pred=pred, gt=mask)

# Calcular resultados finales
res_with = calculate_final_results(metrics_with_weeds)
res_no = calculate_final_results(metrics_no_weeds)

# --- REPORTE FINAL ---
print(f"\n--- REPORTE PARA {method} ---")
print(f"Imágenes con maleza: {count_with} | Imágenes sin maleza: {count_no}")
print("\n[CON MALEZA]:", res_with)
print("\n[SIN MALEZA (Vacías)]:", res_no)

# Guardar en archivo
with open("eval_detailed_results.txt", "a") as file:
    file.write(f"\nMetodo: {method} | Dataset: {dataset}\n")
    file.write(f"CON MALEZA ({count_with}): {str(res_with)}\n")
    file.write(f"SIN MALEZA ({count_no}): {str(res_no)}\n")
    file.write("-" * 50)
