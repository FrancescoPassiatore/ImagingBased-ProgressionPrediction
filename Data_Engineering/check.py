import os
import glob
import numpy as np
import shutil
from collections import Counter
import matplotlib.pyplot as plt

# ==========================
# CONFIGURAZIONE
# ==========================
NPY_DIR = r"C:/Users/frank/OneDrive/Desktop/Thesis - Progress Prediction/extracted_npy"
TARGET_N_SLICES = 11  # Numero target di slice per paziente
MIN_REQUIRED_SLICES = 9  # Minimo accettabile prima di replicare

# ==========================
# FUNZIONI DI ANALISI
# ==========================

def analyze_npy_distribution():
    """
    Analizza la distribuzione del numero di file .npy per paziente.
    
    Returns:
        dict: {patient_id: n_files}
    """
    patients = [d for d in os.listdir(NPY_DIR) if os.path.isdir(os.path.join(NPY_DIR, d))]
    
    distribution = {}
    
    for patient_id in patients:
        patient_folder = os.path.join(NPY_DIR, patient_id)
        npy_files = glob.glob(os.path.join(patient_folder, "*.npy"))
        distribution[patient_id] = len(npy_files)
    
    return distribution

def print_distribution_stats(distribution):
    """Stampa statistiche sulla distribuzione."""
    counts = list(distribution.values())
    
    print("\n" + "="*70)
    print("📊 STATISTICHE DISTRIBUZIONE FILE .NPY")
    print("="*70)
    print(f"Totale pazienti: {len(distribution)}")
    print(f"Min slice per paziente: {min(counts)}")
    print(f"Max slice per paziente: {max(counts)}")
    print(f"Media slice per paziente: {np.mean(counts):.2f}")
    print(f"Mediana slice per paziente: {np.median(counts):.0f}")
    print("="*70)
    
    # Conta quanti pazienti hanno N slice
    counter = Counter(counts)
    print("\n📈 DISTRIBUZIONE FREQUENZE:")
    for n_slices in sorted(counter.keys()):
        n_patients = counter[n_slices]
        print(f"  {n_slices} slice: {n_patients} pazienti ({n_patients/len(distribution)*100:.1f}%)")
    
    # Pazienti sotto la soglia
    below_target = [pid for pid, count in distribution.items() if count < TARGET_N_SLICES]
    print(f"\n⚠️  Pazienti con meno di {TARGET_N_SLICES} slice: {len(below_target)} ({len(below_target)/len(distribution)*100:.1f}%)")
    
    return counter

def visualize_distribution(distribution):
    """Crea un istogramma della distribuzione."""
    counts = list(distribution.values())
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Istogramma
    axes[0].hist(counts, bins=range(min(counts), max(counts)+2), 
                 edgecolor='black', alpha=0.7, color='#74a09e')
    axes[0].axvline(TARGET_N_SLICES, color='red', linestyle='--', 
                    linewidth=2, label=f'Target ({TARGET_N_SLICES})')
    axes[0].axvline(MIN_REQUIRED_SLICES, color='orange', linestyle='--', 
                    linewidth=2, label=f'Min Required ({MIN_REQUIRED_SLICES})')
    axes[0].set_xlabel('Numero di slice .npy', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Numero di pazienti', fontsize=12, fontweight='bold')
    axes[0].set_title('Distribuzione numero slice per paziente', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(counts, vert=True)
    axes[1].axhline(TARGET_N_SLICES, color='red', linestyle='--', 
                   linewidth=2, label=f'Target ({TARGET_N_SLICES})')
    axes[1].axhline(MIN_REQUIRED_SLICES, color='orange', linestyle='--', 
                   linewidth=2, label=f'Min Required ({MIN_REQUIRED_SLICES})')
    axes[1].set_ylabel('Numero di slice .npy', fontsize=12, fontweight='bold')
    axes[1].set_title('Box Plot distribuzione slice', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(NPY_DIR, 'distribution_analysis.png'), dpi=150, bbox_inches='tight')
    print(f"\n📊 Grafico salvato in: {os.path.join(NPY_DIR, 'distribution_analysis.png')}")
    plt.show()

# ==========================
# FUNZIONI DI BILANCIAMENTO
# ==========================

def replicate_to_target(patient_folder, current_n, target_n):
    """
    Replica file .npy per raggiungere il target.
    Strategia: replica l'ultima slice (assumendo sia di buona qualità).
    
    Args:
        patient_folder: path della cartella paziente
        current_n: numero attuale di slice
        target_n: numero target di slice
    
    Returns:
        int: numero di repliche create
    """
    npy_files = sorted(glob.glob(os.path.join(patient_folder, "*.npy")))
    
    if current_n >= target_n or not npy_files:
        return 0
    
    # Prendi l'ultima slice (di solito la migliore)
    last_npy = npy_files[-1]
    last_data = np.load(last_npy)
    last_basename = os.path.splitext(os.path.basename(last_npy))[0]
    
    n_replicas_needed = target_n - current_n
    
    for i in range(n_replicas_needed):
        replica_name = f"{last_basename}_replica_{i+1}.npy"
        replica_path = os.path.join(patient_folder, replica_name)
        np.save(replica_path, last_data)
    
    return n_replicas_needed

def balance_all_patients(distribution, target_n=TARGET_N_SLICES, dry_run=False):
    """
    Bilancia tutti i pazienti al numero target di slice.
    
    Args:
        distribution: dict {patient_id: n_files}
        target_n: numero target di slice
        dry_run: se True, mostra solo cosa verrebbe fatto senza eseguire
    
    Returns:
        dict: statistiche sulle operazioni
    """
    patients_to_balance = {pid: count for pid, count in distribution.items() if count < target_n}
    
    if not patients_to_balance:
        print("\n✅ Tutti i pazienti hanno già il numero target di slice!")
        return {'balanced': 0, 'replicas_created': 0}
    
    print(f"\n{'🔍 DRY RUN - ' if dry_run else '🔧 '}BILANCIAMENTO PAZIENTI")
    print("="*70)
    print(f"Pazienti da bilanciare: {len(patients_to_balance)}")
    print(f"Target: {target_n} slice per paziente")
    print("="*70)
    
    total_replicas = 0
    balanced_count = 0
    
    for patient_id, current_count in patients_to_balance.items():
        patient_folder = os.path.join(NPY_DIR, patient_id)
        n_needed = target_n - current_count
        
        if dry_run:
            print(f"  [{patient_id}] {current_count} -> {target_n} (aggiungerebbe {n_needed} repliche)")
        else:
            n_created = replicate_to_target(patient_folder, current_count, target_n)
            total_replicas += n_created
            balanced_count += 1
            print(f"  ✅ [{patient_id}] {current_count} -> {target_n} ({n_created} repliche create)")
    
    print("="*70)
    
    if not dry_run:
        print(f"\n✅ Bilanciamento completato!")
        print(f"   Pazienti bilanciati: {balanced_count}")
        print(f"   Totale repliche create: {total_replicas}")
    
    return {
        'balanced': balanced_count,
        'replicas_created': total_replicas
    }

def verify_balance():
    """Verifica che tutti i pazienti abbiano lo stesso numero di slice."""
    distribution = analyze_npy_distribution()
    counts = list(distribution.values())
    
    if len(set(counts)) == 1:
        print(f"\n✅ VERIFICA OK: Tutti i {len(distribution)} pazienti hanno {counts[0]} slice!")
        return True
    else:
        print(f"\n⚠️  VERIFICA FALLITA: Trovate {len(set(counts))} diverse quantità di slice")
        counter = Counter(counts)
        for n_slices, n_patients in sorted(counter.items()):
            print(f"   {n_slices} slice: {n_patients} pazienti")
        return False

# ==========================
# MAIN EXECUTION
# ==========================

def main(mode='analyze'):
    """
    Modalità disponibili:
    - 'analyze': solo analisi e visualizzazione
    - 'dry_run': mostra cosa verrebbe fatto senza eseguire
    - 'balance': esegue il bilanciamento
    - 'verify': verifica post-bilanciamento
    """
    
    print("🔍 Analisi distribuzione file .npy...")
    distribution = analyze_npy_distribution()
    
    if not distribution:
        print("❌ Nessun paziente trovato!")
        return
    
    # Statistiche e visualizzazione
    print_distribution_stats(distribution)
    visualize_distribution(distribution)
    
    if mode == 'analyze':
        print("\n💡 Per bilanciare i pazienti, esegui: main(mode='balance')")
        print("   Per vedere cosa verrebbe fatto: main(mode='dry_run')")
    
    elif mode == 'dry_run':
        balance_all_patients(distribution, target_n=TARGET_N_SLICES, dry_run=True)
        print("\n💡 Per eseguire il bilanciamento: main(mode='balance')")
    
    elif mode == 'balance':
        # Conferma
        patients_to_balance = sum(1 for count in distribution.values() if count < TARGET_N_SLICES)
        print(f"\n⚠️  Stai per bilanciare {patients_to_balance} pazienti a {TARGET_N_SLICES} slice.")
        
        response = input("Continuare? (yes/no): ").strip().lower()
        if response != 'yes':
            print("❌ Operazione annullata.")
            return
        
        stats = balance_all_patients(distribution, target_n=TARGET_N_SLICES, dry_run=False)
        
        # Verifica post-bilanciamento
        print("\n🔍 Verifica post-bilanciamento...")
        verify_balance()
    
    elif mode == 'verify':
        verify_balance()
    
    else:
        print(f"❌ Modalità '{mode}' non riconosciuta. Usa: 'analyze', 'dry_run', 'balance', o 'verify'")

# ==========================
# ESECUZIONE
# ==========================

if __name__ == "__main__":
    # Prima esegui l'analisi
    main(mode='dry_run')
    
    # Poi puoi decommentare per eseguire il bilanciamento:
    # main(mode='dry_run')  # Prima fai un dry run
    # main(mode='balance')  # Poi esegui il bilanciamento