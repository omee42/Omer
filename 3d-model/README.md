# Opel Astra – 3D Baskı Maketi (çalışma sürüyor)

Bu klasördeki eski, kutu biçimli model kaldırıldı.

Şu an iki aşama var:

1. **Ara sürüm – Astra L (2022+) taslağı:** `astra_olustur.py`.
   Gerçek ölçülerle (4374 × 1860 × 1470 mm, aks mesafesi 2675 mm) 1:24 ölçekte,
   SDF + marching cubes yöntemiyle pürüzsüz bir gövde üretir.
   Görseller: `onizleme.png` (genel), `detaylar.png` (yakın plan).
   STL'leri yeniden üretmek için:

   ```bash
   pip install numpy scipy scikit-image trimesh manifold3d pyfqmr
   python3 astra_olustur.py            # astra_tek_parca.stl, astra_govde.stl, astra_teker.stl
   ```

2. **Hedef – Astra J GTC (2011–2018, 3 kapı / OPC):** gerçek ölçüler ve teknik çizimler
   (blueprint) toplanıyor. Model bu çizimlerdeki siluetlere oturtulacak ve bu sürümün yerini alacak.

Marka logosu ve yazılar modele işlenmez; rozet yerleri boş halka olarak bırakılır.
