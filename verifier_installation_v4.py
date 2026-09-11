"""À lancer depuis la racine du projet : python verifier_installation_v4.py."""
from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parent
manifest=json.loads((root/'manifest_v4.json').read_text(encoding='utf-8'))
errors=[]
for name,expected in manifest.items():
 p=root/name
 if not p.is_file():errors.append('ABSENT : '+name)
 elif hashlib.sha256(p.read_bytes()).hexdigest()!=expected:errors.append('VERSION DIFFERENTE : '+name)
if errors:
 print('\n'.join(errors));print('Recopier ensemble les fichiers de la livraison V4.');sys.exit(1)
print('OK : les cinq pages et leurs modules correspondent à la même livraison V4.')
